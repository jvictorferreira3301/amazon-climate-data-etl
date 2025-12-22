#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for processing gridded climate data (NetCDF) 
crossed with municipalities of Northern Brazil.

GRANULARITY: ANNUAL

Dataset: Xavier v3.2.3 (BR-DWGD)
Period: 2001-2024
Variables: Tmax, Tmin, pr, RH, ETo, u2, Rs + VPD (calculated)

Auto-optimized based on available system resources.
"""

import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from tqdm import tqdm

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')


# =============================================================================
# AUTO-DETECT SYSTEM RESOURCES
# =============================================================================

def get_system_config():
    """
    Automatically detect system resources and return optimal configuration.
    
    Returns:
        tuple: (num_workers, chunk_size, ram_gb)
    """
    import multiprocessing
    
    # Detect CPU cores
    cpu_cores = multiprocessing.cpu_count()
    num_workers = max(1, cpu_cores - 1)  # Leave 1 core for system
    
    # Detect available RAM
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 ** 3)
    except ImportError:
        # psutil not available, assume conservative 8GB
        ram_gb = 8.0
    
    # Calculate optimal chunk size based on RAM
    # Conservative: ~100 days per GB of RAM, max 730 (2 years)
    chunk_size = min(730, max(100, int(ram_gb * 50)))
    
    return num_workers, chunk_size, ram_gb


# Auto-configure based on system
NUM_WORKERS, CHUNK_SIZE, RAM_GB = get_system_config()
CHUNK_CONFIG = {'time': CHUNK_SIZE}

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directories (relative to scripts/ directory)
BASE_DIR = Path(__file__).parent.parent
SHAPEFILE_PATH = BASE_DIR / "IBGE_data" / "BR_Municipios_2022" / "BR_Municipios_2022.shp"
NETCDF_DIR_1 = BASE_DIR / "climate_data" / "pr_Tmax_Tmin_NetCDF_Files"
NETCDF_DIR_2 = BASE_DIR / "climate_data" / "ETo_u2_RH_Rs_NetCDF_Files"
OUTPUT_DIR = BASE_DIR / "scripts"
OUTPUT_FILE = OUTPUT_DIR / "Climate_Amazon_North_2001-2024.csv"

# Specific NetCDF files (period 2001-2024)
NETCDF_FILES = {
    'Tmax': NETCDF_DIR_1 / "Tmax_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
    'Tmin': NETCDF_DIR_1 / "Tmin_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
    'pr': NETCDF_DIR_1 / "pr_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
    'RH': NETCDF_DIR_2 / "RH_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
    'ETo': NETCDF_DIR_2 / "ETo_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
    'u2': NETCDF_DIR_2 / "u2_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
    'Rs': NETCDF_DIR_2 / "Rs_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc",
}

# Northern Region States
NORTHERN_STATES = ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_vpd(tmax: np.ndarray, tmin: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Calculate Vapor Pressure Deficit (VPD) in kPa.
    
    Formula:
        Tmean = (Tmax + Tmin) / 2
        es = 0.6108 * exp((17.27 * Tmean) / (Tmean + 237.3))  # Saturation vapor pressure
        ea = es * (RH / 100)  # Actual vapor pressure
        VPD = es - ea
    
    Parameters:
        tmax: Maximum temperature (C)
        tmin: Minimum temperature (C)
        rh: Relative humidity (%)
    
    Returns:
        VPD in kPa
    """
    tmean = (tmax + tmin) / 2.0
    es = 0.6108 * np.exp((17.27 * tmean) / (tmean + 237.3))
    ea = es * (rh / 100.0)
    vpd = es - ea
    return vpd


def load_northern_municipalities(shapefile_path: Path) -> gpd.GeoDataFrame:
    """
    Load municipality shapefile and filter only Northern Region.
    
    Returns:
        GeoDataFrame with Northern Region municipalities and their centroids
    """
    print("Loading municipality shapefile...")
    
    gdf = gpd.read_file(shapefile_path)
    
    # Check state column name (may vary)
    state_col = None
    for col in ['SIGLA_UF', 'SIGLA', 'UF', 'sigla_uf']:
        if col in gdf.columns:
            state_col = col
            break
    
    if state_col is None:
        # Try to extract from municipality code (first 2 digits = state code)
        print("⚠️  State column not found. Using municipality code...")
        # IBGE codes for Northern states
        northern_codes = {
            '12': 'AC', '13': 'AM', '16': 'AP', 
            '15': 'PA', '11': 'RO', '14': 'RR', '17': 'TO'
        }
        gdf['SIGLA_UF'] = gdf['CD_MUN'].astype(str).str[:2].map(northern_codes)
        state_col = 'SIGLA_UF'
    
    # Filter Northern Region
    gdf_north = gdf[gdf[state_col].isin(NORTHERN_STATES)].copy()
    
    print(f"✅ {len(gdf_north)} Northern Region municipalities loaded")
    
    # Reproject to WGS84 if necessary
    if gdf_north.crs is None or gdf_north.crs.to_epsg() != 4326:
        gdf_north = gdf_north.to_crs(epsg=4326)
    
    # Calculate centroids
    gdf_north['centroid'] = gdf_north.geometry.centroid
    gdf_north['lon'] = gdf_north['centroid'].x
    gdf_north['lat'] = gdf_north['centroid'].y
    
    return gdf_north


def get_region_bounds(gdf: gpd.GeoDataFrame, buffer: float = 0.5) -> dict:
    """
    Get geographic bounds of the region with a buffer.
    
    Returns:
        Dictionary with lon_min, lon_max, lat_min, lat_max
    """
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    return {
        'lon_min': bounds[0] - buffer,
        'lon_max': bounds[2] + buffer,
        'lat_min': bounds[1] - buffer,
        'lat_max': bounds[3] + buffer
    }


def load_clipped_netcdf(filepath: Path, bounds: dict, var_name: str = None) -> xr.Dataset:
    """
    Load a NetCDF file and clip to region of interest.
    
    Parameters:
        filepath: Path to NetCDF file
        bounds: Geographic bounds for clipping
        var_name: Variable name (optional)
    
    Returns:
        Clipped xarray Dataset
    """
    ds = xr.open_dataset(filepath, chunks=CHUNK_CONFIG)
    
    # Identify coordinate dimension names
    lon_dim = None
    lat_dim = None
    
    for dim in ds.dims:
        dim_lower = dim.lower()
        if 'lon' in dim_lower or dim_lower == 'x':
            lon_dim = dim
        elif 'lat' in dim_lower or dim_lower == 'y':
            lat_dim = dim
    
    # Also check coordinates
    for coord in ds.coords:
        coord_lower = coord.lower()
        if 'lon' in coord_lower and lon_dim is None:
            lon_dim = coord
        elif 'lat' in coord_lower and lat_dim is None:
            lat_dim = coord
    
    if lon_dim is None or lat_dim is None:
        print(f"⚠️  Dimensions of file {filepath.name}:")
        print(f"   Dims: {list(ds.dims)}")
        print(f"   Coords: {list(ds.coords)}")
        # Use default values
        lon_dim = 'longitude' if 'longitude' in ds.dims or 'longitude' in ds.coords else 'lon'
        lat_dim = 'latitude' if 'latitude' in ds.dims or 'latitude' in ds.coords else 'lat'
    
    # Clip the dataset
    try:
        # Check if latitude is descending
        lat_values = ds[lat_dim].values
        if lat_values[0] > lat_values[-1]:
            # Descending latitude
            ds_clipped = ds.sel(
                **{
                    lon_dim: slice(bounds['lon_min'], bounds['lon_max']),
                    lat_dim: slice(bounds['lat_max'], bounds['lat_min'])
                }
            )
        else:
            # Ascending latitude
            ds_clipped = ds.sel(
                **{
                    lon_dim: slice(bounds['lon_min'], bounds['lon_max']),
                    lat_dim: slice(bounds['lat_min'], bounds['lat_max'])
                }
            )
    except Exception as e:
        print(f"⚠️  Clipping error, loading complete dataset: {e}")
        ds_clipped = ds
    
    return ds_clipped, lon_dim, lat_dim


def extract_time_series(ds: xr.Dataset, lon: float, lat: float, 
                        var_name: str, lon_dim: str, lat_dim: str) -> pd.Series:
    """
    Extract time series for a specific point using nearest neighbor.
    
    Returns:
        Pandas Series with daily values
    """
    try:
        # Extract using nearest method and compute immediately
        data = ds[var_name].sel(
            **{lon_dim: lon, lat_dim: lat},
            method='nearest'
        ).compute()  # Materialize in memory for performance
        
        # Convert to pandas series
        series = data.to_series()
        
        return series
    
    except Exception as e:
        return None


def process_municipality(mun_code: str, lon: float, lat: float,
                         datasets: dict, lon_dims: dict, lat_dims: dict) -> pd.DataFrame:
    """
    Process all climate data for a municipality.
    
    Parameters:
        mun_code: Municipality IBGE code
        lon: Longitude of centroid
        lat: Latitude of centroid
        datasets: Dictionary of xarray datasets
        lon_dims: Dictionary of longitude dimension names
        lat_dims: Dictionary of latitude dimension names
    
    Returns:
        DataFrame with annual aggregated data
    """
    try:
        # Extract time series for each variable
        tmax = extract_time_series(
            datasets['Tmax'], lon, lat, 'Tmax', lon_dims['Tmax'], lat_dims['Tmax']
        )
        tmin = extract_time_series(
            datasets['Tmin'], lon, lat, 'Tmin', lon_dims['Tmin'], lat_dims['Tmin']
        )
        pr = extract_time_series(
            datasets['pr'], lon, lat, 'pr', lon_dims['pr'], lat_dims['pr']
        )
        rh = extract_time_series(
            datasets['RH'], lon, lat, 'RH', lon_dims['RH'], lat_dims['RH']
        )
        eto = extract_time_series(
            datasets['ETo'], lon, lat, 'ETo', lon_dims['ETo'], lat_dims['ETo']
        )
        u2 = extract_time_series(
            datasets['u2'], lon, lat, 'u2', lon_dims['u2'], lat_dims['u2']
        )
        rs = extract_time_series(
            datasets['Rs'], lon, lat, 'Rs', lon_dims['Rs'], lat_dims['Rs']
        )
        
        # Check if all series were extracted
        if any(s is None for s in [tmax, tmin, pr, rh, eto, u2, rs]):
            return None
        
        # Create daily DataFrame
        df_daily = pd.DataFrame({
            'Tmax': tmax.values,
            'Tmin': tmin.values,
            'pr': pr.values,
            'RH': rh.values,
            'ETo': eto.values,
            'u2': u2.values,
            'Rs': rs.values
        }, index=tmax.index)
        
        # Calculate daily VPD
        df_daily['VPD'] = calculate_vpd(
            df_daily['Tmax'].values,
            df_daily['Tmin'].values,
            df_daily['RH'].values
        )
        
        # Extract year
        df_daily['year'] = df_daily.index.year
        
        # Aggregate by year
        aggregations = {
            'pr': 'sum',      # Precipitation - annual sum
            'ETo': 'sum',     # Evapotranspiration - annual sum
            'Tmax': 'mean',   # Maximum temperature - annual mean
            'Tmin': 'mean',   # Minimum temperature - annual mean
            'RH': 'mean',     # Relative humidity - annual mean
            'VPD': 'mean',    # VPD - annual mean
            'u2': 'mean',     # Wind speed - annual mean
            'Rs': 'sum'       # Solar radiation - annual sum
        }
        
        df_annual = df_daily.groupby('year').agg(aggregations).reset_index()
        
        # Add municipality code
        df_annual['CD_MUN'] = mun_code
        
        return df_annual
    
    except Exception as e:
        print(f"⚠️  Error processing municipality {mun_code}: {e}")
        return None


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function of the script.
    Auto-optimized based on system resources.
    """
    print("=" * 70)
    print("🌍 CLIMATE DATA PROCESSING - NORTHERN BRAZIL")
    print("   ⚡ Auto-optimized for your system")
    print("   📅 Granularity: ANNUAL")
    print("=" * 70)
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"💾 Detected RAM: {RAM_GB:.1f} GB")
    print(f"🔧 Configuration: {NUM_WORKERS} workers | Chunks: {CHUNK_CONFIG['time']} days")
    print()
    
    # -------------------------------------------------------------------------
    # 1. Check file existence
    # -------------------------------------------------------------------------
    print("📋 Checking input files...")
    
    if not SHAPEFILE_PATH.exists():
        print(f"❌ Shapefile not found: {SHAPEFILE_PATH}")
        sys.exit(1)
    
    for var, filepath in NETCDF_FILES.items():
        if not filepath.exists():
            print(f"❌ NetCDF file not found: {filepath}")
            sys.exit(1)
    
    print("✅ All input files found!")
    print()
    
    # -------------------------------------------------------------------------
    # 2. Load Northern Region municipalities
    # -------------------------------------------------------------------------
    gdf_north = load_northern_municipalities(SHAPEFILE_PATH)
    
    # Get region bounds
    bounds = get_region_bounds(gdf_north)
    print(f"📐 Region bounds: lon [{bounds['lon_min']:.2f}, {bounds['lon_max']:.2f}], "
          f"lat [{bounds['lat_min']:.2f}, {bounds['lat_max']:.2f}]")
    print()
    
    # -------------------------------------------------------------------------
    # 3. Load and clip NetCDF datasets
    # -------------------------------------------------------------------------
    print("📂 Loading NetCDF files...")
    
    datasets = {}
    lon_dims = {}
    lat_dims = {}
    
    for var, filepath in tqdm(NETCDF_FILES.items(), desc="Loading NetCDFs"):
        ds, lon_dim, lat_dim = load_clipped_netcdf(filepath, bounds, var)
        datasets[var] = ds
        lon_dims[var] = lon_dim
        lat_dims[var] = lat_dim
    
    print("✅ All datasets loaded and clipped!")
    print()
    
    # -------------------------------------------------------------------------
    # 4. Process each municipality (with parallelization)
    # -------------------------------------------------------------------------
    print(f"🔄 Processing municipalities with {NUM_WORKERS} parallel workers...")
    
    results = []
    errors = []
    
    # Identify municipality code column
    mun_code_col = 'CD_MUN' if 'CD_MUN' in gdf_north.columns else 'codigo_mun'
    mun_name_col = 'NM_MUN' if 'NM_MUN' in gdf_north.columns else 'nome_mun'
    
    # Prepare task list
    municipalities_info = []
    for idx, row in gdf_north.iterrows():
        mun_code = str(row[mun_code_col])
        lon = row['lon']
        lat = row['lat']
        mun_name = row[mun_name_col] if mun_name_col in row.index else None
        state = None
        for col in ['SIGLA_UF', 'SIGLA', 'UF']:
            if col in row.index:
                state = row[col]
                break
        municipalities_info.append((mun_code, lon, lat, mun_name, state))
    
    def process_wrapper(info):
        """Wrapper for parallel processing."""
        mun_code, lon, lat, mun_name, state = info
        try:
            df_mun = process_municipality(mun_code, lon, lat, datasets, lon_dims, lat_dims)
            if df_mun is not None:
                df_mun['NM_MUN'] = mun_name
                df_mun['UF'] = state
                return ('ok', df_mun)
            return ('error', mun_code)
        except Exception as e:
            return ('error', mun_code)
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_wrapper, info): info for info in municipalities_info}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            status, result = future.result()
            if status == 'ok':
                results.append(result)
            else:
                errors.append(result)
    
    # -------------------------------------------------------------------------
    # 5. Consolidate results
    # -------------------------------------------------------------------------
    print()
    print("📊 Consolidating results...")
    
    if results:
        df_final = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        column_order = ['CD_MUN', 'NM_MUN', 'UF', 'year', 
                        'pr', 'ETo', 'Tmax', 'Tmin', 'RH', 'VPD', 'u2', 'Rs']
        existing_columns = [c for c in column_order if c in df_final.columns]
        df_final = df_final[existing_columns]
        
        # Round numeric values
        numeric_columns = ['pr', 'ETo', 'Tmax', 'Tmin', 'RH', 'VPD', 'u2', 'Rs']
        for col in numeric_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].round(2)
        
        # Save CSV
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print()
        print("=" * 70)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 File saved: {OUTPUT_FILE}")
        print(f"📊 Total records: {len(df_final):,}")
        print(f"🏙️  Municipalities processed: {df_final['CD_MUN'].nunique()}")
        print(f"📅 Years: {df_final['year'].min()} - {df_final['year'].max()}")
        
        if errors:
            print(f"⚠️  Municipalities with errors: {len(errors)}")
        
        print()
        print("📋 Statistical summary:")
        print(df_final[numeric_columns].describe().round(2))
        
    else:
        print("❌ No results generated!")
    
    # -------------------------------------------------------------------------
    # 6. Close datasets
    # -------------------------------------------------------------------------
    for ds in datasets.values():
        ds.close()
    
    print()
    print("🏁 End of processing.")


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
