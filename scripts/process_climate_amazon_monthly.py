#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for processing gridded climate data (NetCDF) 
crossed with municipalities of Northern Brazil.

GRANULARITY: MONTHLY

Dataset: Xavier v3.2.3 (BR-DWGD)
Period: 2001-2024
Variables: Tmax, Tmin, pr, RH, ETo, u2, Rs + VPD (calculated)

Auto-optimized based on system resources
"""

import sys
import warnings
from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SYSTEM DETECTION & CONFIGURATION
# =============================================================================

def get_system_config():
    """
    Auto-detect system resources and configure optimal processing settings.
    
    Returns:
        tuple: (num_workers, chunk_config, ram_gb)
    """
    # Detect CPU cores (cap at 8 to avoid I/O bottleneck)
    cpu_count = min(multiprocessing.cpu_count(), 8)
    
    # Detect available RAM
    if HAS_PSUTIL:
        available_ram_gb = psutil.virtual_memory().total / (1024**3)
    else:
        # Fallback: assume conservative 8GB if psutil not available
        available_ram_gb = 8.0
        print("⚠️  psutil not installed. Using conservative defaults.")
        print("   Install with: pip install psutil")
    
    # Configure chunk size based on RAM
    if available_ram_gb >= 16:
        chunk_days = 730  # 2 years per chunk
    elif available_ram_gb >= 8:
        chunk_days = 365  # 1 year per chunk
    else:
        chunk_days = 180  # 6 months per chunk
    
    return cpu_count, {'time': chunk_days}, available_ram_gb

# Get optimal configuration
NUM_WORKERS, CHUNK_CONFIG, RAM_GB = get_system_config()

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Directories (relative to scripts/ directory)
BASE_DIR = Path(__file__).parent.parent
SHAPEFILE_PATH = BASE_DIR / "IBGE_data" / "BR_Municipios_2022" / "BR_Municipios_2022.shp"
NETCDF_DIR_1 = BASE_DIR / "climate_data" / "pr_Tmax_Tmin_NetCDF_Files"
NETCDF_DIR_2 = BASE_DIR / "climate_data" / "ETo_u2_RH_Rs_NetCDF_Files"
OUTPUT_DIR = BASE_DIR / "scripts"
OUTPUT_FILE = OUTPUT_DIR / "Climate_Amazon_North_Monthly_2001-2024.csv"

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
    print("📍 Loading municipality shapefile...")
    
    gdf = gpd.read_file(shapefile_path)
    
    # Check state column name
    state_col = None
    for col in ['SIGLA_UF', 'SIGLA', 'UF', 'sigla_uf']:
        if col in gdf.columns:
            state_col = col
            break
    
    if state_col is None:
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
    bounds = gdf.total_bounds
    return {
        'lon_min': bounds[0] - buffer,
        'lon_max': bounds[2] + buffer,
        'lat_min': bounds[1] - buffer,
        'lat_max': bounds[3] + buffer
    }


def load_clipped_netcdf(filepath: Path, bounds: dict) -> tuple:
    """
    Load a NetCDF file and clip to region of interest.
    
    Parameters:
        filepath: Path to NetCDF file
        bounds: Geographic bounds for clipping
    
    Returns:
        Tuple of (clipped dataset, lon_dim name, lat_dim name)
    """
    ds = xr.open_dataset(filepath, chunks=CHUNK_CONFIG)
    
    # Identify dimensions
    lon_dim = None
    lat_dim = None
    
    for dim in list(ds.dims) + list(ds.coords):
        dim_lower = dim.lower()
        if 'lon' in dim_lower or dim_lower == 'x':
            lon_dim = dim
        elif 'lat' in dim_lower or dim_lower == 'y':
            lat_dim = dim
    
    if lon_dim is None:
        lon_dim = 'longitude' if 'longitude' in ds.dims or 'longitude' in ds.coords else 'lon'
    if lat_dim is None:
        lat_dim = 'latitude' if 'latitude' in ds.dims or 'latitude' in ds.coords else 'lat'
    
    # Clip the dataset
    try:
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
        print(f"⚠️  Clipping error: {e}")
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
        data = ds[var_name].sel(
            **{lon_dim: lon, lat_dim: lat},
            method='nearest'
        ).compute()  # Materialize data in memory
        
        series = data.to_series()
        return series
    except Exception:
        return None


def process_municipality_monthly(mun_code: str, lon: float, lat: float,
                                 datasets: dict, lon_dims: dict, lat_dims: dict) -> pd.DataFrame:
    """
    Process all climate data for a municipality with MONTHLY aggregation.
    
    Parameters:
        mun_code: Municipality IBGE code
        lon: Longitude of centroid
        lat: Latitude of centroid
        datasets: Dictionary of xarray datasets
        lon_dims: Dictionary of longitude dimension names
        lat_dims: Dictionary of latitude dimension names
    
    Returns:
        DataFrame with monthly aggregated data
    """
    try:
        # Extract time series
        tmax = extract_time_series(datasets['Tmax'], lon, lat, 'Tmax', lon_dims['Tmax'], lat_dims['Tmax'])
        tmin = extract_time_series(datasets['Tmin'], lon, lat, 'Tmin', lon_dims['Tmin'], lat_dims['Tmin'])
        pr = extract_time_series(datasets['pr'], lon, lat, 'pr', lon_dims['pr'], lat_dims['pr'])
        rh = extract_time_series(datasets['RH'], lon, lat, 'RH', lon_dims['RH'], lat_dims['RH'])
        eto = extract_time_series(datasets['ETo'], lon, lat, 'ETo', lon_dims['ETo'], lat_dims['ETo'])
        u2 = extract_time_series(datasets['u2'], lon, lat, 'u2', lon_dims['u2'], lat_dims['u2'])
        rs = extract_time_series(datasets['Rs'], lon, lat, 'Rs', lon_dims['Rs'], lat_dims['Rs'])
        
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
        
        # Create year and month columns
        df_daily['year'] = df_daily.index.year
        df_daily['month'] = df_daily.index.month
        
        # Aggregate by YEAR-MONTH
        aggregations = {
            'pr': 'sum',      # Precipitation - monthly sum
            'ETo': 'sum',     # Evapotranspiration - monthly sum
            'Tmax': 'mean',   # Maximum temperature - monthly mean
            'Tmin': 'mean',   # Minimum temperature - monthly mean
            'RH': 'mean',     # Relative humidity - monthly mean
            'VPD': 'mean',    # VPD - monthly mean
            'u2': 'mean',     # Wind speed - monthly mean
            'Rs': 'sum'       # Solar radiation - monthly sum
        }
        
        df_monthly = df_daily.groupby(['year', 'month']).agg(aggregations).reset_index()
        df_monthly['CD_MUN'] = mun_code
        
        return df_monthly
    
    except Exception as e:
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
    print("   📅 Granularity: MONTHLY")
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
    bounds = get_region_bounds(gdf_north)
    print(f"📐 Bounds: lon [{bounds['lon_min']:.2f}, {bounds['lon_max']:.2f}], "
          f"lat [{bounds['lat_min']:.2f}, {bounds['lat_max']:.2f}]")
    print()
    
    # -------------------------------------------------------------------------
    # 3. Load and clip NetCDF datasets
    # -------------------------------------------------------------------------
    print("📂 Loading NetCDF files (optimized for 32GB RAM)...")
    
    datasets = {}
    lon_dims = {}
    lat_dims = {}
    
    for var, filepath in tqdm(NETCDF_FILES.items(), desc="Loading NetCDFs"):
        ds, lon_dim, lat_dim = load_clipped_netcdf(filepath, bounds)
        datasets[var] = ds
        lon_dims[var] = lon_dim
        lat_dims[var] = lat_dim
    
    print("✅ All datasets loaded!")
    print()
    
    # -------------------------------------------------------------------------
    # 4. Process each municipality
    # -------------------------------------------------------------------------
    print("🔄 Processing municipalities (monthly aggregation)...")
    
    results = []
    errors = []
    
    mun_code_col = 'CD_MUN' if 'CD_MUN' in gdf_north.columns else 'codigo_mun'
    mun_name_col = 'NM_MUN' if 'NM_MUN' in gdf_north.columns else 'nome_mun'
    
    for idx, row in tqdm(gdf_north.iterrows(), total=len(gdf_north), desc="Municipalities"):
        mun_code = str(row[mun_code_col])
        lon = row['lon']
        lat = row['lat']
        
        try:
            df_mun = process_municipality_monthly(
                mun_code, lon, lat, datasets, lon_dims, lat_dims
            )
            
            if df_mun is not None:
                if mun_name_col in row.index:
                    df_mun['NM_MUN'] = row[mun_name_col]
                
                for col in ['SIGLA_UF', 'SIGLA', 'UF']:
                    if col in row.index:
                        df_mun['UF'] = row[col]
                        break
                
                results.append(df_mun)
            else:
                errors.append(mun_code)
                
        except Exception as e:
            errors.append(mun_code)
    
    # -------------------------------------------------------------------------
    # 5. Consolidate results
    # -------------------------------------------------------------------------
    print()
    print("📊 Consolidating results...")
    
    if results:
        df_final = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        column_order = ['CD_MUN', 'NM_MUN', 'UF', 'year', 'month',
                        'pr', 'ETo', 'Tmax', 'Tmin', 'RH', 'VPD', 'u2', 'Rs']
        existing_columns = [c for c in column_order if c in df_final.columns]
        df_final = df_final[existing_columns]
        
        # Round numeric values
        numeric_columns = ['pr', 'ETo', 'Tmax', 'Tmin', 'RH', 'VPD', 'u2', 'Rs']
        for col in numeric_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].round(2)
        
        # Sort by municipality, year and month
        df_final = df_final.sort_values(['CD_MUN', 'year', 'month']).reset_index(drop=True)
        
        # Save CSV
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print()
        print("=" * 70)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 File saved: {OUTPUT_FILE}")
        print(f"📊 Total records: {len(df_final):,}")
        print(f"🏙️  Municipalities processed: {df_final['CD_MUN'].nunique()}")
        print(f"📅 Period: {df_final['year'].min()}/{df_final['month'].min():02d} - "
              f"{df_final['year'].max()}/{df_final['month'].max():02d}")
        
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
