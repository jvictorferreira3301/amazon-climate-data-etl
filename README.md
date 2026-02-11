
# 🌍 Brazilian Amazon Climate Data Processing - Xavier Dataset v3.2.3

ETL pipeline for processing gridded climate data (NetCDF) from Xavier v3.2.3 for municipalities in the Legal Amazon (North Region), merging them with the Brazilian municipal mesh. This project was developed to obtain a database for students of the [EC01019 - Probability and Statistics](https://github.com/glaucogoncalves/p-e) course at the [Faculty of Computer and Telecommunications Engineering](https://fct.ufpa.br/) of the [Federal University of Pará](https://ufpa.br/) (UFPA) in the 2025-4 semester.

## 📊 Dataset

**Source:** BR-DWGD - Brazilian Daily Weather Gridded Data  
**Version:** 3.2.3 (UFES/UTEXAS)  
**Period:** 1961-2024 (full period processed and available)  
**Spatial Resolution:** 0.25° × 0.25° (~28 km)  
**Coverage:** Full Brazil

### Climate Variables

- **pr**: Accumulated precipitation (mm)
- **Tmax/Tmin**: Maximum/Minimum temperature (°C)
- **RH**: Relative Humidity (%)
- **u2**: Wind speed at 2m (m/s)
- **Rs**: Solar radiation (MJ/m²/day)
- **ETo**: Reference evapotranspiration (mm)
- **VPD**: Vapor Pressure Deficit (kPa) - calculated

## 🗂️ Project Structure


```
amazon-climate-data-etl/
├── climate_data/
│   ├── pr_Tmax_Tmin_NetCDF_Files/          # Precipitation and temperatures
│   └── ETo_u2_RH_Rs_NetCDF_Files/          # Humidity, wind, radiation, ETo
├── IBGE_data/
│   └── BR_Municipios_2022/                 # IBGE Municipalities Shapefile
├── processed_output_data/
│   ├── Climate_Amazon_North_1961-1980.csv          # Annual data (1961-1980)
│   ├── Climate_Amazon_North_1981-2000.csv          # Annual data (1981-2000)
│   ├── Climate_Amazon_North_2001-2024.csv          # Annual data (2001-2024)
│   ├── Climate_Amazon_North_1961-2024.csv          # Annual data (all years, concatenated)
│   ├── Climate_Amazon_North_Monthly_2001-2024.csv  # Monthly data (all states)
│   └── by_state/                           # Data split by state (UF)
│       ├── Climate_AC_Annual_2001-2024.csv
│       ├── Climate_AC_Monthly_2001-2024.csv
│       ├── Climate_AM_Annual_2001-2024.csv
│       ├── ...
│       └── Climate_TO_Monthly_2001-2024.csv
├── scripts/
│   ├── process_climate_amazon_annual.py    # Annual pipeline
│   ├── process_climate_amazon_monthly.py   # Monthly pipeline
│   ├── split_by_state.py                   # Split by State (UF)
│   └── validation_northern_capitals.ipynb  # Data validation notebook
├── LICENSE
└── README.md
```

## 🚀 Usage

### Quick Start (Prepared Data)

```bash
# 1. Clone repository
git clone https://github.com/jvictorferreira3301/amazon-climate-data-etl.git
cd amazon-climate-data-etl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (see "Original Data" section below)
# Download and structure files...


# 4. Process annual data (full period)
python scripts/process_climate_amazon_annual.py

```

## Original Data

The NetCDF files (~8-10 GB) and IBGE shapefile (~500 MB) are too large for GitHub and are not included in the repository.

### Data Download

**To reproduce the results, you need to:**

1. **Download Xavier v3.2.3 NetCDF data:**
* Access: https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data
* Select period and variables: pr, Tmax, Tmin, RH, ETo, u2, Rs


2. **Download IBGE Shapefile 2022:**
* Access: https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html
* Download: Municipal Meshes 2022 (Malhas Municipais 2022)


3. **Structure the directories:**

```
amazon-climate-data-etl/
├── climate_data/
│   ├── pr_Tmax_Tmin_NetCDF_Files/
│   │   ├── pr_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmax_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmin_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   └── ETo_u2_RH_Rs_NetCDF_Files/
│       ├── ETo_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── RH_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── u2_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       └── Rs_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
└── IBGE_data/
    └── BR_Municipios_2022/
        ├── BR_Municipios_2022.shp
        ├── BR_Municipios_2022.shx
        ├── BR_Municipios_2022.dbf
        ├── BR_Municipios_2022.prj
        └── BR_Municipios_2022.cpg

```
1. **Download Xavier v3.2.3 NetCDF data:**
  * Access: https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data
  * Download **todos os períodos** e variáveis:
    - 1961-1980: pr, Tmax, Tmin, RH, ETo, u2, Rs
    - 1981-2000: pr, Tmax, Tmin, RH, ETo, u2, Rs
    - 2001-2024: pr, Tmax, Tmin, RH, ETo, u2, Rs

2. **Download IBGE Shapefile 2022:**
  * Access: https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html
  * Download: Municipal Meshes 2022 (Malhas Municipais 2022)

3. **Structure the directories:**

```
amazon-climate-data-etl/
├── climate_data/
│   ├── pr_Tmax_Tmin_NetCDF_Files/
│   │   ├── pr_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── pr_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── pr_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmax_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmax_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmax_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmin_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmin_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   │   ├── Tmin_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│   └── ETo_u2_RH_Rs_NetCDF_Files/
│       ├── ETo_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── ETo_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── ETo_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── RH_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── RH_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── RH_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── u2_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── u2_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── u2_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── Rs_19610101_19801231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── Rs_19810101_20001231_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
│       ├── Rs_20010101_20240320_BR-DWGD_UFES_UTEXAS_v_3.2.3.nc
└── IBGE_data/
   └── BR_Municipios_2022/
      ├── BR_Municipios_2022.shp
      ├── BR_Municipios_2022.shx
      ├── BR_Municipios_2022.dbf
      ├── BR_Municipios_2022.prj
      └── BR_Municipios_2022.cpg

```

## 📦 Dependencies

```bash
pip install -r requirements.txt

```

**Or manually:**

```bash
pip install pandas numpy xarray netCDF4 geopandas matplotlib tqdm

```

**Specific requirements:**

* `xarray` with NetCDF support (`netCDF4` or `h5netcdf`)
* `geopandas` for shapefile manipulation
* `matplotlib` for visualizations
* `tqdm` for progress bars

## 🎯 Analyzed Capitals

| Capital | State (UF) | IBGE Code |
| --- | --- | --- |
| Belém | PA | 1501402 |
| Boa Vista | RR | 1400100 |
| Macapá | AP | 1600303 |
| Manaus | AM | 1302603 |
| Palmas | TO | 1721000 |
| Porto Velho | RO | 1100205 |
| Rio Branco | AC | 1200401 |

## 🔬 Validation

Data was validated across three periods using official references and known climate events:

### 1961-1980 — INMET Climatological Normals (1961-1990)

Annual precipitation means from the Xavier grid were compared against INMET climatological normals for **Belém** and **Manaus**. Differences are expected due to the non-overlapping periods and the grid's spatial averaging (~0.25°).

### 1981-2000 — Known Extreme Events

* **El Niño 1983 (Belém):** Verified below-average rainfall during one of the strongest El Niño events.
* **1996 vs 1997 (Manaus):** Confirmed the expected contrast between a La Niña wet year (1996) and the onset of El Niño (1997).

### 2001-2024 — Spot Check (2015, Strong El Niño)

* **Belém:** Difference of -16.7% vs Embrapa/INMET station (expected underestimation due to grid resolution)
* **Manaus:** Difference of +7.9% vs INMET station (excellent agreement for hydrological data)

See full details in the validation notebook: `processed_output_data/validation_northern_capitals.ipynb`.

## 📚 References

**Dataset:** Xavier, A. C., King, C. W., & Scanlon, B. R. (2016). *Daily gridded meteorological variables in Brazil (1980–2013)*. International Journal of Climatology, 36(6), 2644-2659.

**Climatological Normals (1961-1990):** INMET. *Normais Climatológicas do Brasil 1961-1990*. Instituto Nacional de Meteorologia, Brasília, 2009.

**Belém Validation (2015):** PACHÊCO, N. A. et al. *Boletim agrometeorológico de 2015 para Belém, PA*. Embrapa, 2022. [Link](https://www.infoteca.cnptia.embrapa.br/infoteca/handle/doc/1148466)

**Manaus Validation (2015):** SOUZA, D. C.; ALMEIDA, R. A. *Padrões pluviométricos da cidade de Manaus-AM: 1986 a 2015*. Revista Terra Livre, v. 46, p. 157-194, 2016. [Link](https://publicacoes.agb.org.br/boletim-paulista/article/view/1508)

## ⚠️ Considerations

* **Spatial scale:** Gridded data represents ~780 km² areas, smoothing local extremes.
* **Analysis period:** Full period 1961-2024 disponível (dados anuais e agregados).
* **Missing data:** 2024 NetCDF contains data only up to March 20th.
- **VPD (Vapor Pressure Deficit):** Calculated from Tmax, Tmin and RH using:
  - $e_s = 0.6108 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$ (Saturation vapor pressure, kPa)
  - $e_a = e_s \times \frac{RH}{100}$ (Actual vapor pressure, kPa)
  - $VPD = e_s - e_a$ (Vapor pressure deficit, kPa)
  - Where $T$ is the daily mean temperature $(T_{max} + T_{min})/2$

## 📄 License

This project is under the **MIT** license. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for more details.

### Data Sources (Third Party)

The data used in this project have their own licenses and terms of use:

* **Climate Data (BR-DWGD / Xavier et al.):** Public domain (Unrestricted academic/scientific use with citation).
* **Municipal Mesh (IBGE):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en) (Attribution required).
