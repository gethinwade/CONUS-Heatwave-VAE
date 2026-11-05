R_EARTH = 6.371e6
G = 9.80665


# Lon/Lat Domain
LON_MIN, LON_MAX = -168, -40
LAT_MIN, LAT_MAX = 16, 64


# Onset-detection domain
ONSET_LON_MIN, ONSET_LON_MAX = -125, -66
ONSET_LAT_MIN, ONSET_LAT_MAX = 24, 50


# Target grid shape
TARGET_H, TARGET_W = 48, 128


# Smoothing and decimation
SMOOTH_HALFWIDTH = 2
TARGET_RES = 1.0


# Pressure levels needed
LEVEL_STREAM = 250
LEVEL_Z500 = 500


# ARCO surface variables
ARCO_SURFACE = {
    "msl": ("mean_sea_level_pressure", "mean"),
    "t2m": ("2m_temperature", "max"),
    "d2m": ("2m_dewpoint_temperature", "max"),
    "u10": ("10m_u_component_of_wind", "mean"),
    "v10": ("10m_v_component_of_wind", "mean"),
    "tcc": ("total_cloud_cover", "mean"),
    "ssr": ("surface_net_solar_radiation", "sum"),
}


# ARCO pressure-level variables
ARCO_PLEVEL = {
    "u": ("u_component_of_wind", LEVEL_STREAM),
    "v": ("v_component_of_wind", LEVEL_STREAM),
    "z": ("geopotential", LEVEL_Z500),
}


# Final channel order
CHANNEL_ORDER = ["stream", "msl", "t2m", "r", "u10", "v10", "tcc", "ssr", "z"]


DATA_START, DATA_END = "1980-01-01", "2025-12-31"
BASE_START, BASE_END = 1980, 2014


# Train/val/test splits
TRAIN_YEARS = (1980, 2014)
VAL_YEARS = (2015, 2022)
TEST_YEARS = (2013, 2025)


# URLs
ARCO_URL = "/file/pathway"
SCRATCH = "/file/pathway"