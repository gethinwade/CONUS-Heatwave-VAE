
import pandas as pd
import numpy as np
import xarray as xr
from scipy.fft import dstn, idstn
import config
from collections import deque
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter




# ==============================
# DATA COLLECTION AND PROCESSING
# ==============================

def slice_domain(dataset):
    lon_min_360 = config.LON_MIN % 360
    lon_max_360 = config.LON_MAX % 360
    if lon_min_360 <= lon_max_360:
        domain = dataset.sel(longitude=slice(lon_min_360, lon_max_360))
    else:
        west_piece = dataset.sel(longitude=slice(lon_min_360, 360))
        east_piece = dataset.sel(longitude=slice(0, lon_max_360))
        domain = xr.concat([west_piece, east_piece], dim="longitude")
    domain = domain.sel(latitude=slice(config.LAT_MAX, config.LAT_MIN))
    return domain


def extract_surface_variable(ds, arco_name, out_name, method, scratch_path):
    da = slice_domain(ds[[arco_name]].sel(time=slice(config.DATA_START, config.DATA_END)))[arco_name]
    daily = getattr(da.resample(time="1D"), method)(method="blockwise")
    daily = daily.sel(time=~((daily.time.dt.month == 2) & (daily.time.dt.day == 29)))
    daily = daily.rename(out_name)
    daily = daily.chunk({"time": 365, "latitude": -1, "longitude": -1})
    out = daily.to_dataset()
    for var in out.variables:
        out[var].encoding.clear()
    out.to_zarr(f"{scratch_path}/{out_name}.zarr", mode="w")
    return out_name


def rel_humidity_calc(t2m_kelvin, d2m_kelvin):
    t_celsius = t2m_kelvin - 273.15
    td_celsius = d2m_kelvin - 273.15
    e_t = 6.112 * np.exp((17.62 * t_celsius) / (243.12 + t_celsius))
    e_td = 6.112 * np.exp((17.62 * td_celsius) / (243.12 + td_celsius))
    rh = 100 * (e_td / e_t)
    return rh.clip(0, 100)


def poisson_solve_3d(zeta, dx_m, dy_m):
    """
    zeta: numpy array, shape (time, lat, lon)
    Solves the Poisson equation for every timestep at once
    """
    nt, ny, nx = zeta.shape

    zeta_hat = dstn(zeta, type=1, axes=(-2, -1))

    ky = np.pi * (np.arange(1, ny + 1)) / (ny + 1) / dy_m
    kx = np.pi * (np.arange(1, nx + 1)) / (nx + 1) / dx_m
    KX, KY = np.meshgrid(kx, ky)
    denom = -(KX**2 + KY**2)
    denom[denom == 0] = np.nan
    denom = denom[np.newaxis, :, :]

    psi_hat = zeta_hat / denom
    psi = idstn(psi_hat, type=1, axes=(-2, -1)) / (4 * (nx + 1) * (ny + 1))
    return psi


def compute_streamfunction(u, v):
    """DataArrays with dims (time, latitude, longitude)"""
    lat_rad = np.deg2rad(u.latitude)
    dv_dx = v.differentiate("longitude") / (config.R_EARTH * np.cos(lat_rad) * (np.pi / 180))
    du_dy = u.differentiate("latitude") / (config.R_EARTH * (np.pi / 180))
    vorticity = (dv_dx - du_dy).load()

    dx_m = float(config.R_EARTH * np.cos(lat_rad).mean() * np.deg2rad(u.longitude.diff("longitude").mean()))
    dy_m = float(config.R_EARTH * np.deg2rad(u.latitude.diff("latitude").mean()))

    psi_values = poisson_solve_3d(vorticity.values, dx_m, dy_m)

    psi = xr.DataArray(
        psi_values,
        dims=vorticity.dims,
        coords=vorticity.coords,
        name="stream",
    )
    return psi


def extract_plevel_variable(ds, arco_name, out_name, level, scratch_path):
    da = ds[[arco_name]].sel(time=slice(config.DATA_START, config.DATA_END))
    da = slice_domain(da)[arco_name]
    da = da.sel(level=level)
    daily = da.resample(time="1D").mean(method="blockwise")
    daily = daily.sel(time=~((daily.time.dt.month == 2) & (daily.time.dt.day == 29)))
    daily = daily.rename(out_name)
    daily = daily.chunk({"time": 365, "latitude": -1, "longitude": -1})
    out = daily.to_dataset()
    for var in out.variables:
        out[var].encoding.clear()
    out.to_zarr(f"{scratch_path}/{out_name}.zarr", mode="w")
    return out_name


def extract_land_mask(ds, out_path, threshold=0.5, time="2000-01-01T00"):
    lsm = slice_domain(ds[["land_sea_mask"]].sel(time=time))["land_sea_mask"]
    lsm = lsm.drop_vars("time", errors="ignore")
    lsm = smooth_decimate(lsm)
    land = (lsm >= threshold).rename("land")
    print(f"land cells: {int(land.sum())}/{land.size}")
    out = land.to_dataset()
    for var in out.variables:
        out[var].encoding.clear()
    out.to_zarr(f"{out_path}/land.zarr", mode="w")
    return "land"


def target_grid():
    """Regular 1-degree grid: latitudes descending, longitudes in 0-360."""
    lat = np.arange(config.LAT_MIN, config.LAT_MIN + config.TARGET_H, config.TARGET_RES)[::-1]
    lon = np.arange(config.LON_MIN, config.LON_MIN + config.TARGET_W, config.TARGET_RES) % 360
    return lat.astype(float), lon.astype(float)


def smooth2a(arr, half_r=None, half_c=None):
    """NaN-aware centered sliding-window mean, equivalent to MATLAB smooth2a.
    Window is (2*half_r+1) x (2*half_c+1); shrinks at edges rather than padding."""
    half_r = config.SMOOTH_HALFWIDTH if half_r is None else half_r
    half_c = half_r if half_c is None else half_c
    size = (2*half_r + 1, 2*half_c + 1)
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(valid > 0, arr, 0.0)
    num = uniform_filter(filled, size=size, mode="constant", cval=0.0)
    den = uniform_filter(valid, size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def smooth_decimate(da, lat=None, lon=None):
    """Smooth at native resolution, then take the 1-degree target points."""
    if lat is None or lon is None:
        lat, lon = target_grid()
    dims = da.dims
    smoothed = xr.apply_ufunc(
        smooth2a, da,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["latitude", "longitude"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
    ).transpose(*dims)
    out = smoothed.sel(latitude=lat, longitude=lon, method="nearest")
    out = out.assign_coords(latitude=lat, longitude=lon)
    print(f"smoothed+decimated {da.shape} -> {out.shape}")
    return out




# ========================
# HEATWAVE ONSET DETECTION
# ========================

def compute_cluster_extent(cluster, labels, latitudes, longitudes):
    t_indices, lon_indices, lat_indices = np.where(labels == cluster)
    heatwave_length = np.max(t_indices) - np.min(t_indices) + 1
    if heatwave_length < 3:
        return None
    min_lat_idx = np.clip(np.min(lat_indices), 0, len(latitudes) - 1)
    max_lat_idx = np.clip(np.max(lat_indices), 0, len(latitudes) - 1)
    min_lon_idx = np.clip(np.min(lon_indices), 0, len(longitudes) - 1)
    max_lon_idx = np.clip(np.max(lon_indices), 0, len(longitudes) - 1)
    extent = {
        "time_range": (int(np.min(t_indices)), int(np.max(t_indices))),
        "lat_range": (float(latitudes[min_lat_idx]), float(latitudes[max_lat_idx])),
        "lon_range": (float(longitudes[min_lon_idx]), float(longitudes[max_lon_idx])),
        "heatwave_length": int(heatwave_length),
    }
    return cluster, {"extent": extent}


class GDBSCAN:
    def __init__(self, grid, latitudes, longitudes, min_card=21, connectivity=8, time_window=1):
        self.grid = grid
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.min_card = min_card
        self.connectivity = connectivity
        self.time_window = time_window
        self.NOISE = -1
        self.UNCLASSIFIED = 0
        self.cluster_id = 1
        self.labels = np.zeros_like(grid, dtype=np.int32)

    def get_neighbors(self, t, lon, lat):
        time_steps, lons, lats = self.grid.shape
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dt in range(-self.time_window, self.time_window + 1):
            for dlon, dlat in directions:
                nt, nlon, nlat = t + dt, lon + dlon, lat + dlat
                if (0 <= nt < time_steps and 0 <= nlon < lons and 0 <= nlat < lats
                        and self.grid[nt, nlon, nlat] == 1):
                    neighbors.append((nt, nlon, nlat))
        return neighbors

    def latitude_weighted_min_card(self, lat_idx, total_lat_points):
        lat_index = int((lat_idx / total_lat_points) * (len(self.latitudes) - 1))
        return np.round(self.min_card * np.cos(np.radians(self.latitudes[lat_index])))

    def expand_cluster(self, t, lon, lat):
        neighbors = self.get_neighbors(t, lon, lat)
        local_min_card = self.latitude_weighted_min_card(lat, self.grid.shape[2])
        if len(neighbors) + 1 < local_min_card:
            self.labels[t, lon, lat] = self.NOISE
            return False
        cluster_id = self.cluster_id
        queue = deque([(t, lon, lat)])
        self.labels[t, lon, lat] = cluster_id
        while queue:
            ct, clon, clat = queue.popleft()
            for nt, nlon, nlat in self.get_neighbors(ct, clon, clat):
                if self.labels[nt, nlon, nlat] == self.UNCLASSIFIED:
                    self.labels[nt, nlon, nlat] = cluster_id
                    queue.append((nt, nlon, nlat))
        return True

    def get_cluster_info(self, n_jobs=4):
        unique_clusters = np.unique(self.labels[self.labels > 0])
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_cluster_extent)(cluster, self.labels, self.latitudes, self.longitudes)
            for cluster in unique_clusters
        )
        return {cid: info for res in results if res is not None for cid, info in [res]}

    def run(self, n_jobs=4):
        time_steps, lons, lats = self.grid.shape
        active = np.where(self.grid.reshape(time_steps, -1).any(axis=1))[0]
        for t in active:
            for lon in range(lons):
                for lat in range(lats):
                    if self.grid[t, lon, lat] == 1 and self.labels[t, lon, lat] == self.UNCLASSIFIED:
                        if self.expand_cluster(t, lon, lat):
                            self.cluster_id += 1
        return self.labels, self.get_cluster_info(n_jobs=n_jobs)


def _boxes_overlap(a, b):
    """Do two events' lat/lon bounding boxes intersect?"""
    a_lat, a_lon = a["lat_range"], a["lon_range"]
    b_lat, b_lon = b["lat_range"], b["lon_range"]
    lat_ok = (min(a_lat) <= max(b_lat)) and (min(b_lat) <= max(a_lat))
    lon_ok = (min(a_lon) <= max(b_lon)) and (min(b_lon) <= max(a_lon))
    return lat_ok and lon_ok


def join_across_gaps(df, max_gap=1):
    """Merge events <= max_gap cool days apart whose boxes overlap.
    Iterates to a fixed point."""
    events = df.to_dict("records")

    merged = True
    while merged:
        merged = False
        events.sort(key=lambda e: e["start_date"])
        out = []
        for ev in events:
            joined = False
            for prev in out:
                gap = (pd.Timestamp(ev["start_date"])
                       - pd.Timestamp(prev["end_date"])).days - 1
                if 0 <= gap <= max_gap and _boxes_overlap(prev, ev):
                    prev["end_date"] = max(prev["end_date"], ev["end_date"])
                    prev["lat_range"] = (min(min(prev["lat_range"]), min(ev["lat_range"])),
                                         max(max(prev["lat_range"]), max(ev["lat_range"])))
                    prev["lon_range"] = (min(min(prev["lon_range"]), min(ev["lon_range"])),
                                         max(max(prev["lon_range"]), max(ev["lon_range"])))
                    prev["heatwave_length"] = (pd.Timestamp(prev["end_date"])
                                               - pd.Timestamp(prev["start_date"])).days + 1
                    joined = True
                    merged = True
                    break
            if not joined:
                out.append(dict(ev))
        events = out

    return pd.DataFrame(events).sort_values("start_date").reset_index(drop=True)


def detect_onsets(hot, land, out_dir):
    lon = hot.longitude
    lat = hot.latitude
    lon_min = config.ONSET_LON_MIN % 360
    lon_max = config.ONSET_LON_MAX % 360
    if lon_min <= lon_max:
        lon_sel = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_sel = (lon >= lon_min) | (lon <= lon_max)
    lat_sel = (lat >= config.ONSET_LAT_MIN) & (lat <= config.ONSET_LAT_MAX)

    hot_we = (hot.sel(longitude=lon[lon_sel], latitude=lat[lat_sel])
                 .where(land.sel(longitude=lon[lon_sel], latitude=lat[lat_sel]), other=False)
                 .transpose("time", "longitude", "latitude"))

    grid = hot_we.astype(np.int8).values
    latitudes = hot_we.latitude.values
    longitudes = hot_we.longitude.values
    times = hot_we.time.values

    print(f"grid shape {grid.shape}, lat range {latitudes.min():.2f}-{latitudes.max():.2f}, n_lat {len(latitudes)}")
    
    gdb = GDBSCAN(grid, latitudes, longitudes,
                  min_card=21, connectivity=8, time_window=1)
    labels, info = gdb.run()

    rows = []
    for cid, c in info.items():
        t0, t1 = c["extent"]["time_range"]
        rows.append({
            "event_id": int(cid),
            "start_date": pd.Timestamp(times[t0]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(times[t1]).strftime("%Y-%m-%d"),
            "heatwave_length": c["extent"]["heatwave_length"],
            "lat_range": c["extent"]["lat_range"],
            "lon_range": c["extent"]["lon_range"],
        })
    df = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    print(f"{len(df)} events before gap joining")

    df = join_across_gaps(df, max_gap=1)
    print(f"{len(df)} events after gap joining")

    df.to_csv(f"{out_dir}/events.csv", index=False)
    return df


def smooth_threshold(thresh, width=31):
    """Circular running mean of the day-of-year threshold, per Hobday et al. (2016)."""
    half = width // 2
    padded = xr.concat([thresh.isel(dayofyear=slice(-half, None)),
                        thresh,
                        thresh.isel(dayofyear=slice(0, half))], dim="dayofyear")
    smoothed = padded.rolling(dayofyear=width, center=True).mean()
    out = smoothed.isel(dayofyear=slice(half, half + thresh.sizes["dayofyear"]))
    return out.assign_coords(dayofyear=thresh.dayofyear)


def calculate_spatial_extent(lon_range, lat_range):
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    lat_extent = (lat_max - lat_min) * 111.32
    lat_mean = (lat_max + lat_min) / 2
    lon_extent = (lon_max - lon_min) * 111.32 * np.cos(np.radians(lat_mean))
    return lon_extent * lat_extent




# =================
# DESEASONALIZATION
# =================

def doy_climatology(da, base_start=None, base_end=None, window=15):
    base_start = config.BASE_START if base_start is None else base_start
    base_end = config.BASE_END if base_end is None else base_end
    half = window // 2
    base = da.sel(time=slice(str(base_start), str(base_end)))
    base = base.assign_coords(doy=base.time.dt.dayofyear)

    clim_mean_list = []
    clim_std_list = []
    doys = np.arange(1, 366)
    for d in doys:
        offsets = ((d - 1 + np.arange(-half, half + 1)) % 365) + 1
        sample = base.where(base.doy.isin(offsets), drop=True)
        clim_mean_list.append(sample.mean("time"))
        clim_std_list.append(sample.std("time", ddof=0))

    clim_mean = xr.concat(clim_mean_list, dim="doy").assign_coords(doy=doys)
    clim_std = xr.concat(clim_std_list, dim="doy").assign_coords(doy=doys)
    return clim_mean, clim_std


def standardize_variable(da, out_name, out_dir, base_start=None, base_end=None, window=15):
    clim_mean, clim_std = doy_climatology(da, base_start, base_end, window)

    doy = da.time.dt.dayofyear
    doy = xr.where(doy > 365, 365, doy)
    mu = clim_mean.sel(doy=doy)
    sigma = clim_std.sel(doy=doy)
    std = ((da - mu) / sigma).rename(out_name)

    std = std.chunk({"time": 365, "latitude": -1, "longitude": -1})
    out = std.to_dataset()
    for var in out.variables:
        out[var].encoding.clear()
    out.to_zarr(f"{out_dir}/{out_name}_{config.DATA_START[:4]}-{config.DATA_END[:4]}_standardized.zarr", mode="w")
    return out_name




# =================
# SAMPLE EXTRACTION
# =================


def build_samples(events_csv, data_dir, variables, window=5, image_size=None):
    image_size = image_size or (config.TARGET_H, config.TARGET_W)
    events = pd.read_csv(events_csv)
    event_dates = pd.to_datetime(events["start_date"]).values.astype("datetime64[D]")

    span_tag = f"{config.DATA_START[:4]}-{config.DATA_END[:4]}"

    da0 = xr.open_zarr(f"{data_dir}/{variables[0]}_{span_tag}_standardized.zarr")[variables[0]]
    time_values = da0["time"].values.astype("datetime64[D]")
    n_time = len(time_values)
    pos = {d: i for i, d in enumerate(time_values)}

    seen, idxs, kept = set(), [], []
    for d in event_dates:
        i = pos.get(d)
        if i is None or i - window < 0 or i + window >= n_time:
            continue
        if i in seen:
            continue
        seen.add(i)
        idxs.append(i)
        kept.append(d)

    n = len(idxs)
    span = 2 * window + 1
    X = np.empty((n, len(variables), span, *image_size), dtype=np.float32)
    print(f"{n} samples, allocating {X.nbytes/1e9:.1f} GB")

    for vi, v in enumerate(variables):
        da = xr.open_zarr(f"{data_dir}/{v}_{span_tag}_standardized.zarr")[v]
        arr = da.values.astype(np.float32)
        for si, i in enumerate(idxs):
            X[si, vi] = arr[i - window : i + window + 1]
        del arr
        print(f"{v} done")

    return X, np.array(kept)


def split_by_years(X, dates, year_start, year_end):
    years = np.array([pd.Timestamp(d).year for d in dates])
    mask = (years >= year_start) & (years <= year_end)
    return X[mask], dates[mask]


def year_mask(dates, year_start, year_end):
    years = np.array([pd.Timestamp(d).year for d in dates])
    return np.where((years >= year_start) & (years <= year_end))[0]