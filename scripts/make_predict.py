#!/usr/bin/env python3
"""
process_gewitter.py  –  v5 (bundled files)
Liest ECMWF Open Data aus data/gewitter/<param>/
und berechnet AR-CHaMo-Prädiktoren für Gewittervorhersage.
Speichert Ergebnisse als NetCDF4 in data/output/ mit Zeit-Koordinate.

v5: Unterstützt gebündelte GRIB2-Dateien (alle Steps in einer Datei).
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# -------------------------------------------------------
# Konfiguration
# -------------------------------------------------------
date = os.getenv("DATE", "20260325")
run  = int(os.getenv("RUN", 12))

BASE_DIR   = os.path.join("data", "gewitter")
OUTPUT_DIR = os.path.join("data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nur Steps 0–48h alle 3h (entspricht Download-Script)
steps_all = list(range(0, 49, 3))

# Physikalische Konstanten
g   = 9.80665
rd  = 287.04
rv  = 461.50
eps = rd / rv

# Bounding Box Europa
LAT_MIN, LAT_MAX = 30.0, 72.0
LON_MIN, LON_MAX = -15.0, 35.0

# -------------------------------------------------------
# Dateipfade
# -------------------------------------------------------
def sfc_path(param):
    return os.path.join(BASE_DIR, param, f"{param}_all_steps.grib2")

def pl_path(param):
    return os.path.join(BASE_DIR, param, f"{param}_pl_all_steps.grib2")

def file_ok(*paths):
    return all(os.path.exists(p) for p in paths)

# -------------------------------------------------------
# GRIB lesen  (v5: step-Dimension korrekt auswählen)
# -------------------------------------------------------

# Cache: geöffnete Datasets wiederverwenden (vermeidet wiederholtes Einlesen)
_ds_cache = {}

def _get_sfc_ds(param):
    key = f"sfc_{param}"
    if key not in _ds_cache:
        ds = xr.open_dataset(
            sfc_path(param),
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        ds = ds.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX),
        )
        _ds_cache[key] = ds
    return _ds_cache[key]

def _get_pl_ds(param):
    key = f"pl_{param}"
    if key not in _ds_cache:
        ds = xr.open_dataset(
            pl_path(param),
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            filter_by_keys={"typeOfLevel": "isobaricInhPa"},
        )
        ds = ds.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX),
        )
        _ds_cache[key] = ds
    return _ds_cache[key]


def _find_step_index(step_coord_values, step_h: int) -> int:
    """Findet den Index des gewünschten Steps (Stunden) im step-Array.
    Konvertiert alles nach Nanosekunden – funktioniert mit jeder timedelta64-Auflösung."""
    target_ns = int(step_h) * 3_600_000_000_000
    steps_ns  = step_coord_values.astype("timedelta64[ns]").astype(np.int64)
    return int(np.argmin(np.abs(steps_ns - target_ns)))


def read_sfc(param, step):
    """Liest einen Oberflächenparameter für einen bestimmten Step (h).
    Nutzt .isel() – kein Index auf der step-Koordinate nötig.
    Funktioniert auch ohne step-Dimension (z.B. Orographie).
    """
    ds      = _get_sfc_ds(param)
    varname = [v for v in ds.data_vars][0]
    da      = ds[varname]
    lats    = ds["latitude"].values
    lons    = ds["longitude"].values

    if "step" not in da.dims:
        # Kein step → 2-D Feld direkt zurückgeben (z.B. z/Orographie)
        return da.values, lats, lons

    idx = _find_step_index(da["step"].values, step)
    return da.isel(step=idx).values, lats, lons


def read_pl(param, step):
    """Liest einen Druckniveau-Parameter für einen bestimmten Step (h)."""
    ds      = _get_pl_ds(param)
    varname = [v for v in ds.data_vars][0]
    da      = ds[varname]
    levels  = ds["isobaricInhPa"].values
    lats    = ds["latitude"].values
    lons    = ds["longitude"].values

    idx  = _find_step_index(da["step"].values, step)
    data = da.isel(step=idx).values  # (level, lat, lon)

    if levels[0] > levels[-1]:
        sort_idx = np.argsort(levels)
        levels   = levels[sort_idx]
        data     = data[sort_idx]

    return data, levels, lats, lons


def clear_cache():
    """Cache leeren – kann zwischen Steps aufgerufen werden, um RAM freizugeben."""
    _ds_cache.clear()


# -------------------------------------------------------
# Meteorologische Hilfsfunktionen
# -------------------------------------------------------
def es_buck(T_K):
    Tc = T_K - 273.16
    return 6.1121 * np.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))

def calc_rh_2m(t2m, td2m):
    return np.clip(100.0 * es_buck(td2m) / es_buck(t2m), 0.0, 100.0)

def calc_mixr_2m(td2m, sp_Pa):
    e = es_buck(td2m)
    sp_hPa = sp_Pa / 100.0
    q = 0.622 * e / (sp_hPa - e)
    return np.clip(q * 1000.0, 0.0, 60.0)

def calc_mean_rh(r_pl, levels, p_min=500, p_max=850):
    mask = (levels >= p_min) & (levels <= p_max)
    if not mask.any():
        return np.full(r_pl.shape[1:], np.nan)
    return np.mean(r_pl[mask], axis=0)

def theta_ep_bolton(T_K, p_hPa, q_kgkg):
    r = q_kgkg / np.maximum(1.0 - q_kgkg, 1e-9)
    e = np.maximum(r * p_hPa / (r + eps), 1e-6)
    T_lcl = 2840.0 / (3.5 * np.log(np.maximum(T_K, 1.0)) - np.log(np.maximum(e, 1e-9)) - 4.805) + 55.0
    theta_e = (T_K * (1000.0 / np.maximum(p_hPa, 1.0)) ** (0.2854 * (1.0 - 0.28 * r))
               * np.exp(r * (1.0 + 0.81 * r) * (3376.0 / np.maximum(T_lcl, 1.0) - 2.54)))
    return theta_e

def calc_deg0l(t_pl, z_pl, levels):
    t_c = t_pl - 273.15
    nlev, nlat, nlon = t_pl.shape
    deg0l = np.full((nlat, nlon), np.nan)
    for i in range(nlev - 1, 0, -1):
        j = i - 1
        t_bot = t_c[i]; t_top = t_c[j]
        z_bot = z_pl[i]; z_top = z_pl[j]
        crossing = (t_bot >= 0.0) & (t_top < 0.0) & np.isnan(deg0l)
        dT = t_bot - t_top
        safe_dT = np.where(np.abs(dT) < 1e-6, 1e-6, dT)
        frac = t_bot / safe_dT
        height = z_bot + frac * (z_top - z_bot)
        deg0l = np.where(crossing, height, deg0l)
    deg0l = np.where(np.isnan(deg0l), 0.0, deg0l)
    return deg0l

def calc_mu_li(t_pl, z_pl, q_pl, levels, t2m, td2m, sp_Pa):
    sp_hPa = sp_Pa / 100.0
    idx_500 = np.argmin(np.abs(levels - 500.0))
    t_env_500 = t_pl[idx_500]

    q_sfc = np.clip(0.622 * es_buck(td2m) / (sp_hPa - es_buck(td2m)), 0.0, 0.06)
    theta_e_parcel = theta_ep_bolton(t2m, sp_hPa, q_sfc)

    Lv = 2.5e6
    T_guess = t_env_500.copy()
    for _ in range(6):
        es_500    = es_buck(T_guess)
        q_sat_500 = np.clip(0.622 * es_500 / (500.0 - es_500), 0.0, 0.06)
        theta_e_guess = theta_ep_bolton(T_guess, 500.0, q_sat_500)
        dtheta_dT = theta_e_guess / T_guess * (1.0 + Lv * q_sat_500 / (rd * T_guess))
        T_guess = T_guess + (theta_e_parcel - theta_e_guess) / np.maximum(dtheta_dT, 1e-6)
        T_guess = np.clip(T_guess, 200.0, 320.0)

    mu_li = t_env_500 - T_guess
    return np.clip(mu_li, -20.0, 20.0)

def calc_lcl_height(t2m, td2m):
    return np.maximum((t2m - td2m) * 125.0, 0.0)

def interpolate_to_height(z_3d, param_3d, target_z):
    nlev, nlat, nlon = z_3d.shape
    ngrid = nlat * nlon
    z2 = z_3d.reshape(nlev, ngrid).T
    p2 = param_3d.reshape(nlev, ngrid).T
    tgt = target_z.ravel()
    below = z2 < tgt[:, None]
    idx_bot = np.argmax(below, axis=1)
    no_below = ~below.any(axis=1)
    idx_bot = np.where(no_below, nlev - 1, idx_bot)
    idx_top = np.clip(idx_bot - 1, 0, nlev - 1)
    z_bot = z2[np.arange(ngrid), idx_bot]
    z_top = z2[np.arange(ngrid), idx_top]
    p_bot = p2[np.arange(ngrid), idx_bot]
    p_top = p2[np.arange(ngrid), idx_top]
    dz = z_top - z_bot
    frac = np.where(np.abs(dz) > 1.0, (tgt - z_bot) / dz, 0.5)
    frac = np.clip(frac, 0.0, 1.0)
    result = p_bot + frac * (p_top - p_bot)
    return result.reshape(nlat, nlon)

def calc_mean_wind_1_3km(z_3d, u_3d, v_3d, z_sfc):
    u_sum = np.zeros_like(z_sfc)
    v_sum = np.zeros_like(z_sfc)
    for dz in [1000, 2000, 3000]:
        u_sum += interpolate_to_height(z_3d, u_3d, z_sfc + dz)
        v_sum += interpolate_to_height(z_3d, v_3d, z_sfc + dz)
    return np.sqrt((u_sum / 3.0) ** 2 + (v_sum / 3.0) ** 2)

def calc_eff_bulk_shear(z_3d, u_3d, v_3d, z_sfc, cape):
    u_sfc = interpolate_to_height(z_3d, u_3d, z_sfc)
    v_sfc = interpolate_to_height(z_3d, v_3d, z_sfc)
    u_top = interpolate_to_height(z_3d, u_3d, z_sfc + 3000.0)
    v_top = interpolate_to_height(z_3d, v_3d, z_sfc + 3000.0)
    bs = np.sqrt((u_top - u_sfc) ** 2 + (v_top - v_sfc) ** 2)
    bs = np.where(cape < 10.0, 0.0, bs)
    return bs

# -------------------------------------------------------
# Hauptverarbeitung
# -------------------------------------------------------
def process_step_pair(prev_step, step):
    print(f"\n=== Verarbeite Step {prev_step}h → {step}h ===")

    required = [
        sfc_path("2t"), sfc_path("2d"), sfc_path("sp"),
        sfc_path("tp"), sfc_path("lsm"),
        sfc_path("mucape"), sfc_path("z"),
        pl_path("t"), pl_path("q"), pl_path("r"),
        pl_path("u"), pl_path("v"), pl_path("gh"),
    ]
    if not file_ok(*required):
        missing = [p for p in required if not os.path.exists(p)]
        print(f"  ⚠️ Fehlende Dateien ({len(missing)}), überspringe:")
        for p in missing:
            print(f"     {p}")
        return

    # --- Felder lesen ---
    t2m,    lats, lons = read_sfc("2t",     prev_step)
    td2m,   _,    _    = read_sfc("2d",     prev_step)
    sp,     _,    _    = read_sfc("sp",     prev_step)
    tp0,    _,    _    = read_sfc("tp",     prev_step)
    tp1,    _,    _    = read_sfc("tp",     step)
    lsm,    _,    _    = read_sfc("lsm",    prev_step)
    mucape, _,    _    = read_sfc("mucape", prev_step)
    z_sfc_geo, _, _    = read_sfc("z",      0)         # Orographie immer Step 0
    z_sfc = z_sfc_geo / g

    t_pl,  levels, _, _ = read_pl("t",  prev_step)
    q_pl,  _,      _, _ = read_pl("q",  prev_step)
    r_pl,  _,      _, _ = read_pl("r",  prev_step)
    r_pl = np.clip(r_pl, 0.0, 100.0)
    u_pl,  _,      _, _ = read_pl("u",  prev_step)
    v_pl,  _,      _, _ = read_pl("v",  prev_step)
    z_pl,  _,      _, _ = read_pl("gh", prev_step)

    # --- Prädiktoren ---
    rh_mean = calc_mean_rh(r_pl, levels)

    mcpr_raw = np.maximum((tp1 - tp0) / (step - prev_step), 0.0)
    cape_weight = np.clip(mucape / 100.0, 0.0, 1.0)
    mcpr = np.clip(mcpr_raw * cape_weight, 0.0, 0.00296)

    print(f"  mcpr roh:            min={mcpr_raw.min():.6f}  max={mcpr_raw.max():.6f} m/h")
    print(f"  mcpr nach Gewicht:   min={mcpr.min():.6f}  max={mcpr.max():.6f} m/h")
    print(f"  mcpr >0 Pixel:       {(mcpr > 0).sum()} von {mcpr.size}")
    print(f"  mcpr P90/P95/P99:    {np.percentile(mcpr,90):.6f} / {np.percentile(mcpr,95):.6f} / {np.percentile(mcpr,99):.6f}")

    mu_mixr = calc_mixr_2m(td2m, sp)
    ml_lcl  = calc_lcl_height(t2m, td2m)
    deg0l   = calc_deg0l(t_pl, z_pl, levels)
    mu_li   = calc_mu_li(t_pl, z_pl, q_pl, levels, t2m, td2m, sp)
    mu_eff_bs = calc_eff_bulk_shear(z_pl, u_pl, v_pl, z_sfc, mucape)
    mw_13   = calc_mean_wind_1_3km(z_pl, u_pl, v_pl, z_sfc)
    sb_wmax = np.sqrt(np.maximum(2.0 * mucape, 0.0))
    mu_cape_m10 = np.maximum(mucape * 0.30, 0.0)
    cin = np.full_like(t2m, np.nan)

    print(f"  td2m:    min={td2m.min():.1f}  max={td2m.max():.1f} K")
    print(f"  t2m:     min={t2m.min():.1f}  max={t2m.max():.1f} K")
    print(f"  sp:      min={sp.min():.0f}  max={sp.max():.0f} Pa")
    print(f"  MU_MIXR: min={mu_mixr.min():.2f}  max={mu_mixr.max():.2f} g/kg")
    print(f"  RHmean:  min={rh_mean.min():.1f}  max={rh_mean.max():.1f} %")
    print(f"  td2m 99. Perz.: {np.percentile(td2m, 99):.1f} K = {np.percentile(td2m, 99)-273.15:.1f}°C")

    predictors = {
        "MU_LI":      mu_li,
        "MU_CAPE":    mucape,
        "MU_CAPE_M10":mu_cape_m10,
        "SB_WMAX":    sb_wmax,
        "CIN":        cin,
        "RHmean":     rh_mean,
        "MU_MIXR":    mu_mixr,
        "ML_MIXR":    mu_mixr,
        "ML_LCL":     ml_lcl,
        "ZeroHeight": deg0l,
        "mcpr":       mcpr,
        "MU_EFF_BS":  mu_eff_bs,
        "MW_13":      mw_13,
        "lsm":        lsm,
        "z_sfc":      z_sfc,
    }

    save_predictors(predictors, lats, lons, prev_step, step)

# -------------------------------------------------------
# Speichern mit Zeit-Koordinate
# -------------------------------------------------------
def save_predictors(predictors, lats, lons, prev_step, step):
    outfile = os.path.join(
        OUTPUT_DIR,
        f"predictors_{date}_{run:02d}Z_step{prev_step:03d}-{step:03d}.nc"
    )

    run_time = datetime.strptime(f"{date}{run:02d}", "%Y%m%d%H")
    times = np.array([
        run_time + timedelta(hours=prev_step),
        run_time + timedelta(hours=step),
    ], dtype="datetime64[ns]")

    ds = xr.Dataset(
        {name: (["time", "latitude", "longitude"],
                np.stack([arr, arr], axis=0).astype(np.float32))
         for name, arr in predictors.items()},
        coords={
            "time":      times,
            "latitude":  ("latitude",  lats),
            "longitude": ("longitude", lons),
        },
        attrs={
            "date":           date,
            "run":            f"{run:02d}Z",
            "step_start":     prev_step,
            "step_end":       step,
            "interval_hours": step - prev_step,
            "source":         "ECMWF IFS Open Data",
            "created":        datetime.utcnow().isoformat(),
            "notes":          "AR-CHaMo Prädiktoren v5 (bundled GRIB, Zeit-Koordinate, 3h Intervall)",
        },
    )
    ds.to_netcdf(outfile)
    print(f"  💾 Gespeichert: {outfile}")

# -------------------------------------------------------
# Hauptprogramm
# -------------------------------------------------------
def main():
    print("=== AR-CHaMo Prädiktor-Berechnung (v5 – bundled GRIB) ===")
    print(f"Datum: {date}  Lauf: {run:02d} UTC")
    print(f"Eingabe: {BASE_DIR}/")
    print(f"Ausgabe: {OUTPUT_DIR}/")
    print(f"Steps: {steps_all[0]}–{steps_all[-1]}h")

    for i in range(1, len(steps_all)):
        prev_step = steps_all[i - 1]
        step      = steps_all[i]
        process_step_pair(prev_step, step)

    clear_cache()
    print("\n✅ Alle Steps verarbeitet!")

if __name__ == "__main__":
    main()