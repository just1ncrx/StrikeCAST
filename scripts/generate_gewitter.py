#!/usr/bin/env python3

import os
import glob
import re
import gc
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import ListedColormap, BoundaryNorm

PRED_DIR = "data/output"
LUT_PATH = "data/lut/lightning_lut.nc"
OUT_DIR  = "pngs/gewitter"
os.makedirs(OUT_DIR, exist_ok=True)

# lon_min, lon_max, lat_min, lat_max
EXTENT = [-3.94, 20.34, 43.18, 58.08]
TZ_DE = ZoneInfo("Europe/Berlin")



# -------------------------------------------------------
# Zeit-Hilfsfunktionen (unverändert)
# -------------------------------------------------------

def _to_de_local(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ_DE)


def format_de_datetime(ts):
    """Datum und Uhrzeit in deutscher Ortszeit (ohne Zeitzonen-Label)."""
    t = _to_de_local(ts)
    return f"{t:%d.%m.%Y %H:%M}"


# -------------------------------------------------------
# LUT laden
# -------------------------------------------------------

lut = xr.open_dataset(LUT_PATH)


# -------------------------------------------------------
# Run-Label aus NC holen (ohne Datums-Parsing) - unverändert
# -------------------------------------------------------

def extract_run_label(ds):
    def _hour_to_label(raw):
        if raw is None:
            return None
        s = str(raw).strip()
        m = re.search(r"(?<!\d)(\d{2})\s*(?:Z|z|UTC|utc)\s*$", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}z"
        m = re.search(r"(?<!\d)(\d{2})\s*$", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}z"
        return None

    run_label = _hour_to_label(ds.attrs.get("run", None))
    if run_label is not None:
        return run_label

    for key in ("run_time_utc", "run_time", "forecast_reference_time", "analysis_time"):
        run_label = _hour_to_label(ds.attrs.get(key, None))
        if run_label is not None:
            return run_label

    return "??z"


# -------------------------------------------------------
# Wahrscheinlichkeit (unverändert)
# -------------------------------------------------------

def compute_probability(ds2d, lut, interval_hours=1):

    mu_mixr = np.clip(ds2d["MU_MIXR"].values, float(lut["MU_MIXR"].min()), float(lut["MU_MIXR"].max()))
    lsm     = np.clip(ds2d["lsm"].values,     float(lut["lsm"].min()),     float(lut["lsm"].max()))
    mcpr    = np.clip(ds2d["mcpr"].values,     float(lut["mcpr"].min()),    float(lut["mcpr"].max()))
    rhmean  = np.clip(ds2d["RHmean"].values,   float(lut["meanRH_500-850"].min()), float(lut["meanRH_500-850"].max()))
    mu_li   = np.clip(ds2d["MU_LI"].values,    float(lut["MU_LI"].min()),   float(lut["MU_LI"].max()))

    prob = lut["prob_lightning_lt1h"].interp(
        MU_MIXR=("points", mu_mixr.flatten()),
        lsm=("points", lsm.flatten()),
        mcpr=("points", mcpr.flatten()),
        **{"meanRH_500-850": ("points", rhmean.flatten())},
        MU_LI=("points", mu_li.flatten()),
        method="linear",
    )
    prob = prob.values.reshape(ds2d["MU_LI"].shape)
    prob = np.nan_to_num(prob, nan=0.0)
    prob = np.clip(prob, 0.0, 1.0)

    # Physikalischer Guard: stabiles LI -> Wahrscheinlichkeit dämpfen
    mu_li_raw = ds2d["MU_LI"].values
    stability_weight = np.clip(1.0 - (mu_li_raw - 2.0) / 4.0, 0.0, 1.0)
    prob = prob * stability_weight

    # CIN-Guard: starke Kappung dämpft Blitzwahrscheinlichkeit
    if "CIN" in ds2d:
        cin_raw = ds2d["CIN"].values
        cin_weight = np.where(
            cin_raw >= -15.0,
            1.0,
            np.clip(1.0 - ((-cin_raw) - 15.0) / 135.0, 0.0, 1.0)
        )
        prob = prob * cin_weight

    # Orographie-Maske: NACH LUT und NACH stability_weight
    orog_weight = np.ones_like(prob)
    if "z_sfc" in ds2d:
        z = ds2d["z_sfc"].values
        z_max = maximum_filter(z, size=2, mode="nearest")
        orog_weight = np.clip(1.0 - (z_max - 800.0) / 700.0, 0.0, 1.0)
    prob = prob * orog_weight

    # --- Debug: Ausreißer diagnostizieren ---
    high_mask = prob > 0.5
    if high_mask.any():
        print(f"\n⚠️  {high_mask.sum()} Gitterpunkte mit prob > 50%")
        print(f"   prob:     min={prob[high_mask].min():.3f}  max={prob[high_mask].max():.3f}")
        print(f"   MU_LI:    min={ds2d['MU_LI'].values[high_mask].min():.2f}  max={ds2d['MU_LI'].values[high_mask].max():.2f}")
        print(f"   MU_MIXR:  min={ds2d['MU_MIXR'].values[high_mask].min():.2f}  max={ds2d['MU_MIXR'].values[high_mask].max():.2f}")
        print(f"   mcpr:     min={ds2d['mcpr'].values[high_mask].min():.6f}  max={ds2d['mcpr'].values[high_mask].max():.6f}")
        print(f"   RHmean:   min={ds2d['RHmean'].values[high_mask].min():.1f}  max={ds2d['RHmean'].values[high_mask].max():.1f}")
        print(f"   z_sfc:    min={ds2d['z_sfc'].values[high_mask].min():.0f}  max={ds2d['z_sfc'].values[high_mask].max():.0f}  ← Höhe in m")
        print(f"   CIN:      min={ds2d['CIN'].values[high_mask].min():.1f}  max={ds2d['CIN'].values[high_mask].max():.1f}")

        idx = np.unravel_index(np.argmax(prob), prob.shape)
        print(f"\n   Höchster Wert bei Index {idx}:")
        print(f"   prob={prob[idx]:.3f}  MU_LI={ds2d['MU_LI'].values[idx]:.2f}  MU_MIXR={ds2d['MU_MIXR'].values[idx]:.2f}  mcpr={ds2d['mcpr'].values[idx]:.6f}  z_sfc={ds2d['z_sfc'].values[idx]:.0f}m")

    ih = max(1, int(interval_hours))
    prob_interval = 1.0 - np.power(1.0 - prob, ih)
    return np.clip(prob_interval * 100.0, 0.0, 100.0)


# -------------------------------------------------------
# Diskrete Colormap in 5%-Schritten (unverändert)
# -------------------------------------------------------

bounds = [0,1,2,5,10,20,30,40,50,60,70,80,90,95,98,100]
colors = [
    "#056B6F","#079833","#08B015","#40C50C","#7DD608",
    "#9BE105","#BBEA04","#DBF402","#FFF600","#FEDC00",
    "#FFAD00","#FF6300","#E50014","#BC0035","#930058","#660179"
]
prob_colors = ListedColormap(colors)
prob_norm = BoundaryNorm(bounds, prob_colors.N)


# -------------------------------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator) - wie in Dokument 2
# -------------------------------------------------------

EARTH_RADIUS = 6378137.0  # Meter, WGS84/Web-Mercator-Kugelradius
WEBMERCATOR_WIDTH = 1024  # Ziel-Bildbreite in Pixeln für die Reprojektion


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_target_grid(extent, out_width=WEBMERCATOR_WIDTH):
    lon_min, lon_max, lat_min, lat_max = extent
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_height = max(int(round(out_width * aspect)), 1)
    x_new = np.linspace(x_min, x_max, out_width)
    y_new = np.linspace(y_min, y_max, out_height)  # aufsteigend: Süd -> Nord
    return x_new, y_new


def warp_equirect_to_webmercator(data, lon, lat, extent, method="linear",
                                  out_width=WEBMERCATOR_WIDTH):
    """data/lon/lat: reguläres EPSG:4326-Gitter, lon und lat aufsteigend
    sortiert. Gibt das Datenfeld auf einem regulären EPSG:3857-Pixelraster
    zurück (ebenfalls Süd -> Nord aufsteigend)."""
    x_new, y_new = webmercator_target_grid(extent, out_width=out_width)
    xx, yy = np.meshgrid(x_new, y_new)
    lon_grid = np.degrees(xx / EARTH_RADIUS)
    lat_grid = np.degrees(2 * np.arctan(np.exp(yy / EARTH_RADIUS)) - np.pi / 2)

    interp_func = RegularGridInterpolator(
        (lat, lon), data,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
    pts = np.array([lat_grid.ravel(), lon_grid.ravel()]).T
    warped = interp_func(pts).reshape(lat_grid.shape)
    return warped


# Feste Kartendomäne einmalig nach EPSG:3857 (Meter) umgerechnet - das ist
# der "imageExtent", den OpenLayers'/Leaflets Image-Overlay-Quelle direkt
# in Kartenkoordinaten erwartet.
_dom_x_min, _dom_y_min = lonlat_to_webmercator(EXTENT[0], EXTENT[2])
_dom_x_max, _dom_y_max = lonlat_to_webmercator(EXTENT[1], EXTENT[3])
DOMAIN_EXTENT_3857 = [float(_dom_x_min), float(_dom_y_min), float(_dom_x_max), float(_dom_y_max)]


def data_to_rgba(data, cmap, norm):
    """Wandelt ein 2D-Datenarray in ein RGBA-uint8-Array um.
    NaN-Werte werden komplett transparent."""
    rgba = cmap(norm(data))  # float RGBA in [0,1], shape (H,W,4)
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba


def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")
    img.save(out_path, format="WEBP", lossless=True, method=4)


# -------------------------------------------------------
# Glättung auf ein feineres Gitter (wie zuvor im Matplotlib-Plot,
# nur ohne die Kartendarstellung selbst)
# -------------------------------------------------------

def smooth_probability(lat_1d, lon_1d, prob, target_res=0.01, smooth_km=40.0):
    """lat_1d, lon_1d: aufsteigend sortierte 1D-Koordinaten.
    prob: 2D-Array (lat, lon). Gibt (lat_fine, lon_fine, prob_fine) zurück."""
    lon_new = np.arange(lon_1d.min(), lon_1d.max() + target_res, target_res)
    lat_new = np.arange(lat_1d.min(), lat_1d.max() + target_res, target_res)
    lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)

    interp = RegularGridInterpolator(
        (lat_1d, lon_1d), prob, method="linear", bounds_error=False, fill_value=0.0
    )
    prob_fine = interp((lat2d_new, lon2d_new))
    prob_fine = np.clip(prob_fine, 0.0, 100.0)

    km_per_pixel = target_res * 111.0
    sigma_px = smooth_km / km_per_pixel
    prob_fine = gaussian_filter(prob_fine, sigma=sigma_px, mode="nearest")
    prob_fine = np.clip(prob_fine, 0.0, 100.0)

    return lat_new, lon_new, prob_fine


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "predictors_*.nc")))

    for f in files:
        ds = xr.open_dataset(f)

        run_label = extract_run_label(ds)

        # --- Koordinaten: 1D erwartet, aufsteigend sortiert ---
        lats = ds["latitude"].values
        lons = ds["longitude"].values

        lat_idx = np.where((lats >= EXTENT[2]) & (lats <= EXTENT[3]))[0]
        lon_idx = np.where((lons >= EXTENT[0]) & (lons <= EXTENT[1]))[0]
        lat_1d = lats[lat_idx]
        lon_1d = lons[lon_idx]

        # sicherstellen, dass beide Achsen aufsteigend sortiert sind,
        # da RegularGridInterpolator (in warp_equirect_to_webmercator und
        # smooth_probability) das voraussetzt
        lat_order = np.argsort(lat_1d)
        lon_order = np.argsort(lon_1d)
        lat_1d = lat_1d[lat_order]
        lon_1d = lon_1d[lon_order]
        lat_idx = lat_idx[lat_order]
        lon_idx = lon_idx[lon_order]

        # --- Zeitscheiben auswerten (unverändert) ---
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time = ds["time"].values[0]
            step_time = ds["time"].values[1]
            print(f"Processing step {prev_time} → {step_time}")

            interval_hours = int(ds.attrs.get("interval_hours", 3))
            if interval_hours < 1:
                dt_hours = (step_time - prev_time) / np.timedelta64(1, "h")
                interval_hours = max(1, int(round(float(dt_hours))))

            ds_start = ds.isel(time=0, drop=True)
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            prob = compute_probability(ds_start, lut, interval_hours=interval_hours)
            valid_from = pd.Timestamp(prev_time)
            valid_to   = pd.Timestamp(step_time)

        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            prob = compute_probability(ds_start, lut, interval_hours=interval_hours)

            time_val = ds["time"].values[0] if "time" in ds.dims else None

            if time_val is not None:
                valid_from = pd.Timestamp(time_val)
                valid_to   = valid_from + pd.Timedelta(hours=interval_hours)
            else:
                valid_from = valid_to = None

        print(f"  prob: min={prob.min():.1f}%  max={prob.max():.1f}%  mean={prob.mean():.1f}%")

        # --- Glättung auf feineres Gitter ---
        lat_fine, lon_fine, prob_fine = smooth_probability(lat_1d, lon_1d, prob)


        # --- Nach EPSG:3857 (Web Mercator) umprojizieren ---
        prob_merc = warp_equirect_to_webmercator(
            prob_fine, lon_fine, lat_fine, EXTENT, method="linear"
        )

        # --- Dateiname: gewitter_<Ende des Prognosezeitraums, dt. Ortszeit>.webp ---
        if valid_to is not None:
            vt_de = _to_de_local(valid_to)
            outname = f"gewitter_{vt_de.strftime('%Y%m%d_%H%M')}.webp"
        else:
            outname = "gewitter_unknown.webp"
        outfile = os.path.join(OUT_DIR, outname)

        save_transparent_webp(prob_merc, prob_colors, prob_norm, outfile)
        print(f"  → {outfile}  (run={run_label}, interval={interval_hours}h)")

        ds.close()
        del prob, prob_fine, prob_merc
        gc.collect()

    print("✅ Fertig!")


if __name__ == "__main__":
    main()
