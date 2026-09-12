#!/usr/bin/env python3

import os
import glob
import re
import gc
import struct
import zlib
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr
import pandas as pd
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import ListedColormap, BoundaryNorm

PRED_DIR = "data/output"
OUT_DIR  = "pngs/supercell"
os.makedirs(OUT_DIR, exist_ok=True)

# lon_min, lon_max, lat_min, lat_max - gleiche volle Domäne wie beim
# Gewitter-Produkt, damit beide Layer exakt übereinander liegen.
EXTENT = [-3.94, 20.34, 43.18, 58.08]
TZ_DE = ZoneInfo("Europe/Berlin")

# ------------------------------
# Bounding Box fuer den eingebetteten DVAL-Chunk - Deutschland + etwas
# Rand fuer Grenzregionen beim Hovern; das Farbbild selbst bleibt
# unveraendert auf der vollen Domaene.
# ------------------------------
GERMANY_BBOX_LONLAT = [5.5, 15.3, 47.0, 55.3]  # lon_min, lon_max, lat_min, lat_max

# Rasterschritt fuer den eingebetteten DVAL-Chunk. SCP ist ein
# gebrochener Index (Schwellen u.a. bei 0.2 / 0.5 / 1 / 2 ...), daher
# feiner quantisiert als die Blitz-Wahrscheinlichkeit (dort 1 %).
QUANTUM_STEP = 0.1
NAN_SENTINEL_I16 = -32768
DVAL_FOURCC = b"DVAL"


# -------------------------------------------------------
# Zeit-Hilfsfunktionen
# -------------------------------------------------------

def _to_de_local(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ_DE)


def format_de_datetime(ts):
    t = _to_de_local(ts)
    return f"{t:%d.%m.%Y %H:%M}"


# -------------------------------------------------------
# Run-Label aus NC holen
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
# SCP aus Prädiktoren lesen
# -------------------------------------------------------

def extract_scp(ds2d):
    """Liest SCP direkt aus dem Prädiktor-Dataset."""
    if "SCP" not in ds2d:
        raise KeyError("Variable 'SCP' nicht im Dataset gefunden. "
                       "Bitte process_gewitter.py v6+SRH/SCP zuerst ausführen.")
    scp = ds2d["SCP"].values
    return np.clip(scp, 0.0, None)


# -------------------------------------------------------
# Diskrete Colormap fuer SCP
# -------------------------------------------------------

SCP_BOUNDS = [0, 0.2, 0.5, 1, 2, 3, 4, 8, 10, 15, 20, 25, 30, 40, 50]
SCP_COLORS = [
    "#FFFFFF", "#D3E9FF", "#75BAFF", "#0069D2", "#148F1B", "#64ED07", "#FFF32B",
    "#E9DC01", "#FF7F26", "#F71E53", "#880000", "#64007F", "#C300FC", "#DD66FE",
    "#EBA6FF", "#B97A57"
]
scp_colors = ListedColormap(SCP_COLORS)
scp_norm = BoundaryNorm(SCP_BOUNDS, scp_colors.N)


# -------------------------------------------------------
# EPSG:4326 -> EPSG:3857 (Web Mercator)
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


# Feste Kartendomäne einmalig nach EPSG:3857 (Meter) umgerechnet.
_dom_x_min, _dom_y_min = lonlat_to_webmercator(EXTENT[0], EXTENT[2])
_dom_x_max, _dom_y_max = lonlat_to_webmercator(EXTENT[1], EXTENT[3])
DOMAIN_EXTENT_3857 = [float(_dom_x_min), float(_dom_y_min), float(_dom_x_max), float(_dom_y_max)]

_full_x_new, _full_y_new = webmercator_target_grid(EXTENT, out_width=WEBMERCATOR_WIDTH)

_gbx_min, _gby_min = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[0], GERMANY_BBOX_LONLAT[2])
_gbx_max, _gby_max = lonlat_to_webmercator(GERMANY_BBOX_LONLAT[1], GERMANY_BBOX_LONLAT[3])

_col_i0 = max(0, np.searchsorted(_full_x_new, _gbx_min, side="left") - 1)
_col_i1 = min(len(_full_x_new) - 1, np.searchsorted(_full_x_new, _gbx_max, side="right"))
_row_i0 = max(0, np.searchsorted(_full_y_new, _gby_min, side="left") - 1)
_row_i1 = min(len(_full_y_new) - 1, np.searchsorted(_full_y_new, _gby_max, side="right"))

GERMANY_CROP_EXTENT_3857 = [
    float(_full_x_new[_col_i0]), float(_full_y_new[_row_i0]),
    float(_full_x_new[_col_i1]), float(_full_y_new[_row_i1]),
]


def crop_to_germany(data_south_first):
    return data_south_first[_row_i0:_row_i1 + 1, _col_i0:_col_i1 + 1]


def data_to_rgba(data, cmap, norm):
    """Wandelt ein 2D-Datenarray in ein RGBA-uint8-Array um.
    NaN-Werte werden komplett transparent."""
    rgba = cmap(norm(data))
    rgba = (rgba * 255).astype(np.uint8)
    nan_mask = ~np.isfinite(data)
    rgba[nan_mask, 3] = 0
    return rgba


def save_transparent_webp(data, cmap, norm, out_path):
    rgba = data_to_rgba(data, cmap, norm)
    img = Image.fromarray(rgba[::-1, :, :], mode="RGBA")
    img.save(out_path, format="WEBP", lossless=True, method=4)


def embed_data_chunk(webp_path, data, extent_3857, quantum, fourcc=DVAL_FOURCC):
    """Hängt ein rohes Datenfeld als privaten, int16-quantisierten RIFF-Chunk
    an ein WebP an.

    data: 2D-Array (float), row0 = Norden (also bereits wie fürs Bild
          gespiegelt).
    extent_3857: [x_min, y_min, x_max, y_max] in Web-Mercator-Metern -
                 exakt das Raster, auf dem `data` liegt.
    quantum: Rasterschritt in den Originaleinheiten (hier: SCP-Einheiten).
    """
    height, width = data.shape

    nan_mask = ~np.isfinite(data)
    data_filled = np.where(nan_mask, 0.0, data)
    quant = np.round(data_filled / quantum)
    quant = np.clip(quant, -32767, 32767).astype(np.int16)
    quant[nan_mask] = NAN_SENTINEL_I16

    header = struct.pack("<BBII", 2, 1, width, height)
    header += struct.pack("<4d", *extent_3857)
    header += struct.pack("<d", quantum)
    compressed = zlib.compress(np.ascontiguousarray(quant, dtype="<i2").tobytes(), level=9)
    payload = header + compressed

    size = len(payload)
    chunk = fourcc + struct.pack("<I", size) + payload
    if size % 2 == 1:
        chunk += b"\x00"

    with open(webp_path, "rb") as f:
        content = f.read()

    if content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise ValueError(f"{webp_path} ist keine gültige WebP-Datei (RIFF/WEBP-Header fehlt)")

    riff_size = struct.unpack("<I", content[4:8])[0]
    new_riff_size = riff_size + len(chunk)

    with open(webp_path, "wb") as f:
        f.write(content[:4])
        f.write(struct.pack("<I", new_riff_size))
        f.write(content[8:])
        f.write(chunk)


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

        lat_order = np.argsort(lat_1d)
        lon_order = np.argsort(lon_1d)
        lat_1d = lat_1d[lat_order]
        lon_1d = lon_1d[lon_order]
        lat_idx = lat_idx[lat_order]
        lon_idx = lon_idx[lon_order]

        # --- Zeitscheiben auswerten ---
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time = ds["time"].values[0]
            step_time = ds["time"].values[1]
            print(f"Processing step {prev_time} → {step_time}")

            interval_hours = int(ds.attrs.get("interval_hours", 3))

            ds_start = ds.isel(time=0, drop=True)
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            valid_from = pd.Timestamp(prev_time)
            valid_to   = pd.Timestamp(step_time)

        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_start = ds_start.isel(latitude=lat_idx, longitude=lon_idx)

            time_val = ds["time"].values[0] if "time" in ds.dims else None

            if time_val is not None:
                valid_from = pd.Timestamp(time_val)
                valid_to   = valid_from + pd.Timedelta(hours=interval_hours)
            else:
                valid_from = valid_to = None

        print(f"Processing {f}")

        scp = extract_scp(ds_start)
        print(f"  SCP: min={scp.min():.3f}  max={scp.max():.3f}  mean={scp.mean():.3f}")

        # --- Direkt (ohne Vorinterpolation) nach EPSG:3857 (Web Mercator)
        # umprojizieren, auf Basis des Originalgitters aus der NC-Datei ---
        scp_merc = warp_equirect_to_webmercator(
            scp, lon_1d, lat_1d, EXTENT, method="linear"
        )

        # --- Dateiname: supercell_<Ende des Prognosezeitraums, dt. Ortszeit>.webp ---
        if valid_to is not None:
            vt_de = _to_de_local(valid_to)
            outname = f"scp_{vt_de.strftime('%Y%m%d_%H%M')}.webp"
        else:
            outname = "scp_unknown.webp"
        outfile = os.path.join(OUT_DIR, outname)

        save_transparent_webp(scp_merc, scp_colors, scp_norm, outfile)

        # Zusaetzlich die echten SCP-Werte (nicht die Farben) als privaten
        # RIFF-Chunk direkt ins WebP einbetten - row0 = Norden, damit der
        # Chunk 1:1 zur Bildorientierung passt (das Bild wird in
        # save_transparent_webp beim Speichern gespiegelt, scp_merc selbst
        # hat row0 = Sueden).
        germany_data = crop_to_germany(scp_merc)          # row0 = Süden
        # sehr kleine Werte -> NaN, damit "kein Signal" im Chunk
        # transparent/fehlend ist statt als echter Wert 0 codiert zu werden.
        germany_data = np.where(germany_data < SCP_BOUNDS[1], np.nan, germany_data)
        embed_data_chunk(outfile, germany_data[::-1], GERMANY_CROP_EXTENT_3857, QUANTUM_STEP)  # row0 = Norden

        print(f"  → {outfile}  (run={run_label}, interval={interval_hours}h)")

        ds.close()
        del scp, scp_merc
        gc.collect()

    print("✅ Fertig!")


if __name__ == "__main__":
    main()
