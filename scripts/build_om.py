import bz2
import shutil
import xarray as xr
import struct
import zlib
import json
import os
import re
import sys
import glob
import pandas as pd
import numpy as np
from pyproj import Proj, Transformer
from zoneinfo import ZoneInfo

# -------------------------------
# Konfiguration
# -------------------------------
INPUT_DIR  = "data/warnmos"
OUTPUT_DIR = "warnmos"
CHUNK      = 16
NX, NY     = 900, 900
DX, DY     = 1000.0, 1000.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Raster-Koordinaten berechnen
# -------------------------------
def get_grid():
    proj_ps  = Proj(proj='stere', lat_0=90, lon_0=10, lat_ts=60,
                    x_0=0, y_0=0, ellps='WGS84')
    proj_geo = Proj(proj='latlong', datum='WGS84')
    transformer = Transformer.from_proj(proj_ps, proj_geo, always_xy=True)
    x0, y0 = proj_ps(3.594, 46.957)
    x_arr  = x0 + np.arange(NX) * DX
    y_arr  = y0 + np.arange(NY) * DY
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)
    lon2d, lat2d   = transformer.transform(x_grid, y_grid)
    return lon2d, lat2d

# -------------------------------
# GRIB diagnostisch prüfen
# -------------------------------
def inspect_grib(grib_file):
    try:
        import cfgrib
        datasets = cfgrib.open_datasets(grib_file)
        print(f"\n{'='*50}")
        print(f"Gefundene Datasets ({len(datasets)}):")
        for i, d in enumerate(datasets):
            print(f"  [{i}] vars={list(d.data_vars)}  dims={dict(d.dims)}")
        print('='*50 + "\n")
    except Exception as e:
        print("inspect_grib Fehler:", e)

# -------------------------------
# OM-File + IDX bauen
# -------------------------------
def build_om(grib_file, short_name="W_GEW_01"):
    basename = os.path.splitext(os.path.basename(grib_file))[0]
    om_file  = os.path.join(OUTPUT_DIR, basename + ".om")
    idx_file = om_file + ".idx"

    print(f"Reading GRIB: {grib_file}")
    ds   = xr.open_dataset(grib_file, engine="cfgrib",
                           filter_by_keys={"shortName": short_name})
    data = ds[short_name]

    print(f"  shape : {data.shape}")
    print(f"  dims  : {data.dims}")
    print(f"  min   : {float(np.nanmin(data.values)):.2f}")
    print(f"  max   : {float(np.nanmax(data.values)):.2f}")
    print(f"  NaN%  : {100*np.isnan(data.values).mean():.1f}%")

    # Forecast-Run-Zeit
    if "time" in ds.coords:
        time_val     = ds.time.values
        run_time_utc = (pd.to_datetime(time_val[0])
                        if not np.isscalar(time_val)
                        else pd.to_datetime(time_val))
    else:
        m = re.search(r'WarnMOS(\d{12})', grib_file)
        if m:
            g = m.group(1)
            run_time_utc = pd.Timestamp(
                f"{g[:4]}-{g[4:6]}-{g[6:8]}T{g[8:10]}:{g[10:12]}:00")
        else:
            run_time_utc = pd.Timestamp.now()

    print(f"  run   : {run_time_utc}")

    # Steps sammeln
    all_times  = []
    all_arrays = []

    if data.ndim == 3:
        n_steps = data.shape[0]
        for step in range(n_steps):
            if 'step' in ds.coords:
                step_val = ds['step'].values[step]
                if isinstance(step_val, np.timedelta64):
                    valid_time_utc = run_time_utc + pd.to_timedelta(step_val)
                else:
                    valid_time_utc = run_time_utc + pd.to_timedelta(float(step_val), unit='s')
            else:
                valid_time_utc = run_time_utc + pd.to_timedelta(step, unit='h')

            valid_time_local = valid_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))
            all_times.append(valid_time_local.strftime("%Y-%m-%d %H:%M:%S"))
            all_arrays.append(data[step].values)
    else:
        valid_time_local = run_time_utc.tz_localize("UTC").astimezone(ZoneInfo("Europe/Berlin"))
        all_times.append(valid_time_local.strftime("%Y-%m-%d %H:%M:%S"))
        all_arrays.append(data.values)

    print(f"  steps : {len(all_times)}")

    lon2d, lat2d = get_grid()

    # OM + IDX schreiben
    index = []

    with open(om_file, "wb") as f:
        for idx, array in enumerate(all_arrays):
            chunks        = []
            chunk_offsets = []
            chunk_lens    = []

            for row in range(0, NY, CHUNK):
                for col in range(0, NX, CHUNK):
                    tile = []
                    for cy in range(CHUNK):
                        for cx in range(CHUNK):
                            iy = row + cy
                            ix = col + cx
                            if iy < NY and ix < NX:
                                val = float(array[iy, ix])
                                if np.isnan(val) or val < -8 or val > 110:
                                    val = 0.0
                            else:
                                val = 0.0
                            tile.append(val)

                    buf        = struct.pack(f"{len(tile)}f", *tile)
                    compressed = zlib.compress(buf)
                    chunks.append(compressed)

            header = {
                "nx":         NX,
                "ny":         NY,
                "chunk":      CHUNK,
                "latMin":     float(lat2d.min()),
                "latMax":     float(lat2d.max()),
                "lonMin":     float(lon2d.min()),
                "lonMax":     float(lon2d.max()),
                "chunkCount": len(chunks),
                "timestamp":  all_times[idx],
            }

            header_bytes = json.dumps(header).encode()
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)

            for c in chunks:
                chunk_offsets.append(f.tell())
                chunk_lens.append(len(c))
                f.write(struct.pack("<I", len(c)))
                f.write(c)

            index.append({
                "timestamp":    all_times[idx],
                "chunkOffsets": chunk_offsets,
                "chunkLens":    chunk_lens,
                "header":       header,
            })

            print(f"  Step {idx:03d} – {all_times[idx]}")

    with open(idx_file, "w") as f:
        json.dump(index, f)

    print(f"\nOM  : {om_file}")
    print(f"IDX : {idx_file}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.grb2")))

    if not files:
        print(f"Keine .grb2 Dateien in {INPUT_DIR} gefunden!")
        sys.exit(1)

    print(f"Gefunden: {len(files)} Datei(en) in {INPUT_DIR}")
    for grib_file in files:
        print(f"\n{'='*60}")
        print(f"Verarbeite: {grib_file}")
        print('='*60)
        inspect_grib(grib_file)
        build_om(grib_file)

    print("\nFertig!")
