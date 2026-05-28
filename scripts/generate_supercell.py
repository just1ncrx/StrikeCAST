#!/usr/bin/env python3

import os
import glob
import re
from zoneinfo import ZoneInfo
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import pandas as pd

PRED_DIR = "data/output"
OUT_DIR  = "pngs/supercell"
os.makedirs(OUT_DIR, exist_ok=True)


cities = pd.DataFrame({
    'name': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Dresden', 'Stuttgart', 'Düsseldorf',
             'Nürnberg', 'Erfurt', 'Leipzig', 'Bremen', 'Saarbrücken', 'Hannover'],
    'lat': [52.52, 53.55, 48.14, 50.94, 50.11, 51.05, 48.78, 51.23,
            49.45, 50.98, 51.34, 53.08, 49.24, 52.37],
    'lon': [13.40, 9.99, 11.57, 6.96, 8.68, 13.73, 9.18, 6.78,
            11.08, 11.03, 12.37, 8.80, 6.99, 9.73]
})

EXTENT = [5, 16, 47, 56]
TZ_DE  = ZoneInfo("Europe/Berlin")


def _to_de_local(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(TZ_DE)


def format_de_datetime(ts):
    t = _to_de_local(ts)
    return f"{t:%d.%m.%Y %H:%M}"


def format_forecast_range_de(valid_from, valid_to):
    return (
        f"von {format_de_datetime(valid_from)}\n"
        f"bis {format_de_datetime(valid_to)}"
    )


def format_forecast_range_title_de(valid_from, valid_to):
    return f"{format_de_datetime(valid_from)} - {format_de_datetime(valid_to)}"


# -------------------------------------------------------
# Kartenparameter
# -------------------------------------------------------

FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX    = FIG_H_PX - BOTTOM_AREA_PX


# -------------------------------------------------------
# Hilfsfunktion: Run-Label aus NC holen
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
        raise KeyError("Variable 'scp' nicht im Dataset gefunden. "
                       "Bitte process_gewitter.py v6+SRH/SCP zuerst ausführen.")
    scp = ds2d["SCP"].values
    return np.clip(scp, 0.0, None)


# -------------------------------------------------------
# Plot
# -------------------------------------------------------

# SCP-Bounds und Farben
# Typische Schwellen: <0.5 = niedrig, 1-2 = erhöht, >4 = signifikant
SCP_BOUNDS = [0, 0.2, 0.5, 1, 2, 3, 4, 8, 10, 15, 20, 25, 30, 40, 50]
SCP_COLORS = [
   
    "#FFFFFF", "#D3E9FF", "#75BAFF", "#0069D2", "#148F1B", "#64ED07", "#FFF32B",
    "#E9DC01", "#FF7F26", "#F71E53", "#880000", "#64007F", "#C300FC", "#DD66FE",
    "#EBA6FF", "#B97A57"
]


def plot_scp_png(lats, lons, scp, outfile, interval_hours=3,
                 run_label="??z", valid_from=None, valid_to=None):

    cmap = ListedColormap(SCP_COLORS)
    norm = BoundaryNorm(SCP_BOUNDS, cmap.N)

    # Interpolation auf feineres Gitter
    target_res = 0.025
    lon_min, lon_max = np.min(lons), np.max(lons)
    lat_min, lat_max = np.min(lats), np.max(lats)
    lon_new = np.arange(lon_min, lon_max + target_res, target_res)
    lat_new = np.arange(lat_min, lat_max + target_res, target_res)
    lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)

    interp = RegularGridInterpolator(
        (lats[:, 0], lons[0, :]), scp,
        method="linear", bounds_error=False, fill_value=0.0
    )
    scp_plot = interp((lat2d_new, lon2d_new))
    scp_plot = np.clip(scp_plot, 0.0, SCP_BOUNDS[-1])
    lats, lons = lat2d_new, lon2d_new

    # Figur
    scale = 0.9
    fig = plt.figure(figsize=(FIG_W_PX / 100 * scale, FIG_H_PX / 100 * scale), dpi=100)

    shift_up = 0.02
    ax = fig.add_axes(
        [0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
        projection=ccrs.PlateCarree()
    )
    ax.set_extent(EXTENT)
    ax.set_axis_off()
    ax.set_aspect("auto")

    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#2C2C2C", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="#2C2C2C", linewidth=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")

    for _, city in cities.iterrows():
        ax.plot(city["lon"], city["lat"], "o", markersize=6,
                markerfacecolor="black", markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)
        txt = ax.text(city["lon"] + 0.1, city["lat"] + 0.1, city["name"],
                      fontsize=9, color="black", weight="bold", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])

    im = ax.pcolormesh(
        lons, lats, scp_plot,
        cmap=cmap, norm=norm,
        shading="gouraud", antialiased=True,
        transform=ccrs.PlateCarree(),
    )

    if valid_from is not None and valid_to is not None:
        time_str = format_forecast_range_title_de(valid_from, valid_to)
    else:
        time_str = "-"
    ax.set_title(f"Supercell Composite Parameter  (SCP) - {time_str}", fontsize=14)

    # Colorbar
    legend_h_px      = 50
    legend_bottom_px = 45
    cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=SCP_BOUNDS)
    cbar.ax.tick_params(colors="black", labelsize=7)
    cbar.ax.set_xticklabels([str(int(b)) if b == int(b) else str(b) for b in SCP_BOUNDS], fontsize=7)
    cbar.outline.set_edgecolor("black")
    cbar.ax.set_facecolor("white")

    # Footer
    footer_ax = fig.add_axes([
        0.0,
        (legend_bottom_px + legend_h_px) / FIG_H_PX,
        1.0,
        (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px) / FIG_H_PX,
    ])
    footer_ax.axis("off")

    run_str = f"StrikeCAST ({run_label}), CRX"
    footer_ax.text(
        0.01, 0.85,
        f"Supercell Composite Parameter (SCP), {interval_hours}Std\n{run_str}",
        fontsize=12, fontweight="bold", va="top", ha="left",
    )
    footer_ax.text(
        0.01, 0.25,
        "Daten: ECMWF und AR-CHaMo.",
        fontsize=8, color="black", va="top", ha="left",
    )
    footer_ax.text(0.734, 0.92, "Prognosezeitraum:", fontsize=12,
                   va="top", ha="left", fontweight="bold")

    if valid_from is not None and valid_to is not None:
        range_str = format_forecast_range_de(valid_from, valid_to)
        footer_ax.text(0.962, 0.64, range_str,
                       fontsize=11, va="top", ha="right", fontweight="bold")
    else:
        footer_ax.text(0.962, 0.64, time_str,
                       fontsize=11, va="top", ha="right", fontweight="bold")

    plt.savefig(outfile, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "predictors_*.nc")))

    for f in files:
        ds = xr.open_dataset(f)
        run_label = extract_run_label(ds)

        # Koordinaten
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        if lats.ndim == 1 and lons.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        # Deutschland-Extent zuschneiden
        lat_mask = (lats2d[:, 0] >= 47.0) & (lats2d[:, 0] <= 56.0)
        lon_mask = (lons2d[0, :] >= 5.0)  & (lons2d[0, :] <= 16.0)
        lat_idx  = np.where(lat_mask)[0]
        lon_idx  = np.where(lon_mask)[0]
        lats2d   = lats2d[np.ix_(lat_mask, lon_mask)]
        lons2d   = lons2d[np.ix_(lat_mask, lon_mask)]

        # Zeitscheiben
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time      = ds["time"].values[0]
            step_time      = ds["time"].values[1]
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start       = ds.isel(time=0, drop=True).isel(latitude=lat_idx, longitude=lon_idx)
            valid_from     = pd.Timestamp(prev_time)
            valid_to       = pd.Timestamp(step_time)
        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start       = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_start       = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            time_val       = ds["time"].values[0] if "time" in ds.dims else None
            valid_from     = pd.Timestamp(time_val) if time_val is not None else None
            valid_to       = valid_from + pd.Timedelta(hours=interval_hours) if valid_from else None

        print(f"Processing {f}")

    scp = extract_scp(ds_start)
        print(f"  SCP: min={scp.min():.3f}  max={scp.max():.3f}  mean={scp.mean():.3f}")

        # Ausgabepfad
        if valid_to is not None:
            vt_de   = _to_de_local(valid_to)
            outname = f"supercell_{vt_de.strftime('%Y%m%d_%H%M')}.png"
        else:
            outname = "scp_unknown.png"
        outfile = os.path.join(OUT_DIR, outname)

        plot_scp_png(
            lats2d, lons2d, scp, outfile,
            interval_hours=interval_hours,
            run_label=run_label,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        print(f"  -> {outfile}")

        ds.close()

    print("Fertig!")


if __name__ == "__main__":
    main()
