#!/usr/bin/env python3

import os
import glob
import re
from zoneinfo import ZoneInfo
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict

PRED_DIR      = "data/output"
LUT_HAIL_PATH = "data/lut/hail2cm_lut.nc"
OUT_DIR       = "pngs/hail2cm"
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
    """Datum und Uhrzeit in deutscher Ortszeit (ohne Zeitzonen-Label)."""
    t = _to_de_local(ts)
    return f"{t:%d.%m.%Y %H:%M}"


def format_forecast_range_de(valid_from, valid_to):
    """Prognosezeitraum zweizeilig: von ... / bis ... in deutscher Zeit."""
    return (
        f"von {format_de_datetime(valid_from)}\n"
        f"bis {format_de_datetime(valid_to)}"
    )


def format_forecast_range_title_de(valid_from, valid_to):
    """Einzeilige Bereichsanzeige für den Kartentitel."""
    return f"{format_de_datetime(valid_from)} - {format_de_datetime(valid_to)}"


# -------------------------------------------------------
# LUT laden
# -------------------------------------------------------

lut_hail = xr.open_dataset(LUT_HAIL_PATH)


# -------------------------------------------------------
# Kartenparameter
# -------------------------------------------------------

FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX    = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT  = FIG_W_PX / TOP_AREA_PX


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
# Hagelwahrscheinlichkeit
# -------------------------------------------------------

def compute_hail_probability(ds2d, lut, interval_hours=1):
    """
    Hagelwahrscheinlichkeit >=2cm.
    LUT-Dimensionen: BS_EFF_MU, MU_CAPE_M10, ML_MixingRatio, ZeroHeight

    Prädiktor-Mapping:
      BS_EFF_MU     <- MU_EFF_BS  (identische Größe, nur anderer Variablenname)
      MU_CAPE_M10   <- MU_CAPE_M10
      ML_MixingRatio <- MU_MIXR  (Mixed-Layer Mixing Ratio, identische Größe)
      ZeroHeight    <- ZeroHeight
    """

    # MU_EFF_BS == BS_EFF_MU (effektiver Bulk-Shear Most-Unstable Layer)
    bs_raw = ds2d["MU_EFF_BS"].values if "MU_EFF_BS" in ds2d else ds2d["BS_EFF_MU"].values
    bs   = np.clip(bs_raw,
                   float(lut["BS_EFF_MU"].min()),       float(lut["BS_EFF_MU"].max()))
    cape = np.clip(ds2d["MU_CAPE_M10"].values,
                   float(lut["MU_CAPE_M10"].min()),     float(lut["MU_CAPE_M10"].max()))
    mixr_raw = ds2d["MU_MIXR"].values if "MU_MIXR" in ds2d else ds2d["ML_MixingRatio"].values
    mixr = np.clip(mixr_raw,
                   float(lut["ML_MixingRatio"].min()),  float(lut["ML_MixingRatio"].max()))
    zh   = np.clip(ds2d["ZeroHeight"].values,
                   float(lut["ZeroHeight"].min()),       float(lut["ZeroHeight"].max()))

    prob = lut["prob_hail_ge2cm"].interp(
        BS_EFF_MU      =("points", bs.flatten()),
        MU_CAPE_M10    =("points", cape.flatten()),
        ML_MixingRatio =("points", mixr.flatten()),  # LUT-Koordinate bleibt ML_MixingRatio,
        ZeroHeight     =("points", zh.flatten()),
        method="linear",
    )
    prob = prob.values.reshape(ds2d["ZeroHeight"].shape)
    prob = np.nan_to_num(prob, nan=0.0)
    prob = np.clip(prob, 0.0, 1.0)

    # Physikalischer Guard: kein CAPE -> kein Hagel
    # unter 200 J/kg dämpfen, darunter -> 0
    cape_raw    = ds2d["MU_CAPE_M10"].values
    cape_weight = np.clip(cape_raw / 200.0, 0.0, 1.0)
    prob       *= cape_weight

    # Nullgradgrenze zu tief -> großer Hagel unwahrscheinlicher
    # unter 1500m -> 0, zwischen 1500-2000m -> linear auf 1
    zh_raw    = ds2d["ZeroHeight"].values
    zh_weight = np.clip((zh_raw - 1500.0) / 500.0, 0.0, 1.0)
    prob     *= zh_weight

    # Orographie-Maske
    if "z_sfc" in ds2d:
        z         = ds2d["z_sfc"].values
        z_max     = maximum_filter(z, size=2, mode="nearest")
        orog_weight = np.clip(1.0 - (z_max - 600.0) / 900.0, 0.0, 1.0)
        prob     *= orog_weight

    # --- Debug: Ausreißer diagnostizieren ---
    high_mask = prob > 0.3
    if high_mask.any():
        print(f"\n🧊 {high_mask.sum()} Gitterpunkte mit Hagel-prob > 30%")
        print(f"   prob:         min={prob[high_mask].min():.3f}  max={prob[high_mask].max():.3f}")
        print(f"   MU_CAPE_M10:  min={cape_raw[high_mask].min():.0f}  max={cape_raw[high_mask].max():.0f} J/kg")
        print(f"   ZeroHeight:   min={zh_raw[high_mask].min():.0f}  max={zh_raw[high_mask].max():.0f} m")
        print(f"   BS_EFF_MU:    min={bs_raw[high_mask].min():.1f}  max={bs_raw[high_mask].max():.1f} m/s")
        print(f"   ML_MixingRatio: min={mixr_raw[high_mask].min():.2f}  max={mixr_raw[high_mask].max():.2f}")

        idx = np.unravel_index(np.argmax(prob), prob.shape)
        print(f"\n   Höchster Wert bei Index {idx}:")
        print(f"   prob={prob[idx]:.3f}  MU_CAPE_M10={cape_raw[idx]:.0f}  ZeroHeight={zh_raw[idx]:.0f}m  BS_EFF_MU={bs_raw[idx]:.1f}")

    # Intervall-Skalierung ZULETZT
    ih            = max(1, int(interval_hours))
    prob_interval = 1.0 - np.power(1.0 - prob, ih)
    return np.clip(prob_interval * 100.0, 0.0, 100.0)


# -------------------------------------------------------
# Plot mit Karte
# -------------------------------------------------------

def plot_png(lats, lons, prob, outfile, interval_hours=3,
             run_label="??z", valid_from=None, valid_to=None):
    """
    lats, lons    : 2D Arrays
    prob          : Hail probability in %
    outfile       : Pfad zum PNG
    run_label     : Modell-Run als Label, z.B. "00z"
    valid_from    : Beginn des Prognosezeitraums (pd.Timestamp)
    valid_to      : Ende  des Prognosezeitraums (pd.Timestamp)
    """

    # --- Diskrete Colormap in 5%-Schritten ---
    bounds = [0,1,2,5,10,20,30,40,50,60,70,80,90,95,98,100]
    colors = [
        "#056B6F","#079833","#08B015","#40C50C","#7DD608",
        "#9BE105","#BBEA04","#DBF402","#FFF600","#FEDC00",
        "#FFAD00","#FF6300","#E50014","#BC0035","#930058","#660179"
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- Interpolation auf feineres Gitter ---
    target_res = 0.025
    lon_min, lon_max = np.min(lons), np.max(lons)
    lat_min, lat_max = np.min(lats), np.max(lats)
    lon_new = np.arange(lon_min, lon_max + target_res, target_res)
    lat_new = np.arange(lat_min, lat_max + target_res, target_res)
    lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)

    interp = RegularGridInterpolator(
        (lats[:, 0], lons[0, :]), prob, method="linear", bounds_error=False, fill_value=0.0
    )
    prob_plot = interp((lat2d_new, lon2d_new))
    prob_plot = np.clip(prob_plot, 0.0, 100.0)
    lats, lons = lat2d_new, lon2d_new

    # --- Figurenmaße ---
    scale = 0.9
    fig = plt.figure(figsize=(FIG_W_PX / 100 * scale, FIG_H_PX / 100 * scale), dpi=100)

    # --- Kartenbereich ---
    shift_up = 0.02
    ax = fig.add_axes(
        [0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
        projection=ccrs.PlateCarree()
    )
    ax.set_extent(EXTENT)
    ax.set_axis_off()
    ax.set_aspect('auto')

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

    # Hagel-Mesh
    im = ax.pcolormesh(
        lons, lats, prob_plot,
        cmap=cmap, norm=norm,
        shading="gouraud", antialiased=True,
        transform=ccrs.PlateCarree(),
    )

    # Kartentitel
    if valid_from is not None and valid_to is not None:
        time_str = format_forecast_range_title_de(valid_from, valid_to)
    else:
        time_str = "-"
    ax.set_title(f"Hail Probability >=2cm ({interval_hours}h) - {time_str}", fontsize=14)

    # --- Colorbar ---
    legend_h_px      = 50
    legend_bottom_px = 45
    cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
    cbar.ax.tick_params(colors="black", labelsize=7)
    cbar.outline.set_edgecolor("black")
    cbar.ax.set_facecolor("white")

    # --- Footer ---
    footer_ax = fig.add_axes([
        0.0,
        (legend_bottom_px + legend_h_px) / FIG_H_PX,
        1.0,
        (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px) / FIG_H_PX,
    ])
    footer_ax.axis("off")

    run_str = f"HailCAST ({run_label}), CRX"

    footer_ax.text(
        0.01, 0.85,
        f"Hagelwahrscheinlichkeit (>2cm), {interval_hours}Std (%)\n{run_str}",
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
        footer_ax.text(
            0.962, 0.64, range_str,
            fontsize=11, va="top", ha="right", fontweight="bold",
        )
    else:
        footer_ax.text(
            0.962, 0.64, time_str,
            fontsize=11, va="top", ha="right", fontweight="bold",
        )

    plt.savefig(outfile, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "predictors_*.nc")))

    # -------------------------------------------------------
    # 3h-Karten
    # -------------------------------------------------------
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

        # Auf Deutschland-Extent zuschneiden
        lat_mask = (lats2d[:, 0] >= 47.0) & (lats2d[:, 0] <= 56.0)
        lon_mask = (lons2d[0, :] >= 5.0)  & (lons2d[0, :] <= 16.0)
        lat_idx  = np.where(lat_mask)[0]
        lon_idx  = np.where(lon_mask)[0]
        lats2d   = lats2d[np.ix_(lat_mask, lon_mask)]
        lons2d   = lons2d[np.ix_(lat_mask, lon_mask)]

        # Zeitscheiben auswerten
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time = ds["time"].values[0]
            step_time = ds["time"].values[1]
            print(f"Processing step {prev_time} -> {step_time}")

            interval_hours = int(ds.attrs.get("interval_hours", 3))
            if interval_hours < 1:
                dt_hours = (step_time - prev_time) / np.timedelta64(1, "h")
                interval_hours = max(1, int(round(float(dt_hours))))

            ds_start   = ds.isel(time=0, drop=True)
            ds_start   = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            prob       = compute_hail_probability(ds_start, lut_hail, interval_hours=interval_hours)
            valid_from = pd.Timestamp(prev_time)
            valid_to   = pd.Timestamp(step_time)

        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start   = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            ds_start   = ds_start.isel(latitude=lat_idx, longitude=lon_idx)
            prob       = compute_hail_probability(ds_start, lut_hail, interval_hours=interval_hours)
            time_val   = ds["time"].values[0] if "time" in ds.dims else None

            if time_val is not None:
                valid_from = pd.Timestamp(time_val)
                valid_to   = valid_from + pd.Timedelta(hours=interval_hours)
            else:
                valid_from = valid_to = None

        # PNG: hagel_<Ende des Prognosezeitraums in deutscher Ortszeit>.png
        if valid_to is not None:
            vt_de   = _to_de_local(valid_to)
            outname = f"hagel_{vt_de.strftime('%Y%m%d_%H%M')}.png"
        else:
            outname = "hagel_unknown.png"
        outfile = os.path.join(OUT_DIR, outname)

        print(f"  prob: min={prob.min():.1f}%  max={prob.max():.1f}%  mean={prob.mean():.1f}%")
        print(lats2d.shape, lons2d.shape)

        plot_png(
            lats2d, lons2d, prob, outfile,
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
