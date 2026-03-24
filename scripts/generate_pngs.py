#!/usr/bin/env python3

import os
import glob
import re
from zoneinfo import ZoneInfo
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

PRED_DIR = "data/output"
LUT_PATH = "data/lut/lightning_lut.nc"
OUT_DIR  = "pngs/gewitter"
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
TZ_DE = ZoneInfo("Europe/Berlin")


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

lut = xr.open_dataset(LUT_PATH)


# -------------------------------------------------------
# Kartenparameter
# -------------------------------------------------------

FIG_W_PX, FIG_H_PX = 880, 830
BOTTOM_AREA_PX = 179
TOP_AREA_PX = FIG_H_PX - BOTTOM_AREA_PX
TARGET_ASPECT = FIG_W_PX / TOP_AREA_PX


# -------------------------------------------------------
# Hilfsfunktion: Run-Label aus NC holen (ohne Datums-Parsing)
# -------------------------------------------------------

def extract_run_label(ds):
    def _hour_to_label(raw):
        if raw is None:
            return None
        s = str(raw).strip()
        # Bevorzugt "00Z", "12z", "00 UTC"
        m = re.search(r"(?<!\d)(\d{2})\s*(?:Z|z|UTC|utc)\s*$", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}z"
        # Fallback auf letzte zweistellige Stunde am Stringende
        m = re.search(r"(?<!\d)(\d{2})\s*$", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}z"
        return None

    # process.py schreibt attrs["run"] = "00Z"
    run_label = _hour_to_label(ds.attrs.get("run", None))
    if run_label is not None:
        return run_label

    for key in ("run_time_utc", "run_time", "forecast_reference_time", "analysis_time"):
        run_label = _hour_to_label(ds.attrs.get(key, None))
        if run_label is not None:
            return run_label

    return "??z"


# -------------------------------------------------------
# Wahrscheinlichkeit
# -------------------------------------------------------

def compute_probability(ds2d, interval_hours=1):
    """
    ds2d: Datensatz ohne time-Dimension (nur latitude/longitude).
    """
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

    # nach Battaglioli et al. (2023), Eq. (3): P_nh = 1 - (1 - P_1h)^n
    ih = max(1, int(interval_hours))
    prob_interval = 1.0 - np.power(1.0 - prob, ih)
    return np.clip(prob_interval * 100.0, 0, 100)


def smooth_probability(prob, sigma=1.2):
    if sigma <= 0:
        return prob
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(prob, sigma=sigma, mode="nearest")
    except Exception:
        p = np.pad(prob, ((1, 1), (1, 1)), mode="edge")
        return (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0


# -------------------------------------------------------
# Plot mit Karte
# -------------------------------------------------------

def plot_png(lats, lons, prob, outfile, interval_hours=3,
             run_label="??z", valid_from=None, valid_to=None):
    """
    lats, lons    : 2D Arrays
    prob          : Lightning probability in %
    outfile       : Pfad zum PNG
    run_label     : Modell-Run als Label, z.B. "00z"
    valid_from    : Beginn des Prognosezeitraums (pd.Timestamp)
    valid_to      : Ende  des Prognosezeitraums (pd.Timestamp)
    """

    from matplotlib.colors import ListedColormap, BoundaryNorm

    # --- Diskrete Colormap in 5%-Schritten ---
    bounds = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95, 97, 100]
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
    fig = plt.figure(figsize=(FIG_W_PX/100*scale, FIG_H_PX/100*scale), dpi=100)

    # --- Kartenbereich ---
    shift_up = 0.02
    ax = fig.add_axes([0.0, BOTTOM_AREA_PX / FIG_H_PX + shift_up, 1.0, TOP_AREA_PX / FIG_H_PX],
                    projection=ccrs.PlateCarree())
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

    # Lightning-Mesh
    im = ax.pcolormesh(
        lons, lats, prob_plot,
        cmap=cmap, norm=norm,
        shading="gouraud", antialiased=True,
        transform=ccrs.PlateCarree(),
    )

    # Kartentitel = Prognosezeitraum (deutsche Ortszeit), nur aus valid_from / valid_to
    if valid_from is not None and valid_to is not None:
        time_str = format_forecast_range_title_de(valid_from, valid_to)
    else:
        time_str = "–"
    ax.set_title(f"⚡ Lightning Probability ({interval_hours}h) – {time_str}", fontsize=14)

    # --- Colorbar ---
    legend_h_px      = 50
    legend_bottom_px = 45
    cbar_ax = fig.add_axes([0.03, legend_bottom_px / FIG_H_PX, 0.94, legend_h_px / FIG_H_PX])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=bounds)
    cbar.ax.tick_params(colors="black", labelsize=7)
    cbar.outline.set_edgecolor("black")
    cbar.ax.set_facecolor("white")

    # --- Footer (eine add_axes-Zeile wie gewünscht) ---
    footer_ax = fig.add_axes([
        0.0,
        (legend_bottom_px + legend_h_px) / FIG_H_PX,
        1.0,
        (BOTTOM_AREA_PX - legend_h_px - legend_bottom_px) / FIG_H_PX,
    ])
    footer_ax.axis("off")

    # Linker Text: Produktlabel + Modell-Laufzeit
    run_str = f"StrikeCAST ({run_label}), CRX"

    footer_ax.text(
        0.01, 0.85,
        f"Gewitterwahrscheinlichkeit, 3Std\n{run_str}",
        fontsize=12, fontweight="bold", va="top", ha="left",
    )
    footer_ax.text(
        0.01, 0.25,
        "Daten: ECMWF und AR-CHaMo.",
        fontsize=8, color="black", va="top", ha="left",
    )

    # Rechts: "Prognose für:" + von–bis Zeitraum
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

    for f in files:
        ds = xr.open_dataset(f)

        # --- Run-Label aus NC holen (z.B. '00z') ---
        run_label = extract_run_label(ds)

        # --- Zeitscheiben auswerten ---
        if "time" in ds.dims and ds.sizes["time"] == 2:
            prev_time = ds["time"].values[0]
            step_time = ds["time"].values[1]
            print(f"Processing step {prev_time} → {step_time}")

            interval_hours = int(ds.attrs.get("interval_hours", 3))
            if interval_hours < 1:
                dt_hours = (step_time - prev_time) / np.timedelta64(1, "h")
                interval_hours = max(1, int(round(float(dt_hours))))

            ds_start   = ds.isel(time=0, drop=True)
            prob       = compute_probability(ds_start, interval_hours=interval_hours)
            valid_from = pd.Timestamp(prev_time)
            valid_to   = pd.Timestamp(step_time)

        else:
            interval_hours = int(ds.attrs.get("interval_hours", 3))
            ds_start = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
            prob     = compute_probability(ds_start, interval_hours=interval_hours)

            time_val   = ds["time"].values[0] if "time" in ds.dims else None

            if time_val is not None:
                valid_from = pd.Timestamp(time_val)
                valid_to   = valid_from + pd.Timedelta(hours=interval_hours)
            else:
                valid_from = valid_to = None

        # --- Koordinaten ---
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        if lats.ndim == 1 and lons.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            lons2d, lats2d = lons, lats

        # --- Auf Deutschland-Extent zuschneiden ---
        lat_mask = (lats2d[:, 0] >= 47.0) & (lats2d[:, 0] <= 56.0)
        lon_mask = (lons2d[0, :] >= 5.0)  & (lons2d[0, :] <= 16.0)
        lats2d = lats2d[np.ix_(lat_mask, lon_mask)]
        lons2d = lons2d[np.ix_(lat_mask, lon_mask)]
        prob   = prob[np.ix_(lat_mask, lon_mask)]

        # PNG: gewitter_<Ende des Prognosezeitraums in deutscher Ortszeit>.png
        if valid_to is not None:
            vt_de = _to_de_local(valid_to)
            outname = f"gewitter_{vt_de.strftime('%Y%m%d_%H%M')}.png"
        else:
            outname = "gewitter_unknown.png"
        outfile = os.path.join(OUT_DIR, outname)

        print(lats2d.shape, lons2d.shape)
        plot_png(
            lats2d, lons2d, prob, outfile,
            interval_hours=interval_hours,
            run_label=run_label,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        print(f"  → {outfile}")

    print("✅ Fertig!")


if __name__ == "__main__":
    main()
