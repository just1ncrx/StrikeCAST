# Gemeinsame Kodierung: wird von generate_gewitter.py, generate_supercell.py
# UND generate_metadata.py benutzt (gleiches Format wie ICON-D2).
# Stufen: 0..254 = Wert, 255 = NoData.
# transform:
#   "linear": u = (v - vmin) / (vmax - vmin)
#   "sqrt"  : u = sqrt((v - vmin) / (vmax - vmin))
#   "ssqrt" : u = (sign(v)*sqrt(|v|/vmax) + 1) / 2   (vmin = -vmax)
#   "raw"   : q = Wert direkt
#
# Diese Datei bleibt bewusst stdlib-only (numpy wird nur lazy in encode_uint8
# importiert), damit generate_metadata.py im Deploy-Job ohne pip install läuft.
NODATA_U8 = 255
LEVELS = 254

ENCODING = {
    # Gewitterwahrscheinlichkeit im Prognoseintervall, 0..100 %.
    # sqrt = feinere Auflösung bei kleinen Werten (Schwellen bei 1/2/5 %).
    "gewitter": {"vmin": 0.0, "vmax": 100.0, "transform": "sqrt", "unit": "%"},
    # Supercell Composite Parameter, 0..50 (Index, einheitenlos).
    # sqrt = feinere Auflösung bei kleinen Werten (Schwellen bei 0.2/0.5/1 ...).
    "scp":      {"vmin": 0.0, "vmax": 50.0,  "transform": "sqrt", "unit": ""},
}

# Festes Raster = IDENTISCH zum ICON-D2-Raster (EPSG:4326, 0.02°, row0 = Norden).
# StrikeCAST wird per Interpolation auf genau dieses Gitter gelegt, damit das
# Frontend beide Quellen mit demselben Code lesen kann.
GRID = {
    "extent": [-3.94, 43.18, 20.34, 58.08],   # lon_min, lat_min, lon_max, lat_max
    "width": 1215,
    "height": 746,
    "nodata": NODATA_U8,
    "levels": LEVELS,
}


def encode_uint8(data, enc):
    """float-Array (NaN erlaubt) -> uint8-Array (255 = NoData)."""
    import numpy as np

    nan_mask = ~np.isfinite(data)
    vmin, vmax, tr = enc["vmin"], enc["vmax"], enc["transform"]
    d = np.where(nan_mask, vmin, data).astype(np.float64)

    if tr == "raw":
        q = np.clip(np.rint(d), 0, LEVELS)
    else:
        d = np.clip(d, vmin, vmax)
        if tr == "linear":
            u = (d - vmin) / (vmax - vmin)
        elif tr == "sqrt":
            u = np.sqrt((d - vmin) / (vmax - vmin))
        elif tr == "ssqrt":
            u = (np.sign(d) * np.sqrt(np.abs(d) / vmax) + 1.0) / 2.0
        else:
            raise ValueError(tr)
        q = np.rint(u * LEVELS)

    q = q.astype(np.uint8)
    q[nan_mask] = NODATA_U8
    return q