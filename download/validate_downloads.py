#!/usr/bin/env python3
"""
validate_downloads.py – Prüft ob alle ECMWF-Downloads vollständig sind.

Bricht mit exit(1) ab wenn Dateien fehlen oder unvollständig sind.
Im GitHub Actions Workflow NACH dem Download-Step ausführen.
"""

import os
import sys
import struct

# ── Konfiguration ──────────────────────────────────────────────────────────────

STEPS           = list(range(0, 49, 3))   # 0–48h alle 3h (17 Steps)
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
N_LEVELS        = len(PRESSURE_LEVELS)    # 13

DATA_ROOT = os.path.join("data", "gewitter")

# (param, levtype, expected_messages_per_file, steps)
EXPECTED = [
    # Drucknivau-Parameter: 13 Messages pro Datei (1 pro Level)
    ("r",      "pl", N_LEVELS, STEPS),
    ("t",      "pl", N_LEVELS, STEPS),
    ("q",      "pl", N_LEVELS, STEPS),
    ("u",      "pl", N_LEVELS, STEPS),
    ("v",      "pl", N_LEVELS, STEPS),
    ("gh",     "pl", N_LEVELS, STEPS),
    # Oberflächen-Parameter: 1 Message pro Datei
    ("2t",     "sfc", 1, STEPS),
    ("2d",     "sfc", 1, STEPS),
    ("sp",     "sfc", 1, STEPS),
    ("tp",     "sfc", 1, STEPS),
    ("10u",    "sfc", 1, STEPS),
    ("10v",    "sfc", 1, STEPS),
    ("lsm",    "sfc", 1, STEPS),
    ("mucape", "sfc", 1, STEPS),
    # Orographie: nur Step 0
    ("z",      "sfc", 1, [0]),
]

# ── GRIB2-Message-Zähler ───────────────────────────────────────────────────────

def count_grib2_messages(path: str) -> int:
    """Zählt GRIB2-Messages anhand der Binär-Header (ohne cfgrib/eccodes)."""
    count = 0
    try:
        with open(path, "rb") as f:
            while True:
                header = f.read(16)
                if len(header) < 16 or header[:4] != b"GRIB":
                    break
                msg_len = int.from_bytes(header[8:16], "big")
                count += 1
                f.seek(msg_len - 16, 1)
    except Exception:
        return -1
    return count

# ── Validierung ────────────────────────────────────────────────────────────────

def validate() -> bool:
    errors   = []
    warnings = []
    total_files = 0
    ok_files    = 0

    for param, levtype, expected_msgs, steps in EXPECTED:
        folder = os.path.join(DATA_ROOT, param)

        for step in steps:
            total_files += 1
            if levtype == "pl":
                filename = f"{param}_pl_step_{step:03d}.grib2"
            else:
                filename = f"{param}_step_{step:03d}.grib2"

            path = os.path.join(folder, filename)

            if not os.path.exists(path):
                errors.append(f"FEHLT:        {path}")
                continue

            size = os.path.getsize(path)
            if size == 0:
                errors.append(f"LEER:         {path}")
                continue

            n = count_grib2_messages(path)
            if n == -1:
                errors.append(f"LESEFEHLER:   {path}")
            elif n < expected_msgs:
                errors.append(
                    f"UNVOLLSTÄNDIG: {path}  "
                    f"({n}/{expected_msgs} Messages, {size:,} Bytes)"
                )
            else:
                ok_files += 1

    # ── Bericht ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Download-Validierung")
    print(f"{'='*60}")
    print(f"  Gesamt geprüft : {total_files} Dateien")
    print(f"  OK             : {ok_files}")
    print(f"  Fehler         : {len(errors)}")
    print(f"{'='*60}\n")

    if errors:
        print("❌ FEHLERHAFTE / FEHLENDE DATEIEN:")
        for e in errors:
            print(f"   {e}")
        print()
        return False

    print("✅ Alle Downloads vollständig und valide.")
    return True


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
