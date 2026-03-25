#!/usr/bin/env python3
"""
download_2d.py – Lädt '2d' (2d (2m Taupunkt), Oberfläche, alle Steps) herunter.
"""

import os, time, struct, sys
from ecmwf.opendata import Client

DATE = os.getenv("DATE", "20260325")
TIME = int(os.getenv("RUN", 0))
DATE_ISO = f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:8]}"

STEPS  = list(range(0, 49, 3))
PARAM  = "2d"
FOLDER = os.path.join("data", "gewitter", PARAM)
TARGET   = os.path.join(FOLDER, f"{PARAM}_all_steps.grib2")

EXPECTED_MESSAGES = 1   # Oberfläche: 1 Message pro Step
MAX_RETRIES       = 3
RETRY_DELAY       = 10

client = Client(source="aws")


def count_grib2_messages(path: str) -> int:
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


def is_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    n = count_grib2_messages(path)
    if n < EXPECTED_MESSAGES:
        print(f"  ⚠️  {path}: {n} Messages (erwartet ≥{EXPECTED_MESSAGES}) – unvollständig")
        return False
    return True


def main():
    print(f"=== Download: {PARAM} (Oberfläche, alle Steps in eine Datei) ===")
    print(f"Datum: {DATE_ISO}  Lauf: {TIME:02d} UTC")
    print(f"Steps: {STEPS}")
    print(f"Ziel:  {TARGET}")
    print()
 
    os.makedirs(FOLDER, exist_ok=True)
 
    if is_valid(TARGET):
        print(f"✅ Bereits vorhanden und valide: {TARGET}")
        return
 
    if os.path.exists(TARGET):
        os.remove(TARGET)
 
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(
                date=DATE_ISO,
                time=TIME,
                type="fc",
                step=STEPS,        # alle Steps auf einmal
                param=PARAM,
                levtype="sfc",
                target=TARGET,
            )
            if is_valid(TARGET):
                print(f"✅ Fertig: {TARGET}  ({count_grib2_messages(TARGET)} Messages)")
                return
            print(f"  ⚠️  Versuch {attempt}/{MAX_RETRIES}: Download unvollständig")
        except Exception as e:
            print(f"  ⚠️  Versuch {attempt}/{MAX_RETRIES} Fehler: {e}")
 
        if os.path.exists(TARGET):
            os.remove(TARGET)
        if attempt < MAX_RETRIES:
            print(f"     Warte {RETRY_DELAY}s vor erneutem Versuch...")
            time.sleep(RETRY_DELAY)
 
    print(f"❌ FEHLER: Download nach {MAX_RETRIES} Versuchen fehlgeschlagen!")
    sys.exit(1)
 
 
if __name__ == "__main__":
    main()
