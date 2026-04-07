import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import os
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------
# Konfiguration via Umgebungsvariablen (GitHub Actions)
# -------------------------------------------------------
DATE = os.getenv("DATE", "20250604")
RUN  = f"{int(os.getenv('RUN', 0)):02d}z"
RUN_HHMM = f"{int(os.getenv('RUN', 0)):02d}0000"
BASE = Path("data/gewitter")

BUCKET = "ecmwf-forecasts"

PRESSURE_LEVELS  = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PARAMS_SFC       = ["2t", "2d", "sp", "tp", "lsm", "mucape", "10u", "10v"]
PARAMS_PL        = ["t", "q", "r", "u", "v", "gh"]
STEPS            = list(range(6, 58, 3))

MAX_WORKERS      = 4
DOWNLOAD_RETRIES = 8
BACKOFF_BASE     = 1.5   # Sekunden
BACKOFF_MAX      = 60.0  # Sekunden

# -------------------------------------------------------
# Geteilter S3-Client (ein Client für alle Threads)
# -------------------------------------------------------
_client_lock   = threading.Lock()
_shared_client = None

def get_client():
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = boto3.client(
                "s3",
                region_name="eu-central-1",
                config=Config(
                    signature_version=UNSIGNED,
                    retries={
                        "max_attempts": 3,   # nur kurze botocore-Retries; Hauptlogik unten
                        "mode": "adaptive"
                    }
                )
            )
        return _shared_client

# -------------------------------------------------------
# Index laden
# -------------------------------------------------------
def get_fields_for_step(step):
    step_str = f"{step}h"
    prefix   = f"{DATE}/{RUN}/ifs/0p25/oper"
    idx_key  = f"{prefix}/{DATE}{RUN_HHMM}-{step_str}-oper-fc.index"
    try:
        s3    = get_client()
        obj   = s3.get_object(Bucket=BUCKET, Key=idx_key)
        lines = obj["Body"].read().decode("utf-8").strip().splitlines()
        return [json.loads(l) for l in lines], step_str, prefix
    except Exception as e:
        print(f"  ✗ Index nicht gefunden: {idx_key} ({e})")
        return None, step_str, prefix

# -------------------------------------------------------
# Download eines einzelnen Byte-Range  –  mit Backoff
# -------------------------------------------------------
def download_field(grib_key, field, out_path):
    if out_path.exists():
        return True, str(out_path), field["_length"]

    start = field["_offset"]
    end   = field["_offset"] + field["_length"] - 1

    for attempt in range(DOWNLOAD_RETRIES):
        try:
            s3   = get_client()
            resp = s3.get_object(
                Bucket=BUCKET,
                Key=grib_key,
                Range=f"bytes={start}-{end}"
            )
            chunk = resp["Body"].read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(chunk)
            return True, str(out_path), field["_length"]

        except Exception as e:
            err_str  = str(e)
            is_throttle = any(kw in err_str for kw in (
                "SlowDown", "503", "reduce your request rate",
                "RequestThrottled", "Throttling"
            ))

            last_attempt = (attempt == DOWNLOAD_RETRIES - 1)

            if is_throttle and not last_attempt:
                # Exponential Backoff mit vollem Jitter
                wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
                wait = random.uniform(0, wait)          # Full-Jitter
                print(f"  ⏳ Throttle ({attempt+1}/{DOWNLOAD_RETRIES}) "
                      f"– warte {wait:.1f}s: {out_path.name}")
                time.sleep(wait)
                continue

            if not last_attempt:
                # Anderer Fehler (Netzwerk etc.) – kurze Pause, dann retry
                time.sleep(random.uniform(1, 3))
                continue

            return False, str(out_path), err_str

    return False, str(out_path), "Max retries überschritten"

# -------------------------------------------------------
# Tasks aufbauen – normale Steps (6-144)
# -------------------------------------------------------
def build_download_tasks(fields, step, step_str, prefix):
    grib_key = f"{prefix}/{DATE}{RUN_HHMM}-{step_str}-oper-fc.grib2"
    step_tag = f"step{step:03d}"
    tasks    = []

    for param in PARAMS_SFC:
        for field in fields:
            if field["param"] == param and field.get("levtype") == "sfc":
                out = BASE / param / f"{param}_{step_tag}.grib2"
                tasks.append((grib_key, field, out))

    for param in PARAMS_PL:
        for level in PRESSURE_LEVELS:
            for field in fields:
                if (field["param"] == param
                        and field.get("levtype") == "pl"
                        and field.get("levelist") == str(level)):
                    out = BASE / f"{param}_pl" / f"{param}_pl{level}_{step_tag}.grib2"
                    tasks.append((grib_key, field, out))

    return tasks

# -------------------------------------------------------
# Hauptprogramm
# -------------------------------------------------------
def main():
    print(f"=== ECMWF Download ===")
    print(f"DATE={DATE}  RUN={RUN}  Steps={STEPS[0]}-{STEPS[-1]}h")
    print(f"Ausgabe: {BASE}/\n")

    all_tasks = []

    print("Lade Orographie (z) von Step 0 ...")
    fields_0, _, prefix_0 = get_fields_for_step(0)
    if fields_0 is not None:
        grib_key_0 = f"{prefix_0}/{DATE}{RUN_HHMM}-0h-oper-fc.grib2"
        for field in fields_0:
            if field["param"] == "z":
                out = BASE / "z" / "z_step000.grib2"
                all_tasks.append((grib_key_0, field, out))
                print(f"  z_step000.grib2 geplant")
                break
    else:
        print("  ⚠️  Step 0 Index nicht verfügbar – z_step000.grib2 fehlt!")

    print(f"\nIndizes laden für Steps {STEPS[0]}-{STEPS[-1]}h ...")
    for step in STEPS:
        fields, step_str, prefix = get_fields_for_step(step)
        if fields is None:
            continue
        tasks = build_download_tasks(fields, step, step_str, prefix)
        all_tasks.extend(tasks)
        print(f"  Step {step:3d}h → {len(tasks):3d} Felder geplant")

    if not all_tasks:
        print("Keine Tasks – Abbruch.")
        return

    print(f"\nStarte {len(all_tasks)} Downloads mit {MAX_WORKERS} parallelen Threads ...\n")

    ok = err = total_bytes = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_field, gk, fld, op): op
                   for gk, fld, op in all_tasks}
        for future in as_completed(futures):
            success, path, info = future.result()
            if success:
                ok          += 1
                total_bytes += info
                short = Path(path).relative_to(BASE)
                print(f"  ✓ {short}  ({info/1024:.0f} KB)")
            else:
                err += 1
                print(f"  ✗ {path}: {info}")

    print(f"\nFertig: {ok} OK, {err} Fehler, {total_bytes/1024/1024:.1f} MB gesamt")
    if err > 0:
        raise SystemExit(f"{err} Download-Fehler – Job schlägt fehl")

if __name__ == "__main__":
    main()
