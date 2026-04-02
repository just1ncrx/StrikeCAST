import boto3
from botocore import UNSIGNED
from botocore.config import Config
import json
import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------
# Konfiguration via Umgebungsvariablen (GitHub Actions)
# -------------------------------------------------------
DATE = os.getenv("DATE", "20250604")
RUN  = f"{int(os.getenv('RUN', 0)):02d}z"   # "00z", "06z", "12z", "18z"
BASE = Path("data/gewitter")

BUCKET = "ecmwf-forecasts"

PRESSURE_LEVELS  = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PARAMS_SFC       = ["2t", "2d", "sp", "tp", "lsm", "mucape", "10u", "10v"]
PARAMS_PL        = ["t", "q", "r", "u", "v", "gh"]
STEPS            = list(range(6, 145, 3))   # 6, 9, 12, … 144

# -------------------------------------------------------
# Thread-lokaler S3-Client (thread-safe)
# -------------------------------------------------------
_local = threading.local()

def get_client():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client(
            "s3",
            region_name="eu-central-1",
            config=Config(signature_version=UNSIGNED)
        )
    return _local.s3

# -------------------------------------------------------
# Index laden
# -------------------------------------------------------
def get_fields_for_step(step):
    step_str = f"{step}h"
    prefix   = f"{DATE}/{RUN}/ifs/0p25/oper"
    idx_key  = f"{prefix}/{DATE}000000-{step_str}-oper-fc.index"
    try:
        s3    = get_client()
        obj   = s3.get_object(Bucket=BUCKET, Key=idx_key)
        lines = obj["Body"].read().decode("utf-8").strip().splitlines()
        return [json.loads(l) for l in lines], step_str, prefix
    except Exception as e:
        print(f"  ✗ Index nicht gefunden: {idx_key} ({e})")
        return None, step_str, prefix

# -------------------------------------------------------
# Download eines einzelnen Byte-Range
# -------------------------------------------------------
def download_field(grib_key, field, out_path):
    if out_path.exists():
        return True, str(out_path), field["_length"]

    start = field["_offset"]
    end   = field["_offset"] + field["_length"] - 1
    try:
        s3    = get_client()
        resp  = s3.get_object(Bucket=BUCKET, Key=grib_key, Range=f"bytes={start}-{end}")
        chunk = resp["Body"].read()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(chunk)
        return True, str(out_path), field["_length"]
    except Exception as e:
        return False, str(out_path), str(e)

# -------------------------------------------------------
# Tasks aufbauen – normale Steps (6-144)
# -------------------------------------------------------
def build_download_tasks(fields, step, step_str, prefix):
    grib_key = f"{prefix}/{DATE}000000-{step_str}-oper-fc.grib2"
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

    # ── Orographie: einmalig Step 0 ──────────────────────────────
    print("Lade Orographie (z) von Step 0 ...")
    fields_0, _, prefix_0 = get_fields_for_step(0)
    if fields_0 is not None:
        grib_key_0 = f"{prefix_0}/{DATE}000000-0h-oper-fc.grib2"
        for field in fields_0:
            if field["param"] == "z":
                out = BASE / "z" / "z_step000.grib2"
                all_tasks.append((grib_key_0, field, out))
                print(f"  z_step000.grib2 geplant")
                break
    else:
        print("  ⚠️  Step 0 Index nicht verfügbar – z_step000.grib2 fehlt!")

    # ── Alle anderen Parameter Steps 6-144 ───────────────────────
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

    print(f"\nStarte {len(all_tasks)} Downloads mit 10 parallelen Threads ...\n")

    ok = err = total_bytes = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
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
