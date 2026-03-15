# generate_metadata.py
import json
import os
import sys
import glob
from datetime import datetime, timezone

OUTPUT_DIR   = "warnmos"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

def generate_metadata():
    # IDX-Dateien einlesen
    idx_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.om.idx")))

    if not idx_files:
        print(f"Keine .om.idx Dateien in {OUTPUT_DIR} gefunden!")
        sys.exit(1)

    runs = []
    for idx_file in idx_files:
        with open(idx_file) as f:
            index = json.load(f)

        if not index:
            continue

        timestamps  = [b["timestamp"] for b in index]
        n_steps     = len(timestamps)
        time_start  = min(timestamps)
        time_end    = max(timestamps)
        om_basename = os.path.basename(idx_file).replace(".om.idx", "")

        runs.append({
            "file":       om_basename,
            "timesteps":  n_steps,
            "timeStart":  time_start,
            "timeEnd":    time_end,
        })

        print(f"  {om_basename}: {n_steps} Steps  {time_start} → {time_end}")

    metadata = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "runs":        runs,
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata geschrieben: {METADATA_FILE}")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    generate_metadata()
