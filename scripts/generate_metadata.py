import os
import sys
import json
from datetime import datetime

run  = sys.argv[1]
date = sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().strftime("%Y%m%d")

png_root = "pngs"
var_types = ["gewitter", "hail2cm", "tornado"]

metadata = {
    "run": run,
    "date": date,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "products": {}
}

for var in var_types:
    folder = os.path.join(png_root, var)
    timesteps = []

    if os.path.isdir(folder):
        files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
        for f in files:
            name = f.replace(".png", "")
            parts = name.split("_")
            if len(parts) >= 3:
                timestep = parts[-2] + "_" + parts[-1]
                timesteps.append(timestep)

    metadata["products"][var] = {
        "var_type": var,
        "timesteps": timesteps
    }

meta_path = os.path.join(png_root, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Metadata written to {meta_path}")
