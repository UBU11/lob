"""Export benchmark metrics dictionary to JSON and CSV formats."""
import json
import csv
from typing import Any

def export_to_json(metrics: dict[str, Any], output_path: str) -> None:
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            clean_metrics[k] = {ik: iv for ik, iv in v.items() if not ik.startswith("raw_")}
        else:
            clean_metrics[k] = v
    with open(output_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)

def export_to_csv(metrics: dict[str, dict[str, float]], output_path: str) -> None:
    fieldnames = ["policy", "mean_is_bps", "std_is_bps", "sharpe", "cvar_95_bps"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name, values in metrics.items():
            row = {"policy": policy_name}
            for k in fieldnames[1:]:
                row[k] = values.get(k, 0.0)
            writer.writerow(row)
