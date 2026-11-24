import os
import tempfile
from scripts.export_metrics import export_to_json, export_to_csv

def test_export_metrics_file_creation() -> None:
    sample_metrics = {
        "TWAP": {"mean_is_bps": -31.79, "std_is_bps": 5.28, "sharpe": -6.02, "cvar_95_bps": -42.09},
        "C51": {"mean_is_bps": -9.26, "std_is_bps": 8.15, "sharpe": -1.14, "cvar_95_bps": -26.30},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "metrics.json")
        csv_path = os.path.join(tmpdir, "metrics.csv")
        export_to_json(sample_metrics, json_path)
        export_to_csv(sample_metrics, csv_path)
        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)
