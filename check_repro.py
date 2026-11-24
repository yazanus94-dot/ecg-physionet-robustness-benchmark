import hashlib
import json
from pathlib import Path

files = [
    "reports/results/cv_metrics.csv",
    "reports/results/noise_results.csv",
    "reports/results/noise_config.json",
    "reports/results/nstdb_evaluation.csv",
    "reports/results/dataset_summary.json",
    "reports/results/confusion_matrices.json",
    "reports/results/per_class_oof.json",
    "reports/results/nstdb_details.json",
]
h = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
before = json.loads(Path("checksums_run3.json").read_text())
diffs = {f: (before[f][:12], h[f][:12]) for f in files if before[f] != h[f]}
print("DIFFS:", diffs if diffs else "NONE — reproducible")
