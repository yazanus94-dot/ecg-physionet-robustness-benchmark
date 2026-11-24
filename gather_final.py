import json
from pathlib import Path

pc = json.loads(Path("reports/results/per_class_oof.json").read_text())
for model in sorted(pc):
    print("==", model)
    for c in ["N", "S", "V", "F", "Q"]:
        r = pc[model][c]
        print(
            f"  {c}: P={r['precision']:.4f} R={r['recall']:.4f} "
            f"S={r['sensitivity']:.4f} Spec={r['specificity']:.4f} "
            f"F1={r['f1']:.4f} support={r['support']}"
        )
ds = json.loads(Path("reports/results/dataset_summary.json").read_text())
mitdb = ds["mitdb"]
print("== dataset_summary")
for k in (
    "n_records",
    "total_beats_annotated",
    "total_beats_aami",
    "total_beats_mapped_full",
    "n_reserved_beats",
    "excluded_annotations",
    "unknown_annotations",
    "sampling_frequency_hz",
    "sampling_frequencies_hz",
):
    print(f"  {k}: {mitdb.get(k)}")
print("  class_counts:", mitdb.get("class_counts"))
nc = json.loads(Path("reports/results/noise_config.json").read_text())
print("== noise_config")
print("  noise_rng_seed:", nc["noise_rng_seed"])
print("  snrs:", nc["snrs_db"])
print("  n_noise_realizations:", nc["n_noise_realizations"])
print("  noise_record_ids:", nc["noise_record_ids"])
print("  achieved_snr_db_by_snr:", nc["achieved_snr_db_by_snr"])
