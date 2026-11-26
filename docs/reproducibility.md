# Reproducibility — ECG-ROBUST

## Environment
- Python >= 3.10. Install: `pip install -e .` (and `pip install -r dev-requirements.txt` for dev tools).

## Data
- Place MITDB and NSTDB under `data/raw/mitdb` and `data/raw/nstdb`.
- Raw data is never committed (licensing).

## Run
```
python -m ecg_robust.pipeline
```
Regenerates `reports/results/*.csv`, `reports/TECHNICAL_REPORT.md`, figures.

## Tests
```
python -m pytest -q tests
```
Result: 147 passed, 1 skipped (verified 2026-11-26).
