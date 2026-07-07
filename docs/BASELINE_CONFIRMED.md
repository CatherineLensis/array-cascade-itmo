# BASELINE_CONFIRMED — automotive synth-val (local reproduction)

**Date:** 2026-06-28
**Source:** `output/metrics_doa_*.json`

## DOA metrics (synth-val holdout)

| Method | RMSE |
|---|---|
| siamese | 5.60° |
| lista | 5.23° |
| covrecon | 27.21° |
| hiergru | 9.01° |

## Real-data DOA domain gap (filtered ±70°)

| Method | RMSE | MedAE | SR@3° |
|---|---|---|---|
| siamese | 50.90° | 36.97° | 4.5% |
| lista | 61.61° | 45.76° | 4.0% |
| covrecon | 36.48° | 24.38° | 8.2% |
| hiergru | 56.71° | 40.47° | 3.7% |

## Classifier (real val)

See `output/clf_report.txt` and `output/metrics_classifier.json`.

## Reproduction command

```bash
python -X utf8 scripts/resume_baseline.py
```