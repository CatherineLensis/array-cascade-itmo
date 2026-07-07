# Hypothesis Checks Report

## H1: Angle distribution mismatch
- JS divergence: 0.0497
- KS distance: 0.2232
![Angle distribution](fig_angle_distribution.png)

## H2: Label correctness (beamforming baseline)
- RMSE: 17.54 deg
- MAE: 10.06 deg
- Median AE: 3.40 deg
![Beamforming error](fig_bf_error_hist.png)

## H3: Multipath inside class=0
- proj < 0.8: 0.0%
- omp1 > p95(synth): 22.8%
![mb proj](fig_mb_proj_class0.png)
![mb omp1](fig_mb_omp1_class0.png)

## H4: Noise / phase mismatch
- JS(amp): 15.1134
- JS(phase std): 3.1291
![Amplitude distribution](fig_amplitude_distribution.png)
![Phase std distribution](fig_phase_std_distribution.png)

## H5: edge-effect
![RMSE vs angle](fig_rmse_vs_angle_real.png)

## H6: Root-MUSIC baseline
- RMSE: 16.26 deg
- MAE: 9.11 deg
- Median AE: 2.91 deg
![Root-MUSIC error](fig_rootmusic_error_hist.png)

## Files
- metrics.json: analysis\hypothesis_checks\metrics.json
- report.md: analysis\hypothesis_checks\report.md
