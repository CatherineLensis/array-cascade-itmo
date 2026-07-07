"""
check_hypotheses.py
===================
Hypothesis checks for DOA domain gap (synth vs real).

Outputs:
  - report.md
  - metrics.json
  - fig_*.png

Run:
  python analysis/hypothesis_checks/check_hypotheses.py
  python analysis/hypothesis_checks/check_hypotheses.py --n-samples 5000
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OUTPUT_DIR, SYNTH_NPZ, REAL_NPZ, N_ANT, MIMO_POS
from models import build_steering_matrix, root_music


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _ensure_channels_shape(ch: np.ndarray) -> np.ndarray:
    # synth: [N, 8, 1] → [N, 8]; real: [N, 8]
    if ch.ndim == 3 and ch.shape[-1] == 1:
        return ch[:, :, 0]
    return ch


def _angles_synth_deg(azimuth: np.ndarray) -> np.ndarray:
    # synth: radians, shape [N, 2, 1] or [N, 2]
    if azimuth.ndim == 3:
        azimuth = azimuth[:, :, 0]
    return np.rad2deg(azimuth)


def _angles_real_deg(azimuth: np.ndarray) -> np.ndarray:
    # real: usually degrees; if seems like radians - convert
    if np.max(np.abs(azimuth)) <= 2 * np.pi:
        return np.rad2deg(azimuth)
    return azimuth


def _safe_angles(ang_deg: np.ndarray) -> np.ndarray:
    # Keep only valid angles in [-90, 90]
    return ang_deg[(ang_deg >= -90) & (ang_deg <= 90)]


def _hist_pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(x, bins=bins, density=True)
    return hist + 1e-12


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x)
    y = np.sort(y)
    all_vals = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, all_vals, side="right") / len(x)
    cdf_y = np.searchsorted(y, all_vals, side="right") / len(y)
    return np.max(np.abs(cdf_x - cdf_y))


def beamform_estimate(X: np.ndarray, A: np.ndarray, grid_deg: np.ndarray) -> np.ndarray:
    # X: [N, N_ant], A: [N_ant, N_grid]
    spec = np.abs(np.dot(np.conj(X), A))  # [N, N_grid]
    idx = np.argmax(spec, axis=1)
    return grid_deg[idx]


def root_music_estimate(x: np.ndarray, n_sources: int) -> np.ndarray:
    # x: [N_ant] complex, R = x x^H
    R = np.outer(x, np.conj(x))
    angles = root_music(R, N_sources=n_sources)
    return np.array(angles, dtype=float)


def compute_rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else float("nan")


def angle_error(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0 or true.size == 0:
        return float("nan")
    return float(np.min(np.abs(true - pred)))


def plot_hist_compare(x1, x2, bins, title, label1, label2, path):
    plt.figure(figsize=(6, 4))
    plt.hist(x1, bins=bins, alpha=0.6, density=True, label=label1)
    plt.hist(x2, bins=bins, alpha=0.6, density=True, label=label2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rmse_vs_angle(angle_deg, errors, path, bin_deg=10):
    bins = np.arange(-90, 91, bin_deg)
    centers = (bins[:-1] + bins[1:]) / 2
    rmse = []
    for i in range(len(bins) - 1):
        mask = (angle_deg >= bins[i]) & (angle_deg < bins[i + 1])
        rmse.append(compute_rmse(errors[mask]))
    plt.figure(figsize=(7, 4))
    plt.plot(centers, rmse, marker="o")
    plt.title("RMSE vs True Angle (beamforming baseline)")
    plt.xlabel("True Angle (deg)")
    plt.ylabel("RMSE (deg)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hypothesis checks for DOA domain gap")
    parser.add_argument("--synth-npz", default=SYNTH_NPZ, help="Path to synth_new.npz")
    parser.add_argument("--real-npz", default=REAL_NPZ, help="Path to real_preprocessed.npz")
    parser.add_argument("--output-dir", default=os.path.join("analysis", "hypothesis_checks"),
                        help="Output folder for report and figures")
    parser.add_argument("--n-samples", type=int, default=5000, help="Sample size for heavy checks")
    parser.add_argument("--n-rootmusic", type=int, default=1000, help="Sample size for Root-MUSIC baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    synth = _load_npz(args.synth_npz)
    real = _load_npz(args.real_npz)

    Xs = _ensure_channels_shape(synth["channels"])
    Xr = _ensure_channels_shape(real["channels"])
    ang_s = _angles_synth_deg(synth["azimuth"])
    ang_r = _angles_real_deg(real["azimuth"])
    cls_s = synth["math_class"].astype(int).ravel()
    cls_r = real["math_class"].astype(int).ravel()
    mb_s = synth.get("mb_metrix")
    mb_r = real.get("mb_metrix")

    # H1: Angle distribution mismatch
    ang_s1 = _safe_angles(ang_s[:, 0])
    ang_r1 = _safe_angles(ang_r[:, 0])
    bins = np.linspace(-90, 90, 73)  # 2.5 deg bins
    p = _hist_pdf(ang_s1, bins)
    q = _hist_pdf(ang_r1, bins)
    h1_js = js_divergence(p, q)
    h1_ks = ks_distance(ang_s1, ang_r1)
    plot_hist_compare(ang_s1, ang_r1, bins,
                      "Angle Distribution (angle1)",
                      "synth", "real",
                      os.path.join(args.output_dir, "fig_angle_distribution.png"))

    # H4: Amplitude + phase mismatch
    amp_s = np.mean(np.abs(Xs), axis=1)
    amp_r = np.mean(np.abs(Xr), axis=1)
    plot_hist_compare(amp_s, amp_r, 60,
                      "Amplitude Distribution |signal|",
                      "synth", "real",
                      os.path.join(args.output_dir, "fig_amplitude_distribution.png"))

    # Phase std (relative to first antenna)
    def phase_std(X):
        ref = X[:, [0]]
        rel = X * np.conj(ref)
        phases = np.angle(rel)
        return np.std(phases, axis=1)

    ph_s = phase_std(Xs)
    ph_r = phase_std(Xr)
    plot_hist_compare(ph_s, ph_r, 60,
                      "Phase Std Distribution (relative)",
                      "synth", "real",
                      os.path.join(args.output_dir, "fig_phase_std_distribution.png"))

    h4_js_amp = js_divergence(_hist_pdf(amp_s, np.linspace(0, np.max(amp_s), 60)),
                              _hist_pdf(amp_r, np.linspace(0, np.max(amp_r), 60)))
    h4_js_phase = js_divergence(_hist_pdf(ph_s, np.linspace(0, np.max(ph_s), 60)),
                                _hist_pdf(ph_r, np.linspace(0, np.max(ph_r), 60)))

    # H3: Multipath inside class=0
    h3 = {}
    if mb_s is not None and mb_r is not None:
        # projection threshold heuristic from dataset.py
        proj_s0 = mb_s[cls_s == 0, 0]
        proj_r0 = mb_r[cls_r == 0, 0]
        omp1_s0 = mb_s[cls_s == 0, 1]
        omp1_r0 = mb_r[cls_r == 0, 1]
        omp1_p95 = np.percentile(omp1_s0, 95)
        proj_lt_08 = float(np.mean(proj_r0 < 0.8))
        omp1_gt_p95 = float(np.mean(omp1_r0 > omp1_p95))
        h3 = {
            "proj_lt_0.8_frac_real_class0": proj_lt_08,
            "omp1_gt_p95_frac_real_class0": omp1_gt_p95,
            "omp1_p95_synth_class0": float(omp1_p95),
        }
        plot_hist_compare(proj_s0, proj_r0, 60,
                          "mb_metrix[0] Projection (class 0)",
                          "synth", "real",
                          os.path.join(args.output_dir, "fig_mb_proj_class0.png"))
        plot_hist_compare(omp1_s0, omp1_r0, 60,
                          "mb_metrix[1] OMP norm1 (class 0)",
                          "synth", "real",
                          os.path.join(args.output_dir, "fig_mb_omp1_class0.png"))

    # H2: Label correctness via beamforming baseline
    A, theta_grid = build_steering_matrix(MIMO_POS, n_grid=361)
    grid_deg = np.rad2deg(theta_grid)

    # sample for speed
    idx = rng.choice(len(Xr), size=min(args.n_samples, len(Xr)), replace=False)
    Xr_s = Xr[idx]
    ang_r_s = ang_r[idx]
    cls_r_s = cls_r[idx]

    # Beamforming estimate
    bf_pred = beamform_estimate(Xr_s, A, grid_deg)
    bf_err = []
    bf_err_class0 = []
    for i in range(len(Xr_s)):
        true_deg = _safe_angles(ang_r_s[i])
        err = angle_error(np.array([bf_pred[i]]), true_deg)
        if not np.isnan(err):
            bf_err.append(err)
            if cls_r_s[i] == 0:
                bf_err_class0.append(err)
    bf_err = np.array(bf_err)
    bf_err_class0 = np.array(bf_err_class0)

    plt.figure(figsize=(6, 4))
    plt.hist(bf_err, bins=60, alpha=0.7, color="#4C9F50")
    plt.title("Beamforming Error Histogram (real)")
    plt.xlabel("Absolute Error (deg)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fig_bf_error_hist.png"), dpi=150)
    plt.close()

    # H5: Edge effect (RMSE vs angle) using beamforming baseline
    angle1 = _safe_angles(ang_r_s[:, 0])
    valid_mask = (ang_r_s[:, 0] >= -90) & (ang_r_s[:, 0] <= 90)
    plot_rmse_vs_angle(ang_r_s[:, 0][valid_mask], bf_err[valid_mask],
                       os.path.join(args.output_dir, "fig_rmse_vs_angle_real.png"))

    # H6: Root-MUSIC baseline (real)
    idx_rm = rng.choice(len(Xr), size=min(args.n_rootmusic, len(Xr)), replace=False)
    rm_err = []
    for i in idx_rm:
        x = Xr[i]
        true_deg = _safe_angles(ang_r[i])
        if len(true_deg) == 0:
            continue
        n_src = 1 if cls_r[i] == 0 else 2
        pred = root_music_estimate(x, n_sources=n_src)
        err = angle_error(pred, true_deg)
        if not np.isnan(err):
            rm_err.append(err)
    rm_err = np.array(rm_err)

    plt.figure(figsize=(6, 4))
    plt.hist(rm_err, bins=60, alpha=0.7, color="#9C27B0")
    plt.title("Root-MUSIC Error Histogram (real)")
    plt.xlabel("Absolute Error (deg)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "fig_rootmusic_error_hist.png"), dpi=150)
    plt.close()

    # metrics.json
    metrics = {
        "samples": {
            "synth_total": int(len(Xs)),
            "real_total": int(len(Xr)),
            "beamforming_sample": int(len(Xr_s)),
            "rootmusic_sample": int(len(rm_err)),
        },
        "H1_angle_distribution": {
            "js_div": float(h1_js),
            "ks_dist": float(h1_ks),
            "synth_mean": float(np.mean(ang_s1)),
            "real_mean": float(np.mean(ang_r1)),
            "synth_std": float(np.std(ang_s1)),
            "real_std": float(np.std(ang_r1)),
        },
        "H2_beamforming_label_check": {
            "rmse": compute_rmse(bf_err),
            "mae": float(np.mean(bf_err)),
            "median_ae": float(np.median(bf_err)),
            "rmse_class0": compute_rmse(bf_err_class0),
            "n_errors": int(len(bf_err)),
        },
        "H3_multipath_in_class0": h3,
        "H4_noise_phase_mismatch": {
            "amp_mean_synth": float(np.mean(amp_s)),
            "amp_mean_real": float(np.mean(amp_r)),
            "amp_std_synth": float(np.std(amp_s)),
            "amp_std_real": float(np.std(amp_r)),
            "phase_std_mean_synth": float(np.mean(ph_s)),
            "phase_std_mean_real": float(np.mean(ph_r)),
            "js_amp": float(h4_js_amp),
            "js_phase_std": float(h4_js_phase),
        },
        "H5_edge_effect": {
            "rmse_vs_angle_plot": "fig_rmse_vs_angle_real.png",
        },
        "H6_rootmusic_baseline": {
            "rmse": compute_rmse(rm_err),
            "mae": float(np.mean(rm_err)) if len(rm_err) > 0 else float("nan"),
            "median_ae": float(np.median(rm_err)) if len(rm_err) > 0 else float("nan"),
            "n_errors": int(len(rm_err)),
        },
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # report.md
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Hypothesis Checks Report\n\n")
        f.write("## H1: Angle distribution mismatch\n")
        f.write(f"- JS divergence: {metrics['H1_angle_distribution']['js_div']:.4f}\n")
        f.write(f"- KS distance: {metrics['H1_angle_distribution']['ks_dist']:.4f}\n")
        f.write("![Angle distribution](fig_angle_distribution.png)\n\n")

        f.write("## H2: Label correctness (beamforming baseline)\n")
        f.write(f"- RMSE: {metrics['H2_beamforming_label_check']['rmse']:.2f} deg\n")
        f.write(f"- MAE: {metrics['H2_beamforming_label_check']['mae']:.2f} deg\n")
        f.write(f"- Median AE: {metrics['H2_beamforming_label_check']['median_ae']:.2f} deg\n")
        f.write("![Beamforming error](fig_bf_error_hist.png)\n\n")

        f.write("## H3: Multipath inside class=0\n")
        if h3:
            f.write(f"- proj < 0.8: {h3['proj_lt_0.8_frac_real_class0']*100:.1f}%\n")
            f.write(f"- omp1 > p95(synth): {h3['omp1_gt_p95_frac_real_class0']*100:.1f}%\n")
            f.write("![mb proj](fig_mb_proj_class0.png)\n")
            f.write("![mb omp1](fig_mb_omp1_class0.png)\n\n")
        else:
            f.write("- mb_metrix is missing in NPZ\n\n")

        f.write("## H4: Noise / phase mismatch\n")
        f.write(f"- JS(amp): {metrics['H4_noise_phase_mismatch']['js_amp']:.4f}\n")
        f.write(f"- JS(phase std): {metrics['H4_noise_phase_mismatch']['js_phase_std']:.4f}\n")
        f.write("![Amplitude distribution](fig_amplitude_distribution.png)\n")
        f.write("![Phase std distribution](fig_phase_std_distribution.png)\n\n")

        f.write("## H5: edge-effect\n")
        f.write("![RMSE vs angle](fig_rmse_vs_angle_real.png)\n\n")

        f.write("## H6: Root-MUSIC baseline\n")
        f.write(f"- RMSE: {metrics['H6_rootmusic_baseline']['rmse']:.2f} deg\n")
        f.write(f"- MAE: {metrics['H6_rootmusic_baseline']['mae']:.2f} deg\n")
        f.write(f"- Median AE: {metrics['H6_rootmusic_baseline']['median_ae']:.2f} deg\n")
        f.write("![Root-MUSIC error](fig_rootmusic_error_hist.png)\n\n")

        f.write("## Files\n")
        f.write(f"- metrics.json: {metrics_path}\n")
        f.write(f"- report.md: {report_path}\n")

    print(f"[OK] report.md: {report_path}")
    print(f"[OK] metrics.json: {metrics_path}")
    print(f"[OK] figures: {args.output_dir}")


if __name__ == "__main__":
    main()
