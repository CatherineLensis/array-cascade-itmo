"""
plot_for_paper_targets.py
=========================
Visualize real vs predicted targets in XY plane and in target scatter.

Uses weights from output_real_data_sh/real_only and data from for_paper.txt.

Run:
  python analysis/real_only/plot_for_paper_targets.py --method lista
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DEVICE, N_ANT, N_GRID, MIMO_POS
from preprocessing import max_normalize
from models import (
    ComplexNet,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA, build_steering_matrix,
)


DEFAULT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "output_real_data_sh", "real_only")
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "for_paper", "for_paper.txt")
DEFAULT_OUT = os.path.join(DEFAULT_WEIGHTS_DIR, "for_paper_eval")


def load_for_paper_txt(txt_path):
    raw = np.loadtxt(txt_path)
    ch = raw[:, 3:11] + 1j * raw[:, 11:19]
    az = raw[:, 22:24]  # deg
    cls = raw[:, 24].astype(int)
    r = raw[:, 2]  # range
    return ch, az, cls, r


def build_model(method_name):
    A_np, _ = build_steering_matrix(MIMO_POS, N_GRID)
    if method_name == "siamese":
        return SiameseDOAModel(N_GRID, N_ANT)
    if method_name == "lista":
        return DeepLISTA_DOA(A_np, n_layers=10)
    if method_name == "covrecon":
        return CovReconstructor(N_ANT, hidden=128)
    if method_name == "hiergru":
        return HierarchicalGRU_DOA()
    raise ValueError(f"Unknown method: {method_name}")


def load_weights(model, method_name, weights_dir):
    path = os.path.join(weights_dir, f"doa_{method_name}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing weights: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_spectrum(model, method_name, signal_np):
    with torch.no_grad():
        if method_name in ("siamese", "hiergru"):
            x = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            if method_name == "siamese":
                _, doa_logits, _ = model(x)
            else:
                doa_logits, _ = model(x)
            spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
            return spectrum
        if method_name == "lista":
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            power, _ = model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            return spectrum
    return None


def predict_angles(signal_np, method_name, model, grid_deg, k=1):
    if method_name == "covrecon":
        R_flat = cov_to_flat(signal_np.reshape(1, -1), N_ANT)
        R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
        out = model(R_t)
        R_rec = flat_to_hermitian(out, N_ANT)
        R_np = R_rec[0].detach().cpu().numpy()
        angles = root_music(R_np, N_sources=max(k, 2))
        return np.array(angles[:k], dtype=float)

    spectrum = get_spectrum(model, method_name, signal_np)
    if spectrum is None:
        return np.array([])
    if k == 1:
        return np.array([grid_deg[np.argmax(spectrum)]])
    idx = np.argpartition(spectrum, -k)[-k:]
    idx = idx[np.argsort(-spectrum[idx])]
    return grid_deg[idx]


def polar_to_xy(r, az_deg):
    az_rad = np.deg2rad(az_deg)
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Plot real vs predicted targets for for_paper.txt")
    parser.add_argument("--input-txt", default=DEFAULT_INPUT)
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--method", default="lista", choices=["siamese", "lista", "covrecon", "hiergru"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ch_raw, az_deg, cls, rng = load_for_paper_txt(args.input_txt)
    ch = max_normalize(ch_raw)

    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    model = build_model(args.method)
    model = load_weights(model, args.method, args.weights_dir)

    true_xy_1 = []
    true_xy_2 = []
    pred_xy_1 = []
    pred_xy_2 = []
    scatter_true_1 = []
    scatter_pred_1 = []
    scatter_true_2 = []
    scatter_pred_2 = []

    for i in range(len(ch)):
        r = rng[i]
        true_angles = az_deg[i].ravel()
        if cls[i] == 1:
            continue

        k = 2 if cls[i] == 2 else 1
        pred_angles = predict_angles(ch[i], args.method, model, grid_deg, k=k)

        # True target 1
        x1, y1 = polar_to_xy(r, true_angles[0])
        true_xy_1.append((x1, y1))
        scatter_true_1.append(true_angles[0])

        if k == 2:
            x2, y2 = polar_to_xy(r, true_angles[1])
            true_xy_2.append((x2, y2))
            scatter_true_2.append(true_angles[1])

        if len(pred_angles) >= 1:
            xp1, yp1 = polar_to_xy(r, pred_angles[0])
            pred_xy_1.append((xp1, yp1))
            scatter_pred_1.append(pred_angles[0])
        if k == 2 and len(pred_angles) >= 2:
            xp2, yp2 = polar_to_xy(r, pred_angles[1])
            pred_xy_2.append((xp2, yp2))
            scatter_pred_2.append(pred_angles[1])

    # Graph 1: XY positions
    # Keep portrait style close to targets_xy_covrecon_0.png
    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(111)
    if true_xy_1:
        x, y = zip(*true_xy_1)
        ax.scatter(x, y, facecolors="none", edgecolors="blue", label="true target1")
    if pred_xy_1:
        x, y = zip(*pred_xy_1)
        ax.scatter(x, y, c="blue", marker="x", label="pred target1")
    if true_xy_2:
        x, y = zip(*true_xy_2)
        ax.scatter(x, y, facecolors="none", edgecolors="magenta", label="true target2")
    if pred_xy_2:
        x, y = zip(*pred_xy_2)
        ax.scatter(x, y, c="magenta", marker="x", label="pred target2")
    ax.set_xlabel("longitude, m", fontsize=16)
    ax.set_ylabel("latitude, m", fontsize=16)
    if true_xy_1 or true_xy_2 or pred_xy_1 or pred_xy_2:
        all_pts = np.array(
            true_xy_1 + true_xy_2 + pred_xy_1 + pred_xy_2, dtype=float
        )
        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
        dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
        ax.set_xlim([xmin - dx, xmax + dx])
        ax.set_ylim([ymin - dy, ymax + dy])
    ax.set_title(f"Targets XY (method={args.method})", fontsize=22, pad=10)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.95)
    fig.tight_layout()
    out1 = os.path.join(args.output_dir, f"targets_xy_{args.method}.png")
    plt.savefig(out1, dpi=150)
    plt.close()

    # Graph 2: real vs pred (scatter)
    plt.figure(figsize=(5, 5))
    if scatter_true_1:
        plt.scatter(scatter_true_1, scatter_pred_1, c="blue", label="target1")
    if scatter_true_2:
        plt.scatter(scatter_true_2, scatter_pred_2, c="magenta", label="target2")
    lims = [-90, 90]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("true angle (deg)")
    plt.ylabel("pred angle (deg)")
    plt.title(f"True vs Pred Angle (method={args.method})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out2 = os.path.join(args.output_dir, f"targets_scatter_{args.method}.png")
    plt.savefig(out2, dpi=150)
    plt.close()

    print(f"Saved:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
