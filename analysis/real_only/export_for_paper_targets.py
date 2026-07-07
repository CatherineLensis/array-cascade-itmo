"""
export_for_paper_targets.py
===========================
Export real vs predicted targets to a file with columns: source, prediction_*.

Runs all four models (siamese, lista, covrecon, hiergru) and writes predictions
to a single CSV file for scientific analysis.

Run:
  python analysis/real_only/export_for_paper_targets.py
  python analysis/real_only/export_for_paper_targets.py --output data_predictions.csv
"""

import argparse
import os
import sys
import numpy as np
import torch

# Ensure project root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DEVICE, N_ANT, N_GRID, MIMO_POS
from preprocessing import max_normalize
from models import (
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA, build_steering_matrix,
)


DEFAULT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "output_real_data_sh", "real_only")
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "for_paper", "for_paper.txt")
DEFAULT_OUTPUT = os.path.join(DEFAULT_WEIGHTS_DIR, "predictions_source_vs_pred.csv")


def load_for_paper_txt(txt_path):
    raw = np.loadtxt(txt_path)
    ch = raw[:, 3:11] + 1j * raw[:, 11:19]
    az = raw[:, 22:24]  # deg
    cls = raw[:, 24].astype(int)
    return ch, az, cls


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


def _interpolate_peak(spectrum, grid_deg, idx):
    """Quadratic interpolation around peak for sub-grid resolution."""
    n = len(spectrum)
    step = grid_deg[1] - grid_deg[0] if len(grid_deg) > 1 else 0.5
    if idx <= 0 or idx >= n - 1:
        return np.float64(grid_deg[idx])
    s0, s1, s2 = spectrum[idx - 1], spectrum[idx], spectrum[idx + 1]
    denom = s0 - 2 * s1 + s2
    if abs(denom) < 1e-12:
        return np.float64(grid_deg[idx])
    delta = 0.5 * (s0 - s2) / denom
    delta = np.clip(delta, -1.0, 1.0)  # keep within neighbor
    return np.float64(grid_deg[idx] + delta * step)


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
        idx = np.argmax(spectrum)
        ang = _interpolate_peak(spectrum, grid_deg, idx)
        return np.array([ang])
    top_idx = np.argpartition(spectrum, -k)[-k:]
    top_idx = top_idx[np.argsort(-spectrum[top_idx])]
    angles = np.array([_interpolate_peak(spectrum, grid_deg, i) for i in top_idx])
    return angles


METHODS = ["siamese", "lista", "covrecon", "hiergru"]


def main():
    parser = argparse.ArgumentParser(
        description="Export real vs predicted targets to CSV (source + all methods)"
    )
    parser.add_argument("--input-txt", default=DEFAULT_INPUT)
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--sep", default=",", help="Column separator (default: comma)")
    args = parser.parse_args()

    ch_raw, az_deg, cls = load_for_paper_txt(args.input_txt)
    ch = max_normalize(ch_raw)

    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    # Load all models
    models = {}
    for method in METHODS:
        model = build_model(method)
        model = load_weights(model, method, args.weights_dir)
        models[method] = model

    source_list = []
    predictions_by_method = {m: [] for m in METHODS}

    for i in range(len(ch)):
        true_angles = az_deg[i].ravel()
        if cls[i] == 1:
            continue

        k = 2 if cls[i] == 2 else 1

        # One inference per method per sample
        pred_by_method = {}
        for method in METHODS:
            pred_angles = predict_angles(ch[i], method, models[method], grid_deg, k=k)
            pred_by_method[method] = pred_angles

        # Target 1
        source_list.append(true_angles[0])
        for method in METHODS:
            pa = pred_by_method[method]
            pred = pa[0] if len(pa) >= 1 else np.nan
            predictions_by_method[method].append(pred)

        # Target 2 (if two-target case)
        if k == 2:
            source_list.append(true_angles[1])
            for method in METHODS:
                pa = pred_by_method[method]
                pred = pa[1] if len(pa) >= 2 else np.nan
                predictions_by_method[method].append(pred)

    # Write to file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    pred_cols = [f"prediction_{m}" for m in METHODS]
    header = args.sep.join(["source"] + pred_cols)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for j, src in enumerate(source_list):
            pred_strs = []
            for method in METHODS:
                p = predictions_by_method[method][j]
                pred_strs.append(f"{p:.6f}" if not np.isnan(p) else "")
            row = args.sep.join([f"{src:.6f}"] + pred_strs)
            f.write(row + "\n")

    print(f"Exported {len(source_list)} pairs to: {args.output}")
    print(f"Columns: source, {', '.join(pred_cols)}")


if __name__ == "__main__":
    main()
