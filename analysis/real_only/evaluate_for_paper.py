"""
evaluate_for_paper.py
=====================
Evaluate trained real-only models on external test data (for_paper.txt).

Uses weights from: output_real_data/
Writes results to: output_real_data/for_paper_eval/

Run:
  python analysis/real_only/evaluate_for_paper.py
  python analysis/real_only/evaluate_for_paper.py --input-txt for_paper/for_paper.txt
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

from config import (
    DEVICE, N_ANT, N_GRID, MIMO_POS, WHITEN_K
)
from preprocessing import preprocess_channels
from models import (
    ComplexNet, to_complex_tensor,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA, build_steering_matrix
)
from metrics import classifier_full_report, DOAEvaluator


OUTPUT_DIR_REAL = os.path.join(PROJECT_ROOT, "output_real_data")
EVAL_DIR = os.path.join(OUTPUT_DIR_REAL, "for_paper_eval")


def load_for_paper_txt(txt_path):
    raw = np.loadtxt(txt_path)
    # columns:
    # [0] doppler, [1] range bin, [2] range
    # [3:11] Re(ch0..7), [11:19] Im(ch0..7)
    # [19:22] mb_metrix, [22:24] azimuth (deg), [24] class
    ch = raw[:, 3:11] + 1j * raw[:, 11:19]
    mb = raw[:, 19:22]
    az = raw[:, 22:24]
    cls = raw[:, 24].astype(int)
    return {
        "channels_raw": ch,
        "mb_metrix": mb,
        "azimuth_deg": az,
        "classes": cls,
    }


def predict_doa_single(signal_np, method_name, doa_model, grid_deg):
    doa_model.eval()
    with torch.no_grad():
        if method_name in ("siamese", "hiergru"):
            x = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            if method_name == "siamese":
                _, doa_logits, _ = doa_model(x)
            else:
                doa_logits, _ = doa_model(x)
            spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]]), spectrum

        if method_name == "lista":
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            power, _ = doa_model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]]), spectrum

        if method_name == "covrecon":
            R_flat = cov_to_flat(signal_np.reshape(1, -1), N_ANT)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            out = doa_model(R_t)
            R_rec = flat_to_hermitian(out, N_ANT)
            R_np = R_rec[0].cpu().numpy()
            pred_deg = root_music(R_np, N_sources=2)
            return pred_deg, None

    return np.array([]), None


def main():
    parser = argparse.ArgumentParser(description="Evaluate real-only models on for_paper.txt")
    parser.add_argument("--input-txt", type=str,
                        default=os.path.join(PROJECT_ROOT, "for_paper", "for_paper.txt"),
                        help="Path to for_paper.txt")
    parser.add_argument("--output-dir", type=str, default=EVAL_DIR,
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.input_txt):
        print(f"[!] File not found: {args.input_txt}")
        return

    # Load and preprocess
    data = load_for_paper_txt(args.input_txt)
    ch_raw = data["channels_raw"]
    ch = preprocess_channels(ch_raw, K=WHITEN_K, normalize="rms")
    az_deg = data["azimuth_deg"]
    true_cls = data["classes"]

    # Load classifier
    clf_path = os.path.join(OUTPUT_DIR_REAL, "classifier.pth")
    if not os.path.isfile(clf_path):
        print(f"[!] Classifier weights not found: {clf_path}")
        return

    classifier = ComplexNet(input_length=N_ANT).to(DEVICE)
    classifier.load_state_dict(torch.load(clf_path, map_location=DEVICE))
    classifier.eval()

    X = to_complex_tensor(ch[:, :, None]).to(DEVICE)
    with torch.no_grad():
        logits = classifier(X)
        pred_cls = logits.argmax(dim=1).cpu().numpy()
        proba = torch.softmax(logits.float(), dim=1).cpu().numpy()

    # Classifier report
    clf_result = classifier_full_report(
        true_cls, pred_cls, y_proba=proba,
        output_dir=args.output_dir, show=False
    )
    print(f"\nClassifier accuracy: {clf_result['accuracy']:.4f}")
    print(f"Classifier F1 macro: {clf_result['f1_macro']:.4f}")

    # DOA models
    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    doa_configs = {
        "siamese": (os.path.join(OUTPUT_DIR_REAL, "doa_siamese.pth"),
                    lambda: SiameseDOAModel(N_GRID, N_ANT)),
        "lista": (os.path.join(OUTPUT_DIR_REAL, "doa_lista.pth"),
                  lambda: DeepLISTA_DOA(A_np, n_layers=10)),
        "covrecon": (os.path.join(OUTPUT_DIR_REAL, "doa_covrecon.pth"),
                     lambda: CovReconstructor(N_ANT, hidden=128)),
        "hiergru": (os.path.join(OUTPUT_DIR_REAL, "doa_hiergru.pth"),
                    lambda: HierarchicalGRU_DOA()),
    }

    evaluator = DOAEvaluator(grid_deg)
    for name, (path, factory) in doa_configs.items():
        if not os.path.isfile(path):
            print(f"[!] Missing DOA weights: {path}")
            continue
        model = factory().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

        errors = []
        pred_classes_list = []
        true_angles_list = []
        pred_angles_list = []

        for i in range(len(ch)):
            if pred_cls[i] == 1:
                continue
            pred_angles, _ = predict_doa_single(ch[i], name, model, grid_deg)
            true_deg = az_deg[i].ravel()
            valid = true_deg[(true_deg >= -90) & (true_deg <= 90)]
            if len(valid) == 0 or len(pred_angles) == 0:
                continue
            min_err = min(np.min(np.abs(valid - p)) for p in pred_angles)
            errors.append(min_err)
            pred_classes_list.append(pred_cls[i])
            true_angles_list.append(valid[0])
            pred_angles_list.append(pred_angles[0])

        evaluator.add_method(
            name, errors,
            pred_classes=pred_classes_list,
            true_angles=true_angles_list,
            pred_angles=pred_angles_list,
        )

    evaluator.print_summary()
    evaluator.plot_all(args.output_dir, show=False)
    evaluator.save_json(args.output_dir, prefix="doa_")

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
