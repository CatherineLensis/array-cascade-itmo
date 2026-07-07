"""
run_real_only.py
================
Real-only training and evaluation:
  - Classifier trained on real data only
  - DOA models trained on real data only (class 0 + class 2)
Outputs go to output_real_data/

Run:
  python analysis/real_only/run_real_only.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Ensure project root on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    REAL_NPZ, REAL_DATA_DIR, WHITEN_K,
    CLASSIFIER_LR, CLASSIFIER_WD, FINETUNE_EPOCHS, BATCH_SIZE,
    CLASS_WEIGHT_CAP, DEVICE, N_ANT, N_GRID, MIMO_POS,
    DOA_EPOCHS, DOA_EPOCHS_HIERGRU,
)
from dataset import dataset_from_real_data
from preprocessing import preprocess_channels
from models import (
    ComplexNet, to_complex_tensor,
    build_steering_matrix,
)
from metrics import MetricsTracker, DOAEvaluator, classifier_full_report, plot_data_eda
from run_full_pipeline import (
    make_doa_labels, predict_doa_single, doa_rmse,
    train_method1_siamese, train_method2_lista,
    train_method3_covrecon, train_method4_hiergru,
)


OUTPUT_DIR_REAL = os.path.join(PROJECT_ROOT, "output_real_data")


def _load_real_data():
    if os.path.isfile(REAL_NPZ):
        data = np.load(REAL_NPZ, allow_pickle=True)
        channels = data["channels"]
        angles = data["azimuth"]
        classes = data["math_class"].astype(int)
        mb_metrix = data.get("mb_metrix", None)
    elif os.path.isdir(REAL_DATA_DIR):
        raw = dataset_from_real_data(REAL_DATA_DIR, normalize="none")
        channels = preprocess_channels(raw["channels"], K=WHITEN_K)
        angles = raw["azimuth"]
        classes = raw["math class"].astype(int)
        mb_metrix = raw.get("mb_metrix", None)
    else:
        raise FileNotFoundError("Real data not found. Provide REAL_NPZ or REAL_DATA_DIR.")

    # Normalize angle units (deg -> rad)
    if np.max(np.abs(angles)) > 2 * np.pi:
        angles = np.deg2rad(angles)

    # Channels shape: [N, 8] expected
    if channels.ndim == 3 and channels.shape[-1] == 1:
        channels = channels[:, :, 0]

    return {
        "channels": channels,
        "angles": angles,
        "classes": classes,
        "mb_metrix": mb_metrix,
    }


def train_classifier_real(data):
    print("\n" + "=" * 60)
    print("REAL-ONLY: Classifier training")
    print("=" * 60)

    X = data["channels"]
    y = data["classes"]

    idx_all = np.arange(len(X))
    idx_train, idx_val = train_test_split(
        idx_all, test_size=0.2, stratify=y, random_state=42
    )

    X_train = to_complex_tensor(X[idx_train][:, :, None]).to(DEVICE)
    y_train = torch.LongTensor(y[idx_train]).to(DEVICE)
    X_val = to_complex_tensor(X[idx_val][:, :, None]).to(DEVICE)
    y_val = torch.LongTensor(y[idx_val]).to(DEVICE)

    model = ComplexNet(input_length=N_ANT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WD)

    # class weights with cap
    class_counts = np.bincount(y[idx_train], minlength=3).astype(float)
    class_counts[class_counts == 0] = 1.0
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.min()
    class_weights = np.clip(class_weights, 1.0, CLASS_WEIGHT_CAP)
    cw_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}")
    print(f"  Class weights: [{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}]")

    tracker = MetricsTracker("classifier_real_only")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            out = model(X_train[idx])
            loss = criterion(out.float(), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        model.eval()
        with torch.no_grad():
            pred_train = model(X_train).argmax(dim=1)
            acc_train = (pred_train == y_train).float().mean().item()
            out_val = model(X_val)
            val_loss = criterion(out_val.float(), y_val).item()
            pred_val = out_val.argmax(dim=1)
            acc_val = (pred_val == y_val).float().mean().item()

        tracker.log_train(epoch, loss=avg_loss, acc=acc_train)
        tracker.log_val(epoch, loss=val_loss, acc=acc_val)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} acc={acc_train:.4f}"
                  f" | val_loss={val_loss:.4f} acc={acc_val:.4f}")

    os.makedirs(OUTPUT_DIR_REAL, exist_ok=True)
    tracker.plot_curves(OUTPUT_DIR_REAL, show=False)
    tracker.save_json(OUTPUT_DIR_REAL)

    # Full report on val
    model.eval()
    with torch.no_grad():
        out_val = model(X_val).float()
        pred_val = out_val.argmax(dim=1).cpu().numpy()
        proba_val = torch.softmax(out_val, dim=1).cpu().numpy()
    clf_result = classifier_full_report(
        y[idx_val], pred_val, y_proba=proba_val,
        output_dir=OUTPUT_DIR_REAL, show=False
    )
    print(f"  Real-only Val Accuracy: {clf_result['accuracy']:.4f}")
    print(f"  Real-only Val F1 macro: {clf_result['f1_macro']:.4f}")

    # Confusion matrix on all real
    X_all = to_complex_tensor(X[:, :, None]).to(DEVICE)
    with torch.no_grad():
        pred_all = model(X_all).argmax(dim=1).cpu().numpy()
    cm = confusion_matrix(y, pred_all)
    print(f"  Confusion matrix (all real):\n{cm}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR_REAL, "classifier.pth"))
    return model


def train_doa_real(data):
    print("\n" + "=" * 60)
    print("REAL-ONLY: DOA training")
    print("=" * 60)

    ch = data["channels"]
    ang = data["angles"]
    cls = data["classes"]

    # Use only class 0 and class 2 for DOA
    mask = (cls == 0) | (cls == 2)
    ch_doa = ch[mask]
    ang_doa = ang[mask]
    cls_doa = cls[mask]

    idx_all = np.arange(len(ch_doa))
    idx_tr, idx_vl = train_test_split(idx_all, test_size=0.15, random_state=42)
    ch_tr, ch_vl = ch_doa[idx_tr], ch_doa[idx_vl]
    ang_tr, ang_vl = ang_doa[idx_tr], ang_doa[idx_vl]
    cls_tr, cls_vl = cls_doa[idx_tr], cls_doa[idx_vl]

    print(f"  DOA: train={len(ch_tr)}, val={len(ch_vl)} (class 0 + class 2)")

    _, grid_deg = make_doa_labels(ang_tr[:1], N_GRID)

    m1, l1 = train_method1_siamese(ch_tr, ang_tr, cls_tr, grid_deg)
    m2, l2 = train_method2_lista(ch_tr, ang_tr, grid_deg)
    m3, l3 = train_method3_covrecon(ch_tr, ang_tr, grid_deg)
    m4, l4 = train_method4_hiergru(ch_tr, ang_tr, cls_tr, grid_deg)

    doa_models = {"siamese": m1, "lista": m2, "covrecon": m3, "hiergru": m4}
    loss_dict = {"siamese": l1, "lista": l2, "covrecon": l3, "hiergru": l4}

    # Val RMSE
    print("\n  Val RMSE:")
    for name, model in doa_models.items():
        rmse_list = []
        for i in range(len(ch_vl)):
            pred_angles, _ = predict_doa_single(ch_vl[i], name, model, grid_deg)
            true_deg = np.rad2deg(ang_vl[i].ravel())
            err = doa_rmse(pred_angles, true_deg)
            if not np.isnan(err):
                rmse_list.append(err)
        val_rmse = np.mean(rmse_list) if rmse_list else float("nan")
        print(f"    {name:12s} | val RMSE = {val_rmse:.2f} deg ({len(rmse_list)} samples)")

        tracker = MetricsTracker(f"doa_{name}_real_only")
        for ep, lv in enumerate(loss_dict[name]):
            tracker.log_train(ep, loss=lv)
        tracker.log_val(0, rmse=val_rmse)
        tracker.plot_curves(OUTPUT_DIR_REAL, show=False)
        tracker.save_json(OUTPUT_DIR_REAL)

    # Save models
    for name, model in doa_models.items():
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR_REAL, f"doa_{name}.pth"))

    return doa_models


def evaluate_pipeline(data, classifier, doa_models):
    print("\n" + "=" * 60)
    print("REAL-ONLY: Evaluation (pipeline + DOA)")
    print("=" * 60)

    ch = data["channels"]
    ang = data["angles"]
    cls = data["classes"]

    _, grid_deg = make_doa_labels(ang[:1], N_GRID)

    # Pipeline evaluation: predicted classes
    classifier.eval()
    X_t = to_complex_tensor(ch[:, :, None]).to(DEVICE)
    with torch.no_grad():
        pred_cls = classifier(X_t).argmax(dim=1).cpu().numpy()
    cm = confusion_matrix(cls, pred_cls)
    acc = np.mean(pred_cls == cls)
    print(f"  Pipeline classifier accuracy: {acc:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # DOA evaluation using true classes (clean evaluation)
    evaluator = DOAEvaluator(grid_deg)
    for name, model in doa_models.items():
        errors = []
        pred_classes_list = []
        true_angles_list = []
        pred_angles_list = []

        for i in range(len(ch)):
            if cls[i] == 1:
                continue
            pred_angles, _ = predict_doa_single(ch[i], name, model, grid_deg)
            true_deg = np.rad2deg(ang[i].ravel())
            valid = true_deg[(true_deg >= -90) & (true_deg <= 90)]
            if len(valid) == 0 or len(pred_angles) == 0:
                continue
            min_err = min(np.min(np.abs(valid - p)) for p in pred_angles)
            errors.append(min_err)
            pred_classes_list.append(cls[i])
            true_angles_list.append(valid[0])
            pred_angles_list.append(pred_angles[0])

        evaluator.add_method(
            name, errors,
            pred_classes=pred_classes_list,
            true_angles=true_angles_list,
            pred_angles=pred_angles_list,
        )

    evaluator.print_summary()
    evaluator.plot_all(OUTPUT_DIR_REAL, show=False)
    evaluator.save_json(OUTPUT_DIR_REAL, prefix="doa_")

    # EDA
    real_eda = {
        "angles": ang,
        "classes": cls,
        "channels": ch,
        "mb_metrix": data.get("mb_metrix", None),
    }
    plot_data_eda({"angles": ang, "classes": cls, "channels": ch}, real_eda, output_dir=OUTPUT_DIR_REAL, show=False)

    print(f"  Outputs saved to: {OUTPUT_DIR_REAL}")


def main():
    print("REAL-ONLY PIPELINE")
    print(f"  Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR_REAL, exist_ok=True)

    data = _load_real_data()
    classifier = train_classifier_real(data)
    doa_models = train_doa_real(data)
    evaluate_pipeline(data, classifier, doa_models)


if __name__ == "__main__":
    main()
