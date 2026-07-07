"""
domains/counter_uav/pipeline.py — Train classifier + DOA models on counter-UAV synth data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEVICE, LISTA_LAYERS, N_GRID
from core.classifier import ComplexNet, canonicalize_counter_uav_batch, to_complex_tensor
from core.doa import DeepLISTA_DOA, SiameseDOAModel
from domains.counter_uav.config import (
    CLASSIFIER_PATH,
    CLASSES,
    DOA_GRID_DEG,
    DOA_LISTA_PATH,
    DOA_SIAMESE_PATH,
    METRICS_JSON,
    N_ANT,
    OUTPUT_DIR,
    SYNTH_NPZ,
)
from domains.counter_uav.physics_xband import steering_matrix_xband
from run_full_pipeline import predict_doa_single, doa_rmse


PREPROCESSING_CONTRACT = {
    "train_input": "raw",
    "label_grid": "counter-UAV steering-domain grid [-70, 70] deg",
    "inference_required": "DomainSpec.counter_uav().preprocessing == 'raw'",
}


def make_counter_uav_doa_labels(
    angles_np: np.ndarray,
    grid_deg: np.ndarray | None = None,
    sigma_deg: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian soft labels on the physical counter-UAV angle grid."""
    grid_deg = np.asarray(grid_deg if grid_deg is not None else DOA_GRID_DEG, dtype=np.float32)
    labels = np.zeros((len(angles_np), len(grid_deg)), dtype=np.float32)
    lo = float(grid_deg[0])
    hi = float(grid_deg[-1])
    for i, angs in enumerate(angles_np):
        for a_rad in np.atleast_1d(angs).ravel():
            a_deg = np.rad2deg(a_rad)
            if lo <= a_deg <= hi:
                gauss = np.exp(-0.5 * ((grid_deg - a_deg) / sigma_deg) ** 2)
                labels[i] = np.maximum(labels[i], gauss)
    row_max = labels.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    labels = labels / row_max
    return labels, grid_deg


class ClfDataset(Dataset):
    def __init__(self, X, y):
        self.X = to_complex_tensor(X[:, :, None])
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleDOADataset(Dataset):
    def __init__(self, channels, doa_labels):
        self.X = torch.complex(
            torch.tensor(channels.real, dtype=torch.float32),
            torch.tensor(channels.imag, dtype=torch.float32),
        ).unsqueeze(1)
        self.doa = torch.tensor(doa_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.doa[idx]


def load_synth_data(npz_path: str | None = None):
    path = npz_path or SYNTH_NPZ
    if not Path(path).exists():
        from domains.counter_uav.dataset_synth import generate_dataset
        generate_dataset(n_samples=2048, output_path=path, seed=42)
    npz = np.load(path, allow_pickle=True)
    signals = npz["signals"]
    labels = npz["labels"].astype(int)
    angles = npz["angles_deg"]
    n_walls = npz["n_walls"].astype(int) if "n_walls" in npz else np.zeros(len(labels), dtype=np.int32)
    return signals, labels, angles, n_walls


def train_classifier(signals, labels, n_walls, epochs: int = 30):
    signals = canonicalize_counter_uav_batch(signals)
    idx = np.arange(len(signals))
    tr, vl = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    ds_tr = ClfDataset(signals[tr], labels[tr])
    ds_vl = ClfDataset(signals[vl], labels[vl])
    sample_weights = np.ones(len(tr), dtype=np.float32)
    hard_bird_mask = (labels[tr] == 2) & (n_walls[tr] >= 4)
    sample_weights[hard_bird_mask] = 1.25
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    loader_tr = DataLoader(ds_tr, batch_size=128, sampler=sampler)
    loader_vl = DataLoader(ds_vl, batch_size=256)

    model = ComplexNet(input_length=N_ANT, n_classes=len(CLASSES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.0, 1.2, 1.0], dtype=torch.float32, device=DEVICE)
    )

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = crit(out.float(), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader_vl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0:
            print(f"  clf epoch {epoch+1}: val acc={acc:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_PATH)

    with torch.no_grad():
        preds, ys = [], []
        for xb, yb in loader_vl:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            ys.extend(yb.numpy())
    report = classification_report(ys, preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(ys, preds).tolist()
    return model, {"accuracy": best_acc, "report": report, "confusion_matrix": cm}


def train_doa_siamese(channels, angles_rad, epochs: int = 40):
    doa_labels, grid_deg = make_counter_uav_doa_labels(angles_rad)
    ds = SimpleDOADataset(channels, doa_labels)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
    model = SiameseDOAModel(n_grid=N_GRID, N_ant=N_ANT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, doa_b in loader:
            xb, doa_b = xb.to(DEVICE), doa_b.to(DEVICE)
            _, doa_logits, _ = model(xb)
            loss = bce(doa_logits, doa_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n += 1
        if (epoch + 1) % 20 == 0:
            print(f"  siamese epoch {epoch+1}: loss={loss_sum/max(n,1):.4f}")
    torch.save(model.state_dict(), DOA_SIAMESE_PATH)
    return model, grid_deg


def train_doa_lista(channels, angles_rad, epochs: int = 40):
    doa_labels, grid_deg = make_counter_uav_doa_labels(angles_rad)
    A_np = steering_matrix_xband(grid_deg)
    ds = SimpleDOADataset(channels, doa_labels)
    loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
    model = DeepLISTA_DOA(A_np, n_layers=LISTA_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        for xb, doa_b in loader:
            xb, doa_b = xb.to(DEVICE), doa_b.to(DEVICE)
            power, _ = model(xb.squeeze(1))
            loss = bce(power, doa_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), DOA_LISTA_PATH)
    return model, grid_deg


def evaluate_doa(models, channels, angles_deg, grid_deg):
    metrics = {}
    angles_rad = np.deg2rad(angles_deg)
    for name, model in models.items():
        rmses = []
        for i in range(min(len(channels), 200)):
            pred, _ = predict_doa_single(channels[i], name, model, grid_deg)
            err = doa_rmse(pred, angles_deg[i].ravel() if angles_deg[i].ndim else np.array([angles_deg[i]]))
            if not np.isnan(err):
                rmses.append(err)
        metrics[name] = {"rmse_deg": float(np.mean(rmses)) if rmses else None}
    return metrics


def main(npz_path: str | None = None, quick: bool = False, classifier_only: bool = False):
    print("=" * 60)
    print("COUNTER-UAV TRAINING PIPELINE")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    signals, labels, angles_deg, n_walls = load_synth_data(npz_path)

    clf_epochs = 15 if quick else 30
    doa_epochs = 20 if quick else 40

    print(f"\nSamples: {len(signals)}, classes: {np.bincount(labels)}")
    clf_model, clf_metrics = train_classifier(signals, labels, n_walls, epochs=clf_epochs)
    print(f"Classifier accuracy: {clf_metrics['accuracy']:.4f}")

    doa_metrics: dict[str, dict[str, float | None]] = {}
    if classifier_only:
        if Path(METRICS_JSON).exists():
            try:
                previous = json.loads(Path(METRICS_JSON).read_text(encoding="utf-8"))
                doa_metrics = previous.get("doa", {})
            except json.JSONDecodeError:
                doa_metrics = {}
        print("\nDOA retraining skipped: classifier_only mode")
    else:
        doa_mask = labels != 0
        ch_doa = signals[doa_mask]
        ang_doa = np.deg2rad(angles_deg[doa_mask])

        print(f"\nDOA training on {len(ch_doa)} samples (non-clutter)...")
        m1, grid = train_doa_siamese(ch_doa, ang_doa, epochs=doa_epochs)
        m2, _ = train_doa_lista(ch_doa, ang_doa, epochs=doa_epochs)
        doa_models = {"siamese": m1, "lista": m2}
        doa_metrics = evaluate_doa(doa_models, ch_doa[:200], angles_deg[doa_mask][:200], grid)

    out = {
        "classifier": clf_metrics,
        "doa": doa_metrics,
        "n_samples": int(len(signals)),
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "doa_grid_deg": {
            "min": float(DOA_GRID_DEG[0]),
            "max": float(DOA_GRID_DEG[-1]),
            "n_grid": int(len(DOA_GRID_DEG)),
        },
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved: {METRICS_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    main(args.npz, quick=args.quick)
