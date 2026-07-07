from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import DEVICE
from core.classifier import (
    ComplexNet,
    canonicalize_counter_uav_batch,
    canonicalize_counter_uav_snapshot,
    to_complex_tensor,
)
from domains.counter_uav.config import CLASSIFIER_PATH, CLASSES, N_ANT, SYNTH_NPZ
from domains.counter_uav.dataset_synth import build_bird_snapshot, build_drone_snapshot


def _load_counter_uav_classifier() -> ComplexNet:
    model = ComplexNet(input_length=N_ANT, n_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.eval()
    return model


def _holdout_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(SYNTH_NPZ, allow_pickle=True)
    signals = npz["signals"]
    labels = npz["labels"].astype(int)
    n_walls = npz["n_walls"].astype(int)
    idx = np.arange(len(signals))
    _, vl = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    return signals[vl], labels[vl], n_walls[vl]


def _predict_labels(model: ComplexNet, signals: np.ndarray) -> np.ndarray:
    x = canonicalize_counter_uav_batch(np.asarray(signals, dtype=np.complex64))
    xb = to_complex_tensor(x[:, :, None]).to(DEVICE)
    with torch.no_grad():
        return model(xb).argmax(1).cpu().numpy()


def analyze_counter_uav_classifier_failure() -> dict[str, float]:
    model = _load_counter_uav_classifier()
    signals, labels, n_walls = _holdout_split()
    pred = _predict_labels(model, signals)

    bird_mask = labels == 2
    strong_mp_mask = bird_mask & np.isin(n_walls, [4, 6])

    u2_signal = build_bird_snapshot(-20.0, 15.0, 4, np.random.default_rng(60), velocity_mps=12.0)
    u2_norm = canonicalize_counter_uav_snapshot(u2_signal)
    xb = to_complex_tensor(u2_norm.reshape(1, -1, 1)).to(DEVICE)
    with torch.no_grad():
        logits = model(xb).squeeze(0).cpu().numpy()
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    return {
        "bird_recall_nwalls_4_6": float((pred[strong_mp_mask] == 2).mean()),
        "bird_recall_overall": float((pred[bird_mask] == 2).mean()),
        "u2_margin_vs_drone": float(probs[2] - max(probs[0], probs[1], probs[3])),
    }


def _probe_score(samples: list[np.ndarray], labels: list[int]) -> float:
    X = canonicalize_counter_uav_batch(np.asarray(samples, dtype=np.complex64))
    feats = np.concatenate([X.real, X.imag], axis=1)
    y = np.asarray(labels, dtype=np.int32)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(feats[tr], y[tr])
    return float(clf.score(feats[te], y[te]))


def analyze_generator_separability() -> dict[str, float]:
    clean_samples: list[np.ndarray] = []
    clean_labels: list[int] = []
    multipath_samples: list[np.ndarray] = []
    multipath_labels: list[int] = []

    for i in range(200):
        rng_drone = np.random.default_rng(i)
        rng_bird = np.random.default_rng(1000 + i)

        clean_samples.append(build_drone_snapshot(20.0, 25.0, 0, rng_drone, velocity_mps=12.0))
        clean_labels.append(0)
        clean_samples.append(build_bird_snapshot(20.0, 25.0, 0, rng_bird, velocity_mps=12.0))
        clean_labels.append(1)

        multipath_samples.append(build_drone_snapshot(20.0, 15.0, 4, np.random.default_rng(i), velocity_mps=12.0))
        multipath_labels.append(0)
        multipath_samples.append(build_bird_snapshot(20.0, 15.0, 4, np.random.default_rng(1000 + i), velocity_mps=12.0))
        multipath_labels.append(1)

    return {
        "clean_sep_lr_score": _probe_score(clean_samples, clean_labels),
        "multipath_sep_lr_score": _probe_score(multipath_samples, multipath_labels),
    }


__all__ = [
    "analyze_counter_uav_classifier_failure",
    "analyze_generator_separability",
]
