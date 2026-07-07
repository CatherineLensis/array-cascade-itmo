"""
core/classifier.py — ComplexNet with configurable n_classes.
"""

from __future__ import annotations

import numpy as np

from models import ComplexNet, to_complex_tensor


_COUNTER_UAV_STEERING = None
_COUNTER_UAV_GRID = None


def normalize_counter_uav_snapshot(signal: np.ndarray) -> np.ndarray:
    """Apply only global phase alignment + RMS normalization for classifier stability."""
    x = np.asarray(signal, dtype=np.complex64).reshape(-1)
    aligned = x * np.exp(-1j * np.angle(x[0] + 1e-12))
    rms = np.sqrt(np.mean(np.abs(aligned) ** 2)) + 1e-12
    return (aligned / rms).astype(np.complex64)


def normalize_counter_uav_batch(signals: np.ndarray) -> np.ndarray:
    signals = np.asarray(signals)
    return np.stack([normalize_counter_uav_snapshot(x) for x in signals], axis=0)


def canonicalize_counter_uav_snapshot(signal: np.ndarray) -> np.ndarray:
    """De-steer counter-UAV snapshots to reduce angle variance for classification."""
    global _COUNTER_UAV_STEERING, _COUNTER_UAV_GRID

    from domains.counter_uav.physics_xband import (
        build_steering_grid,
        estimate_doa_beamforming,
        steering_vector,
    )

    x = np.asarray(signal, dtype=np.complex64).reshape(-1)
    if _COUNTER_UAV_STEERING is None or _COUNTER_UAV_GRID is None:
        _COUNTER_UAV_STEERING, _COUNTER_UAV_GRID = build_steering_grid()

    angle = estimate_doa_beamforming(x, A=_COUNTER_UAV_STEERING, grid_deg=_COUNTER_UAV_GRID)
    canonical = x * np.conj(steering_vector(float(angle)))
    canonical = canonical * np.exp(-1j * np.angle(canonical[0] + 1e-12))
    rms = np.sqrt(np.mean(np.abs(canonical) ** 2)) + 1e-12
    return (canonical / rms).astype(np.complex64)


def canonicalize_counter_uav_batch(signals: np.ndarray) -> np.ndarray:
    signals = np.asarray(signals)
    return np.stack([canonicalize_counter_uav_snapshot(x) for x in signals], axis=0)


__all__ = [
    "ComplexNet",
    "to_complex_tensor",
    "normalize_counter_uav_snapshot",
    "normalize_counter_uav_batch",
    "canonicalize_counter_uav_snapshot",
    "canonicalize_counter_uav_batch",
]
