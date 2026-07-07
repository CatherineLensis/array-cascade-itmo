"""
core/preprocessing.py — Wiener-Hopf whitening + RMS normalization.
"""

from __future__ import annotations

import numpy as np

from preprocessing import preprocess_channels, rms_normalize, max_normalize


class Preprocessor:
    """Domain-agnostic preprocessing wrapper."""

    def __init__(self, whiten_k: int = 64, normalize: str = "rms"):
        self.whiten_k = whiten_k
        self.normalize = normalize

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Process single signal [n_ant] or batch [N, n_ant]."""
        if signal.ndim == 1:
            return preprocess_channels(
                signal.reshape(1, -1),
                K=self.whiten_k,
                normalize=self.normalize,
            )[0]
        return preprocess_channels(signal, K=self.whiten_k, normalize=self.normalize)


class IdentityPreprocessor:
    """No-op preprocessing for models trained on raw steering-domain signals."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return np.asarray(signal).copy()


__all__ = [
    "IdentityPreprocessor",
    "Preprocessor",
    "preprocess_channels",
    "rms_normalize",
    "max_normalize",
]
