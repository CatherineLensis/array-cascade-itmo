"""
domains/counter_uav/physics_xband.py
====================================
Steering-матрица, классический beamforming и предел Рэлея для X-band ULA.

Переиспользует `core.physics` (SPEC §7.1).
"""

from __future__ import annotations

import numpy as np

from core.physics import rayleigh_limit_deg, steering_matrix
from domains.counter_uav.config import (
    ARRAY_POS,
    DOA_ANGLE_LIMIT_DEG,
    DOA_GRID_DEG,
    N_ANT,
    N_GRID,
    WAVELENGTH_M,
)


def angle_grid_deg(n_grid: int = N_GRID, limit_deg: float = DOA_ANGLE_LIMIT_DEG) -> np.ndarray:
    """Равномерная угловая сетка для DOA."""
    return np.linspace(-limit_deg, limit_deg, n_grid, dtype=np.float64)


def steering_matrix_xband(angles_deg: np.ndarray) -> np.ndarray:
    """Steering-матрица A [n_ant, n_angles] для X-band ULA."""
    return steering_matrix(ARRAY_POS, WAVELENGTH_M, np.asarray(angles_deg, dtype=np.float64))


def steering_vector(azimuth_deg: float) -> np.ndarray:
    """Steering-вектор для одного угла, форма (n_ant,)."""
    return steering_matrix_xband(np.array([azimuth_deg], dtype=np.float64))[:, 0]


def build_steering_grid(
    n_grid: int = N_GRID,
    limit_deg: float = DOA_ANGLE_LIMIT_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (A, grid_deg), A формы (n_ant, n_grid)."""
    grid_deg = angle_grid_deg(n_grid, limit_deg)
    return steering_matrix_xband(grid_deg), grid_deg


def classical_beamforming_spectrum(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Классический (conventional) beamformer.

    Args:
        x: комплексный snapshot (n_ant,)
        A: steering-матрица (n_ant, n_grid)

    Returns:
        spectrum: (n_grid,) неотрицательные значения
    """
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    if x.shape[0] != A.shape[0]:
        raise ValueError(f"Expected {A.shape[0]} antennas, got {x.shape[0]}")
    return np.abs(np.conj(A).T @ x)


def estimate_doa_beamforming(
    x: np.ndarray,
    A: np.ndarray | None = None,
    grid_deg: np.ndarray | None = None,
) -> float:
    """Оценка DOA argmax классического beamforming, возвращает угол в градусах."""
    if A is None or grid_deg is None:
        A, grid_deg = build_steering_grid()
    spectrum = classical_beamforming_spectrum(x, A)
    return float(grid_deg[int(np.argmax(spectrum))])


def rayleigh_limit() -> float:
    """Теоретический предел Рэлея для текущей ULA (градусы)."""
    return rayleigh_limit_deg(ARRAY_POS)


def synth_single_target_snapshot(
    azimuth_deg: float,
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Синтетический snapshot одной цели: steering(θ) + AWGN.

    Используется для sanity-тестов beamforming и baseline DOA.
    """
    if rng is None:
        rng = np.random.default_rng()
    a = steering_vector(azimuth_deg)
    signal = np.exp(1j * 2.0 * np.pi * rng.random()) * a
    noise_power = 10.0 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(N_ANT) + 1j * rng.standard_normal(N_ANT)
    )
    x = signal + noise
    rms = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
    return (x / rms).astype(np.complex64)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Добавляет AWGN к уже сформированному сигналу и RMS-нормирует."""
    x = np.asarray(x, dtype=np.complex64)
    signal_power = np.mean(np.abs(x) ** 2) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    y = x + noise
    rms = np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12
    return (y / rms).astype(np.complex64)
