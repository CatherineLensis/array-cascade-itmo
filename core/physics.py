"""
core/physics.py
================
Абстракция steering-матрицы и связанной физики (углы → комплексный отклик).

В этот модуль выносим то, что вычисляется через `DomainSpec` без
domain-специфичных особенностей:
  - steering_matrix(domain, angles_deg) → np.ndarray (n_ant, len(angles))
  - rayleigh_limit_deg(domain) → float
  - crb_at_snr(domain, snr_db) → float

Физика конкретного домена (urban multipath для counter-UAV, например)
живёт в `domains/<domain>/physics_*.py`, а не здесь.

Status: SKELETON. День 4 — реализация по мере необходимости при переносе
core/cascade.py.

См. SPEC_ITMO_V2.md §5, §6, §7.
"""

from __future__ import annotations

import numpy as np


def steering_matrix(
    array_pos_lambda_half: np.ndarray,
    wavelength_m: float,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """
    Steering-матрица ULA для заданных углов.

    Args:
        array_pos_lambda_half: позиции антенн в долях λ/2 (форма: (n_ant,))
        wavelength_m:          длина волны в метрах (для документации, не используется
                               в формуле т.к. array_pos уже нормирован λ/2)
        angles_deg:            углы в градусах (форма: (n_angles,))

    Returns:
        A: complex64 (n_ant, n_angles), где A[k, i] = exp(j·π·k·sin(θ_i))
    """
    angles_rad = np.deg2rad(angles_deg).astype(np.float64)
    # phase = π · pos · sin(θ)   (т.к. step = λ/2 → 2π·d/λ·sinθ = π·sinθ для d=λ/2)
    phase = np.pi * np.outer(array_pos_lambda_half, np.sin(angles_rad))
    return np.exp(1j * phase).astype(np.complex64)


def rayleigh_limit_deg(array_pos_lambda_half: np.ndarray) -> float:
    """
    Теоретический предел Рэлея для классического beamforming.

    Δθ_min ≈ 0.886 · λ / D, где D — апертура массива.

    Считается напрямую из array_pos (в единицах λ/2):
        D[λ] = (max(pos) - min(pos)) · (λ/2) / λ = span / 2

    Для array_pos=[0, 0.5, ..., 3.5] (как у TI AWR1843):
        span = 3.5, D = 1.75λ → Δθ ≈ 29°  (совпадает с источником правды Шишанова)
    """
    span_lambda_half = float(array_pos_lambda_half.max() - array_pos_lambda_half.min())
    aperture_lambda = span_lambda_half * 0.5
    delta_rad = 0.886 / aperture_lambda
    return float(np.rad2deg(delta_rad))
