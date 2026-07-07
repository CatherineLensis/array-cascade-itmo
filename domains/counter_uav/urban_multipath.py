"""
domains/counter_uav/urban_multipath.py
======================================
Упрощённая urban multipath модель (method of images).

См. SPEC §7.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domains.counter_uav.classes import N_WALLS_CHOICES
from domains.counter_uav.config import DOA_ANGLE_LIMIT_DEG, WAVELENGTH_M
from domains.counter_uav.physics_xband import add_awgn, steering_vector


@dataclass(frozen=True)
class MultipathComponent:
    azimuth_deg: float
    amplitude: float
    phase_rad: float
    is_direct: bool = False


def _clip_azimuth(azimuth_deg: float) -> float:
    return float(np.clip(azimuth_deg, -85.0, 85.0))


def sample_n_walls(rng: np.random.Generator) -> int:
    """Случайное число стен из допустимого множества {0, 2, 4, 6}."""
    return int(rng.choice(N_WALLS_CHOICES))


def build_multipath_components(
    azimuth_deg: float,
    n_walls: int,
    rng: np.random.Generator,
    alpha_range: tuple[float, float] = (0.1, 0.6),
) -> list[MultipathComponent]:
    """
    Method of images: direct path + отражения от n_walls «стен».

    Каждая стена задаётся азимутом «зеркала»; image angle = 2·az_wall − θ_target.
    """
    if n_walls not in N_WALLS_CHOICES:
        raise ValueError(f"n_walls must be one of {N_WALLS_CHOICES}, got {n_walls}")

    components = [
        MultipathComponent(
            azimuth_deg=_clip_azimuth(azimuth_deg),
            amplitude=1.0,
            phase_rad=0.0,
            is_direct=True,
        )
    ]

    for _ in range(n_walls):
        az_wall = rng.uniform(-DOA_ANGLE_LIMIT_DEG, DOA_ANGLE_LIMIT_DEG)
        az_image = _clip_azimuth(2.0 * az_wall - azimuth_deg)
        alpha = rng.uniform(*alpha_range)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        components.append(
            MultipathComponent(
                azimuth_deg=az_image,
                amplitude=alpha,
                phase_rad=phi,
                is_direct=False,
            )
        )

    return components


def synthesize_multipath_snapshot(
    components: list[MultipathComponent],
    micro_phase: complex = 1.0 + 0.0j,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Суммирует вклады direct + image paths в пространственный snapshot."""
    if rng is None:
        rng = np.random.default_rng()

    snapshot = np.zeros(steering_vector(0.0).shape, dtype=np.complex64)
    for comp in components:
        a = steering_vector(comp.azimuth_deg)
        snapshot += comp.amplitude * np.exp(1j * comp.phase_rad) * micro_phase * a
    return snapshot.astype(np.complex64)


def apply_urban_multipath(
    azimuth_deg: float,
    n_walls: int,
    micro_phase: complex,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[MultipathComponent]]:
    """
    Полный pipeline: images → snapshot → AWGN.

    Returns:
        snapshot: (n_ant,)
        components: список лучей (direct + images)
    """
    components = build_multipath_components(azimuth_deg, n_walls, rng)
    snapshot = synthesize_multipath_snapshot(components, micro_phase=micro_phase, rng=rng)
    snapshot = add_awgn(snapshot, snr_db, rng)
    return snapshot, components


def extra_delay_m(component: MultipathComponent, range_m: float) -> float:
    """
    Дополнительная задержка image-path относительно direct (упрощённо).

    τ ≈ 2·Δr/c, где Δr моделируется через угловое отличие и дальность.
    """
    if component.is_direct:
        return 0.0
    delta_deg = abs(component.azimuth_deg) * 0.01
    delta_r = range_m * np.tan(np.deg2rad(delta_deg))
    return 2.0 * delta_r / 3.0e8


def path_phase_from_delay(delay_s: float, fc_hz: float | None = None) -> float:
    """Фазовый сдвиг exp(j·2π·fc·τ)."""
    if fc_hz is None:
        fc_hz = 3.0e8 / WAVELENGTH_M
    return float(2.0 * np.pi * fc_hz * delay_s)
