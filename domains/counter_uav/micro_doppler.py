"""
domains/counter_uav/micro_doppler.py
===================================
Micro-Doppler модели дрона (4×2 лопасти) и птицы (wing-beat 3–8 Hz).

См. SPEC §7.3.
"""

from __future__ import annotations

import numpy as np

from domains.counter_uav.config import N_CHIRPS, CPI_S, WAVELENGTH_M
from domains.counter_uav.physics_xband import add_awgn, steering_vector


def _time_axis(n_samples: int | None = None) -> np.ndarray:
    n = n_samples if n_samples is not None else N_CHIRPS
    return np.linspace(0.0, CPI_S, n, endpoint=False)


def _bulk_phase_series(t: np.ndarray, velocity_mps: float) -> np.ndarray:
    fd_bulk = 2.0 * velocity_mps / WAVELENGTH_M
    return np.exp(1j * 2.0 * np.pi * fd_bulk * t)


def _collapse_spatial_series(spatial: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    snapshot = np.mean(spatial, axis=1)
    snapshot = add_awgn(snapshot, snr_db, rng)
    micro_series = np.mean(spatial, axis=0)
    return snapshot.astype(np.complex64), micro_series.astype(np.complex64)


def drone_micro_doppler_phase(
    t: float | np.ndarray,
    rng: np.random.Generator,
    n_motors: int = 4,
    n_blades_per_motor: int = 2,
    blade_radius_m: float = 0.10,
) -> np.complex128 | np.ndarray:
    """
    Композитная micro-Doppler фаза квадрокоптера.

    v_blade = ω_k · r · sin(ω_k·t + θ_k + φ_l)
    fd ≈ 2·v_blade / λ
    """
    t_arr = np.atleast_1d(t).astype(np.float64)
    lam = WAVELENGTH_M
    acc = np.zeros_like(t_arr, dtype=np.complex128)

    for _ in range(n_motors):
        omega_k = rng.uniform(150.0, 250.0)
        theta_k = rng.uniform(0.0, 2.0 * np.pi)
        for _ in range(n_blades_per_motor):
            phi_l = rng.uniform(0.0, 2.0 * np.pi)
            v_blade = blade_radius_m * omega_k * np.sin(omega_k * t_arr + theta_k + phi_l)
            fd = 2.0 * v_blade / lam
            acc += np.exp(1j * 2.0 * np.pi * fd * t_arr)

    norm = n_motors * n_blades_per_motor
    result = acc / norm
    return result[0] if np.ndim(t) == 0 else result


def bird_micro_doppler_phase(
    t: float | np.ndarray,
    rng: np.random.Generator,
    f_wb: float | None = None,
    a_wing_m: float | None = None,
) -> np.complex128 | np.ndarray:
    """
    Micro-Doppler фаза птицы (wing-beat).

    v_wing = A_wing · 2π·f_wb · cos(2π·f_wb·t + φ)
    """
    t_arr = np.atleast_1d(t).astype(np.float64)
    if f_wb is None:
        f_wb = rng.uniform(3.0, 8.0)
    if a_wing_m is None:
        a_wing_m = rng.uniform(0.1, 0.3)

    phi = rng.uniform(0.0, 2.0 * np.pi)
    v_wing = a_wing_m * 2.0 * np.pi * f_wb * np.cos(2.0 * np.pi * f_wb * t_arr + phi)
    fd = 2.0 * v_wing / WAVELENGTH_M
    result = np.exp(1j * 2.0 * np.pi * fd * t_arr)
    return result[0] if np.ndim(t) == 0 else result


def synth_drone_signal(
    azimuth_deg: float,
    snr_db: float = 15.0,
    rng: np.random.Generator | None = None,
    n_time: int = N_CHIRPS,
    velocity_mps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Синтез micro-Doppler сигнала дрона.

    Returns:
        snapshot: комплексный вектор (n_ant,) — усреднение по CPI с micro-Doppler
        micro_series: (n_time,) комплексная slow-time последовательность (для STFT)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = _time_axis(n_time)
    if velocity_mps is None:
        velocity_mps = rng.uniform(5.0, 25.0)
    bulk_phase = _bulk_phase_series(t, velocity_mps)

    spatial = 0.35 * steering_vector(azimuth_deg)[:, None] * bulk_phase[None, :]
    n_rotors = 4
    rotor_offsets = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) + rng.normal(0.0, 0.15, size=n_rotors)
    rotor_coeffs = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j], dtype=np.complex64)

    for offset_deg, rotor_coeff in zip(rotor_offsets, rotor_coeffs, strict=True):
        a_rotor = steering_vector(float(azimuth_deg + offset_deg))
        rotor_micro = drone_micro_doppler_phase(
            t,
            rng,
            n_motors=1,
            n_blades_per_motor=2,
            blade_radius_m=float(rng.uniform(0.08, 0.14)),
        )
        amp = (0.45 + 0.05 * rng.random()) * rotor_coeff
        spatial += amp * a_rotor[:, None] * rotor_micro[None, :] * bulk_phase[None, :]

    spatial /= 0.35 + n_rotors
    micro_series = np.mean(spatial, axis=0).astype(np.complex64)

    drone_proxy = 0.28 * steering_vector(azimuth_deg)
    for offset_deg, rotor_coeff in zip(rotor_offsets, rotor_coeffs, strict=True):
        drone_proxy = drone_proxy + 0.42 * rotor_coeff * steering_vector(float(azimuth_deg + offset_deg))
    drone_proxy = drone_proxy / (np.sqrt(np.mean(np.abs(drone_proxy) ** 2)) + 1e-12)
    snapshot = add_awgn(drone_proxy.astype(np.complex64), snr_db, rng)
    return snapshot.astype(np.complex64), micro_series


def synth_bird_signal(
    azimuth_deg: float,
    snr_db: float = 15.0,
    rng: np.random.Generator | None = None,
    n_time: int = N_CHIRPS,
    velocity_mps: float | None = None,
    f_wb: float | None = None,
    a_wing_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Синтез micro-Doppler сигнала птицы.

    Returns:
        snapshot: (n_ant,)
        micro_series: (n_time,)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = _time_axis(n_time)
    if velocity_mps is None:
        velocity_mps = rng.uniform(5.0, 25.0)
    bulk_phase = _bulk_phase_series(t, velocity_mps)

    if f_wb is None:
        f_wb = rng.uniform(3.0, 8.0)
    if a_wing_m is None:
        a_wing_m = rng.uniform(0.1, 0.3)

    body = 0.8 * steering_vector(azimuth_deg)[:, None] * bulk_phase[None, :]

    wing_span_deg = float(rng.uniform(6.0, 9.0))
    left_micro = bird_micro_doppler_phase(t, rng, f_wb=f_wb, a_wing_m=a_wing_m)
    right_micro = bird_micro_doppler_phase(t, rng, f_wb=f_wb, a_wing_m=a_wing_m)
    left = 0.55 * steering_vector(float(azimuth_deg - wing_span_deg))[:, None] * left_micro[None, :] * bulk_phase[None, :]
    right = 0.55 * steering_vector(float(azimuth_deg + wing_span_deg))[:, None] * right_micro[None, :] * bulk_phase[None, :]

    spatial = body + left + right
    spatial /= 1.9
    micro_series = np.mean(spatial, axis=0).astype(np.complex64)

    bird_proxy = 0.95 * steering_vector(azimuth_deg)
    bird_proxy = bird_proxy + 0.65 * steering_vector(float(azimuth_deg - wing_span_deg))
    bird_proxy = bird_proxy - 0.65 * steering_vector(float(azimuth_deg + wing_span_deg))
    bird_proxy = bird_proxy + 0.12j * steering_vector(azimuth_deg)
    bird_proxy = bird_proxy / (np.sqrt(np.mean(np.abs(bird_proxy) ** 2)) + 1e-12)
    snapshot = add_awgn(bird_proxy.astype(np.complex64), snr_db, rng)
    return snapshot.astype(np.complex64), micro_series


def micro_doppler_spectrogram(
    micro_series: np.ndarray,
    fs_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT спектрограмма slow-time micro-Doppler (для EDA / UI).

    Returns:
        f_hz, t_s, Sxx (амплитуда)
    """
    from scipy.signal import stft

    if fs_hz is None:
        fs_hz = len(micro_series) / CPI_S
    f_hz, t_s, zxx = stft(micro_series, fs=fs_hz, nperseg=min(32, len(micro_series)))
    return f_hz, t_s, np.abs(zxx)
