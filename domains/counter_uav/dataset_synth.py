"""
domains/counter_uav/dataset_synth.py
====================================
Генератор 4-классового синтетического NPZ-датасета counter-UAV.

Классы: clutter / drone / bird / multi (SPEC §7.5).

Пример:
    python -m domains.counter_uav.dataset_synth --n-samples 256 --output output_counter_uav/synth_small.npz
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
from tqdm import tqdm

from domains.counter_uav.classes import BIRD, CLUTTER, DRONE, MULTI
from domains.counter_uav.config import (
    DEFAULT_SYNTH_NPZ,
    DOA_ANGLE_LIMIT_DEG,
    N_ANT,
    N_SYNTH_DEFAULT,
    N_SYNTH_SMALL,
    OUTPUT_DIR,
    RANGE_M,
    SMALL_SYNTH_NPZ,
    SNR_DB,
    VELOCITY_MPS,
)
from domains.counter_uav.micro_doppler import synth_bird_signal, synth_drone_signal
from domains.counter_uav.physics_xband import add_awgn
from domains.counter_uav.urban_multipath import apply_urban_multipath, build_multipath_components, sample_n_walls


def _sample_azimuth(rng: np.random.Generator) -> float:
    return float(rng.uniform(-DOA_ANGLE_LIMIT_DEG, DOA_ANGLE_LIMIT_DEG))


def _sample_snr(rng: np.random.Generator) -> float:
    return float(rng.uniform(SNR_DB[0], SNR_DB[1]))


def _sample_range(rng: np.random.Generator) -> float:
    return float(rng.uniform(RANGE_M[0], RANGE_M[1]))


def _sample_velocity(rng: np.random.Generator) -> float:
    return float(rng.uniform(VELOCITY_MPS[0], VELOCITY_MPS[1]))


def _two_separated_azimuths(rng: np.random.Generator, min_sep_deg: float = 8.0) -> tuple[float, float]:
    for _ in range(32):
        a1 = _sample_azimuth(rng)
        a2 = _sample_azimuth(rng)
        if abs(a1 - a2) >= min_sep_deg:
            return a1, a2
    center = _sample_azimuth(rng)
    return center - min_sep_deg / 2.0, center + min_sep_deg / 2.0


def generate_clutter_sample(rng: np.random.Generator, snr_db: float, n_walls: int) -> dict[str, Any]:
    """Clutter: только шум + urban multipath без целей (ground-like reflections)."""
    az_ghost = _sample_azimuth(rng)
    micro_phase = 0.3 * np.exp(1j * 2.0 * np.pi * rng.random())
    snapshot, _ = apply_urban_multipath(
        azimuth_deg=az_ghost,
        n_walls=n_walls,
        micro_phase=micro_phase,
        snr_db=snr_db,
        rng=rng,
    )
    # Ослабляем «целевой» вклад — остаётся преимущественно шум/мультипас
    snapshot = 0.15 * snapshot + add_awgn(np.zeros(N_ANT, dtype=np.complex64), snr_db, rng)
    return {
        "signal": snapshot,
        "angles_deg": np.array([np.nan, np.nan], dtype=np.float32),
        "target_types": np.array(["none", "none"], dtype=object),
    }


def _apply_snapshot_multipath(
    snapshot: np.ndarray,
    azimuth_deg: float,
    n_walls: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject image-path copies while preserving class-specific target residual."""
    from domains.counter_uav.physics_xband import steering_vector

    residual_scale = 1.75
    ghost_scale = 0.7

    components = build_multipath_components(azimuth_deg, n_walls, rng)
    a_direct = steering_vector(azimuth_deg)
    direct_coeff = np.vdot(a_direct, snapshot) / (np.vdot(a_direct, a_direct) + 1e-12)
    residual = snapshot - direct_coeff * a_direct

    mixed = (residual_scale * residual).astype(np.complex64)
    for comp in components:
        a = steering_vector(comp.azimuth_deg)
        coeff = direct_coeff
        if not comp.is_direct:
            coeff = coeff * ghost_scale * comp.amplitude * np.exp(1j * comp.phase_rad)
        mixed = mixed + coeff * a
    return mixed.astype(np.complex64)


def generate_drone_sample(
    rng: np.random.Generator,
    snr_db: float,
    n_walls: int,
    velocity_mps: float | None = None,
) -> dict[str, Any]:
    az = _sample_azimuth(rng)
    return {
        "signal": build_drone_snapshot(az, snr_db, n_walls, rng, velocity_mps=velocity_mps),
        "angles_deg": np.array([az, np.nan], dtype=np.float32),
        "target_types": np.array(["drone", "none"], dtype=object),
    }


def build_drone_snapshot(
    azimuth_deg: float,
    snr_db: float,
    n_walls: int,
    rng: np.random.Generator,
    velocity_mps: float | None = None,
) -> np.ndarray:
    snapshot, _ = synth_drone_signal(azimuth_deg, snr_db=snr_db + 6.0, rng=rng, velocity_mps=velocity_mps)
    snapshot = _apply_snapshot_multipath(snapshot, azimuth_deg, n_walls, rng)
    snapshot = add_awgn(snapshot, snr_db, rng)
    return snapshot.astype(np.complex64)


def generate_bird_sample(
    rng: np.random.Generator,
    snr_db: float,
    n_walls: int,
    velocity_mps: float | None = None,
) -> dict[str, Any]:
    az = _sample_azimuth(rng)
    return {
        "signal": build_bird_snapshot(az, snr_db, n_walls, rng, velocity_mps=velocity_mps),
        "angles_deg": np.array([az, np.nan], dtype=np.float32),
        "target_types": np.array(["bird", "none"], dtype=object),
    }


def build_bird_snapshot(
    azimuth_deg: float,
    snr_db: float,
    n_walls: int,
    rng: np.random.Generator,
    velocity_mps: float | None = None,
) -> np.ndarray:
    snapshot, _ = synth_bird_signal(azimuth_deg, snr_db=snr_db + 6.0, rng=rng, velocity_mps=velocity_mps)
    snapshot = _apply_snapshot_multipath(snapshot, azimuth_deg, n_walls, rng)
    snapshot = add_awgn(snapshot, snr_db, rng)
    return snapshot.astype(np.complex64)


def generate_multi_sample(
    rng: np.random.Generator,
    snr_db: float,
    n_walls: int,
    velocity_mps: float | None = None,
) -> dict[str, Any]:
    az1, az2 = _two_separated_azimuths(rng)
    type1, type2 = rng.choice(["drone", "bird"], size=2, replace=True)
    vel1 = velocity_mps if velocity_mps is not None else _sample_velocity(rng)
    vel2 = velocity_mps if velocity_mps is not None else _sample_velocity(rng)
    snapshot = build_multi_snapshot(
        az1,
        type1,
        az2,
        type2,
        snr_db,
        n_walls,
        rng,
        velocity_mps_1=vel1,
        velocity_mps_2=vel2,
    )

    return {
        "signal": snapshot,
        "angles_deg": np.array([az1, az2], dtype=np.float32),
        "target_types": np.array([type1, type2], dtype=object),
        "n_walls": n_walls,
    }


def build_multi_snapshot(
    azimuth_deg_1: float,
    type_1: str,
    azimuth_deg_2: float,
    type_2: str,
    snr_db: float,
    n_walls: int,
    rng: np.random.Generator,
    velocity_mps_1: float | None = None,
    velocity_mps_2: float | None = None,
) -> np.ndarray:
    if type_1 == "drone":
        snap1, _ = synth_drone_signal(azimuth_deg_1, snr_db=snr_db + 6.0, rng=rng, velocity_mps=velocity_mps_1)
    else:
        snap1, _ = synth_bird_signal(azimuth_deg_1, snr_db=snr_db + 6.0, rng=rng, velocity_mps=velocity_mps_1)

    if type_2 == "drone":
        snap2, _ = synth_drone_signal(azimuth_deg_2, snr_db=snr_db + 6.0, rng=rng, velocity_mps=velocity_mps_2)
    else:
        snap2, _ = synth_bird_signal(azimuth_deg_2, snr_db=snr_db + 6.0, rng=rng, velocity_mps=velocity_mps_2)

    snap1 = _apply_snapshot_multipath(snap1, azimuth_deg_1, n_walls, rng)
    snap2 = _apply_snapshot_multipath(snap2, azimuth_deg_2, n_walls, rng)
    snapshot = add_awgn(snap1 + snap2, snr_db, rng)
    return snapshot.astype(np.complex64)


def generate_sample(class_id: int, rng: np.random.Generator) -> dict[str, Any]:
    """Генерирует один сэмпл заданного класса."""
    snr_db = _sample_snr(rng)
    n_walls = sample_n_walls(rng)
    range_m = _sample_range(rng)
    velocity_mps = _sample_velocity(rng)

    if class_id == CLUTTER:
        sample = generate_clutter_sample(rng, snr_db, n_walls)
    elif class_id == DRONE:
        sample = generate_drone_sample(rng, snr_db, n_walls, velocity_mps=velocity_mps)
    elif class_id == BIRD:
        sample = generate_bird_sample(rng, snr_db, n_walls, velocity_mps=velocity_mps)
    elif class_id == MULTI:
        sample = generate_multi_sample(rng, snr_db, n_walls, velocity_mps=velocity_mps)
    else:
        raise ValueError(f"Unknown class_id: {class_id}")

    sample["label"] = class_id
    sample["snr_db"] = snr_db
    sample["n_walls"] = sample.get("n_walls", n_walls)
    sample["range_m"] = range_m
    sample["velocity_mps"] = velocity_mps
    return sample


def generate_dataset(
    n_samples: int,
    output_path: str,
    seed: int = 42,
    class_balance: bool = True,
) -> str:
    """
    Генерирует NPZ с ключами:
        signals      (N, n_ant) complex64
        labels       (N,) int32
        angles_deg   (N, 2) float32
        snr_db       (N,) float32
        n_walls      (N,) int32
        range_m      (N,) float32
        velocity_mps (N,) float32
        target_types (N, 2) str
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if class_balance:
        per_class = n_samples // 4
        class_ids = np.array([CLUTTER, DRONE, BIRD, MULTI] * per_class, dtype=np.int32)
        remainder = n_samples - len(class_ids)
        if remainder > 0:
            extra = rng.choice([CLUTTER, DRONE, BIRD, MULTI], size=remainder, replace=True)
            class_ids = np.concatenate([class_ids, extra.astype(np.int32)])
        rng.shuffle(class_ids)
    else:
        class_ids = rng.integers(0, 4, size=n_samples, dtype=np.int32)

    signals = np.zeros((n_samples, N_ANT), dtype=np.complex64)
    labels = np.zeros(n_samples, dtype=np.int32)
    angles_deg = np.full((n_samples, 2), np.nan, dtype=np.float32)
    snr_db_arr = np.zeros(n_samples, dtype=np.float32)
    n_walls_arr = np.zeros(n_samples, dtype=np.int32)
    range_m_arr = np.zeros(n_samples, dtype=np.float32)
    velocity_mps_arr = np.zeros(n_samples, dtype=np.float32)
    target_types = np.empty((n_samples, 2), dtype=object)

    for i in tqdm(range(n_samples), desc="Generating counter-UAV synth"):
        sample = generate_sample(int(class_ids[i]), rng)
        signals[i] = sample["signal"]
        labels[i] = sample["label"]
        angles_deg[i] = sample["angles_deg"]
        snr_db_arr[i] = sample["snr_db"]
        n_walls_arr[i] = sample["n_walls"]
        range_m_arr[i] = sample["range_m"]
        velocity_mps_arr[i] = sample["velocity_mps"]
        target_types[i] = sample["target_types"]

    np.savez_compressed(
        output_path,
        signals=signals,
        labels=labels,
        angles_deg=angles_deg,
        snr_db=snr_db_arr,
        n_walls=n_walls_arr,
        range_m=range_m_arr,
        velocity_mps=velocity_mps_arr,
        target_types=target_types,
        meta=np.array(
            {
                "domain": "counter_uav",
                "n_samples": n_samples,
                "seed": seed,
                "class_names": ["clutter", "drone", "bird", "multi"],
            },
            dtype=object,
        ),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate counter-UAV synthetic NPZ dataset")
    parser.add_argument("--n-samples", type=int, default=N_SYNTH_SMALL)
    parser.add_argument("--output", type=str, default=SMALL_SYNTH_NPZ)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Generate full {N_SYNTH_DEFAULT}-sample dataset to {DEFAULT_SYNTH_NPZ}",
    )
    args = parser.parse_args()

    if args.full:
        args.n_samples = N_SYNTH_DEFAULT
        args.output = DEFAULT_SYNTH_NPZ

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = generate_dataset(args.n_samples, args.output, seed=args.seed)
    print(f"Saved {args.n_samples} samples to {path}")


if __name__ == "__main__":
    main()
