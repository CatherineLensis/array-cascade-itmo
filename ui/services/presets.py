"""
ui/services/presets.py — built-in preset signals for /demo.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import dataset_gen_final_experiment
from domains.counter_uav.dataset_synth import build_bird_snapshot, build_drone_snapshot, build_multi_snapshot
from domains.counter_uav.classes import DRONE, BIRD, MULTI


@dataclass(frozen=True)
class Preset:
    id: str
    domain: str
    title: str
    description: str
    true_angle_deg: float | None
    n_targets: int
    scene_class: int | None = None


PRESETS: dict[str, Preset] = {
    "a1": Preset("a1", "automotive", "A1 — single direct target",
                  "Один прямой сигнал, θ=+15°, SNR=20dB. Класс 0 (direct).", 15.0, 1, 0),
    "a2": Preset("a2", "automotive", "A2 — multipath ghost",
                  "Прямой сигнал θ=-10° + переотражение. Класс 1 (multipath).", -10.0, 1, 1),
    "a3": Preset("a3", "automotive", "A3 — two targets",
                  "Две цели θ₁=-25°, θ₂=+40°. Класс 2 (two_targets).", None, 2, 2),
    "u1": Preset("u1", "counter_uav", "U1 — quadcopter (DJI Mavic class)",
                  "Дрон θ=+30°, 4 мотора, urban multipath 4 walls.", 30.0, 1, DRONE),
    "u2": Preset("u2", "counter_uav", "U2 — bird (wing-beat 5 Hz)",
                  "Птица θ=-20°, wing-beat 5 Hz.", -20.0, 1, BIRD),
    "u3": Preset("u3", "counter_uav", "U3 — drone + bird (multi)",
                  "Дрон θ=+10° и птица θ=-35° одновременно.", None, 2, MULTI),
}


def get_preset(preset_id: str) -> Preset:
    if preset_id not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_id}")
    return PRESETS[preset_id]


def list_presets(domain: str | None = None) -> list[Preset]:
    if domain is None:
        return list(PRESETS.values())
    return [p for p in PRESETS.values() if p.domain == domain]


_AUTOMOTIVE_DEMO_SPECS = {
    "a1": {"seed": 1348, "sample_idx": 0, "true_angle_deg": 14.963080007889024},
    "a2": {"seed": 271, "sample_idx": 2, "true_angle_deg": None},
    "a3": {"seed": 709, "sample_idx": 1, "true_angle_deg": None},
}


def _build_automotive_demo_bundle(preset: Preset, snr_db: float) -> tuple[np.ndarray, float | None] | None:
    spec = _AUTOMOTIVE_DEMO_SPECS.get(preset.id)
    if spec is None:
        return None
    np.random.seed(spec["seed"])
    signals, _, _, _ = dataset_gen_final_experiment(1, 0)
    signal = np.asarray(signals[spec["sample_idx"]]).reshape(-1).astype(np.complex64)
    return signal, spec["true_angle_deg"]


def get_signal(preset_id: str, snr_db: float | None = None) -> np.ndarray:
    preset = get_preset(preset_id)
    snr = snr_db if snr_db is not None else 20.0

    if preset.domain == "automotive":
        demo_bundle = _build_automotive_demo_bundle(preset, snr)
        if demo_bundle is not None:
            return demo_bundle[0]
        raise KeyError(f"Unsupported automotive preset: {preset.id}")

    if preset.id == "u1":
        return build_drone_snapshot(
            preset.true_angle_deg or 30.0,
            snr_db=snr,
            n_walls=4,
            rng=np.random.default_rng(123),
            velocity_mps=12.0,
        ).reshape(-1)
    if preset.id == "u2":
        return build_bird_snapshot(
            preset.true_angle_deg or -20.0,
            snr_db=snr,
            n_walls=4,
            rng=np.random.default_rng(60),
            velocity_mps=12.0,
        ).reshape(-1)
    if preset.id == "u3":
        return build_multi_snapshot(
            10.0,
            "drone",
            -35.0,
            "bird",
            snr_db=snr,
            n_walls=4,
            rng=np.random.default_rng(789),
            velocity_mps_1=12.0,
            velocity_mps_2=12.0,
        ).reshape(-1)
    raise KeyError(f"Unsupported preset: {preset.id}")


def get_signal_bundle(preset_id: str, snr_db: float | None = None) -> tuple[np.ndarray, float | None]:
    preset = get_preset(preset_id)
    snr = snr_db if snr_db is not None else 20.0

    if preset.domain == "automotive":
        demo_bundle = _build_automotive_demo_bundle(preset, snr)
        if demo_bundle is not None:
            return demo_bundle
        raise KeyError(f"Unsupported automotive preset: {preset.id}")

    return get_signal(preset_id, snr_db=snr), preset.true_angle_deg
