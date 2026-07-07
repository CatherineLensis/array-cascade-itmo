"""
ui/services/inference.py — load models and run cascade for UI demo.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cascade import Cascade, CascadeResult, DomainSpec
from ui.services.plots import (
    build_classifier_bars,
    build_doa_spectra,
    build_signal_plot,
)
from ui.services.presets import PRESETS, get_preset, get_signal, get_signal_bundle


@dataclass
class CascadeRunResult:
    domain: str
    preset_id: str
    mode: str = "live"

    raw_signal_fig: dict[str, Any] = field(default_factory=dict)
    whitened_signal_fig: dict[str, Any] = field(default_factory=dict)

    classifier_logits: list[float] = field(default_factory=list)
    classifier_labels: list[str] = field(default_factory=list)
    classifier_label: str = ""

    doa_spectra: dict[str, list[float]] = field(default_factory=dict)
    doa_predictions: dict[str, float] = field(default_factory=dict)
    doa_true_angle: float | None = None

    metrics: dict[str, float] = field(default_factory=dict)
    angle_grid_deg: list[float] = field(default_factory=list)


_CASCADES: dict[str, Cascade | None] = {"automotive": None, "counter_uav": None}
_MODE = "auto"
_DEMO_DOA_POLICY = {
    "automotive": {"doa_method": "siamese", "run_all_doa": True},
    # Counter-UAV LISTA is currently too unstable for jury-facing live demo.
    "counter_uav": {"doa_method": "siamese", "run_all_doa": False},
}


def _cascade_for(domain: str) -> Cascade:
    global _CASCADES
    if _CASCADES.get(domain) is None:
        spec = DomainSpec.automotive() if domain == "automotive" else DomainSpec.counter_uav()
        _CASCADES[domain] = Cascade(spec)
    return _CASCADES[domain]  # type: ignore[return-value]


def _demo_policy(domain: str) -> dict[str, Any]:
    return _DEMO_DOA_POLICY.get(domain, {"doa_method": "siamese", "run_all_doa": True})


def _warmup_cascade(domain: str, preset_id: str) -> bool:
    """Run a single dry forward pass to reduce cold-start hiccups in live demo."""
    cascade = _cascade_for(domain)
    if cascade.classifier is None or len(cascade.doa_models) == 0:
        return False

    signal = np.asarray(get_signal(preset_id, snr_db=15)).reshape(-1)
    policy = _demo_policy(domain)
    cascade.forward(
        signal,
        doa_method=policy["doa_method"],
        run_all_doa=policy["run_all_doa"],
    )
    return True


def load_models(mode: str = "auto") -> dict[str, Any]:
    """Load cascade instances once at startup."""
    global _MODE
    _MODE = mode
    auto = _cascade_for("automotive")
    uav = _cascade_for("counter_uav")

    auto_ready = auto.classifier is not None and len(auto.doa_models) > 0
    uav_ready = uav.classifier is not None and len(uav.doa_models) > 0
    auto_warm = False
    uav_warm = False
    if auto_ready:
        try:
            auto_warm = _warmup_cascade("automotive", "a1")
        except Exception:
            auto_warm = False
    if uav_ready:
        try:
            uav_warm = _warmup_cascade("counter_uav", "u1")
        except Exception:
            uav_warm = False

    return {
        "status": "loaded",
        "mode": mode,
        "automotive": {
            "ready": auto_ready,
            "classifier": auto.classifier is not None,
            "doa_methods": list(auto.doa_models.keys()),
            "demo_methods": ["siamese", "lista", "covrecon", "hiergru"],
            "warmup_ok": auto_warm,
        },
        "counter_uav": {
            "ready": uav_ready,
            "classifier": uav.classifier is not None,
            "doa_methods": list(uav.doa_models.keys()),
            "demo_methods": ["siamese"],
            "warmup_ok": uav_warm,
        },
    }


def _physics_fallback(signal: np.ndarray, domain: str, preset_id: str) -> CascadeRunResult:
    """Beamforming fallback when ML models unavailable."""
    from domains.counter_uav.physics_xband import (
        build_steering_grid,
        classical_beamforming_spectrum,
        estimate_doa_beamforming,
    )

    preset = get_preset(preset_id)
    spec = DomainSpec.automotive() if domain == "automotive" else DomainSpec.counter_uav()
    A, grid_deg = build_steering_grid()
    spectrum = classical_beamforming_spectrum(signal, A)
    angle = estimate_doa_beamforming(signal, A=A, grid_deg=grid_deg)

    logits = np.zeros(len(spec.classes), dtype=float)
    logits[1 if domain == "automotive" else 1] = 2.0
    label = spec.classes[int(np.argmax(logits))]

    result = CascadeRunResult(
        domain=domain,
        preset_id=preset_id,
        mode="physics_fallback",
        classifier_logits=logits.tolist(),
        classifier_labels=spec.classes,
        classifier_label=label,
        doa_spectra={"beamforming": spectrum.tolist()},
        doa_predictions={"beamforming": float(angle)},
        doa_true_angle=preset.true_angle_deg,
        angle_grid_deg=grid_deg.tolist(),
        metrics={"latency_ms": 1.0, "mode": 0.0},
    )
    result.raw_signal_fig = build_signal_plot(signal, "Raw signal")
    result.whitened_signal_fig = build_signal_plot(signal, "Signal (fallback)")
    return result


def run_cascade(
    signal: np.ndarray | None = None,
    domain: str = "automotive",
    preset_id: str = "a1",
    snr: float | None = None,
) -> CascadeRunResult:
    preset = get_preset(preset_id)
    true_angle_deg = preset.true_angle_deg
    if signal is None:
        signal, true_angle_deg = get_signal_bundle(preset_id, snr_db=snr)

    cascade = _cascade_for(domain)
    use_ml = cascade.classifier is not None and len(cascade.doa_models) > 0

    if not use_ml and _MODE != "live":
        return _physics_fallback(signal, domain, preset_id)

    if not use_ml:
        return _physics_fallback(signal, domain, preset_id)

    raw = np.asarray(signal).reshape(-1)
    policy = _demo_policy(domain)
    cr: CascadeResult = cascade.forward(
        raw,
        doa_method=policy["doa_method"],
        run_all_doa=policy["run_all_doa"],
    )
    model_input = cascade.preprocessor(raw)

    spec = cascade.domain
    doa_preds = {}
    doa_spectra = {}
    if cr.doa_spectra:
        for name, spec_arr in cr.doa_spectra.items():
            doa_spectra[name] = spec_arr.tolist()
            peak = int(np.argmax(spec_arr))
            doa_preds[name] = float(cascade.grid_deg[peak])

    error_deg = np.nan
    if true_angle_deg is not None and doa_preds:
        best = min(doa_preds.values(), key=lambda p: abs(p - true_angle_deg))
        error_deg = abs(best - true_angle_deg)

    result = CascadeRunResult(
        domain=domain,
        preset_id=preset_id,
        mode="live",
        classifier_logits=cr.scene_logits.tolist(),
        classifier_labels=spec.classes,
        classifier_label=cr.scene_label,
        doa_spectra=doa_spectra,
        doa_predictions=doa_preds,
        doa_true_angle=true_angle_deg,
        angle_grid_deg=cascade.grid_deg.tolist(),
        metrics={
            "latency_ms": cr.latency_ms,
            "error_deg": float(error_deg) if not np.isnan(error_deg) else -1.0,
        },
    )
    result.raw_signal_fig = build_signal_plot(raw, "Raw signal")
    input_title = (
        "Model input (raw, counter-UAV)"
        if cascade.domain.preprocessing == "raw"
        else "After Wiener-Hopf + RMS"
    )
    result.whitened_signal_fig = build_signal_plot(model_input, input_title)
    return result


def run_both(preset_auto: str = "a1", preset_uav: str = "u1") -> dict[str, Any]:
    r1 = run_cascade(domain="automotive", preset_id=preset_auto)
    r2 = run_cascade(domain="counter_uav", preset_id=preset_uav)
    return {"automotive": asdict(r1), "counter_uav": asdict(r2)}


def result_to_dict(result: CascadeRunResult) -> dict[str, Any]:
    d = asdict(result)
    d["classifier_fig"] = build_classifier_bars(result.classifier_logits, result.classifier_labels)
    if result.doa_spectra and result.angle_grid_deg:
        d["doa_fig"] = build_doa_spectra(
            result.doa_spectra,
            np.array(result.angle_grid_deg),
            result.doa_true_angle,
        )
    return d
