"""Smoke tests for universal Cascade interface."""

from __future__ import annotations

import numpy as np
import pytest

from core.cascade import Cascade, DomainSpec


def test_domain_specs_instantiate():
    auto = DomainSpec.automotive()
    uav = DomainSpec.counter_uav()
    assert auto.name == "automotive"
    assert uav.name == "counter_uav"
    assert len(auto.classes) == 3
    assert len(uav.classes) == 4


def test_cascade_automotive_instantiates():
    cascade = Cascade(DomainSpec.automotive())
    assert cascade.domain.name == "automotive"


def test_cascade_forward_runs_without_models():
    """Forward should not crash even if models missing (graceful degrade)."""
    cascade = Cascade(DomainSpec.automotive(), classifier=None, doa_models={})
    signal = np.exp(1j * np.linspace(0, 2 * np.pi, 8))
    result = cascade.forward(signal)
    assert result.scene_label in DomainSpec.automotive().classes
    assert result.latency_ms >= 0


def test_counter_uav_uses_raw_preprocessing_contract():
    spec = DomainSpec.counter_uav()
    cascade = Cascade(spec, classifier=None, doa_models={})
    signal = np.exp(1j * np.linspace(0, 2 * np.pi, 8))

    processed = cascade.preprocessor(signal)
    result = cascade.forward(signal)

    assert spec.preprocessing == "raw"
    assert np.allclose(processed, signal)
    assert result.metadata["preprocessing"] == "raw"


def test_counter_uav_uses_physical_angle_grid():
    spec = DomainSpec.counter_uav()
    cascade = Cascade(spec, classifier=None, doa_models={})

    assert cascade.grid_deg[0] == pytest.approx(-70.0)
    assert cascade.grid_deg[-1] == pytest.approx(70.0)
    assert len(cascade.grid_deg) == spec.n_grid
