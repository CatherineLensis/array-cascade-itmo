"""UI route smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))

from app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/health"])
def test_routes_200(client, path):
    resp = client.get(path)
    assert resp.status_code == 200


def test_html_pages_smoke(client):
    """Smoke GET for main pages."""
    for path in ["/", "/demo", "/methodology", "/counter_uav", "/roadmap", "/about"]:
        resp = client.get(path)
        assert resp.status_code == 200, path


def test_api_run_cascade(client):
    resp = client.post("/api/run_cascade", json={"domain": "counter_uav", "preset": "u1", "snr": 15})
    assert resp.status_code == 200
    data = resp.json()
    assert "classifier_label" in data
    assert "raw_signal_fig" in data


def test_api_run_cascade_u2_classifies_bird(client):
    resp = client.post("/api/run_cascade", json={"domain": "counter_uav", "preset": "u2", "snr": 15})
    assert resp.status_code == 200
    data = resp.json()
    assert data["classifier_label"] == "bird"


def test_api_run_cascade_a3_is_two_targets_without_single_angle_error(client):
    resp = client.post("/api/run_cascade", json={"domain": "automotive", "preset": "a3", "snr": 8})
    assert resp.status_code == 200
    data = resp.json()
    assert data["classifier_label"] == "two_targets"
    assert data["doa_true_angle"] is None
    assert data["metrics"]["error_deg"] == -1.0


def test_automotive_a1_preset_uses_stable_demo_angle(client):
    resp = client.post("/api/run_cascade", json={"domain": "automotive", "preset": "a1", "snr": 15})
    assert resp.status_code == 200
    data = resp.json()
    assert data["doa_true_angle"] == pytest.approx(15.0, abs=0.1)


def test_automotive_a2_preset_is_multipath_without_single_angle_error(client):
    resp = client.post("/api/run_cascade", json={"domain": "automotive", "preset": "a2", "snr": 15})
    assert resp.status_code == 200
    data = resp.json()
    assert data["classifier_label"] == "multipath"
    assert data["doa_true_angle"] is None
    assert data["metrics"]["error_deg"] == -1.0


def test_about_page_hides_application_text_block(client):
    resp = client.get("/about")
    assert resp.status_code == 200
    assert "Текст заявки" not in resp.text


def test_counter_uav_page_shows_temporal_recovery_result(client):
    """Design A: counter-UAV page presents the corrected temporal-recovery
    classification result. Snapshot-only baseline numbers and developer-internal
    labels are hidden from jury-facing copy."""
    resp = client.get("/counter_uav")
    assert resp.status_code == 200
    assert "best checkpoint" not in resp.text
    assert "Различение дрона и птицы" in resp.text
    assert "Полная четырёхклассовая задача" in resp.text
    # internal labels must not leak
    assert "Live smoke" not in resp.text
    assert "Phase B" not in resp.text and "Phase C" not in resp.text
    assert "DomainSpec" not in resp.text
