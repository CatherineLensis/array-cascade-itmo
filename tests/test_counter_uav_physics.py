"""
tests/test_counter_uav_physics.py
=================================
Юнит-тесты физического слоя counter-UAV (AC-2).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core.classifier import canonicalize_counter_uav_snapshot
import domains.counter_uav.dataset_synth as dataset_synth
from domains.counter_uav.classes import BIRD, CLUTTER, DRONE, MULTI
from domains.counter_uav.config import DOA_ANGLE_LIMIT_DEG, N_ANT, WAVELENGTH_M
from domains.counter_uav.dataset_synth import generate_dataset, generate_sample
from domains.counter_uav.micro_doppler import synth_bird_signal, synth_drone_signal
from domains.counter_uav.physics_xband import (
    build_steering_grid,
    estimate_doa_beamforming,
    rayleigh_limit,
    steering_matrix_xband,
    steering_vector,
    synth_single_target_snapshot,
)
from domains.counter_uav.urban_multipath import build_multipath_components, sample_n_walls
from ui.services.presets import get_signal


def _phase_invariant_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    y = np.asarray(y, dtype=np.complex64).reshape(-1)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = y / (np.linalg.norm(y) + 1e-12)
    phase = np.vdot(y, x)
    x_aligned = x * np.exp(-1j * np.angle(phase))
    return float(np.linalg.norm(x_aligned - y))


class TestPhysicsXband:
    def test_steering_matrix_shape(self):
        angles = np.array([-30.0, 0.0, 30.0])
        A = steering_matrix_xband(angles)
        assert A.shape == (N_ANT, 3)
        assert A.dtype == np.complex64

    def test_steering_vector_unit_phase_origin(self):
        a0 = steering_vector(0.0)
        np.testing.assert_allclose(a0, np.ones(N_ANT, dtype=np.complex64))

    def test_rayleigh_limit_positive(self):
        limit = rayleigh_limit()
        assert limit > 0.0
        assert limit < 90.0

    def test_beamforming_single_target_error_le_2_deg(self):
        """AC-2: classical beamforming on single target gives error <= 2 degrees."""
        rng = np.random.default_rng(123)
        A, grid_deg = build_steering_grid()
        errors = []

        for _ in range(64):
            az_true = rng.uniform(-DOA_ANGLE_LIMIT_DEG + 5.0, DOA_ANGLE_LIMIT_DEG - 5.0)
            x = synth_single_target_snapshot(az_true, snr_db=25.0, rng=rng)
            az_est = estimate_doa_beamforming(x, A=A, grid_deg=grid_deg)
            errors.append(abs(az_est - az_true))

        errors = np.array(errors)
        assert np.max(errors) <= 2.0, f"max error {np.max(errors):.3f}° > 2°"
        assert np.mean(errors) <= 1.0, f"mean error {np.mean(errors):.3f}° > 1°"

    def test_counter_uav_canonicalization_reduces_angle_variance(self):
        x1 = synth_single_target_snapshot(15.0, snr_db=60.0, rng=np.random.default_rng(1))
        x2 = synth_single_target_snapshot(-25.0, snr_db=60.0, rng=np.random.default_rng(2))

        c1 = canonicalize_counter_uav_snapshot(x1)
        c2 = canonicalize_counter_uav_snapshot(x2)

        assert _phase_invariant_distance(c1, c2) < 0.2


class TestMicroDoppler:
    def test_drone_signal_shape(self):
        snap, series = synth_drone_signal(15.0, snr_db=20.0, rng=np.random.default_rng(0))
        assert snap.shape == (N_ANT,)
        assert series.ndim == 1
        assert np.isfinite(snap).all()

    def test_bird_signal_shape(self):
        snap, series = synth_bird_signal(-20.0, snr_db=18.0, rng=np.random.default_rng(1))
        assert snap.shape == (N_ANT,)
        assert series.ndim == 1
        assert np.isfinite(snap).all()

    def test_drone_and_bird_snapshots_are_not_phase_degenerate(self):
        drone_snap, _ = synth_drone_signal(
            20.0,
            snr_db=60.0,
            rng=np.random.default_rng(10),
            velocity_mps=12.0,
        )
        bird_snap, _ = synth_bird_signal(
            20.0,
            snr_db=60.0,
            rng=np.random.default_rng(20),
            velocity_mps=12.0,
            f_wb=5.0,
            a_wing_m=0.2,
        )

        assert _phase_invariant_distance(drone_snap, bird_snap) > 0.05

    def test_drone_and_bird_clean_snapshots_are_linearly_separable(self):
        X = []
        y = []
        for i in range(96):
            drone_snap, _ = synth_drone_signal(
                20.0,
                snr_db=25.0,
                rng=np.random.default_rng(i),
                velocity_mps=12.0,
            )
            bird_snap, _ = synth_bird_signal(
                20.0,
                snr_db=25.0,
                rng=np.random.default_rng(1000 + i),
                velocity_mps=12.0,
                f_wb=5.0,
                a_wing_m=0.2,
            )
            X.extend([drone_snap, bird_snap])
            y.extend([0, 1])

        X = np.asarray(X, dtype=np.complex64)
        y = np.asarray(y, dtype=np.int32)
        X = X * np.exp(-1j * np.angle(X[:, :1]))
        feats = np.concatenate([X.real, X.imag], axis=1)

        tr, te = train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        clf = LogisticRegression(max_iter=2000)
        clf.fit(feats[tr], y[tr])

        assert clf.score(feats[te], y[te]) >= 0.70


class TestUrbanMultipath:
    def test_n_walls_choices(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            assert sample_n_walls(rng) in (0, 2, 4, 6)

    def test_multipath_component_count(self):
        rng = np.random.default_rng(8)
        comps = build_multipath_components(10.0, n_walls=4, rng=rng)
        assert len(comps) == 5
        assert comps[0].is_direct


class TestDatasetSynth:
    def test_generate_sample_all_classes(self):
        rng = np.random.default_rng(99)
        for cls in (CLUTTER, DRONE, BIRD, MULTI):
            sample = generate_sample(cls, rng)
            assert sample["signal"].shape == (N_ANT,)
            assert sample["label"] == cls

    def test_generate_small_npz(self, tmp_path):
        out = tmp_path / "test_synth.npz"
        path = generate_dataset(n_samples=16, output_path=str(out), seed=0)
        data = np.load(path, allow_pickle=True)
        assert data["signals"].shape == (16, N_ANT)
        assert set(np.unique(data["labels"])) <= {0, 1, 2, 3}

    def test_demo_preset_u1_uses_train_aligned_drone_path(self, monkeypatch):
        monkeypatch.setattr(dataset_synth, "_sample_azimuth", lambda rng: 30.0)
        rng = np.random.default_rng(123)
        expected = dataset_synth.generate_drone_sample(rng, 15.0, n_walls=4, velocity_mps=12.0)["signal"]
        actual = get_signal("u1", 15.0)
        np.testing.assert_allclose(actual, expected)

    def test_demo_preset_u2_uses_train_aligned_bird_path(self, monkeypatch):
        monkeypatch.setattr(dataset_synth, "_sample_azimuth", lambda rng: -20.0)
        rng = np.random.default_rng(60)
        expected = dataset_synth.generate_bird_sample(rng, 15.0, n_walls=4, velocity_mps=12.0)["signal"]
        actual = get_signal("u2", 15.0)
        np.testing.assert_allclose(actual, expected)

    def test_multi_sample_depends_on_n_walls(self):
        rng0 = np.random.default_rng(777)
        rng4 = np.random.default_rng(777)
        s0 = dataset_synth.generate_multi_sample(rng0, 25.0, n_walls=0, velocity_mps=12.0)["signal"]
        s4 = dataset_synth.generate_multi_sample(rng4, 25.0, n_walls=4, velocity_mps=12.0)["signal"]
        assert _phase_invariant_distance(s0, s4) > 0.01


class TestConfig:
    def test_wavelength_xband(self):
        assert abs(WAVELENGTH_M - 0.024) < 0.001
