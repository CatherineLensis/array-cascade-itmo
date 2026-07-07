"""
domains/counter_uav/config.py
=============================
Конфигурация физической модели counter-UAV (SPEC §7.1–§7.2).
"""

from __future__ import annotations

import os

import numpy as np

from domains.counter_uav.classes import CLASS_NAMES

# ── Радар X-band ──────────────────────────────────────────────
FC_HZ = 12.5e9
C_LIGHT = 299_792_458.0
WAVELENGTH_M = C_LIGHT / FC_HZ  # ≈ 0.024 м

N_ANT = 8
ARRAY_POS = np.linspace(0.0, 3.5, N_ANT, dtype=np.float64)  # ULA, шаг λ/2

N_GRID = 361
DOA_ANGLE_LIMIT_DEG = 70.0
DOA_GRID_DEG = np.linspace(-DOA_ANGLE_LIMIT_DEG, DOA_ANGLE_LIMIT_DEG, N_GRID)

# ── Сцена ─────────────────────────────────────────────────────
RANGE_M = (50.0, 1000.0)
VELOCITY_MPS = (5.0, 25.0)
SNR_DB = (-5.0, 25.0)
CPI_S = 0.100
N_CHIRPS = 50
RANGE_RES_M = 0.5
DOPPLER_RES_MPS = 0.5
N_SNAPSHOTS = (1, 4)

# ── Пути ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_counter_uav")
DEFAULT_SYNTH_NPZ = os.path.join(OUTPUT_DIR, "synth_10k.npz")
SMALL_SYNTH_NPZ = os.path.join(OUTPUT_DIR, "synth_small.npz")
SYNTH_NPZ = DEFAULT_SYNTH_NPZ
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier.pth")
DOA_SIAMESE_PATH = os.path.join(OUTPUT_DIR, "doa_siamese.pth")
DOA_LISTA_PATH = os.path.join(OUTPUT_DIR, "doa_lista.pth")
DOA_COVRECON_PATH = os.path.join(OUTPUT_DIR, "doa_covrecon.pth")
DOA_HIERGRU_PATH = os.path.join(OUTPUT_DIR, "doa_hiergru.pth")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")
CLASSES = CLASS_NAMES_LIST = list(CLASS_NAMES)

# ── Датасет ───────────────────────────────────────────────────
N_SYNTH_DEFAULT = 10_000
N_SYNTH_SMALL = 256
