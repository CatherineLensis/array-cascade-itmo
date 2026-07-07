"""Automotive domain configuration (TI AWR1843, 77 GHz)."""

from config import (
    CLASSIFIER_PATH,
    DOA_COVRECON_PATH,
    DOA_HIERGRU_PATH,
    DOA_LISTA_PATH,
    DOA_SIAMESE_PATH,
    MIMO_POS,
    N_ANT,
    N_GRID,
    OUTPUT_DIR,
    REAL_NPZ,
    SYNTH_NPZ,
)

DOMAIN_NAME = "automotive"
WAVELENGTH_M = 0.0039
CLASSES = ["direct", "multipath", "two_targets"]
DOA_CLASSES = ["direct", "two_targets"]

__all__ = [
    "DOMAIN_NAME",
    "WAVELENGTH_M",
    "CLASSES",
    "DOA_CLASSES",
    "N_ANT",
    "N_GRID",
    "MIMO_POS",
    "OUTPUT_DIR",
    "SYNTH_NPZ",
    "REAL_NPZ",
    "CLASSIFIER_PATH",
    "DOA_SIAMESE_PATH",
    "DOA_LISTA_PATH",
    "DOA_COVRECON_PATH",
    "DOA_HIERGRU_PATH",
]
