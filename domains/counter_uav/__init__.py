"""
domains/counter_uav
=================
Counter-UAV физический слой: X-band ULA, micro-Doppler, urban multipath, synth dataset.

См. SPEC_ITMO_V2.md §7.
"""

from domains.counter_uav.classes import (
    BIRD,
    CLASS_NAMES,
    CLUTTER,
    DOA_CLASS_IDS,
    DOA_CLASS_NAMES,
    DRONE,
    MULTI,
    NAME_TO_CLASS,
    N_WALLS_CHOICES,
)
from domains.counter_uav.config import (
    ARRAY_POS,
    FC_HZ,
    N_ANT,
    OUTPUT_DIR,
    WAVELENGTH_M,
)
__all__ = [
    "ARRAY_POS",
    "BIRD",
    "CLASS_NAMES",
    "CLUTTER",
    "DOA_CLASS_IDS",
    "DOA_CLASS_NAMES",
    "DRONE",
    "FC_HZ",
    "MULTI",
    "NAME_TO_CLASS",
    "N_ANT",
    "N_WALLS_CHOICES",
    "OUTPUT_DIR",
    "WAVELENGTH_M",
]
