"""
domains/counter_uav/classes.py
==============================
Метки классов и константы сцены counter-UAV (SPEC §7.5).
"""

from __future__ import annotations

CLUTTER = 0
DRONE = 1
BIRD = 2
MULTI = 3

CLASS_NAMES: tuple[str, ...] = ("clutter", "drone", "bird", "multi")
CLASS_TO_NAME: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}
NAME_TO_CLASS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

DOA_CLASS_IDS: tuple[int, ...] = (DRONE, BIRD, MULTI)
DOA_CLASS_NAMES: tuple[str, ...] = tuple(CLASS_NAMES[i] for i in DOA_CLASS_IDS)

TARGET_TYPES: tuple[str, ...] = ("drone", "bird")
N_WALLS_CHOICES: tuple[int, ...] = (0, 2, 4, 6)

N_TARGETS_BY_CLASS: dict[int, tuple[int, int]] = {
    CLUTTER: (0, 0),
    DRONE: (1, 1),
    BIRD: (1, 1),
    MULTI: (2, 2),
}
