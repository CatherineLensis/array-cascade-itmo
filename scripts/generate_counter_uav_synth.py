"""Generate counter-UAV synthetic dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domains.counter_uav.dataset_synth import main

if __name__ == "__main__":
    main()
