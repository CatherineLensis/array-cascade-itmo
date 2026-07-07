"""Train counter-UAV cascade on synthetic data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domains.counter_uav.pipeline import main

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None)
    p.add_argument("--quick", action="store_true", help="Fast training for demo")
    p.add_argument("--full", action="store_true", help="Full 10k dataset")
    p.add_argument(
        "--classifier-only",
        action="store_true",
        help="Retrain classifier and keep existing DOA checkpoints/metrics intact",
    )
    args = p.parse_args()
    npz = args.npz
    if args.full and npz is None:
        npz = str(ROOT / "output_counter_uav" / "synth_10k.npz")
    main(npz, quick=not args.full, classifier_only=args.classifier_only)
