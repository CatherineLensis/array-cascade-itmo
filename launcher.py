"""
launcher.py
===========
Unified CLI launcher for two methodologies:
  1) whitening + RMS (new)
  2) max-normalize (old)

Examples:
  python launcher.py --mode new
  python launcher.py --mode old
  python launcher.py --mode real-only
  python launcher.py --mode eval-for-paper
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Launcher for pipeline variants (new/old/real-only/eval)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["new", "old", "real-only", "real-only-sh", "eval-for-paper", "hypo-synth"],
        help="Which pipeline to run",
    )
    parser.add_argument(
        "--for-paper",
        default=os.path.join("for_paper", "for_paper.txt"),
        help="Path to for_paper.txt (used by eval-for-paper)",
    )
    args = parser.parse_args()

    if args.mode == "new":
        # Whitening + RMS on synth + real
        run("python prepare_data.py")
        run("python run_full_pipeline.py")
    elif args.mode == "old":
        # Max-norm on synth + real
        run("python run_full_pipeline_sh.py")
    elif args.mode == "real-only":
        # Real-only training and evaluation
        run("python analysis/real_only/run_real_only.py")
    elif args.mode == "real-only-sh":
        # Real-only training and evaluation (old max-normalize)
        run("python analysis/real_only/run_real_only_sh.py")
    elif args.mode == "eval-for-paper":
        # Evaluate trained real-only weights on external txt
        run(f"python analysis/real_only/evaluate_for_paper.py --input-txt \"{args.for_paper}\"")
    elif args.mode == "hypo-synth":
        # Generate hypothesis-based synthetic data only
        run("python prepare_data_hypo.py")
    else:
        raise SystemExit("Unsupported mode")

    print("\nNotes:")
    print("  --mode new         : whitening + RMS (prepare_data.py + run_full_pipeline.py)")
    print("  --mode old         : max-normalize (run_full_pipeline_sh.py)")
    print("  --mode real-only   : train+eval on real only (analysis/real_only/run_real_only.py)")
    print("  --mode real-only-sh: train+eval on real only, max-norm (analysis/real_only/run_real_only_sh.py)")
    print("  --mode eval-for-paper : evaluate weights from output_real_data on for_paper.txt")
    print("  --mode hypo-synth  : generate hypothesis-based synth (prepare_data_hypo.py)")
    print("  --for-paper <path> : path to for_paper.txt for eval-for-paper")


if __name__ == "__main__":
    main()
