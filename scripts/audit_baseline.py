"""Update docs/BASELINE_CONFIRMED.md from local baseline artifacts."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "output"
OUT = ROOT / "docs" / "BASELINE_CONFIRMED.md"
SYNTH_METRICS = {
    "siamese": OUTPUT / "metrics_doa_siamese.json",
    "lista": OUTPUT / "metrics_doa_lista.json",
    "covrecon": OUTPUT / "metrics_doa_covrecon.json",
    "hiergru": OUTPUT / "metrics_doa_hiergru.json",
}
REAL_FILTERED = OUTPUT / "doa_filtered_metrics.json"


def main():
    missing = [str(path) for path in SYNTH_METRICS.values() if not path.exists()]
    if missing:
        print("Missing synth-val metric files — run scripts/resume_baseline.py first")
        for path in missing:
            print(f"  - {path}")
        return 1

    synth_summary = {}
    for method, path in SYNTH_METRICS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        rmse_values = data.get("val", {}).get("rmse", [])
        synth_summary[method] = {
            "rmse": rmse_values[-1] if rmse_values else None,
        }

    real_summary = {}
    if REAL_FILTERED.exists():
        real_summary = json.loads(REAL_FILTERED.read_text(encoding="utf-8"))

    lines = [
        "# BASELINE_CONFIRMED — automotive synth-val (local reproduction)",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Source:** `output/metrics_doa_*.json`",
        "",
        "## DOA metrics (synth-val holdout)",
        "",
        "| Method | RMSE |",
        "|---|---|",
    ]
    for method in ["siamese", "lista", "covrecon", "hiergru"]:
        rmse = synth_summary.get(method, {}).get("rmse")
        rmse_str = f"{rmse:.2f}°" if isinstance(rmse, (int, float)) else "—"
        lines.append(f"| {method} | {rmse_str} |")

    lines += [
        "",
        "## Real-data DOA domain gap (filtered ±70°)",
        "",
        "| Method | RMSE | MedAE | SR@3° |",
        "|---|---|---|---|",
    ]
    for method in ["siamese", "lista", "covrecon", "hiergru"]:
        m = real_summary.get(method, {})
        rmse = m.get("rmse")
        med = m.get("median_ae")
        sr3 = m.get("success_rate_3deg")
        lines.append(
            f"| {method} | "
            f"{rmse:.2f}°" if isinstance(rmse, (int, float)) else f"| {method} | —"
        )
        if isinstance(rmse, (int, float)):
            lines[-1] += f" | {med:.2f}°" if isinstance(med, (int, float)) else " | —"
            lines[-1] += f" | {sr3 * 100:.1f}%" if isinstance(sr3, (int, float)) else " | —"
            lines[-1] += " |"

    lines += [
        "",
        "## Classifier (real val)",
        "",
        "See `output/clf_report.txt` and `output/metrics_classifier.json`.",
        "",
        "## Reproduction command",
        "",
        "```bash",
        "python -X utf8 scripts/resume_baseline.py",
        "```",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
