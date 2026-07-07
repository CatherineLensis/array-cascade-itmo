"""Evaluate live counter-UAV runtime presets after preprocessing-contract fixes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.services.inference import load_models, run_cascade


OUTPUT = ROOT / "output_counter_uav" / "runtime_smoke_metrics.json"


def main() -> int:
    load_models(mode="auto")
    results = {}
    for preset_id in ["u1", "u2", "u3"]:
        result = run_cascade(domain="counter_uav", preset_id=preset_id, snr=15)
        results[preset_id] = {
            "mode": result.mode,
            "classifier_label": result.classifier_label,
            "true_angle_deg": result.doa_true_angle,
            "doa_predictions": result.doa_predictions,
            "metrics": result.metrics,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
