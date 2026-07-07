from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domains.counter_uav import pipeline


def test_classifier_only_retrain_preserves_existing_doa_metrics(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "classifier": {"accuracy": 0.1},
                "doa": {
                    "siamese": {"rmse_deg": 4.55},
                    "lista": {"rmse_deg": 63.87},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "METRICS_JSON", str(metrics_path))
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        pipeline,
        "load_synth_data",
        lambda npz_path=None: (
            np.zeros((8, 8), dtype=np.complex64),
            np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32),
            np.zeros((8, 2), dtype=np.float32),
            np.zeros(8, dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_classifier",
        lambda signals, labels, n_walls, epochs=30: (
            object(),
            {
                "accuracy": 0.52,
                "report": {"accuracy": 0.51, "macro avg": {"f1-score": 0.50}},
                "confusion_matrix": [[1, 0], [0, 1]],
            },
        ),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("DOA retraining must be skipped in classifier_only mode")

    monkeypatch.setattr(pipeline, "train_doa_siamese", _unexpected)
    monkeypatch.setattr(pipeline, "train_doa_lista", _unexpected)
    monkeypatch.setattr(pipeline, "evaluate_doa", _unexpected)

    pipeline.main(quick=True, classifier_only=True)

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["classifier"]["accuracy"] == 0.52
    assert saved["doa"]["siamese"]["rmse_deg"] == 4.55
    assert saved["doa"]["lista"]["rmse_deg"] == 63.87
