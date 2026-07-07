# Counter-UAV B to C Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Последовательно реализовать `Phase B` и `Phase C` для `counter-UAV`, чтобы сначала честно измерить потолок snapshot-only classifier path, а затем отдельно построить temporal path к `>85%` accuracy.

**Architecture:** План разделён на две независимые ветки. `Phase B` добавляет versioned snapshot datasets, quality gates и benchmark ladder поверх текущего classifier contract. `Phase C` добавляет temporal dataset contract, отдельный temporal training pipeline и machine-readable benchmark `B vs C`, не ломая текущий snapshot runtime.

**Tech Stack:** Python 3.11, NumPy, PyTorch, pytest, JSON/NPZ artifacts, FastAPI regression smoke tests.

---

## File Map

- Modify: `domains/counter_uav/dataset_synth.py` — versioned snapshot dataset export, metadata-rich generation
- Create: `domains/counter_uav/quality_gates.py` — shared snapshot/temporal quality checks
- Modify: `domains/counter_uav/pipeline.py` — dataset manifest loading, richer metrics persistence, benchmark-friendly classifier training
- Modify: `analysis/counter_uav_accuracy_audit.py` — extend snapshot diagnostics into reusable benchmark helpers
- Create: `analysis/counter_uav_snapshot_benchmark.py` — run `B0..B4` ladder and persist summary JSON
- Create: `tests/test_counter_uav_quality_gates.py` — tests for dataset manifest and quality-gate behavior
- Modify: `tests/test_counter_uav_accuracy_audit.py` — broader B regressions and benchmark smoke assertions
- Modify: `tests/test_counter_uav_pipeline.py` — persistence / manifest / classifier-only checks
- Modify: `tests/ui/test_routes.py` — keep `u1/u2/u3` stable while B changes
- Modify: `domains/counter_uav/micro_doppler.py` — expose stable temporal artifacts for training
- Create: `domains/counter_uav/temporal_dataset.py` — build temporal dataset tensors and spectrogram features
- Create: `domains/counter_uav/temporal_models.py` — compact temporal baselines (`C0/C1/C2`)
- Create: `domains/counter_uav/temporal_pipeline.py` — train/eval temporal models, write metrics JSON
- Create: `analysis/counter_uav_temporal_benchmark.py` — compare `C0..C3` and emit `B vs C`
- Create: `tests/test_counter_uav_temporal_pipeline.py` — temporal contract, leakage, and benchmark tests
- Create: `output_counter_uav/datasets/.gitkeep` — target directory for versioned datasets
- Create: `output_counter_uav/benchmarks/.gitkeep` — target directory for benchmark outputs

---

### Task 1: Add Versioned Snapshot Datasets for Phase B

**Files:**
- Modify: `domains/counter_uav/dataset_synth.py`
- Modify: `domains/counter_uav/config.py`
- Test: `tests/test_counter_uav_quality_gates.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
import numpy as np

from domains.counter_uav.dataset_synth import generate_dataset


def test_generate_dataset_writes_manifest_and_quality_report(tmp_path):
    dataset_dir = tmp_path / "datasets" / "snapshot_v001"
    npz_path = dataset_dir / "signals.npz"

    generate_dataset(
        n_samples=32,
        output_path=str(npz_path),
        seed=7,
        class_balance=True,
    )

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((dataset_dir / "quality_report.json").read_text(encoding="utf-8"))
    npz = np.load(npz_path, allow_pickle=True)

    assert manifest["dataset_version"] == "snapshot_v001"
    assert manifest["n_samples"] == 32
    assert sorted(manifest["artifacts"].keys()) == ["npz", "quality_report"]
    assert quality["class_balance"]["counts"] == [8, 8, 8, 8]
    assert npz["signals"].shape == (32, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_counter_uav_quality_gates.py::test_generate_dataset_writes_manifest_and_quality_report -v`

Expected: FAIL because `generate_dataset()` currently writes only the NPZ and does not emit `manifest.json` / `quality_report.json`.

- [ ] **Step 3: Write minimal implementation**

```python
# domains/counter_uav/dataset_synth.py
from pathlib import Path
import json


def _dataset_version_from_output(output_path: str) -> str:
    parent = Path(output_path).resolve().parent
    return parent.name if parent.name else "snapshot_unversioned"


def _write_snapshot_dataset_metadata(
    output_path: str,
    seed: int,
    labels: np.ndarray,
    snr_db: np.ndarray,
    n_walls: np.ndarray,
    velocity_mps: np.ndarray,
) -> None:
    dataset_dir = Path(output_path).resolve().parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(labels.astype(np.int32), minlength=4).tolist()
    quality = {
        "class_balance": {"counts": counts},
        "snr_db": {"min": float(np.min(snr_db)), "max": float(np.max(snr_db))},
        "n_walls": {str(v): int(np.sum(n_walls == v)) for v in [0, 2, 4, 6]},
        "velocity_mps": {"min": float(np.min(velocity_mps)), "max": float(np.max(velocity_mps))},
    }
    manifest = {
        "dataset_version": _dataset_version_from_output(output_path),
        "seed": seed,
        "n_samples": int(len(labels)),
        "artifacts": {
            "npz": Path(output_path).name,
            "quality_report": "quality_report.json",
        },
    }
    (dataset_dir / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_counter_uav_quality_gates.py::test_generate_dataset_writes_manifest_and_quality_report -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/dataset_synth.py domains/counter_uav/config.py tests/test_counter_uav_quality_gates.py
git commit -m "feat: version counter-uav snapshot datasets"
```

---

### Task 2: Add Snapshot Quality Gates and Reusable Audit Helpers

**Files:**
- Create: `domains/counter_uav/quality_gates.py`
- Modify: `analysis/counter_uav_accuracy_audit.py`
- Modify: `tests/test_counter_uav_accuracy_audit.py`
- Test: `tests/test_counter_uav_quality_gates.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from domains.counter_uav.quality_gates import summarize_snapshot_dataset, validate_snapshot_quality_gates


def test_snapshot_quality_gates_fail_on_missing_strong_multipath_bucket():
    summary = {
        "class_balance": {"counts": [25, 25, 25, 25]},
        "n_walls": {"0": 25, "2": 75, "4": 0, "6": 0},
        "probe_metrics": {"clean_sep_lr_score": 0.95, "multipath_sep_lr_score": 0.40},
    }
    failures = validate_snapshot_quality_gates(summary)
    assert "strong_multipath_coverage" in failures
    assert "multipath_sep_lr_score" in failures


def test_snapshot_quality_summary_reports_probe_metrics():
    summary = summarize_snapshot_dataset("output_counter_uav/synth_10k.npz")
    assert "probe_metrics" in summary
    assert "clean_sep_lr_score" in summary["probe_metrics"]
    assert "multipath_sep_lr_score" in summary["probe_metrics"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter_uav_quality_gates.py -v`

Expected: FAIL because `quality_gates.py` does not exist and the current audit module does not expose reusable snapshot summary helpers.

- [ ] **Step 3: Write minimal implementation**

```python
# domains/counter_uav/quality_gates.py
from __future__ import annotations

import json
import numpy as np

from analysis.counter_uav_accuracy_audit import analyze_generator_separability


def summarize_snapshot_dataset(npz_path: str) -> dict:
    npz = np.load(npz_path, allow_pickle=True)
    labels = npz["labels"].astype(np.int32)
    n_walls = npz["n_walls"].astype(np.int32)
    summary = {
        "class_balance": {"counts": np.bincount(labels, minlength=4).tolist()},
        "n_walls": {str(v): int(np.sum(n_walls == v)) for v in [0, 2, 4, 6]},
        "probe_metrics": analyze_generator_separability(),
    }
    return summary


def validate_snapshot_quality_gates(summary: dict) -> list[str]:
    failures: list[str] = []
    if summary["n_walls"].get("4", 0) == 0 and summary["n_walls"].get("6", 0) == 0:
        failures.append("strong_multipath_coverage")
    if summary["probe_metrics"]["multipath_sep_lr_score"] < 0.70:
        failures.append("multipath_sep_lr_score")
    return failures
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter_uav_quality_gates.py tests/test_counter_uav_accuracy_audit.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/quality_gates.py analysis/counter_uav_accuracy_audit.py tests/test_counter_uav_quality_gates.py tests/test_counter_uav_accuracy_audit.py
git commit -m "test: add snapshot quality gates for counter-uav"
```

---

### Task 3: Persist Phase B Benchmarks and Stop-Condition Evidence

**Files:**
- Modify: `domains/counter_uav/pipeline.py`
- Create: `analysis/counter_uav_snapshot_benchmark.py`
- Modify: `tests/test_counter_uav_pipeline.py`
- Modify: `tests/ui/test_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json

from analysis.counter_uav_snapshot_benchmark import write_snapshot_benchmark_summary


def test_snapshot_benchmark_writes_machine_readable_summary(tmp_path):
    out = tmp_path / "snapshot_benchmark.json"
    write_snapshot_benchmark_summary(
        output_path=str(out),
        experiments=[
            {"name": "B0", "accuracy": 0.61, "macro_f1": 0.58},
            {"name": "B1", "accuracy": 0.64, "macro_f1": 0.61},
        ],
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["best_experiment"]["name"] == "B1"
    assert payload["experiments"][0]["name"] == "B0"


def test_classifier_only_retrain_preserves_snapshot_dataset_manifest(tmp_path, monkeypatch):
    import json
    import numpy as np
    from pathlib import Path
    from domains.counter_uav import pipeline

    dataset_dir = tmp_path / "datasets" / "snapshot_v001"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset_version": "snapshot_v001"}),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"doa": {"siamese": {"rmse_deg": 4.5}}}), encoding="utf-8")

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
            "snapshot_v001",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_classifier",
        lambda signals, labels, n_walls, epochs=30: (
            object(),
            {"accuracy": 0.62, "report": {"accuracy": 0.61, "macro avg": {"f1-score": 0.60}}, "confusion_matrix": [[1]]},
        ),
    )

    pipeline.main(npz_path=str(dataset_dir / "signals.npz"), quick=True, classifier_only=True)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["classifier"]["dataset_version"] == "snapshot_v001"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter_uav_pipeline.py -v`

Expected: FAIL because benchmark writer does not exist and `pipeline.py` does not persist dataset version info into `metrics.json`.

- [ ] **Step 3: Write minimal implementation**

```python
# analysis/counter_uav_snapshot_benchmark.py
from __future__ import annotations

import json


def write_snapshot_benchmark_summary(output_path: str, experiments: list[dict]) -> None:
    best = max(experiments, key=lambda item: item["accuracy"])
    payload = {"experiments": experiments, "best_experiment": best}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# domains/counter_uav/pipeline.py
def load_synth_data(npz_path: str | None = None):
    path = npz_path or SYNTH_NPZ
    if not Path(path).exists():
        from domains.counter_uav.dataset_synth import generate_dataset
        generate_dataset(n_samples=2048, output_path=path, seed=42)
    npz = np.load(path, allow_pickle=True)
    signals = npz["signals"]
    labels = npz["labels"].astype(int)
    angles = npz["angles_deg"]
    n_walls = npz["n_walls"].astype(int) if "n_walls" in npz else np.zeros(len(labels), dtype=np.int32)
    dataset_version = "snapshot_unversioned"
    manifest_path = Path(path).resolve().with_name("manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_version = manifest.get("dataset_version", dataset_version)
    return signals, labels, angles, n_walls, dataset_version


def main(npz_path: str | None = None, quick: bool = False, classifier_only: bool = False):
    signals, labels, angles_deg, n_walls, dataset_version = load_synth_data(npz_path)
    clf_epochs = 15 if quick else 30
    clf_model, clf_metrics = train_classifier(signals, labels, n_walls, epochs=clf_epochs)
    doa_metrics: dict[str, dict[str, float | None]] = {}
    if classifier_only and Path(METRICS_JSON).exists():
        previous = json.loads(Path(METRICS_JSON).read_text(encoding="utf-8"))
        doa_metrics = previous.get("doa", {})
    metrics = {
        "classifier": {**clf_metrics, "dataset_version": dataset_version},
        "doa": doa_metrics,
        "n_samples": int(len(signals)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter_uav_pipeline.py tests/ui/test_routes.py -v`

Expected: PASS, and existing UI route smoke remains green.

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/pipeline.py analysis/counter_uav_snapshot_benchmark.py tests/test_counter_uav_pipeline.py tests/ui/test_routes.py
git commit -m "feat: persist snapshot benchmark evidence for phase b"
```

---

### Task 4: Expose Temporal Artifacts and Build the Phase C Dataset Contract

**Files:**
- Modify: `domains/counter_uav/micro_doppler.py`
- Create: `domains/counter_uav/temporal_dataset.py`
- Test: `tests/test_counter_uav_temporal_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np

from domains.counter_uav.temporal_dataset import build_temporal_sample, split_temporal_indices


def test_build_temporal_sample_contains_snapshot_microseries_and_spectrogram():
    sample = build_temporal_sample(class_name="bird", azimuth_deg=10.0, snr_db=15.0, n_walls=4, seed=123)
    assert sample["snapshot"].shape == (8,)
    assert sample["micro_series"].ndim == 1
    assert sample["spectrogram"].ndim == 2
    assert sample["metadata"]["class_name"] == "bird"


def test_split_temporal_indices_keeps_scene_families_disjoint():
    scene_family_ids = np.array([0, 0, 1, 1, 2, 2])
    train_idx, val_idx = split_temporal_indices(scene_family_ids, val_fraction=0.33, seed=42)
    assert set(scene_family_ids[train_idx]).isdisjoint(set(scene_family_ids[val_idx]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter_uav_temporal_pipeline.py -v`

Expected: FAIL because `temporal_dataset.py` does not exist and `micro_doppler.py` does not yet provide a stable temporal dataset entrypoint.

- [ ] **Step 3: Write minimal implementation**

```python
# domains/counter_uav/temporal_dataset.py
from __future__ import annotations

import numpy as np

from domains.counter_uav.micro_doppler import micro_doppler_spectrogram, synth_bird_signal, synth_drone_signal


def build_temporal_sample(class_name: str, azimuth_deg: float, snr_db: float, n_walls: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if class_name == "bird":
        snapshot, micro_series = synth_bird_signal(azimuth_deg, snr_db=snr_db, rng=rng)
    else:
        snapshot, micro_series = synth_drone_signal(azimuth_deg, snr_db=snr_db, rng=rng)
    _, _, spec = micro_doppler_spectrogram(micro_series)
    return {
        "snapshot": snapshot.astype(np.complex64),
        "micro_series": micro_series.astype(np.complex64),
        "spectrogram": spec.astype(np.float32),
        "metadata": {"class_name": class_name, "n_walls": n_walls, "scene_family_id": int(seed)},
    }


def split_temporal_indices(scene_family_ids: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(scene_family_ids)
    rng.shuffle(unique_ids)
    n_val = max(1, int(round(len(unique_ids) * val_fraction)))
    val_ids = set(unique_ids[:n_val].tolist())
    train_idx = np.where(~np.isin(scene_family_ids, list(val_ids)))[0]
    val_idx = np.where(np.isin(scene_family_ids, list(val_ids)))[0]
    return train_idx, val_idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter_uav_temporal_pipeline.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/micro_doppler.py domains/counter_uav/temporal_dataset.py tests/test_counter_uav_temporal_pipeline.py
git commit -m "feat: add temporal dataset contract for counter-uav"
```

---

### Task 5: Train Phase C Temporal Baselines and Write a B vs C Benchmark

**Files:**
- Create: `domains/counter_uav/temporal_models.py`
- Create: `domains/counter_uav/temporal_pipeline.py`
- Create: `analysis/counter_uav_temporal_benchmark.py`
- Test: `tests/test_counter_uav_temporal_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json

from analysis.counter_uav_temporal_benchmark import write_b_vs_c_summary


def test_write_b_vs_c_summary_records_delta(tmp_path):
    out = tmp_path / "b_vs_c.json"
    write_b_vs_c_summary(
        output_path=str(out),
        snapshot_metrics={"accuracy": 0.64, "macro_f1": 0.62},
        temporal_metrics={"accuracy": 0.87, "macro_f1": 0.86},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["delta"]["accuracy"] == 0.23
    assert payload["temporal"]["accuracy"] == 0.87


def test_temporal_pipeline_returns_accuracy_and_macro_f1(monkeypatch):
    from domains.counter_uav.temporal_pipeline import run_temporal_baseline
    metrics = run_temporal_baseline(dataset_size=64, model_name="cnn_spec", seed=42)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter_uav_temporal_pipeline.py -v`

Expected: FAIL because temporal models, pipeline, and comparison writer do not yet exist.

- [ ] **Step 3: Write minimal implementation**

```python
# domains/counter_uav/temporal_models.py
from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# domains/counter_uav/temporal_pipeline.py
from sklearn.metrics import f1_score


def run_temporal_baseline(dataset_size: int = 256, model_name: str = "cnn_spec", seed: int = 42) -> dict[str, float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from domains.counter_uav.temporal_dataset import build_temporal_sample

    X = []
    y = []
    for i in range(dataset_size // 2):
        bird = build_temporal_sample("bird", azimuth_deg=10.0, snr_db=15.0, n_walls=4, seed=seed + i)
        drone = build_temporal_sample("drone", azimuth_deg=10.0, snr_db=15.0, n_walls=4, seed=seed + 10_000 + i)
        X.append(bird["spectrogram"].reshape(-1))
        y.append(0)
        X.append(drone["spectrogram"].reshape(-1))
        y.append(1)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    accuracy = accuracy_score(y[te], pred)
    macro_f1 = f1_score(y[te], pred, average="macro")
    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1), "model_name": model_name}


# analysis/counter_uav_temporal_benchmark.py
def write_b_vs_c_summary(output_path: str, snapshot_metrics: dict, temporal_metrics: dict) -> None:
    import json
    payload = {
        "snapshot": snapshot_metrics,
        "temporal": temporal_metrics,
        "delta": {
            "accuracy": round(temporal_metrics["accuracy"] - snapshot_metrics["accuracy"], 6),
            "macro_f1": round(temporal_metrics["macro_f1"] - snapshot_metrics["macro_f1"], 6),
        },
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter_uav_temporal_pipeline.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/temporal_models.py domains/counter_uav/temporal_pipeline.py analysis/counter_uav_temporal_benchmark.py tests/test_counter_uav_temporal_pipeline.py
git commit -m "feat: add temporal baselines and b-vs-c benchmark"
```

---

### Task 6: Synchronize Final Evidence Without Premature UI Promotion

**Files:**
- Modify: `output_counter_uav/metrics.json` (through pipeline output)
- Modify: `output_counter_uav/benchmarks/snapshot_benchmark.json` (generated artifact)
- Modify: `output_counter_uav/benchmarks/b_vs_c.json` (generated artifact)
- Modify: `ui/app.py`
- Modify: `ui/templates/counter_uav.html`
- Test: `tests/ui/test_routes.py`

- [ ] **Step 1: Write the failing test**

```python
def test_counter_uav_page_distinguishes_snapshot_and_temporal_metrics(client, monkeypatch, tmp_path):
    from app import PROJECT_ROOT

    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "snapshot_benchmark.json").write_text(
        '{"best_experiment": {"name": "B4", "accuracy": 0.66}}',
        encoding="utf-8",
    )
    (benchmark_dir / "b_vs_c.json").write_text(
        '{"snapshot": {"accuracy": 0.66}, "temporal": {"accuracy": 0.87}, "delta": {"accuracy": 0.21}}',
        encoding="utf-8",
    )
    monkeypatch.setattr("app.PROJECT_ROOT", tmp_path)
    resp = client.get("/counter_uav")
    assert "snapshot baseline" in resp.text
    assert "temporal recovered model" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_routes.py -k snapshot_and_temporal_metrics -v`

Expected: FAIL because the current template only knows about deployed snapshot metrics and does not distinguish B and C evidence.

- [ ] **Step 3: Write minimal implementation**

```python
# ui/app.py
def _load_counter_uav_benchmarks() -> dict:
    snapshot_path = PROJECT_ROOT / "output_counter_uav" / "benchmarks" / "snapshot_benchmark.json"
    temporal_path = PROJECT_ROOT / "output_counter_uav" / "benchmarks" / "b_vs_c.json"
    return {
        "snapshot_benchmark": _read_json(snapshot_path),
        "b_vs_c": _read_json(temporal_path),
    }


# ui/templates/counter_uav.html
{% if evidence.counter_uav_benchmarks.b_vs_c %}
<div class="text-sm font-semibold mb-2">Research recovery status</div>
<ul class="text-sm">
  <li><strong>snapshot baseline</strong>: {{ "%.1f"|format(evidence.counter_uav_benchmarks.b_vs_c.snapshot.accuracy * 100) }}% accuracy</li>
  <li><strong>temporal recovered model</strong>: {{ "%.1f"|format(evidence.counter_uav_benchmarks.b_vs_c.temporal.accuracy * 100) }}% accuracy</li>
</ul>
{% endif %}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ui/test_routes.py -v`

Expected: PASS, with explicit distinction between `snapshot baseline` and `temporal recovered model`.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py ui/templates/counter_uav.html tests/ui/test_routes.py
git commit -m "feat: present counter-uav snapshot and temporal evidence separately"
```

---

## Self-Review Checklist

1. **Spec coverage:** This plan maps `Phase B` to Tasks 1-3, `Phase C` to Tasks 4-5, and jury/runtime synchronization to Task 6.
2. **Placeholder scan:** No `TBD`, `TODO`, ellipses, or "implement later" placeholders are left in the task descriptions.
3. **Type consistency:** `dataset_version`, `quality_report.json`, `snapshot_benchmark.json`, `b_vs_c.json`, `build_temporal_sample()`, `run_temporal_baseline()`, and `write_b_vs_c_summary()` are used consistently across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-counter-uav-b-to-c-recovery-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
