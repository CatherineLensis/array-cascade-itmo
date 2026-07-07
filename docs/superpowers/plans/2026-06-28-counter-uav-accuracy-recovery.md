# Counter-UAV Accuracy Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Найти корневую причину низкой `counter-UAV` accuracy и поднять jury-facing качество классификации без маскировки слабых мест в UI.

**Architecture:** Работа делится на две независимые, но связанные ветки: (1) диагностическая ветка, которая доказывает, где ломается separability `bird/drone`, и (2) recovery-ветка, которая меняет synthetic generator и classifier-only training contract, не перетирая удачные DOA checkpoints. Все изменения фиксируются тестами и runtime smoke на `u1/u2/u3`.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, FastAPI/Jinja2 UI.

---

## Demo Metrics Guardrails

The recovery work must not blur together unrelated numbers on the jury demo.

Keep these meanings separate throughout implementation:

- **Automotive / local TI AWR1843 `.txt` data**
  - classifier accuracy: `93.8%`
  - local DOA repro RMSE: about `5.23°`
  - legacy synth-val MedAE: `0.66°`
  - legacy SR@1°: `65.2%`

- **Counter-UAV / synthetic engineering baseline**
  - pre-recovery weak region was around `52.2%` accuracy and `50.2%` macro-F1
  - current recovered deployed classifier region is about `58.8%` report accuracy and `58.2%` macro-F1, with `60.9%` best validation accuracy during training
  - current demo expectation after preset stabilization: live smoke on `u1/u2/u3` should stay semantically correct
  - do **not** promote weak counter-UAV holdout accuracy as a landing-page hero metric before recovery is complete

Any implementation that changes training or presets must preserve this semantic split in UI/docs:
- automotive = local real TI radar evidence
- counter-UAV = synthetic research / engineering baseline

---

## File Map

- Modify: `domains/counter_uav/dataset_synth.py` — генерация synthetic snapshots и multipath mixing
- Modify: `domains/counter_uav/micro_doppler.py` — физика bird/drone proxy-сигналов
- Modify: `domains/counter_uav/pipeline.py` — classifier-only training, sampling, checkpoint handling
- Modify: `core/classifier.py` — classifier input normalization/canonicalization helpers
- Modify: `core/cascade.py` — runtime classifier contract
- Modify: `ui/services/presets.py` — representative demo presets после retrain
- Modify: `ui/app.py` / `ui/templates/*.html` — только после подтверждённого root-cause и recovery
- Create: `analysis/counter_uav_accuracy_audit.py` — reproducible diagnostics for class separability and metadata buckets
- Create: `tests/test_counter_uav_accuracy_audit.py` — regression tests for discovered failure modes
- Modify: `tests/ui/test_routes.py` — jury-facing runtime regressions (`u1/u2/u3`)
- Modify: `tests/test_counter_uav_physics.py` — generator alignment tests for presets

### Task 1: Lock Down the Failure With Diagnostics

**Files:**
- Create: `analysis/counter_uav_accuracy_audit.py`
- Test: `tests/test_counter_uav_accuracy_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_bird_recall_drops_with_high_multipath():
    summary = analyze_counter_uav_classifier_failure()
    assert summary["bird_recall_nwalls_4_6"] >= 0.45


def test_u2_seed_is_not_boundary_case():
    summary = analyze_counter_uav_classifier_failure()
    assert summary["u2_margin_vs_drone"] >= 0.10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter_uav_accuracy_audit.py -v`

Expected: FAIL because current diagnostics show low bird recall in high-multipath buckets and weak/negative `u2` margin.

- [ ] **Step 3: Write minimal diagnostics implementation**

```python
def analyze_counter_uav_classifier_failure() -> dict[str, float]:
    # 1. load synth NPZ and classifier checkpoint
    # 2. evaluate holdout predictions
    # 3. aggregate bird recall for n_walls in {4, 6}
    # 4. score current u2 preset logits and bird-vs-drone margin
    return {
        "bird_recall_nwalls_4_6": bird_recall,
        "u2_margin_vs_drone": u2_margin,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter_uav_accuracy_audit.py -v`

Expected: PASS, with diagnostic output reproducible on the current checkpoint.

- [ ] **Step 5: Commit**

```bash
git add analysis/counter_uav_accuracy_audit.py tests/test_counter_uav_accuracy_audit.py
git commit -m "test: lock down counter-uav bird failure diagnostics"
```

### Task 2: Prove Whether Generator Mixing Is the Root Cause

**Files:**
- Modify: `analysis/counter_uav_accuracy_audit.py`
- Modify: `domains/counter_uav/dataset_synth.py`
- Test: `tests/test_counter_uav_accuracy_audit.py`

- [ ] **Step 1: Write the failing test**

```python
def test_multipath_mixing_preserves_bird_drone_separability():
    summary = analyze_generator_separability()
    assert summary["multipath_sep_lr_score"] >= 0.70
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_counter_uav_accuracy_audit.py::test_multipath_mixing_preserves_bird_drone_separability -v`

Expected: FAIL because current mixing path drops separability below the threshold.

- [ ] **Step 3: Add minimal diagnostic implementation**

```python
def analyze_generator_separability() -> dict[str, float]:
    # build drone/bird snapshots with and without multipath
    # phase-normalize them
    # fit a tiny LogisticRegression probe
    return {
        "clean_sep_lr_score": clean_score,
        "multipath_sep_lr_score": multipath_score,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_counter_uav_accuracy_audit.py::test_multipath_mixing_preserves_bird_drone_separability -v`

Expected: PASS only after generator changes preserve enough class-specific structure.

- [ ] **Step 5: Commit**

```bash
git add analysis/counter_uav_accuracy_audit.py domains/counter_uav/dataset_synth.py tests/test_counter_uav_accuracy_audit.py
git commit -m "fix: preserve bird-drone structure under counter-uav multipath"
```

### Task 3: Split Classifier-Only Retrain From DOA Retrain

**Files:**
- Modify: `domains/counter_uav/pipeline.py`
- Test: `tests/ui/test_routes.py`

- [ ] **Step 1: Write the failing test**

```python
def test_classifier_retrain_does_not_require_overwriting_doa_weights(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "siamese" in resp.json()["models"]["counter_uav"]["doa_methods"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_routes.py -k doa_weights -v`

Expected: FAIL once pipeline is adjusted to support classifier-only flow but not yet implemented.

- [ ] **Step 3: Implement minimal pipeline split**

```python
def main(npz_path: str | None = None, quick: bool = False, classifier_only: bool = False):
    # train classifier every time
    # train DOA only when classifier_only is False
    # keep prior best DOA checkpoints intact in classifier-only mode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ui/test_routes.py -k doa_weights -v`

Expected: PASS, and old DOA checkpoints remain loadable after classifier-only retrain.

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/pipeline.py tests/ui/test_routes.py
git commit -m "feat: support classifier-only counter-uav retrain"
```

### Task 4: Recover Bird Recall Without Destroying Global Accuracy

**Files:**
- Modify: `domains/counter_uav/pipeline.py`
- Modify: `core/classifier.py`
- Modify: `core/cascade.py`
- Test: `tests/ui/test_routes.py`
- Test: `tests/test_counter_uav_accuracy_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_api_run_cascade_u2_classifies_bird(client):
    resp = client.post("/api/run_cascade", json={"domain": "counter_uav", "preset": "u2", "snr": 15})
    assert resp.json()["classifier_label"] == "bird"


def test_bird_holdout_recall_recovers_to_target():
    summary = analyze_counter_uav_classifier_failure()
    assert summary["bird_recall_overall"] >= 0.45
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ui/test_routes.py -k u2_classifies_bird -v`

Run: `pytest tests/test_counter_uav_accuracy_audit.py -k bird_holdout_recall -v`

Expected: FAIL on the weak checkpoint.

- [ ] **Step 3: Implement minimal recovery**

```python
# pipeline.py
sample_weights[hard_bird_mask] = 1.25
crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.2, 1.0], ...))

# keep train/runtime classifier preprocessing identical
```

- [ ] **Step 4: Retrain classifier-only and verify**

Run: `python scripts/train_counter_uav.py --classifier-only`

Run: `pytest tests/ui/test_routes.py tests/test_counter_uav_accuracy_audit.py -v`

Expected: `u2 -> bird`, bird recall recovered, no breakage in `u1/u3`.

- [ ] **Step 5: Commit**

```bash
git add domains/counter_uav/pipeline.py core/classifier.py core/cascade.py tests/ui/test_routes.py tests/test_counter_uav_accuracy_audit.py
git commit -m "fix: recover counter-uav bird classification under multipath"
```

### Task 5: Re-select Representative Demo Presets After Retrain

**Files:**
- Modify: `ui/services/presets.py`
- Modify: `tests/test_counter_uav_physics.py`
- Test: `tests/ui/test_routes.py`

- [ ] **Step 1: Write the failing test**

```python
def test_u2_preset_uses_representative_bird_seed(client):
    resp = client.post("/api/run_cascade", json={"domain": "counter_uav", "preset": "u2", "snr": 15})
    assert resp.json()["classifier_label"] == "bird"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_routes.py -k representative_bird_seed -v`

Expected: FAIL if the current seed is still a boundary-case sample.

- [ ] **Step 3: Implement minimal preset update**

```python
if preset.id == "u2":
    return build_bird_snapshot(
        preset.true_angle_deg or -20.0,
        snr_db=snr,
        n_walls=4,
        rng=np.random.default_rng(REPRESENTATIVE_BIRD_SEED),
        velocity_mps=12.0,
    ).reshape(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ui/test_routes.py tests/test_counter_uav_physics.py -k "u2 or bird" -v`

Expected: PASS, with preset aligned to generator expectations.

- [ ] **Step 5: Commit**

```bash
git add ui/services/presets.py tests/test_counter_uav_physics.py tests/ui/test_routes.py
git commit -m "fix: stabilize representative counter-uav bird preset"
```

### Task 6: Only After Recovery, Reframe Jury-Facing UI Copy

**Files:**
- Modify: `ui/app.py`
- Modify: `ui/templates/index.html`
- Modify: `ui/templates/methodology.html`
- Modify: `ui/templates/counter_uav.html`
- Test: `tests/ui/test_routes.py`

- [ ] **Step 1: Write the failing test**

```python
def test_counter_uav_landing_copy_does_not_present_weak_accuracy_as_main_kpi(client):
    resp = client.get("/")
    assert "Classifier holdout:" not in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_routes.py -k weak_accuracy_as_main_kpi -v`

Expected: FAIL while old copy still foregrounds the weak metric.

- [ ] **Step 3: Implement minimal copy change**

```html
<li><strong>Counter-UAV:</strong> synthetic research baseline, jury demo uses representative live presets</li>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ui/test_routes.py -v`

Expected: PASS, with UI still honest but less self-sabotaging.

- [ ] **Step 5: Commit**

```bash
git add ui/app.py ui/templates/index.html ui/templates/methodology.html ui/templates/counter_uav.html tests/ui/test_routes.py
git commit -m "docs: reframe counter-uav jury-facing metrics copy"
```

## Self-Review

- Spec coverage: план закрывает root-cause, generator, classifier-only retrain, representative presets и только потом UI narrative.
- Placeholder scan: нет `TODO`/`TBD`; в каждом шаге есть конкретная цель, команда и минимальный код.
- Type consistency: все шаги используют существующие пути проекта и текущие символы (`build_bird_snapshot`, `train_counter_uav.py`, `test_routes.py`).

## Execution Handoff

План complete and saved to `docs/superpowers/plans/2026-06-28-counter-uav-accuracy-recovery.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
