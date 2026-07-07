# Gap Analysis — было → будет

**Проект:** Array Cascade ITMO  
**Дата:** июнь 2026  
**Принцип:** честная фиксация текущего состояния vs целевого; без overclaim

---

## 1. Executive summary

| Измерение | Было (legacy Shishanov) | Сейчас (спринт, июнь 2026) | Будет (магистратура) |
|---|---|---|---|
| **Архитектура** | Monolithic scripts в корне репо | `Cascade.forward()` работает, но core частично re-export legacy | `core/` + `domains/*` unified |
| **Домены** | 1 (automotive AWR1843) | automotive working + counter-UAV physics/synth/first ML | 2+ validated |
| **DOA synth-val** | MedAE 0.66° (Siamese legacy) | Local repro: LISTA 5.23°, Siamese 5.60° RMSE | Counter-UAV metrics improved after preprocessing fix |
| **DOA real** | RMSE ~51° (domain gap) | Признан, не «баг» | RMSE target < 15° (sem 1–2) |
| **Classifier real** | Acc 94.5%, F1 0.724 | Local repro: Acc 95.54% | Improved class-2 precision |
| **Tracker** | Нет | Нет (out of sprint scope) | IMM+JPDA (sem 2) |
| **Fusion** | Нет | Нет | RF+acoustic (sem 3) |
| **UI demo** | Нет | Live FastAPI + Plotly + API | Screenshots/video backup |
| **Публикации** | Internal reports | Заявка ИТМО | arxiv → conf → journal → Q1 |

**Главный вывод:** проект не «сломан» — он **находится в переходе** от working prototype одного домена к universal framework с honest documentation domain gap.

---

## 2. Технический gap: synth → real (DOA)

### 2.1. Факты (источник: `_legacy_docs/source_papers/Описание_результатов_качество.txt`)

| Этап | Siamese RMSE | Siamese MedAE | Интерпретация |
|---|---|---|---|
| Val (synth) | **4.57°** | **0.66°** | Модели обучаются корректно |
| Inference (real, ±70°) | **41.73°** | — | Domain gap |
| Inference (real, full) | **52.49°** | — | + edge angles |
| All methods on real | 41–55° | — | Не одна модель — все падают одинаково |

**Вывод:** это **distribution mismatch**, не архитектурный дефект одного метода.

### 2.2. Диагностированные факторы gap

| Параметр | Synth | Real | Влияние |
|---|---|---|---|
| Угловое распределение | Квазиравномерное [-90°, +90°] | Сконцентрировано [-30°, +30°] с пиками | Сеть редко видит real-like углы |
| Баланс классов | ~33/33/33 (39K каждый) | 86/12/2 % | Class 2 (two-targets) почти absent на real |
| Амплитуды | Peak ~1.0 | Peak ~1.0 | ✓ OK |
| SNR / phase / hardware | Идеализирован | Реальные искажения | Гипотеза, требует sem 1 audit |
| Edge angles ±75–85° | Много synth samples | Мало на real | 5–10% samples → 90% RMSE inflation |

### 2.3. План закрытия (sem 1–2)

| Приоритет | Действие | Ожидаемый эффект |
|---|---|---|
| P0 | Matching angular distribution synth→real | −30–50% real RMSE |
| P0 | Weighted sampling / focal loss для class 2 | Лучший routing для two-target |
| P1 | Semi-supervised finetune на unlabeled real | Domain adaptation |
| P1 | Confidence rejection (low spectrum → skip) | SR@3° на confident subset → 95%+ |
| P2 | Multi-snapshot M>1 (если radar format allows) | 2–3× accuracy gain |

### 2.4. Что НЕ делаем

- Не называем 51° RMSE «багом, который починим за неделю».
- Не публикуем «RMSE < 1°» без указания synth/real и snapshot count.
- Не скрываем gap от жюри — он central research question магистратуры.

---

## 3. Архитектурный gap: monolith → framework

### 3.1. Было (legacy root)

```
Shishanov_paper/
├── config.py, dataset.py, models.py      # automotive-only
├── preprocessing.py, metrics.py
├── run_full_pipeline.py
├── output/                                # trained weights
└── train_validate/                        # real .txt data
```

**Проблема:** каждый новый домен = копипaste + правки по всему репо.

### 3.2. Сейчас (Array_Cascade_ITMO, июнь 2026)

```
Array_Cascade_ITMO/
├── core/                    # ✓ Cascade.forward + re-export integration layer
├── domains/                 # ✓ automotive config + counter_uav implementation
├── config.py, dataset.py  # ✓ legacy root (still working)
├── ui/                      # ✓ 6 routes, templates, live/fallback inference
└── docs/                    # ✓ in progress
```

| Компонент | Статус | Gap |
|---|---|---|
| `core/doa/*` | Re-export from legacy | Clean migration still pending |
| `core/cascade.py` | Working DomainSpec + `forward()` | Needs cleaner separation from root legacy |
| `core/classifier.py` | Present | Re-export; usable in cascade |
| `domains/automotive/` | Minimal config | Full dataset/pipeline migration pending |
| `domains/counter_uav/` | Implemented first version | ML quality below AC targets |
| `ui/services/inference.py` | Live/fallback | Loads models and serves UI |

### 3.3. Будет (target, end of sprint → sem 1)

```
core/           # domain-agnostic, tested
domains/
  automotive/   # migrated pipeline
  counter_uav/  # synth gen + train
scripts/        # thin wrappers
ui/             # live inference
```

**Acceptance (AC-5):** один `Cascade` инстанс на `DomainSpec.automotive()` и `DomainSpec.counter_uav()` без diff в `core/`.

---

## 4. Доменный gap: automotive → counter-UAV

| Аспект | Automotive (legacy) | Counter-UAV (target) | Gap |
|---|---|---|---|
| Частота | 77 GHz | 12.5 GHz X-band | New `physics_xband.py` |
| RCS | 1–10 m² | 0.001–0.01 m² | SNR range −5..25 dB |
| Multipath | Road, vehicles | Urban walls (method of images) | New `urban_multipath.py` |
| Classes | 3 (direct/mp/2tgt) | 4 (clutter/drone/bird/multi) | Classifier retrain |
| Micro-Doppler | Minimal | Core feature (drone vs bird) | New `micro_doppler.py` |
| Data | synth_new.npz + real txt | synth_10k.npz generated | Generator built |
| Metrics | Known (table §4.2 SPEC) | classifier 59.9%, Siamese RMSE 4.49° | Better after data-contract fixes, but still below AC |

**Honest status:** counter-UAV — physics/synth layer is working; ML baseline exists but **does not yet pass AC-3/AC-4**.

---

## 5. ML quality gap

### 5.1. Automotive classifier

| Метрика | Synth-val | Real-val | Gap action |
|---|---|---|---|
| Accuracy | ~89.6% (overfit risk) | **94.5%** | Early stopping (P0) |
| F1 macro | 0.658 | **0.724** | Acceptable on real |
| Class 2 precision | 0.18 (broken) | Better on real | Reduce class-2 weight 41× → 10–15× |

**Insight:** automotive classifier **works on real** for routing; DOA is the bottleneck there, not classification. Это не относится к counter-UAV classifier, где even после data-generator fixes full 10K baseline пока даёт 59.9%.

### 5.2. DOA methods

| Метод | Synth MedAE | Issue | Sprint/MA action |
|---|---|---|---|
| Siamese | 0.66° | Real gap | Domain matching |
| LISTA | 0.73° | Init spike epoch 1 | Warmup LR |
| HierGRU | 1.58° | Undertrained (60 ep) | 120+ epochs |
| CovRecon | 2.03° | Phantom roots | Adaptive N_sources |

### 5.3. Counter-UAV targets (AC from SPEC §11)

| Criterion | Target | Status |
|---|---|---|
| AC-3 Classifier accuracy | ≥85% synth-val | **FAIL:** full 10K accuracy 59.9% |
| AC-4 DOA RMSE single-target | ≤3° synth-val | **FAIL:** Siamese 4.49°, LISTA 63.36° |
| AC-2 Beamforming unit test | ≤2° single target | **PASS:** 11/11 tests |

### 5.4. Preprocessing / grid contract status

Критический mismatch между train/runtime preprocessing path для counter-UAV уже частично устранён. Сейчас `DomainSpec.counter_uav()` использует `raw` contract, classifier-path canonicalizes snapshot только для классификации, а DOA остаётся на raw steering-domain grid `[-70, 70]` deg. Это убрало самый грубый источник ложного ухудшения качества из-за сдвига peak/bin после лишней ортогонализации.

Оставшаяся проблема уже не в формальном несоответствии train/runtime contract, а в качестве самого baseline: drone/bird discrimination и multi-target semantics всё ещё недостаточно сильны, поэтому даже после фикса контракта метрики остаются ниже AC-3/AC-4.

---

## 6. Infrastructure gap

| Item | Было | Сейчас | Будет |
|---|---|---|---|
| Local baseline repro | On original machine | **Done** on RTX 2060 SUPER | BASELINE_CONFIRMED.md filled |
| GPU training | Validated legacy | Same GPU, env setup TBD | Documented in README |
| Public datasets | None | None | DIAT-µSAT, RAD-DAR (sem 1) |
| Hardware testbed | AWR1843 recordings only | Same | Pi + SDR + mic (sem 4) |
| CI/CD | None | None | Optional pytest for core |

---

## 7. Documentation & demo gap

| Artifact | Было | Сейчас | Будет |
|---|---|---|---|
| SPEC | — | SPEC_ITMO_V2.md ✓ | Frozen snapshot post-sprint |
| Application pack | — | **This folder** ✓ | PDF export if needed |
| UI | — | Live demo ✓ | Screenshots/video backup |
| Screenshots | — | Pending | docs/ui_screenshots/ |
| Video backup | — | Pending | 2-min screen recording |

---

## 8. Risk-adjusted timeline

```
                    CONFIDENCE
High ─────────────────────────────────────────
     │  Synth baseline (legacy numbers)
     │  Classifier on real
     │  UI routes exist
     │
Med  │  Local repro baseline
     │  core/cascade wired
     │  counter-UAV synth gen
     │
Low  │  counter-UAV metrics AC-3/4
     │  Real DOA gap closed
     │  Tracker, fusion, MVP
     └────────────────────────────────────────▶ TIME
        Sprint    Sem1    Sem2    Sem3    Sem4
```

---

## 9. Честные формулировки для жюри

### ✓ Можно говорить

- «MedAE 0.66° на synth-val, single-snapshot, 8-ant ULA — ~76% от CRB при SNR 20 dB»
- «Превосходим предел Рэлея (~29°) на порядок на синтетике»
- «Classifier на real data: 95.54% accuracy локально — cascade routing работает»
- «Domain gap synth→real для DOA задокументирован; это research agenda магистратуры»
- «Один cascade interface работает для двух доменов; UI демонстрирует архитектуру»

### ✗ Нельзя говорить

- «RMSE < 1°» без контекста synth/real
- «Production-ready counter-UAV system»
- «Counter-UAV ML уже достиг 85% accuracy» или «DOA RMSE ≤3°»
- «Domain gap решён» или «это был баг»
- «Tracker и fusion уже реализованы»
- «Q1 paper уже submitted»

---

## 10. Summary matrix

| Gap type | Severity | Owner phase | Blocker for application? |
|---|---|---|---|
| Synth→real DOA | **High** (research) | Sem 1–2 | **No** — honestly documented |
| domains/ migration | Medium | Sprint | No — integration layer proves intent |
| counter-UAV impl | Medium | Sprint | No — physics/synth exists, ML WIP |
| Local baseline repro | Low | Sprint day 1 | No — reproduced locally |
| Tracker/fusion/MVP | N/A (planned) | Sem 2–4 | No — in roadmap |

**Итог:** заявка опирается на **подтверждённый automotive baseline + working two-domain architecture + честный gap analysis**, а не на скрытие проблем real-data DOA и counter-UAV ML.
