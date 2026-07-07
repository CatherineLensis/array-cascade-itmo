# Counter-UAV 85%+ Recovery Design

> **Назначение.** Дизайн-документ для поэтапного повышения качества `counter-UAV` классификации до уровня `>85%` через честный ML/DS workflow, без смешивания baseline-ветки и новой temporal-ветки.
>
> **Статус.** Draft after brainstorming confirmation-gates. Следующий шаг после review этого документа: записать implementation plan и только затем переходить к коду.

---

## 0. Журнал решений brainstorming-сессии (2026-06-28)

| # | Вопрос | Выбрано | Обоснование |
|---|---|---|---|
| 1 | Целевой путь к `>85%` | **Выполнять отдельно B и отдельно C** | Нужно получить честную сравнительную оценку текущего snapshot-contract и нового temporal-contract |
| 2 | Порядок исполнения | **Сначала B, затем C** | Иначе потом нельзя будет доказать, что именно дало прирост: улучшение baseline или смена представления |
| 3 | Роль B | **Bounded baseline experiment** | Проверяет потолок `8`-антенного `single-snapshot` classifier path без подмены контракта |
| 4 | Роль C | **Основной кандидат на `>85%`** | Open-dataset audit показывает, что discriminative cue живёт в slow-time / spectrogram morphology |
| 5 | UI/runtime стратегия | **Не смешивать метрики B и C в одном пути** | Jury-facing narrative должен различать `snapshot baseline` и `temporal recovered model` |

---

## 1. Контекст и проблема

Текущий `counter-UAV` classifier работает на `8`-антенном complex `single snapshot` и после последних исправлений поднялся примерно до:

- `60.9%` best validation accuracy during training,
- `58.8%` deployed report accuracy,
- `58.2%` macro-F1,
- при этом live smoke по `u1/u2/u3` уже семантически корректен.

Этот результат полезен как engineering baseline, но недостаточен для цели `>85%`.

Согласно `docs/counter_uav_open_dataset_audit.md`, публичные radar bird-vs-drone references указывают, что ключевой discriminative signal находится не только в статическом spatial snapshot, а прежде всего в:

- slow-time micro-Doppler dynamics,
- spectrogram morphology,
- burst-like wingbeat behavior for birds,
- rotor-like regular sideband structure for drones.

Следовательно, путь к `>85%` нельзя строить как один непрозрачный retrain. Нужна двухфазная программа:

1. честно проверить пределы текущего snapshot-contract;
2. отдельно построить temporal classifier path, если snapshot-contract оказывается узким местом.

---

## 2. Цели и не-цели

### Цели

1. Получить воспроизводимый ML/DS workflow для `counter-UAV` classification recovery.
2. Проверить, достижимо ли `>85%` внутри текущего `single-snapshot` contract.
3. Если нет, построить отдельный temporal / hybrid path, максимально близкий по структуре к open-dataset reality-check.
4. Сохранить честное разделение смыслов:
   - automotive = local real TI radar evidence,
   - counter-UAV = synthetic research baseline или recovered temporal model.
5. Не ломать текущий demo/runtime во время research-итераций.

### Не-цели

1. Не маскировать слабую offline accuracy косметикой в UI.
2. Не объявлять snapshot baseline solved-product решением.
3. Не смешивать результаты B и C в одну метрику.
4. Не переписывать весь UI/runtime до появления сильного offline benchmark для C.

---

## 3. Архитектура recovery-программы

```
counter_uav recovery
│
├── Phase B: snapshot-only baseline recovery
│   ├── versioned snapshot datasets
│   ├── data quality gates
│   ├── experiment ladder B0..B4
│   ├── bounded stop-conditions
│   └── verdict: viable or ceiling confirmed
│
└── Phase C: temporal / hybrid recovery
    ├── versioned temporal datasets
    ├── temporal quality gates
    ├── experiment ladder C0..C3
    ├── direct B vs C benchmark
    └── runtime promotion allowed only after offline proof
```

Ключевой архитектурный принцип: B и C независимы по данным, артефактам, метрикам и narrative. Единственная общая точка между ними - единый evaluation protocol на сопоставимых сценах, чтобы можно было честно сравнивать результаты.

---

## 4. Фаза B: snapshot-only recovery

### 4.1 Цель фазы B

Проверить максимум качества, достижимый внутри текущего classifier contract:

`8 complex antennas -> preprocessing -> snapshot classifier`.

B нужна не только как попытка улучшить accuracy, но и как доказательство потолка, если `>85%` не достигается даже после дисциплинированного ML/DS workflow.

### 4.2 Data contract для B

Каждая snapshot-версия датасета должна быть versioned artifact с manifest, в котором фиксируются:

- генератор и версия кода,
- seed policy,
- class balance,
- bucket coverage по `n_walls`, `snr_db`, `velocity_mps`, azimuth,
- subtype knobs для drone/bird,
- separability and quality summary.

Датасет перестаёт быть просто `synth_10k.npz`; он становится воспроизводимой версией вида:

- `output_counter_uav/datasets/<dataset_version>/signals.npz`
- `output_counter_uav/datasets/<dataset_version>/manifest.json`
- `output_counter_uav/datasets/<dataset_version>/quality_report.json`

### 4.3 Quality gates для B

Перед retrain snapshot-пайплайн должен проходить минимальный набор gates:

1. class balance sanity;
2. coverage sanity по ключевым buckets;
3. separability probe:
   - clean bird vs drone,
   - strong-multipath bird vs drone;
4. regression gates:
   - `bird_recall_nwalls_4_6`,
   - `bird_recall_overall`,
   - `u2_margin_vs_drone`;
5. demo preset stability:
   - `u1 -> drone`,
   - `u2 -> bird`,
   - `u3 -> multi`.

Если gate падает, retrain считается невалидным.

### 4.4 Experiment ladder для B

B должен идти по контролируемым ступеням:

- `B0`: current recovered baseline;
- `B1`: только data improvements;
- `B2`: data improvements + sampler / hard-negative mining;
- `B3`: feature engineering внутри snapshot-only;
- `B4`: model sweep при том же input contract.

На каждой ступени фиксируются:

- dataset version,
- train config,
- holdout accuracy,
- macro-F1,
- bird recall,
- bird recall at strong multipath,
- `u2` margin,
- live smoke outcome.

### 4.5 Stop conditions для B

Фаза B не должна превращаться в бесконечный тюнинг. Stop-condition наступает, если выполняется хотя бы одно из условий:

1. после 2-3 полных dataset/model итераций accuracy остаётся существенно ниже `85%`, а прирост стабилизируется;
2. улучшения касаются только demo presets, но не честного holdout;
3. accuracy растёт ценой деградации live smoke или воспроизводимости;
4. probe-анализ показывает, что snapshot-only separability насыщается ниже целевого уровня.

### 4.6 Deliverables фазы B

- versioned snapshot datasets,
- quality reports,
- benchmark report `B0..B4`,
- итоговый verdict:
  - `snapshot-only viable`, или
  - `snapshot-only ceiling confirmed`.

---

## 5. Фаза C: temporal / hybrid recovery

### 5.1 Цель фазы C

Построить новый classifier path, использующий temporal information как основной источник discriminative signal для `bird vs drone`, при этом не ломая snapshot baseline runtime.

### 5.2 Data contract для C

Генератор должен выпускать versioned temporal artifacts, например:

- `snapshot`,
- `micro_series`,
- derived `spectrogram` representation for temporal models,
- scene metadata:
  - class,
  - `n_walls`,
  - `snr_db`,
  - `velocity_mps`,
  - temporal parameters,
  - subtype knobs.

Важно: snapshot и temporal artifacts должны быть синхронизированы на уровне одной и той же scene-family, чтобы comparison `B vs C` был честным.

### 5.3 Quality gates для C

Temporal dataset обязан проходить отдельный набор gates:

1. band sanity для wingbeat frequency;
2. regularity sanity для rotor-like signatures;
3. spectrogram/probe separability between bird and drone;
4. robustness under multipath;
5. leakage prevention:
   - no split leakage by seed,
   - no split leakage by scene-family,
   - no split leakage by generator subtype template.

### 5.4 Experiment ladder для C

Фаза C идёт по ступеням от простого к сложному:

- `C0`: probe on handcrafted temporal features or flattened spectrogram;
- `C1`: compact CNN on spectrogram;
- `C2`: CRNN or compact transformer on temporal representation;
- `C3`: hybrid model with temporal main branch and snapshot auxiliary branch.

Эта лестница нужна для раннего доказательства bottleneck: если даже `C0/C1` заметно обгоняют B, значит узкое место действительно в информационном представлении.

### 5.5 Evaluation protocol для C

Для каждой temporal-модели фиксируются:

- overall accuracy,
- macro-F1,
- bird recall,
- bird recall for `n_walls in {4, 6}`,
- robustness by SNR buckets,
- comparison to B on the same holdout scenes.

Главный сравнительный артефакт C - отчёт `B vs C`, где на одинаковых сценах видны:

- snapshot baseline score,
- temporal score,
- absolute and relative delta.

### 5.6 Runtime strategy для C

До появления устойчивого offline benchmark C не внедряется в основной UI-path.

Порядок такой:

1. snapshot runtime сохраняется как baseline/demo-compatible path;
2. temporal classifier живёт отдельно как research/benchmark path;
3. только после устойчивого offline преимущества можно обсуждать promotion temporal model в runtime.

### 5.7 Deliverables фазы C

- temporal dataset format,
- temporal quality-gate suite,
- benchmark `C0..C3`,
- final `B vs C` comparison report,
- решение:
  - можно ли честно заявлять `>85%`,
  - и в каком именно classifier contract.

---

## 6. Порядок реализации

Реализация обязана идти строго по порядку:

1. завершить B как автономный research cycle;
2. зафиксировать B verdict;
3. только затем запускать C;
4. сначала получить offline benchmark для C;
5. только затем обсуждать runtime integration;
6. UI/docs/jury narrative синхронизировать только в самом конце.

Это критично, потому что смешивание B и C на раннем этапе разрушит причинно-следственную интерпретацию результатов.

---

## 7. File map

### Existing files primarily affected by B

- `domains/counter_uav/dataset_synth.py`
- `domains/counter_uav/pipeline.py`
- `analysis/counter_uav_accuracy_audit.py`
- `tests/test_counter_uav_accuracy_audit.py`
- `tests/test_counter_uav_pipeline.py`
- `ui/services/presets.py`
- `tests/ui/test_routes.py`

### New artifacts recommended for B

- `analysis/counter_uav_snapshot_benchmark.py`
- `output_counter_uav/datasets/<version>/manifest.json`
- `output_counter_uav/benchmarks/snapshot_benchmark.json`

### Existing and new files primarily affected by C

- `domains/counter_uav/micro_doppler.py`
- `domains/counter_uav/temporal_dataset.py`
- `domains/counter_uav/temporal_pipeline.py`
- `analysis/counter_uav_temporal_benchmark.py`
- `tests/test_counter_uav_temporal_pipeline.py`

### Runtime/UI files that should change only after offline proof

- `core/cascade.py`
- `ui/app.py`
- `ui/templates/counter_uav.html`
- `ui/templates/index.html`
- `ui/templates/methodology.html`

---

## 8. Метрики и acceptance criteria

### Acceptance for B

Фаза B считается успешной, если выполняется одно из двух:

1. snapshot-only path честно достигает `>= 85%` на зафиксированном holdout;
2. либо формально подтверждён ceiling, и это подтверждение воспроизводимо через benchmark + probes + stop-condition.

### Acceptance for C

Фаза C считается успешной, если:

1. temporal path достигает `>= 85%` на заранее зафиксированном protocol;
2. improvement over B виден на одном и том же holdout;
3. quality gates подтверждают отсутствие leakage и synthetic cheating;
4. итоговая цифра может быть честно объяснена пользователю и жюри.

---

## 9. Риски и митигация

| Риск | Где возникает | Митигация |
|---|---|---|
| Snapshot-only path никогда не приблизится к `85%` | B | bounded stop-condition + ранний verdict вместо бесконечного retrain |
| Temporal path даст искусственно завышенную accuracy на слишком удобной синтетике | C | temporal quality gates + leakage protection + cross-version checks |
| Улучшения окажутся demo-specific | B и C | всегда сравнивать holdout metrics и live smoke отдельно |
| UI narrative начнёт опережать реальные артефакты | финальная синхронизация | обновлять UI/docs только после offline proof |
| B и C станут неразличимы по данным и метрикам | вся программа | раздельные datasets, manifests, benchmark outputs и отчёты |

---

## 10. Testing strategy

Минимальный testing stack для recovery-программы:

1. unit/regression tests для dataset generation;
2. audit tests для separability and recall buckets;
3. pipeline tests для versioned datasets and metrics persistence;
4. benchmark scripts with machine-readable outputs;
5. UI smoke tests only as regression guard, not as primary evidence of classifier quality.

---

## 11. Итоговая рекомендация

Практический курс действий должен быть таким:

- использовать B как честный bounded baseline experiment,
- использовать C как основной путь к `>85%`,
- сохранять явное разделение:
  - `snapshot baseline`,
  - `temporal recovered model`,
  - `jury-facing runtime`.

Только такой порядок даёт одновременно инженерную честность, исследовательскую ценность и защищаемую narrative для ИТМО.
