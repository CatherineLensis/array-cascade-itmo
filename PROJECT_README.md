# Array Cascade ITMO — Universal Array Signal Cascade + Counter-UAV

> **Кратко.** Универсальная cascade-архитектура для multipath-rich array signals (V1 — методология) с прикладным кейсом детекции малых БпЛА в городе (V3 — counter-UAV). Проект подаётся в магистратуру ИТМО (Лазько Екатерина).

---

## Зачем этот проект

В современных сенсорных системах (автомобильные радары ADAS, counter-UAV, indoor positioning, акустические массивы) **multipath и single-snapshot regime — главные блокеры точности**. Классические алгоритмы (MUSIC, ESPRIT) ломаются при M=1 snapshot, а SOTA-нейросети обычно строятся под один конкретный сенсор и не переносятся.

**Этот проект показывает, что одна и та же cascade-архитектура (preprocess → classify scene → route → DOA) переносится между доменами с заменой только физмодели.** Демонстрация: автомобильный радар TI AWR1843 (77 GHz) и counter-UAV X-band (12.5 GHz) на одном scaffold'е.

---

## Two-domain proof at a glance

| | Automotive radar (TI AWR1843) | Counter-UAV (X-band synth) |
|---|---|---|
| Частота | 77 GHz | 12.5 GHz |
| RCS цели | 1–10 м² | 0.001–0.01 м² |
| Multipath | от дороги, машин | от зданий (urban) |
| Классы | direct / multipath / two-targets | clutter / drone / bird / multi |
| **Cascade core** | **тот же** | **тот же** |
| Лучший DOA метод | LISTA / Siamese (локальный synth-val ~5.23° / 5.60° RMSE) | improved full-train baseline: Siamese RMSE 4.49° |
| Лучший MedAE | **0.66°** (legacy synth-val), локально подтверждён RMSE ~5° | не заявляем как закрытый AC; требуется preprocessing/grid alignment |
| Превосходит Rayleigh limit (29°) | да, на порядок | (ожидаем) |

---

## Структура репозитория

```
Array_Cascade_ITMO/
├── SPEC_ITMO_V2.md                 # ← главная спецификация (источник правды)
├── STAGES.md                       # ← операционный чек-лист по 10 дням
├── PROJECT_README.md               # ← этот файл (entry point)
│
├── core/                           # домен-агностичный слой
│   └── doa/                        # 4 SOTA-метода (Siamese, LISTA, CovRecon, HierGRU)
│
├── domains/
│   ├── automotive/                 # TI AWR1843, 77 GHz, 3-class (current code)
│   └── counter_uav/                # X-band, 12.5 GHz, 4-class (new in this sprint)
│
├── ui/                             # FastAPI + Jinja2 UI для жюри
│   ├── app.py                      # запуск: python ui/app.py
│   ├── templates/                  # 6 страниц
│   ├── static/
│   └── services/                   # inference, presets, plots
│
├── scripts/                        # runtime скрипты (train / evaluate)
├── notebooks/                      # demo + EDA + sanity
├── analysis/                       # cascade_error, snr_analysis, angular_resolution, inference_benchmark
├── study/                          # 4 SOTA-method исследовательские реализации
├── docs/
│   ├── ITMO_application/           # пакет для заявки в ИТМО
│   ├── BASELINE_CONFIRMED.md       # подтверждение baseline на synth-val
│   ├── ARCHITECTURE.md             # архитектурный README
│   └── ui_screenshots/             # backup-скрины UI
│
├── _legacy_docs/                   # source papers + оригинальный README Шишанова
├── train_validate/                 # реальные данные (.txt из TI AWR1843)
└── requirements.txt
```

---

## Что уже знаем (baseline)

Из исходной работы (Шишанов + Лазько, см. `_legacy_docs/source_papers/Описание_результатов_качество.txt`) и локального воспроизведения baseline:

| Метод | Val RMSE (synth) | MedAE | SR@1° | SR@3° |
|---|---|---|---|---|
| **Siamese** | **4.57°** | **0.66°** | **65.2%** | **88.1%** |
| LISTA | 4.97° | 0.73° | 62.2% | **90.6%** |
| HierGRU | 6.77° | 1.58° | 37.0% | 69.6% |
| CovRecon | 34.96° | 2.03° | 26.5% | 62.9% |

Classifier на real-data: Acc **94.5%** в legacy-отчёте, локально воспроизведено **95.54%**.

**Domain gap synth ↔ real** (real RMSE ~50°): задокументирован, не баг. Решение — тема магистратуры (семестр 1-2).

---

## Быстрый старт (для разработчика)

```bash
cd E:/ML_works_folder/Array_Cascade_ITMO

# 1) Установить
pip install -r requirements.txt

# 2) Подготовить automotive baseline (если не сделано)
python prepare_data.py
python scripts/resume_baseline.py

# 3) Counter-UAV (quick demo или full train)
python scripts/generate_counter_uav_synth.py
python scripts/train_counter_uav.py --quick
# или:
python scripts/train_counter_uav.py --full

# 4) После Дня 6 — UI
python ui/app.py
# → http://localhost:8080
```

### Важно про артефакты

- ML-веса и `.npz` **не лежат в git** и должны быть собраны локально.
- Без `output/classifier.pth`, `output/doa_*.pth`, `output_counter_uav/classifier.pth`, `output_counter_uav/doa_siamese.pth` UI запустится, но inference уйдёт в `physics_fallback`.
- Для jury-demo это допустимый safety net, но не эквивалент live ML demo.

### Быстрый запуск demo на Windows

```bat
run_demo.bat
```

Дополнительно:

```bat
run_demo.bat smoke
run_demo.bat assets
```

---

## Для жюри ИТМО (5-минутный сценарий)

1. Открыть http://localhost:8080 — landing с диаграммой cascade и 4 KPI
2. `/demo` → automotive preset → live cascade на локально воспроизведённом baseline
3. `/demo` → counter-UAV preset → live demo на том же core через стабильный `Siamese` path (first full-train baseline)
4. `/methodology` → формальное доказательство переноса + `Run both domains`
5. `/about` → текст заявки

> Automotive baseline подтверждён локально; counter-UAV full-train уже прогнан, но в jury-demo по умолчанию показывается только самый стабильный путь, потому что остальной baseline пока ниже целевых AC.

**Оговорка по counter-UAV ML:** обновлённый full 10K baseline уже прогнан, но пока не проходит целевые AC-3/AC-4 (classifier 59.9%, Siamese DOA RMSE 4.49°, LISTA RMSE 63.36°). Последний итеративный прирост был получен после фикса synthetic generator, grid contract и classifier canonicalization. Это улучшение подтверждает, что проблема была не только в сети, но и в данных/контракте; при этом финальные AC всё ещё не закрыты.

Подробный сценарий: `docs/ITMO_application/jury_demo_script.md`.

---

## Технологический стек

| Слой | Выбор |
|---|---|
| ML | PyTorch (complex64 native, mixed precision) |
| Web | FastAPI + Jinja2 + Tailwind via CDN |
| Plots | Plotly (interactive, no node.js) |
| GPU | NVIDIA RTX 2060 SUPER, 8 GB VRAM |
| Python | 3.11+ |

---

## Roadmap магистратуры (2 года)

- **Семестр 1** — micro-Doppler ветка + валидация на DIAT-µSAT / RAD-DAR → arxiv tutorial
- **Семестр 2** — tracker (IMM + JPDA) → RadarConf / EuRAD paper
- **Семестр 3** — sensor fusion (RF + acoustic) → IEEE Sensors Journal / Drones (MDPI)
- **Семестр 4** — MVP сенсор-нода + защита + Q1 paper (IEEE TGRS)

Полный roadmap: `docs/ITMO_application/roadmap_2years.md`.

---

## Ссылки

- Источник кода: `C:/Users/User/Documents/LEMZ_project_trajectory/Shishanov_paper/` (legacy)
- Спецификация: [SPEC_ITMO_V2.md](SPEC_ITMO_V2.md)
- Чек-лист: [STAGES.md](STAGES.md)
- UI референс: `E:/ML_works_folder/AI_agent/Startup_Lens_mirror/`
