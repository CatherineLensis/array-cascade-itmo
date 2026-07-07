# Roadmap магистратуры — 2 года, 4 семестра

**Проект:** Array Cascade ITMO  
**Период:** осень 2026 — весна 2028 (2026/27 + 2027/28 учебные годы)  
**Источник:** SPEC_ITMO_V2.md §10, адаптировано с указанием зависимостей и статуса на июнь 2026

---

## Обзор

```
2026 Q3-Q4          2027 Q1-Q2          2027 Q3-Q4          2028 Q1-Q2
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Семестр 1   │    │ Семестр 2   │    │ Семестр 3   │    │ Семестр 4   │
│ Synth→Real  │───▶│ Tracker     │───▶│ Fusion      │───▶│ MVP + Q1    │
│ µSAT/RAD-DAR│    │ RadarConf   │    │ Journal     │    │ Диссертация │
│ arxiv tut.  │    │ EuRAD 2027  │    │ Sensors/    │    │ Live demo   │
│             │    │             │    │ Drones      │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      ▲
      │ Спринт заявки (июнь–июль 2026):
      │ baseline + counter-UAV synth + UI demo
```

**Предпосылка спринта (до осени 2026):** automotive baseline локально воспроизведён; counter-UAV synth-генератор и первичное обучение реализованы; UI-демо работает; `core/cascade.py` выполняет live/fallback inference. Открытый технический долг: counter-UAV ML ниже AC-3/AC-4, screenshots/video backup не готовы, полный перенос automotive в `domains/` не завершён.

---

## Семестр 1 — осень 2026

**Тема:** Закрытие domain gap и валидация на публичных данных

### Цели

1. Систематически диагностировать и снижать synth→real gap для automotive DOA.
2. Добавить micro-Doppler как отдельную модальность (signal + spectrogram fusion).
3. Валидировать counter-UAV pipeline на публичных датасетах DIAT-µSAT и RAD-DAR.
4. Опубликовать arxiv tutorial-paper по universal cascade framework.

### Задачи по месяцам

| Месяц | Задача | Deliverable |
|---|---|---|
| Сент | Аудит domain gap: угловое распределение, class balance, SNR, phase | `docs/domain_gap_report_v1.md` |
| Сент–Окт | Matching angular distribution synth→real; weighted sampling | Улучшение real RMSE DOA (цель: < 20°) |
| Окт | Micro-Doppler 2D-CNN branch + late fusion с spatial DOA | Ablation study |
| Нояб | DIAT-µSAT ingest + adapter DomainSpec | Benchmark table |
| Нояб–Дек | RAD-DAR cross-validation | Comparison synth vs public real |
| Дек | Tutorial paper draft | arxiv submission |

### Критерии успеха

- [ ] Real DOA RMSE снижен минимум на 30% vs baseline 51° (или задокументированы блокеры).
- [ ] Counter-UAV: результаты на ≥1 публичном датасете опубликованы в репо.
- [ ] arxiv tutorial принят (preprint online).

### Риски

| Риск | Митигация |
|---|---|
| DIAT/RAD-DAR format mismatch | Adapter layer в `domains/counter_uav/` |
| Gap не снижается | Pivot на semi-supervised / domain adaptation paper |

---

## Семестр 2 — весна 2027

**Тема:** Tracker и end-to-end бенчмарк

### Цели

1. Реализовать tracker (IMM + JPDA) поверх cascade outputs.
2. End-to-end бенчмарк: drone vs bird vs clutter с dwell time analysis.
3. Real-data experiments на automotive (если gap частично закрыт).
4. Conference paper: **IEEE RadarConf 2027** или **EuRAD 2027**.

### Задачи

| Блок | Содержание |
|---|---|
| **Tracker** | IMM (Interacting Multiple Model) для maneuvering targets; JPDA для data association в multi-target |
| **Integration** | Cascade output (angles + confidence) → tracker input; latency budget |
| **Benchmark** | Единый scaffold: automotive + counter-UAV + (optional) DIAT subset |
| **Dwell time** | Анализ: сколько CPI нужно для stable track при single-snapshot DOA |
| **Paper** | 6–8 стр., focus на cascade+tracker architecture |

### Критерии успеха

- [ ] Tracker работает на synth sequences (≥100 trajectories).
- [ ] MOTA/MOTP или аналоги для counter-UAV synth benchmark.
- [ ] Conference submission до дедлайна RadarConf/EuRAD.

### Зависимости от семестра 1

- Стабильные DOA predictions (real или public real) для feeding tracker.
- Counter-UAV classifier accuracy ≥85% на synth-val (AC-3 спринта) — пока не достигнуто, переносится в sem 1 как приоритетная engineering/research задача.

---

## Семестр 3 — осень 2027

**Тема:** Multi-sensor fusion

### Цели

1. Добавить acoustic array как вторую модальность.
2. Сравнить late fusion vs early fusion vs end-to-end multimodal transformer.
3. Journal paper: **IEEE Sensors Journal** или **Drones (MDPI)**.

### Задачи

| Блок | Содержание |
|---|---|
| **Hardware** | USB mic array (~$50), синхронизация с RF (software timestamp) |
| **Acoustic DOA** | Адаптация Cascade + DomainSpec для acoustic ULA |
| **Fusion strategies** | Late (decision-level), early (feature-level), E2E transformer |
| **Urban scenario** | RF weak + acoustic strong / vice versa — complementarity |
| **Paper** | Journal format, fusion ablation + counter-UAV case study |

### Критерии успеха

- [ ] Fusion улучшает drone/bird discrimination vs RF-only на ≥5% F1.
- [ ] Reproducible fusion pipeline в репо.
- [ ] Journal submission.

### Ограничения

- Нет claim «production fusion product» — research prototype.
- Синхронизация RF↔acoustic — software-level, не hardware trigger (sem 4 upgrade).

---

## Семестр 4 — весна 2028 (диплом)

**Тема:** MVP сенсор-нода, Q1 paper, защита

### Цели

1. MVP сенсор-нода: mic array + RTL-SDR/HackRF + Raspberry Pi 5.
2. Live demo: дрон в комнате → detection + localization → video recording.
3. Q1 paper submission: **IEEE TGRS** или **Pattern Recognition**.
4. Защита магистерской диссертации.

### MVP спецификация

| Компонент | Выбор | Стоимость (ориентир) |
|---|---|---|
| Compute | Raspberry Pi 5 (8 GB) | ~$80 |
| RF | RTL-SDR v3 или HackRF One | ~$30–350 |
| Acoustic | ReSpeaker 4-Mic Array USB | ~$50 |
| Optional radar frontend | Partner-provided или SDR-only demo | TBD |

**Demo script:** indoor quadcopter (≤100 g, caged) → cascade + tracker → azimuth estimate → overlay on video.

### Критерии успеха

- [ ] Live demo работает ≥3 мин без сбоев (rehearsal ≥10 раз).
- [ ] Q1 paper submitted (acceptance — bonus, не blocker защиты).
- [ ] Диссертация сдана, защита пройдена.

---

## Сводная таблица deliverables

| Семестр | Научный output | Инженерный output | Коммерческий след |
|---|---|---|---|
| **1** | arxiv tutorial | Domain gap fixes, DIAT/RAD-DAR adapters | Credibility для B2B |
| **2** | RadarConf / EuRAD | IMM+JPDA tracker, E2E benchmark | Demo-ready pipeline |
| **3** | Sensors / Drones journal | RF+acoustic fusion | Differentiated vs RF-only |
| **4** | IEEE TGRS / PR submission | MVP sensor node | Pilot case study |

---

## Ресурсы по семестрам

| Ресурс | Sem 1 | Sem 2 | Sem 3 | Sem 4 |
|---|---|---|---|---|
| GPU (RTX 2060 SUPER) | ✓ train | ✓ train | ✓ train | ✓ inference |
| TI AWR1843 real data | ✓ gap work | ✓ optional | — | — |
| DIAT/RAD-DAR | ✓ download | ✓ benchmark | ✓ fusion eval | — |
| Mic array + SDR | — | — | ✓ acquire | ✓ MVP |
| Raspberry Pi 5 | — | — | optional | ✓ required |

---

## Что сознательно вне roadmap

- Active counter-UAV (jamming, kinetic)
- TI C66x DSP / TFLite edge deploy как главный вклад
- Knowledge distillation как core contribution
- Покупка дорогого radar frontend (IWR6843AOP и т.п.) без grant funding

---

## Связь со спринтом заявки (июнь–июль 2026)

| Спринт deliverable | Как поддерживает sem 1 |
|---|---|
| `core/cascade.py` + DomainSpec | Foundation для всех семестров |
| Counter-UAV synth generator | Training data для sem 1–2 benchmarks |
| UI demo | Communication tool для advisor/partners |
| `gap_analysis.md` | Honest baseline для sem 1 gap work |
| BASELINE_CONFIRMED | Starting point metrics |

**Статус на июнь 2026:** automotive baseline подтверждён; UI live demo готов; `domains/counter_uav/` создан; counter-UAV full 10K baseline после фикса synthetic generator и classifier-path улучшен, но качество всё ещё ниже целевых AC. Следующий фокус — дальше улучшать drone/bird discrimination, multi-target semantics и packaging для показа.
