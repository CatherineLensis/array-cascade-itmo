# Заявка в магистратуру ИТМО — текст проекта

**Заявитель:** Лазько Екатерина  
**Программа:** Магистратура, направление «Искусственный интеллект / Прикладная математика и информатика» (уточняется по конкретному треку ИТМО)  
**Тема проекта:** Универсальный cascade для многоканальной обработки сигналов в условиях multipath: от автомобильного радара к counter-UAV  
**Рабочее репозиторий:** `E:/ML_works_folder/Array_Cascade_ITMO/`  
**Дата подготовки:** июнь 2026

---

## 1. Аннотация

Современные сенсорные системы — автомобильные радары ADAS, системы обнаружения малых БпЛА, indoor positioning, акустические массивы — сталкиваются с одной и той же фундаментальной проблемой: **multipath и режим single-snapshot** разрушают точность классических алгоритмов (MUSIC, ESPRIT, beamforming). Существующие нейросетевые решения, как правило, жёстко привязаны к конкретному сенсору и не переносятся между доменами.

Предлагаемый проект развивает **универсальную cascade-архитектуру** `Preprocess → Classify(scene) → Route → DOA → Track`, где домен-специфичной остаётся только физическая модель антенной решётки и генератор синтетических данных. Ядро ML (классификатор ComplexNet, четыре SOTA-метода DOA: Siamese, LISTA, CovRecon, HierGRU) переиспользуется без изменений.

**Демонстрация переносимости:** два прикладных домена на одном scaffold'е:
- **Automotive radar** — TI AWR1843, 77 GHz, 3 класса сцены (direct / multipath / two-targets), legacy-код и подтверждённый baseline;
- **Counter-UAV** — X-band 12.5 GHz, 4 класса (clutter / drone / bird / multi), micro-Doppler модели и urban multipath — новый applied-кейс спринта.

На синтетических данных automotive-домена лучший legacy-прогон (Siamese) достигает **MedAE 0.66°** при single-snapshot режиме с 8-антенной ULA — это ~76% от теоретической границы Крамера-Рао при SNR=20 dB и превосходит классический предел Рэлея (~29°) на порядок. Локально baseline воспроизведён по RMSE на synth-val: LISTA **5.23°**, Siamese **5.60°**. Классификатор на реальных данных показывает **Accuracy 95.54%** в локальном прогоне.

**Честная оговорка:** переход synth→real для DOA даёт RMSE ~50° — задокументированный domain gap, не архитектурный баг. Его закрытие — центральная задача первых двух семестров магистратуры.

---

## 2. Актуальность и мотивация

### 2.1. Научная актуальность

Single-snapshot DOA — один из наиболее жёстких режимов array processing: при M=1 невозможно усреднить ковариационную матрицу, и классические subspace-методы не работают. Нейросетевые подходы (physics-informed unfolding, siamese embeddings, covariance reconstruction) показывают SOTA-результаты, но почти всегда обучаются и оцениваются **внутри одного домена** без систематического анализа переносимости.

Универсальный cascade, где routing по классу сцены определяет, нужен ли DOA и сколько углов искать, — это **переносимый паттерн**, применимый к radar, acoustic, WiFi CSI, passive RF. Проект формализует этот паттерн и доказывает перенос на двух радикально разных доменах.

### 2.2. Прикладная актуальность (РФ-контекст 2026)

Counter-UAV — одна из наиболее востребованных прикладных тем: малые БпЛА (RCS 0.001–0.01 м²) в городской среде с жёстким multipath от зданий. Критична **дискриминация дрон vs птица vs clutter** — без неё растёт число ложных тревог (PFA). Проект фокусируется на **пассивном обнаружении и локализации**, без активного противодействия (jamming, kinetic) — этически и юридически безопасная постановка.

### 2.3. Личная мотивация

Проект вырос из совместной работы с А. Шишановым над FMCW MIMO-радаром TI AWR1843: реализован полный pipeline классификации + 4 DOA-метода, проведён детальный анализ качества и domain gap. Магистратура в ИТМО — логичное продолжение: превратить working prototype в **open framework** с публикациями Q1 и прикладным counter-UAV кейсом.

---

## 3. Цели и задачи

### 3.1. Цель магистерской диссертации

Разработать, формализовать и экспериментально подтвердить **универсальный cascade-framework** для multipath-rich array signals с прикладной валидацией на counter-UAV и публикацией результатов в рецензируемых журналах Q1.

### 3.2. Задачи спринта (до подачи заявки, ~2026-07-04)

| # | Задача | Статус на момент подачи |
|---|---|---|
| 1 | Подтвердить baseline automotive на synth-val | Выполнено: локальные DOA-веса и `BASELINE_CONFIRMED.md` |
| 2 | Рефакторинг `core/` + `domains/` | Частично: `Cascade.forward()` работает; полный перенос automotive в `domains/` не завершён |
| 3 | Counter-UAV синтетический генератор (micro-Doppler, urban multipath) | Выполнено: `synth_10k.npz`, physics tests pass |
| 4 | Обучение cascade на counter-UAV synth | Improved baseline after synthetic/data-contract fixes: classifier 59.9%, Siamese DOA RMSE 4.49°; AC-3/AC-4 пока не достигнуты |
| 5 | UI-демо для жюри (FastAPI, 6 страниц) | Live demo реализован: `/api/run_cascade`, `/api/run_both`, Plotly |
| 6 | Пакет документов для заявки | Черновик готов; синхронизация с фактическими метриками продолжается |

### 3.3. Задачи магистратуры (2 года, 4 семестра)

1. **Семестр 1:** micro-Doppler как отдельная модальность; валидация на DIAT-µSAT, RAD-DAR; закрытие domain gap synth→real; arxiv tutorial-paper.
2. **Семестр 2:** tracker (IMM + JPDA) поверх cascade; end-to-end бенчмарк drone/bird/clutter; conference paper (RadarConf / EuRAD 2027).
3. **Семестр 3:** sensor fusion (RF + acoustic array); сравнение late/early/end-to-end fusion; journal paper (IEEE Sensors Journal / Drones MDPI).
4. **Семестр 4:** MVP сенсор-нода (mic array + RTL-SDR + Raspberry Pi 5); live demo; Q1 paper (IEEE TGRS / Pattern Recognition); защита диссертации.

---

## 4. Научная новизна

1. **Формализация универсального cascade** с `DomainSpec` — единый интерфейс, подменяемый между доменами без изменения core ML.
2. **Two-domain proof:** automotive (77 GHz, 3 cls) + counter-UAV (12.5 GHz, 4 cls) на одном scaffold'е — эмпирическое доказательство переносимости, а не декларация.
3. **Micro-Doppler-aware генератор и постановка drone vs bird** в urban multipath — прикладной вклад; ML-классификатор сейчас является engineering baseline и требует улучшения в первом семестре.
4. **Честная документация domain gap** synth→real с планом его закрытия — методологическая зрелость, отличающая проект от «overclaim» заявок.

---

## 5. Методология

### 5.1. Cascade pipeline

```
Raw signal (8 complex channels, 1 snapshot)
    → Domain preprocessing
        automotive: Wiener-Hopf + RMS
        counter-UAV: raw steering-domain input (для текущих весов)
    → Scene classifier (ComplexNet, N classes)
    → Router (skip DOA for clutter/multipath-only scenes)
    → DOA estimation (4 methods in parallel)
    → Result: scene_class + angles + confidence
```

### 5.2. DOA-методы (SOTA, single-snapshot regime)

| Метод | Принцип | Сильная сторона |
|---|---|---|
| Siamese | Embedding + angular regression | Лучший MedAE (0.66° synth-val) |
| LISTA | Physics-informed unfolding (10 layers) | Лучший SR@3° (90.6%) |
| CovRecon | Learned covariance + Root-MUSIC | Интерпретируемость |
| HierGRU | Coarse-to-fine BiGRU | Потенциал при >120 эпох |

### 5.3. Counter-UAV физическая модель

- **Радар:** X-band 12.5 GHz, 8-элементная ULA, λ/2 шаг, дальность до 1 км.
- **Micro-Doppler:** 4-моторный квадрокоптер (дискретные узкие пики) vs птица (широкополосный wing-beat 3–8 Hz).
- **Urban multipath:** method of images, N_walls ∈ {0, 2, 4, 6}.
- **4 класса:** clutter / drone / bird / multi — type discrimination снижает PFA.

### 5.4. Обучение

- Pretrain на синтетике → finetune на real (automotive).
- Mixed precision (AMP) на NVIDIA RTX 2060 SUPER (8 GB VRAM).
- Метрики: RMSE, MedAE, SR@1°/3°/5°, confusion matrix, latency.

**Ограничение текущей counter-UAV оценки:** even после последних фиксов counter-UAV ML нельзя интерпретировать как финальный предел качества. Улучшение до 59.9% classifier accuracy и 4.49° Siamese RMSE показывает, что данные и контракт действительно влияли на качество, но AC-3/AC-4 пока всё равно не закрыты. Это engineering baseline и диагностический сигнал, а не окончательный научный результат.

---

## 6. Ожидаемые результаты

### К концу спринта (заявка)

- Рабочий UI-демо с двумя доменами (automotive + counter-UAV presets).
- Counter-UAV synth-генератор и первичные метрики cascade.
- Документированный baseline automotive (synth-val + honest domain gap writeup).
- Репозиторий с `core/`, `domains/`, `ui/`, `docs/`.

### К концу магистратуры

- Open-source framework с ≥2 доменами и tracker + fusion.
- 3–4 публикации (arxiv tutorial + conference + journal + Q1).
- MVP сенсор-нода с live demo.
- Магистерская диссертация.

---

## 7. Ресурсы и инфраструктура

| Ресурс | Наличие |
|---|---|
| GPU | NVIDIA RTX 2060 SUPER, 8 GB VRAM, CUDA 12.x |
| Legacy код + real data | TI AWR1843, ~67K сэмплов в `train_validate/` |
| Публичные датасеты (магистратура) | DIAT-µSAT, RAD-DAR |
| Сенсор-нода (семестр 4) | Mic array USB (~$50) + RTL-SDR/HackRF + Raspberry Pi 5 |

**Не требуется на этапе заявки:** покупка нового радара, edge-deploy на DSP, активное противодействие БпЛА.

---

## 8. Риски и их митигация

| Риск | Митигация |
|---|---|
| Domain gap synth→real не закрывается быстро | Запланирован на семестры 1–2; честно документирован с первого дня |
| Counter-UAV synth нефизичен | Юнит-тест: classical beamforming на одиночной цели ≤ 2° error |
| Refactor ломает legacy pipeline | Fallback: дубль кода, прогресс не блокируется |
| Overclaim метрик | Формулировка «MedAE 0.66° на synth-val», не «RMSE < 1° в общем виде» |

---

## 9. Связь с программой ИТМО

Проект пересекается с компетенциями ИТМО в области:
- машинного обучения для сигналов и изображений;
- embedded AI и edge inference (latency < 10 ms — proxy-метрика);
- прикладных систем безопасности (counter-UAV без active countermeasures).

Готовность к коллаборации с лабораториями ИТМО по computer vision, robotics, sensor networks.

---

## 10. Заключение

Проект сочетает **научную глубину** (универсальный cascade, single-snapshot DOA, physics-informed ML) и **прикладную актуальность** (counter-UAV в городе). Baseline на automotive synth уже подтверждён; domain gap synth→real честно зафиксирован как задача магистратуры, а не скрытый дефект. Спринт до подачи заявки создаёт working demo и counter-UAV proof-of-concept; два года магистратуры — путь к open framework, публикациям Q1 и MVP сенсор-ноде.

**Ключевое сообщение для жюри:** одна архитектура — два домена — ноль переписывания ядра. Это не маркетинговый слоган, а инженерный контракт `DomainSpec` + `Cascade`, который можно увидеть в live demo.

---

*Приложения: `motivation_commercial.md`, `roadmap_2years.md`, `gap_analysis.md`, `jury_demo_script.md`, `docs/ARCHITECTURE.md`, `docs/BASELINE_CONFIRMED.md`*
