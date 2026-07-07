# Коммерческая мотивация и B2B-контекст

**Проект:** Array Cascade ITMO — Universal Array Signal Cascade + Counter-UAV  
**Автор:** Лазько Екатерина  
**Дата:** июнь 2026

> Документ описывает прикладной и коммерческий потенциал **без привязки к конкретному чипу или hardware SKU**. Фокус — методология и software framework, лицензируемый или внедряемый партнёрами с собственным железом.

---

## 1. Проблема рынка

### 1.1. Counter-UAV (основной B2B-кейс)

| Факт | Источник / контекст |
|---|---|
| Рынок counter-drone растёт двузначными темпами | Глобальный тренд 2024–2028; в РФ — ускорение после 2022 |
| Городская среда — worst case для радара | Multipath от зданий, низкий RCS (0.001–0.01 м²), дрон ≈ птица по допpler-профилю |
| Ложные тревоги (PFA) — главная боль заказчика | Без type discrimination (drone/bird/clutter) система не масштабируется |
| Active countermeasures (jamming, kinetic) — регуляторный риск | Пассивное RF-обнаружение + fusion — «безопасная» продуктовая ниша |

**Наш вклад:** cascade с 4-классовой классификацией + DOA снижает PFA за счёт routing (clutter → skip DOA) и micro-Doppler-aware discrimination (drone vs bird).

### 1.2. Automotive ADAS (legacy-домен, вторичный B2B)

- TI AWR1843 / аналоги — стандарт de facto для short-range automotive radar.
- Multipath от дороги и соседних машин — та же cascade-логика (direct / multipath / two-targets).
- **Ограничение:** domain gap synth→real для DOA (~50° RMSE) — продуктовое внедрение требует закрытия gap (магистратура, семестры 1–2). Классификатор (legacy 94.5%, local repro 95.54% Acc on real) уже пригоден для routing.

---

## 2. Ценностное предложение (Value Proposition)

```
Для интегратора counter-UAV / ADAS,
которому нужна точная локализация в multipath без переписывания ML под каждый сенсор,

Array Cascade — это software framework,
который даёт единый pipeline Preprocess → Classify → DOA
с подменой только физической модели (DomainSpec).

В отличие от monolithic SOTA-моделей «под один радар»,
наш cascade переносится между доменами (automotive ↔ counter-UAV)
и масштабируется на acoustic / WiFi (roadmap семестр 3).
```

### Ключевые метрики для pitch (честные формулировки)

| Метрика | Значение | Контекст |
|---|---|---|
| MedAE DOA | **0.66°** | Synth-val, Siamese, single-snapshot, 8 ant |
| SR@1° | **65.2%** | Synth-val |
| Classifier Acc (real) | **95.54%** | Automotive, TI AWR1843 real data, local repro |
| Cascade latency (target) | **< 10 ms** | Batch=1, RTX 2060 SUPER; edge — магистратура |
| Число DOA-методов | **4** | Параллельное сравнение, ансамбль — roadmap |

**Не заявляем:** «RMSE < 1° на real data», «production-ready counter-UAV», «готовый продукт».

---

## 3. Целевые сегменты и use cases

### 3.1. B2B-сегменты

| Сегмент | Use case | Наш fit |
|---|---|---|
| **Counter-UAV интеграторы** | Аэропорты, стадионы, критическая инфраструктура | Drone/bird/clutter + DOA, urban multipath |
| **Оборонно-промышленные НИОКР** | Пассивное RF-обнаружение малых целей | Framework + synth benchmark, без jamming |
| **Automotive Tier-1/2** | ADAS multipath routing | Legacy baseline, classifier ready |
| **Smart city / perimeter** | Мультисенсорная охрана периметра | Fusion roadmap (RF + acoustic) |
| **Research labs / вузы** | Обучение array processing + ML | Open framework, reproducible benchmarks |

### 3.2. B2G (госзаказ, гранты)

| Источник | Релевантность |
|---|---|
| **ФПИ** (Фонд перспективных исследований) | ML для сигналов, counter-UAV, sensor fusion |
| **Сколтех, ЦКР** | Applied R&D, hardware-software co-design |
| **ЛЭТИ / Almaz-Antey ecosystem** | Radar signal processing, defense-adjacent R&D |
| **Кronstadt / Геоскан ecosystem** | Drone detection, не kinetic countermeasures |

*Конкретные заявки — после магистратуры и MVP demo; на этапе заявки в ИТМО — указание траектории, не обещание контрактов.*

---

## 4. Потенциальные партнёры (для коллаборации)

| Организация | Тип партнёрства | Что даём | Что получаем |
|---|---|---|---|
| **ИТМО** (лаборатории CV/robotics) | Научное руководство, co-authorship | Framework, публикации | Инфраструктура, экспертиза |
| **ЛЭТИ** | Radar lab, validation data | DOA methods, cascade | Real radar data, peer review |
| **DIAT / RAD-DAR authors** | Dataset access, benchmark | Reproducible pipeline | Public validation, citations |
| **Kronstadt, ZALA Aero** | Applied feedback (не NDAs на спринте) | Synth benchmark methodology | Domain requirements |
| **RTL-SDR / HackRF community** | Low-cost hardware path | Fusion MVP (sem 4) | User base, open hardware |

**Принцип:** партнёрства строятся вокруг **методологии и benchmark**, не вокруг продажи «чёрного ящика» без воспроизводимости.

---

## 5. Конкурентный ландшафт

| Подход | Примеры | Слабость | Наше отличие |
|---|---|---|---|
| Classical DOA (MUSIC, ESPRIT) | Embedded radar stacks | M=1 snapshot — не работает | 4 NN-метода, обучены на synth |
| Monolithic end-to-end NN | Per-sensor custom models | Нет переноса между доменами | DomainSpec + universal Cascade |
| Optical-only drone detection | CV cameras | Night, fog, small RCS | RF + micro-Doppler complement |
| Acoustic-only | Mic arrays | Range, wind, urban noise | RF primary; acoustic fusion — sem 3 |
| «Black box» commercial C-UAS | Various vendors | Непрозрачность, vendor lock | Open framework, honest metrics |

---

## 6. Модели монетизации (post-магистратура)

### 6.1. Краткосрочные (0–12 мес после защиты)

1. **Consulting / integration** — адаптация DomainSpec под сенсор заказчика, обучение cascade на их synth/real mix.
2. **Benchmark-as-a-Service** — воспроизводимый counter-UAV synth benchmark для сравнения алгоритмов заказчика с нашим baseline.
3. **Grant-funded R&D** — ФПИ, Сколтех, отраслевые НИОКР.

### 6.2. Среднесрочные (12–24 мес)

1. **Open-core licensing** — core cascade MIT/Apache; domain packs (counter-UAV, automotive) — commercial license.
2. **Sensor node kit** — Raspberry Pi + mic array + SDR + pre-trained models (~$200–500 BOM).
3. **Publication-driven leads** — RadarConf, IEEE Sensors → inbound от интеграторов.

### 6.3. Что сознательно не monetize в спринте

- Продажа «готовой системы counter-UAV» — нет real validation, нет сертификации.
- Edge deploy на TI C66x DSP — вне scope, инженерия > наука для магистратуры.
- Active countermeasures — этически/юридически исключены.

---

## 7. ROI для партнёра-интегратора (illustrative)

**Сценарий:** интегратор counter-UAV имеет X-band radar frontend, нужен software stack для detection + DOA.

| Без Array Cascade | С Array Cascade |
|---|---|
| 12–18 мес разработки custom ML под один радар | 3–6 мес адаптация DomainSpec + finetune |
| Переписывание при смене частоты/антенн | Подмена `physics_*.py`, core без изменений |
| Нет публичного benchmark | Synth benchmark + DIAT/RAD-DAR validation (roadmap) |
| Риск overclaim без domain gap analysis | Documented gap + migration path |

*Цифры — порядок величин для pitch deck, не contractual SLA.*

---

## 8. Регуляторный и этический контекст

- **Только passive sensing:** detection, classification, localization, tracking.
- **Без:** jamming, GPS spoofing, RF poisoning, kinetic interception — явно вне scope проекта и заявки.
- **GDPR / персональные данные:** RF pipeline не обрабатывает PII; acoustic fusion (sem 3) — локальная обработка, без cloud по умолчанию.
- **Export control:** software framework; партнёры сами оценивают compliance для своего hardware.

---

## 9. Связь B2B ↔ магистерская программа

| Семестр | B2B-deliverable |
|---|---|
| 1 | Public dataset validation → credibility для integrators |
| 2 | Tracker + end-to-end benchmark → demo-ready pipeline |
| 3 | Multi-sensor fusion → differentiated product vs RF-only |
| 4 | MVP sensor node → pilot с партнёром, case study |

Магистратура — не «отложенный стартап», а **научно обоснованный путь к воспроизводимому B2B-активу** с публикациями и open framework.

---

## 10. Резюме для жюри

1. **Горячий рынок:** counter-UAV в РФ/EU с фокусом на passive RF + type discrimination.
2. **Честный продукт:** framework, не «волшебная коробка»; метрики с контекстом (synth vs real).
3. **Партнёрский потенциал:** ИТМО, ЛЭТИ, DIAT, отраслевые НИОКР — без преждевременных обещаний контрактов.
4. **Масштабируемость:** один cascade → automotive, counter-UAV, далее acoustic/WiFi.
5. **Этика:** detection only, без active countermeasures.
