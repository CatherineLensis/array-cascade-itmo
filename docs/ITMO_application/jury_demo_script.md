# Сценарий демонстрации для жюри ИТМО (5 минут)

**Проект:** Array Cascade ITMO  
**Формат:** live presentation + UI walkthrough  
**URL:** http://localhost:8080  
**Запуск:** `python ui/app.py` из корня репозитория  
**Дата:** июнь 2026

---

## Подготовка (до входа в аудиторию)

| # | Действие | Время |
|---|---|---|
| 1 | `cd E:/ML_works_folder/Array_Cascade_ITMO && python ui/app.py` | −5 мин |
| 2 | Открыть Chrome/Firefox: http://localhost:8080 | −3 мин |
| 3 | Smoke-test: пройти все 6 routes, убедиться в отсутствии console.error | −2 мин |
| 4 | Backup: видео 2 мин в `docs/ui_screenshots/` или offline MP4 | заранее |
| 5 | Разрешение экрана ≥1280×720, масштаб браузера 100% | −1 мин |

**Примечание на июнь 2026:** inference-сервис уже работает в live/fallback режиме. Automotive использует локально воспроизведённые веса; counter-UAV использует первый full-train baseline, но в live demo по умолчанию показывается только самый стабильный DOA path (`Siamese`). `LISTA` не выносится в основной live-показ, потому что его качество на counter-UAV пока ниже AC-3/AC-4. Это нужно проговаривать как research gap, а не скрывать.

---

## Тайминг (300 секунд)

```
0:00 ────── landing (/)
0:45 ────── live demo automotive (/demo)
2:00 ────── live demo counter-UAV (/demo)
2:45 ────── methodology (/methodology)
3:30 ────── counter-UAV applied (/counter_uav)
4:15 ────── roadmap (/roadmap) — опционально
4:45 ────── about (/about) + closing
5:00 ────── END
```

---

## Минутa 0:00–0:45 — Landing `/`

**URL:** http://localhost:8080/

**Что показать:**
- Заголовок: *«Один cascade, два домена, ноль переписывания»*
- 4 KPI-карточки: MedAE 0.66°, SR@1° 65.2%, latency <10 ms (target), 4 метода
- Диаграмма cascade: Raw → Whitening → Classify → DOA → Result

**Текст диктора (~30 сек):**

> «Добрый день. Это Array Cascade — универсальная архитектура для multipath-rich array signals. Один pipeline: предобработка, классификация сцены, routing и оценка направления прихода. Ядро ML одинаково для автомобильного радара 77 GHz и counter-UAV X-band. На синтетике лучший метод — MedAE 0.66 градуса при одном snapshot и восьми антеннах. Это в шесть раз ближе к границе Крамера-Рао, чем классический предел Рэлея в 29 градусов.»

**Действие:** кликнуть **Live demo** или перейти `/demo`.

---

## Минутa 0:45–2:00 — Demo automotive `/demo`

**URL:** http://localhost:8080/demo

**Что показать:**
- Selector domain: **Automotive**
- Preset **A1 — single direct target** (θ=+15°, SNR=20 dB, class direct)
- 5 шагов Plotly (signal → whitening → logits → DOA spectra → metrics)
- Статус backend: automotive live; counter-UAV live/fallback в зависимости от наличия артефактов

**Текст диктора (~40 сек):**

> «Automotive domain — радар TI AWR1843, три класса: прямой сигнал, multipath, две цели. Preset A1 — одна цель под плюс 15 градусов. Cascade сначала классифицирует сцену — ComplexNet на восьми комплексных каналах. Если сцена требует DOA, параллельно работают четыре метода: Siamese, LISTA, CovRecon, HierGRU. Siamese даёт лучший MedAE 0.66° на synth validation.»

**Действие:** переключить preset **A2 — multipath ghost** (θ=−10°, class multipath).

**Текст (~20 сек):**

> «A2 — multipath: прямой луч плюс переотражение. Классификатор на real data показывает 94.5% accuracy — routing работает. DOA на real — отдельная история: domain gap около 50° RMSE, мы это не скрываем, это задача первого семестра магистратуры.»

**Money moment M1 (если inference live):** side-by-side 4 DOA spectra на одном сигнале.

---

## Минутa 2:00–2:45 — Demo counter-UAV `/demo`

**URL:** http://localhost:8080/demo (domain = Counter-UAV)

**Что показать:**
- Preset **U1 — quadcopter DJI Mavic class** (θ=+30°, 4 motors, urban 4 walls)
- Preset **U2 — bird wing-beat 5 Hz** (θ=−20°)

**Текст диктора (~45 сек):**

> «Тот же cascade, другой DomainSpec: X-band 12.5 GHz, четыре класса — clutter, drone, bird, multi. Ключевое для counter-UAV — отличить дрон от птицы. После фикса synthetic generator и classifier-path обновлённый full-train baseline уже даёт classifier 59.9% и Siamese DOA RMSE 4.49°. Это всё ещё не закрывает AC-3/AC-4, но уже показывает, что главная проблема была не только в модели, а в корректности данных и train/inference contract. В live demo я показываю только самый стабильный counter-UAV path, чтобы не подменять исследовательский baseline ложной production-картинкой.»

**Money moment M4:** U1 vs U2 — контраст drone/bird; если live-графики не успевают, перейти на `/counter_uav` и проговорить физику micro-Doppler.

---

## Минутa 2:45–3:30 — Methodology `/methodology`

**URL:** http://localhost:8080/methodology

**Что показать:**
- Mermaid-диаграмма universal cascade
- Два DomainSpec рядом: automotive vs counter_uav
- Таблица «что переиспользуется на 100%»: preprocessing, classifier architecture, 4 DOA methods, metrics, cascade interface

**Текст диктора (~45 сек):**

> «Формальное доказательство V1. DomainSpec описывает физику: число антенн, wavelength, классы. Cascade class принимает любой DomainSpec. Меняется только domains/automotive или domains/counter_uav — steering matrix, synth generator. Core не трогаем. Правило для каждого решения: можно ли переиспользовать в acoustic или WiFi без переписывания? Если нет — код идёт в domains, не в core.»

---

## Минутa 3:30–4:15 — Counter-UAV applied `/counter_uav`

**URL:** http://localhost:8080/counter_uav

**Что показать:**
- Micro-Doppler спектрограммы: 4 примера (2 drone + 2 bird)
- Urban multipath beam-pattern visualization
- Confusion matrix drone/bird/clutter/multi (static или post-training)

**Текст диктора (~45 сек):**

> «Прикладной слой V3. Город, малый RCS, multipath от зданий. Без type discrimination растёт PFA — ложные тревоги на птиц. Мы сознательно не делаем jamming и kinetic countermeasures — только passive detection и localization. Партнёрский потенциал: интеграторы counter-UAV, ФПИ, отраслевые НИОКР — framework, не чёрный ящик.»

---

## Минутa 4:15–4:45 — Roadmap (опционально) `/roadmap`

**URL:** http://localhost:8080/roadmap

**Пропустить, если не хватает времени.** Иначе — 30 сек:

> «Четыре семестра: первый — закрытие domain gap и DIAT-µSAT; второй — tracker IMM+JPDA; третий — RF плюс acoustic fusion; четвёртый — MVP на Raspberry Pi и защита. Три–четыре публикации по пути.»

---

## Минутa 4:45–5:00 — About + closing `/about`

**URL:** http://localhost:8080/about

**Что показать:**
- HTML-рендер текста заявки (application_text.md)

**Closing (~15 сек):**

> «Резюме: одна cascade-архитектура, два домена, честные метрики, domain gap как research agenda. Готов ответить на вопросы. Спасибо.»

---

## Карта маршрутов UI

| Route | Страница | Роль в сценарии | Время |
|---|---|---|---|
| `/` | index.html | Hook + KPI | 0:00–0:45 |
| `/demo` | demo.html | Live cascade (M1, M3) | 0:45–2:45 |
| `/methodology` | methodology.html | V1 proof | 2:45–3:30 |
| `/counter_uav` | counter_uav.html | V3 applied (M4) | 3:30–4:15 |
| `/roadmap` | roadmap.html | 2-year plan (optional) | 4:15–4:45 |
| `/about` | about.html | Application text | 4:45–5:00 |
| `/health` | JSON | Smoke-test only | prep |

---

## Preset reference (для `/demo`)

| ID | Domain | Title | True angle | Class |
|---|---|---|---|---|
| a1 | automotive | Single direct | +15° | direct |
| a2 | automotive | Multipath ghost | −10° | multipath |
| a3 | automotive | Two targets | −25°, +40° | two_targets |
| u1 | counter_uav | Quadcopter | +30° | drone |
| u2 | counter_uav | Bird 5 Hz | −20° | bird |
| u3 | counter_uav | Drone + bird | +10°, −35° | multi |

---

## FAQ — ответы на likely questions

| Вопрос | Ответ (15–30 сек) |
|---|---|
| «51° RMSE — это провал?» | «Нет, это synth→real domain gap. На synth RMSE 4–5°. Все 4 метода падают одинаково на real — mismatch распределений, не баг одной сети. Sem 1 — закрываем.» |
| «Почему не купили радар?» | «Сначала методология и synth benchmark. Real validation — DIAT-µSAT, sem 1. MVP sensor node — sem 4, ~200 долларов BOM.» |
| «Чем лучше MUSIC/ESPRIT?» | «При M=1 snapshot классические subspace-методы не работают — нет устойчивой covariance. Мы в hardest regime.» |
| «Production ready?» | «Research framework + demo. Automotive baseline подтверждён локально: classifier 95.54%, synth-val DOA ~5°. Counter-UAV ML — first baseline, не production.» |
| «Jamming?» | «Explicitly out of scope. Passive sensing only — этика и регуляторика.» |

---

## Fallback plan

| Ситуация | Действие |
|---|---|
| Нет интернета (CDN Tailwind/Plotly) | Offline video + screenshots из `docs/ui_screenshots/` |
| `python ui/app.py` crash | Показать PROJECT_README.md + ARCHITECTURE.md + gap_analysis |
| Inference timeout | Physics fallback + verbal walkthrough presets |
| Жюри просит код | Открыть `core/cascade.py` DomainSpec.automotive() vs counter_uav() |

---

## Checklist после demo

- [ ] `/health` → `{"status":"ok"}`
- [ ] Все 6 страниц без 500 error
- [ ] Screenshot каждой страницы сохранён
- [ ] 2-min backup video записан
