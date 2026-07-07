# Jury UI Design — Array Cascade ITMO

> **Назначение.** Детальный дизайн UI (Дни 6–7 спринта), который выступает витриной проекта для приёмной комиссии магистратуры ИТМО. Документ — результат brainstorming-сессии 2026-06-24, продолжает `SPEC_ITMO_V2.md` §16 (концептуальный уровень) и заземляет `STAGES.md` дни 6–7 (операционный уровень).
>
> **Статус.** Approved через 4 confirmation-gates (Architecture / Pages / Data Flow / Testing). Следующий шаг — invoke writing-plans skill для разбиения на task'и реализации.

---

## 0. Журнал решений brainstorming-сессии (2026-06-24)

| # | Вопрос | Выбрано | Обоснование |
|---|---|---|---|
| 1 | Режим демонстрации | **C — asymmetric hybrid** (A-flow live + self-drive harness) | Реальный формат ИТМО двойной: материалы заявки + интервью. UI обслуживает оба, при этом основной поток — live presentation |
| 2 | One big takeaway | **A primary + B co-variant** (universal cascade spear + counter-UAV co-variant) | ИТМО — про научный потенциал. Жюри предпочтёт «думатель архитектур» над «бенчмаркером». V3 защищает от обвинения «абстрактно» |
| 3 | Money moments | **M1 + M3 + M4** | M1 visual proof V1, M3 интерактивная глубина методов, M4 applied wow. M2 (code diff) отвергнут как «требует от жюри читать код» |
| Approach | Общая композиция UI | **A2 + teaser** (six-page narrative с teaser M1 на landing) | Максимальное переиспользование scaffold + минимальный risk surface + семантически правильное размещение money moments |

---

## 1. Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Chrome/Firefox)                 │
│              Tailwind CDN + Alpine.js + Plotly.js               │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP localhost:8080
┌─────────────────────────────▼───────────────────────────────────┐
│                       ui/app.py (FastAPI, async)                │
│   Routes: / · /demo · /methodology · /counter_uav ·             │
│           /roadmap · /about · /api/run_cascade ·                │
│           /api/run_both · /health                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │ await run_in_threadpool(...)
┌─────────────────────────────▼───────────────────────────────────┐
│                       ui/services/ (sync, torch)                │
│   inference.py  ── run_cascade(signal, domain) → ResultDTO      │
│   presets.py    ── 6 preset signals (3 auto + 3 c-UAV)          │
│   plots.py      ── Plotly figure builders → dict for template   │
│   cache.py      ── LRU(maxsize=64) for preset×SNR×domain        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       core/ + domains/                          │
│   core.cascade.Cascade(DomainSpec, models)                      │
│   models loaded ONCE in FastAPI lifespan → MODELS dict          │
└─────────────────────────────────────────────────────────────────┘
```

### Ключевые архитектурные решения

| # | Решение | Обоснование |
|---|---|---|
| A1 | **Sync inference в async-сервере** через `run_in_threadpool` | torch — sync; FastAPI — async; single-user demo не требует Celery/RQ |
| A2 | **Eager model loading в lifespan** (~3–5 сек при старте) | жюри не ждёт первого inference |
| A3 | **LRU(64) кеш на `(preset_id, snr_rounded, domain)`** | повторные клики — instant; ~3 MB RAM суммарно |
| A4 | **Plotly через server-side `to_json()` → browser `Plotly.react`** | zero-build, нет node.js, нет webpack |
| A5 | **Flat pages topology** (нет sub-routes) | self-drive trivially работает; навбар везде одинаковый |
| A6 | **Стейтлесс backend** (нет session/DB/auth) | перезагрузка = свежий запрос |
| A7 | **Localhost only, без CORS** | если нужен внешний доступ — `ssh -L` или ngrok руками |

### Архитектурные риски и митигация

| Риск | Митигация |
|---|---|
| Torch+Plotly server render одновременно тормозят на 2060 SUPER | Day 6 smoke-тест измерит latency; fallback — pre-computed static Plotly для M1 |
| Lifespan startup > 5 сек | Eager loading логирует время каждой модели; если > 5 сек — параллельная загрузка через `asyncio.gather` |
| Cache poisoning одинаковыми ключами разных вычислений | Key включает `(preset_id, snr_rounded:int, domain)` + versioning через model checksum |

---

## 2. Page-by-page contents

| Route | Роль | Money moment | Содержимое |
|---|---|---|---|
| `/` | landing — first impression + tour entry | **M1 teaser** (статичный) | Hero «one cascade, two domains»; **2 мини-card бок-о-бок** с pre-rendered результатом cascade (Siamese spectrum на automotive | counter-UAV); подпись «see live in Methodology →»; 4 KPI cards (MedAE 0.66°, SR@1° 65.2%, latency <10 ms, n_methods 4); 3 CTA (Demo/Methodology/Counter-UAV); **«5-min tour →»** кнопка запускает self-drive |
| `/methodology` | V1 spear page | **M1 live** | Universal cascade ASCII-диаграмма; два DomainSpec code-блока side-by-side; **большая кнопка «Run cascade on both domains»** → 2 параллельные колонки с 5 шагами; tooltip «click — same cascade runs on both signals»; таблица переиспользования core/ vs domains/ |
| `/demo` | методы deep dive | **M3 SNR slider** | Selector domain → preset → SNR-слайдер (−5..30 dB); single-domain run; Plotly 5 шагов cascade; **4 DOA spectra на одном графике с overlay** (видно, как Siamese/LISTA держат пик, CovRecon разваливается при низком SNR); tooltip «move the slider to see physics-informed superiority» |
| `/counter_uav` | V3 applied page | **M4 drone vs bird** | Problem framing; 4 классов cards (clutter/drone/bird/multi); **side-by-side**: STFT drone квадрокоптера vs bird wing-beat + classifier confidence bars + caption «discrete blade peaks vs broadband wing-beat»; urban multipath beam pattern; 4×4 confusion matrix на synth-val |
| `/roadmap` | план магистратуры | — | 4 семестра timeline (Plotly Gantt или vertical timeline); deliverables на каждом семестре; партнёры (академические + индустриальные) |
| `/about` | заявка ИТМО | — | Рендеренный markdown `docs/ITMO_application/application_text.md`; блок «предшествующая работа» с baseline-числами; ссылки на репо |

### Live presentation timing (5 минут)

| Время | Страница | Действие | Доминирующий message |
|---|---|---|---|
| 0:00 – 0:30 | `/` | Pitch + KPIs | «универсальный, превосходит классику» |
| 0:30 – 2:00 | `/methodology` | Click «Run both» → M1 parallel cascade | **V1 wow** |
| 2:00 – 3:30 | `/demo` | Slider SNR −5..25 dB | методы в действии |
| 3:30 – 4:30 | `/counter_uav` | M4 drone vs bird | **V3 wow** |
| 4:30 – 5:00 | `/about` | Q&A на тексте заявки | applied плюс публикации |

### Self-drive UX hooks

1. Каждая интерактивная зона имеет `title="..."` tooltip
2. Кнопка **«Reset to default»** на `/demo`
3. **Sticky next-page hint** в правом нижнем углу: «next: Demo →»
4. На landing — **«5-min tour»** кнопка, ведёт по 4-странице sequence с highlights (state в `localStorage`)
5. Default preset выбран при загрузке `/demo` — пустого состояния нет
6. Все кнопки **disabled во время inference + spinner** — спам-клик не ломает

---

## 3. Data flow & inference

### Базовый request lifecycle (`/demo` single cascade)

```
Browser ─ POST /api/run_cascade {preset_id, domain, snr} ─► FastAPI
                                                              │
                                                  ┌───────────┴────────────┐
                                                  │ cache.get(key)?        │
                                                  │   hit  → return DTO    │
                                                  │   miss ↓               │
                                                  └───────────┬────────────┘
                                                              │
                                                  await run_in_threadpool:
                                                     presets.get_signal(preset_id)
                                                     regenerate at requested SNR
                                                     cascade.forward(signal)
                                                       • Wiener-Hopf + RMS
                                                       • classifier (n_classes)
                                                       • 4 DOA methods
                                                     plots.build_figures(...) → 5 dicts
                                                     wrap into ResultDTO
                                                  cache.put(key, DTO)
                                                              │
Browser ◄─ 200 JSON {plotly figs, metrics} ─────── FastAPI
   │
   └─ Plotly.react(divs, fig.data, fig.layout)   # inplace update
```

### Спецслучаи money moments

**M1 (`/methodology` — Run both domains):**
- POST `/api/run_both` → server делает `asyncio.gather(run_cascade(a_sig, "automotive"), run_cascade(u_sig, "counter_uav"))`
- Возвращает `{automotive: ResultDTO, counter_uav: ResultDTO}`
- Browser рендерит 2 колонки. Обе видны одновременно — V1 visual proof.

**M3 (`/demo` — SNR slider):**
- Alpine.js `@change.debounce.200ms`
- Каждое движение → POST `/api/run_cascade` с новым snr
- Browser `Plotly.react(div, fig.data, fig.layout)` — обновление **inplace**, без перезагрузки

**M4 (`/counter_uav`):**
- **Полностью pre-computed при старте app в lifespan**: 4 примера drone + 4 bird, STFT-спектрограммы, Plotly figures, сохраняются в `MODELS["c_uav_examples"]`
- Confusion matrix pre-computed один раз после загрузки моделей на synth-val
- Страница рендерит из памяти, нулевой inference на запрос

### Контракт ResultDTO

```python
{
  "domain": "automotive" | "counter_uav",
  "preset_id": "a1",
  "true_angle_deg": 15.0,            # null для multi-target
  "scene_label": "direct",
  "step1_raw_fig":      {plotly data/layout},
  "step2_white_fig":    {plotly data/layout},
  "step3_class_fig":    {plotly data/layout},   # bar chart logits
  "step4_doa_fig":      {plotly data/layout},   # 4 spectra overlay
  "step5_metrics": {
    "best_method": "siamese",
    "best_pred_deg": 14.87,
    "error_deg": 0.13,
    "latency_ms": 142
  }
}
```

### Latency budget (2060 SUPER, целевая)

| Этап | Бюджет |
|---|---|
| Preset signal regen (new SNR) | < 30 мс |
| Wiener-Hopf + RMS | < 10 мс |
| Classifier forward | < 5 мс |
| 4 DOA forwards (последовательно) | < 30 мс |
| Plotly figures build (5 шт) | < 50 мс |
| JSON serialize + network localhost | < 30 мс |
| **End-to-end (cache miss)** | **< 200 мс** |
| End-to-end (cache hit) | < 15 мс |

Соответствует UI-AC-2 (< 500 мс) с запасом 2.5×.

### Error handling

1. **Cascade throws** → catch в endpoint → 500 + `{"error": "Cascade failed", "detail": "..."}`. Browser показывает Tailwind toast «Что-то пошло не так. Попробуй Reset to default»; страница не падает.
2. **Inference > 2 сек** → AbortController на fetch + кнопка Cancel.
3. **Модели не загрузились** (lifespan exception) → `/health` возвращает `{"status": "models_failed"}`, баннер «Demo mode degraded», все интерактивные кнопки disabled, графики из `static/plotly_charts/*.json`.

---

## 4. Testing strategy

### Уровни тестирования

| Уровень | Что | Когда | Инструмент |
|---|---|---|---|
| Unit | сервисы (presets, plots, cache, inference) | каждый коммит | pytest |
| Smoke auto | App стартует, все 6 страниц → 200 | каждый коммит | pytest + httpx |
| Integration | M1/M3/M4 endpoints возвращают валидный DTO | дни 6–7 | pytest + httpx |
| Visual / browser | Plotly рендерится, нет console.error | вручную, дни 6–7 | Chrome DevTools |
| Jury-rehearsal | 5-минутный сценарий без затыков | день 10 | секундомер + чек-лист |

### Unit-тесты — что покрываем

```
tests/ui/
├── test_presets.py        # 6 preset: правильный shape, true_angle ∈ [-90,90]
├── test_inference.py      # run_cascade на каждом preset → ResultDTO с обязательными ключами
├── test_plots.py          # build_*_fig → dict с ["data", "layout"]
├── test_cache.py          # LRU eviction, key roundtrip
└── test_routes.py         # 6 GET → 200, /api/run_cascade → валидный JSON
```

Минимум 1 unit-тест на функцию. Не покрываем: template rendering (визуальное), Plotly корректность (доверяем библиотеке), Alpine.js (визуальное).

### Smoke-тест

```python
def test_app_starts(): ...                           # lifespan не падает
def test_all_pages_200(client):
    for route in ["/", "/demo", "/methodology",
                  "/counter_uav", "/roadmap", "/about", "/health"]:
        assert client.get(route).status_code == 200
def test_run_cascade_each_preset(client):
    for pid in PRESETS:
        r = client.post("/api/run_cascade", json={"preset_id": pid, "snr": 15})
        assert r.status_code == 200
        assert "step4_doa_fig" in r.json()
def test_run_both_domains(client):
    r = client.post("/api/run_both", json={"snr": 15})
    body = r.json()
    assert "automotive" in body and "counter_uav" in body
def test_models_loaded(client):
    assert client.get("/health").json()["models"] != "stub"
```

### Mapping на acceptance criteria (SPEC §16.6)

| AC | Как проверяем |
|---|---|
| UI-AC-1: старт < 5 сек | `time python ui/app.py` + ручной таймер до первого GET / |
| UI-AC-2: inference < 500 мс | smoke-тест с `time.perf_counter` |
| UI-AC-3: открывается без console.error | вручную, день 7 |
| UI-AC-4: 5-мин jury-сценарий | репетиция по чек-листу `docs/ITMO_application/jury_demo_script.md` |
| UI-AC-5: screenshots в `docs/ui_screenshots/` | вручную, день 7 |

### Jury-rehearsal чек-лист (день 10)

```
[ ] Холодный старт: python ui/app.py → http://localhost:8080 за ≤ 15 сек

[ ] Live flow 5 мин:
    [ ] 0:00–0:30 / — KPI читаемы, teaser виден без скролла
    [ ] 0:30–2:00 /methodology — Run M1 → результат ≤ 1 сек, 2 колонки одновременно
    [ ] 2:00–3:30 /demo — slider −5..25 dB, 4 спектра обновляются inplace
    [ ] 3:30–4:30 /counter_uav — 4 примера drone/bird мгновенно
    [ ] 4:30–5:00 /about — текст прокручивается, ссылки кликабельны

[ ] Self-drive flow:
    [ ] Перезагрузить браузер → «5-min tour» → пройти 6 страниц без проводника
    [ ] Reset to default работает на /demo
    [ ] Tailwind CDN недоступен → degraded UI читаем (inline критические классы)

[ ] Failure modes:
    [ ] Cancel inference — UI не залипает
    [ ] Перезапуск app — все preset'ы работают
```

### Backup-сценарий для защиты

`docs/jury_demo.mp4` (день 10) — screen-recording полного 5-мин flow, на случай отсутствия интернета / сбоя ноутбука / онлайн-формата.

---

## 5. Acceptance criteria (полный список)

| AC | Описание | Гард |
|---|---|---|
| UI-AC-1 | `python ui/app.py` стартует за < 5 сек, первая страница загружается < 200 мс | MUST |
| UI-AC-2 | `/demo` любой preset → результат < 500 мс на 2060 SUPER (cache miss) | MUST |
| UI-AC-3 | Все 6 страниц открываются в Chrome/Firefox/Edge без console.error | MUST |
| UI-AC-4 | Жюри-сценарий 5 минут без поломок | MUST |
| UI-AC-5 | По 1 PNG screenshot каждой страницы в `docs/ui_screenshots/` | MUST |
| UI-AC-6 | Smoke-тесты (`pytest tests/ui/`) проходят зелёным после Дня 7 | MUST |
| UI-AC-7 | Pre-computed M4 examples — нулевой inference на `/counter_uav` запрос | SHOULD |
| UI-AC-8 | LRU cache hit отображает результат < 15 мс | SHOULD |
| UI-AC-9 | Backup-MP4 готов и проверен в день 10 | SHOULD |

---

## 6. Out of scope

- WebSockets / live-streaming сигнала
- Загрузка пользовательских файлов (preset'ы достаточны)
- Multi-tenant / auth / accounts / DB
- Адаптивный мобильный layout (защита на ноутбуке)
- A/B-тестирование UI вариантов
- i18n (русский по умолчанию, английский — только для ключевых терминов)
- Telemetry / analytics

---

## 7. Open questions (для уточнения в реализации)

1. **CovRecon в demo:** показывать как «плохой пример при низком SNR» (educational moment) или скрыть, чтобы не подсвечивать слабую модель? — рекомендую показать с подписью «physics edge case».
2. **Цветовая схема Plotly:** Plotly default vs custom palette под Tailwind blue/emerald? — default достаточно, ноль времени на бренд-работу.
3. **English переводы UI:** нужны полностью или только для money moments? — переводы ключевых KPI и заголовков, иначе русский (комиссия русскоязычная).

---

## 8. Self-review (inline)

| Проверка | Результат |
|---|---|
| Placeholder scan (TBD/TODO/vague) | Найдены 3 в §7 — переформулированы как явные «open questions» с recommended defaults, не блокеры |
| Internal consistency (sections vs architecture) | OK: A1–A7 решения проявляются в Section 3 (run_in_threadpool, LRU, lifespan) и Section 4 (как тестировать) |
| Scope check (focused для одного плана) | OK: 6 страниц + сервисы + тесты — реально для Дней 6–7 одним планом |
| Ambiguity check | M3 «slider обновление» уточнён как `@change.debounce.200ms` + `Plotly.react`. ResultDTO имеет явную JSON-схему. Latency budget числовой. |

Все findings зафиксированы inline, документ self-consistent.

---

## 9. История изменений

| Дата | Изменение | Автор |
|---|---|---|
| 2026-06-24 | Создание design doc после brainstorming-сессии (4 confirmation-gates) | Claude / SP |
