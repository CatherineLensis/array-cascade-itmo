# UI — jury demo

Простое FastAPI + Jinja2 + Tailwind веб-приложение для показа cascade-проекта жюри ИТМО.

## Запуск

```bash
cd E:/ML_works_folder/Array_Cascade_ITMO
pip install -r requirements.txt
python ui/app.py
```

Открыть: http://localhost:8080

## Требования для live ML demo

- Артефакты `output/*.pth` и `output_counter_uav/*.pth` не коммитятся в репозиторий.
- Если локальные веса отсутствуют, `/demo` и `/api/run_cascade` автоматически используют `physics_fallback`.
- Перед показом стоит проверить `/health` и убедиться, что `models.*.ready == true`.
- Для backup-пакета можно запустить `python scripts/package_demo_assets.py` на поднятом локальном UI.

## Страницы

| Route | Назначение |
|---|---|
| `/` | Landing с pitch, KPI, диаграммой cascade |
| `/demo` | Живая прогонка cascade (после Дня 7 — с графиками) |
| `/methodology` | V1 — universal cascade, два DomainSpec рядом |
| `/counter_uav` | V3 — применение, micro-Doppler, urban multipath |
| `/roadmap` | План магистратуры на 4 семестра |
| `/about` | Текст заявки ИТМО |
| `/health` | JSON health-check для smoke-тестов |

## Структура

```
ui/
├── app.py                  # FastAPI entry
├── templates/              # Jinja2 (6 страниц + base)
├── static/                 # CSS / IMG / fallback charts
├── services/
│   ├── inference.py        # load_models() + run_cascade(...)
│   ├── presets.py          # 6 preset-сигналов
│   └── plots.py            # Plotly figure builders
└── README.md
```

## Статус разработки

| Этап | Что готово |
|---|---|
| ✅ День 6 | FastAPI каркас, 6 шаблонов, API /api/run_cascade, inference live/fallback |
| ✅ День 7 | Plotly charts on /demo, M1 on /methodology, about reads application_text.md |

## Backend контракт

`run_cascade(signal, domain, snr)` → `CascadeRunResult`:

```python
@dataclass
class CascadeRunResult:
    domain: str
    preset_id: str
    raw_signal_fig: dict          # Plotly figure
    whitened_signal_fig: dict
    classifier_logits: list[float]
    classifier_label: str
    doa_spectra: dict[str, list[float]]   # method → spectrum
    doa_predictions: dict[str, float]     # method → angle (deg)
    doa_true_angle: float | None
    metrics: dict[str, float]             # MedAE, error_deg, latency_ms
```

## Acceptance criteria (из SPEC §16.6)

- UI-AC-1: старт < 5 сек, landing загружается < 200 мс
- UI-AC-2: `/demo` inference < 500 мс
- UI-AC-3: открывается в Chrome / Firefox / Edge без console.error
- UI-AC-4: жюри-сценарий 5 минут без поломок
- UI-AC-5: скриншоты всех страниц в `docs/ui_screenshots/`
