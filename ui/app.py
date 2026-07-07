"""
ui/app.py — Array Cascade ITMO jury demo UI.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MODELS: dict = {}


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_accuracy_from_report(path: Path) -> float | None:
    if not path.exists():
        return None
    match = re.search(r"Accuracy:\s*([0-9.]+)", path.read_text(encoding="utf-8"))
    if not match:
        return None
    return float(match.group(1))


def _expected_counter_uav_labels() -> dict[str, str]:
    return {"u1": "drone", "u2": "bird", "u3": "multi"}


def _compute_counter_uav_live_smoke() -> dict | None:
    try:
        from services.inference import run_cascade
    except Exception:
        return None

    results: dict[str, dict] = {}
    expected = _expected_counter_uav_labels()
    correct = 0
    latencies: list[float] = []
    for preset_id, expected_label in expected.items():
        try:
            run = run_cascade(domain="counter_uav", preset_id=preset_id, snr=15)
        except Exception:
            return None
        latency_ms = run.metrics.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            latencies.append(float(latency_ms))
        is_correct = run.classifier_label == expected_label
        correct += int(is_correct)
        results[preset_id] = {
            "expected_label": expected_label,
            "predicted_label": run.classifier_label,
            "correct": is_correct,
            "latency_ms": latency_ms,
            "error_deg": run.metrics.get("error_deg"),
        }

    total = len(expected)
    return {
        "presets": results,
        "ok_count": correct,
        "total": total,
        "accuracy_pct": (100.0 * correct / total) if total else None,
        "latency_avg_ms": (sum(latencies) / len(latencies)) if latencies else None,
    }


def _load_demo_evidence() -> dict:
    automotive_clf = _extract_accuracy_from_report(PROJECT_ROOT / "output" / "clf_report.txt")
    automotive_siamese = _read_json(PROJECT_ROOT / "output" / "metrics_doa_siamese.json")
    automotive_lista = _read_json(PROJECT_ROOT / "output" / "metrics_doa_lista.json")
    counter_uav = _read_json(PROJECT_ROOT / "output_counter_uav" / "metrics.json")
    runtime = _read_json(PROJECT_ROOT / "output_counter_uav" / "runtime_smoke_metrics.json")
    live_smoke = _compute_counter_uav_live_smoke()

    bench_dir = PROJECT_ROOT / "output_counter_uav" / "benchmarks"
    summary = _read_json(bench_dir / "counter_uav_summary.json") or {}
    b_vs_c = _read_json(bench_dir / "b_vs_c.json") or {}
    layers = summary.get("truth_layers", {})
    snapshot_layer = layers.get("runtime_snapshot", {})
    temp_binary = layers.get("temporal_2class", {})
    temp_4cls = layers.get("temporal_4class", {})

    local_rmse = None
    if automotive_lista:
        local_rmse = automotive_lista.get("val", {}).get("rmse", [None])[-1]
    legacy_rmse = None
    if automotive_siamese:
        legacy_rmse = automotive_siamese.get("val", {}).get("rmse", [None])[-1]

    runtime_values = []
    if live_smoke and live_smoke.get("latency_avg_ms") is not None:
        runtime_values.append(float(live_smoke["latency_avg_ms"]))
    elif runtime:
        for preset in runtime.values():
            latency = preset.get("metrics", {}).get("latency_ms")
            if isinstance(latency, (int, float)):
                runtime_values.append(float(latency))

    return {
        "automotive_classifier_real_pct": automotive_clf * 100 if automotive_clf is not None else None,
        "automotive_local_best_rmse_deg": local_rmse,
        "automotive_local_siamese_rmse_deg": legacy_rmse,
        "automotive_legacy_medae_deg": 0.66,
        "automotive_legacy_sr1_pct": 65.2,
        "counter_uav_historical_best_val_pct": (
            counter_uav.get("classifier", {}).get("accuracy", 0.0) * 100 if counter_uav else None
        ),
        "counter_uav_deployed_report_pct": (
            counter_uav.get("classifier", {}).get("report", {}).get("accuracy", 0.0) * 100 if counter_uav else None
        ),
        "counter_uav_deployed_macro_f1_pct": (
            counter_uav.get("classifier", {}).get("report", {}).get("macro avg", {}).get("f1-score", 0.0) * 100
            if counter_uav
            else None
        ),
        "counter_uav_siamese_rmse_deg": (
            counter_uav.get("doa", {}).get("siamese", {}).get("rmse_deg") if counter_uav else None
        ),
        "counter_uav_lista_rmse_deg": (
            counter_uav.get("doa", {}).get("lista", {}).get("rmse_deg") if counter_uav else None
        ),
        "counter_uav_latency_avg_ms": (
            sum(runtime_values) / len(runtime_values) if runtime_values else None
        ),
        "counter_uav_live_smoke": live_smoke,
        # offline temporal recovery (Phase C)
        "snapshot_runtime_acc_pct": (snapshot_layer.get("accuracy", 0.0) * 100 if snapshot_layer else None),
        "snapshot_runtime_macro_f1_pct": (snapshot_layer.get("macro_f1", 0.0) * 100 if snapshot_layer else None),
        "temporal_binary_acc_pct": (temp_binary.get("accuracy", 0.0) * 100 if temp_binary else None),
        "temporal_binary_macro_f1_pct": (temp_binary.get("macro_f1", 0.0) * 100 if temp_binary else None),
        "temporal_binary_std_pct": (temp_binary.get("accuracy_std", 0.0) * 100 if temp_binary else None),
        "temporal_4class_acc_pct": (temp_4cls.get("accuracy", 0.0) * 100 if temp_4cls else None),
        "temporal_4class_macro_f1_pct": (temp_4cls.get("macro_f1", 0.0) * 100 if temp_4cls else None),
        "temporal_4class_std_pct": (temp_4cls.get("accuracy_std", 0.0) * 100 if temp_4cls else None),
        "temporal_n_seeds": temp_binary.get("n_seeds"),
        "temporal_model_name": temp_binary.get("model_name"),
        "temporal_feature_set": temp_binary.get("feature_set"),
        "b_vs_c_delta_acc_pct": (b_vs_c.get("delta", {}).get("accuracy", 0.0) * 100 if b_vs_c else None),
        "temporal_4class_confusion": temp_4cls.get("confusion_matrix") if temp_4cls else None,
        "temporal_4class_per_class": temp_4cls.get("per_class") if temp_4cls else None,
        "temporal_binary_confusion": temp_binary.get("confusion_matrix") if temp_binary else None,
    }


class RunCascadeRequest(BaseModel):
    domain: str = "automotive"
    preset: str = "a1"
    snr: float | None = 15.0


class RunBothRequest(BaseModel):
    preset_auto: str = "a1"
    preset_uav: str = "u1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from services.inference import load_models

    MODELS.update(load_models(mode="auto"))
    yield
    MODELS.clear()


app = FastAPI(
    title="Array Cascade ITMO",
    description="Universal array signal cascade — jury demo for ITMO master's application",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    evidence = _load_demo_evidence()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"page": "index", "evidence": evidence},
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    auto = MODELS.get("automotive", {})
    uav = MODELS.get("counter_uav", {})
    status = f"automotive={'ready' if auto.get('ready') else 'fallback'} | counter_uav={'ready' if uav.get('ready') else 'fallback'}"
    return templates.TemplateResponse(
        request,
        "demo.html",
        {"page": "demo", "models_status": status},
    )


@app.get("/methodology", response_class=HTMLResponse)
async def methodology(request: Request):
    evidence = _load_demo_evidence()
    return templates.TemplateResponse(
        request,
        "methodology.html",
        {"page": "methodology", "evidence": evidence},
    )


@app.get("/counter_uav", response_class=HTMLResponse)
async def counter_uav(request: Request):
    metrics = None
    evidence = _load_demo_evidence()
    metrics_path = PROJECT_ROOT / "output_counter_uav" / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = None
    return templates.TemplateResponse(
        request,
        "counter_uav.html",
        {"page": "counter_uav", "counter_uav_metrics": metrics, "evidence": evidence},
    )


@app.get("/roadmap", response_class=HTMLResponse)
async def roadmap(request: Request):
    return templates.TemplateResponse(
        request,
        "roadmap.html",
        {"page": "roadmap"},
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    app_text = ""
    app_path = PROJECT_ROOT / "docs" / "ITMO_application" / "application_text.md"
    if app_path.exists():
        app_text = app_path.read_text(encoding="utf-8")
    return templates.TemplateResponse(
        request,
        "about.html",
        {"page": "about", "application_text": app_text},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "models": MODELS}


@app.post("/api/run_cascade")
async def api_run_cascade(body: RunCascadeRequest):
    from services.inference import result_to_dict, run_cascade

    result = await run_in_threadpool(
        run_cascade,
        None,
        body.domain,
        body.preset,
        body.snr,
    )
    return JSONResponse(result_to_dict(result))


@app.post("/api/run_both")
async def api_run_both(body: RunBothRequest):
    from services.inference import result_to_dict, run_cascade

    def _run():
        r1 = run_cascade(domain="automotive", preset_id=body.preset_auto)
        r2 = run_cascade(domain="counter_uav", preset_id=body.preset_uav)
        return {
            "automotive": result_to_dict(r1),
            "counter_uav": result_to_dict(r2),
        }

    return JSONResponse(await run_in_threadpool(_run))


@app.post("/api/run_both_v2")
async def api_run_both_v2(body: RunBothRequest):
    return await api_run_both(body)


@app.get("/api/counter_uav_plots")
async def api_counter_uav_plots():
    """Pre-rendered Plotly figures for counter-UAV result section."""
    def _build():
        from services.plots import build_confusion_heatmap, build_seed_stability, build_per_class_bars

        plots_dir = PROJECT_ROOT / "output_counter_uav" / "plots"
        bench_dir = PROJECT_ROOT / "output_counter_uav" / "benchmarks"
        conf4 = _read_json(plots_dir / "temporal_4class_confusion.json") or {}
        conf2 = _read_json(plots_dir / "temporal_2class_confusion.json") or {}
        stab = _read_json(plots_dir / "temporal_seed_stability.json") or {}
        temp4 = _read_json(bench_dir / "temporal_4class.json") or {}

        out: dict[str, dict] = {}
        if conf4.get("labels") and conf4.get("matrix"):
            out["confusion_4class"] = build_confusion_heatmap(
                conf4["labels"], conf4["matrix"], "Полная задача · средняя матрица ошибок"
            )
        if conf2.get("labels") and conf2.get("matrix"):
            out["confusion_binary"] = build_confusion_heatmap(
                conf2["labels"], conf2["matrix"], "Дрон против птицы · средняя матрица ошибок"
            )
        if stab.get("points"):
            out["seed_stability"] = build_seed_stability(
                stab["points"], "Стабильность результата на пяти независимых запусках"
            )
        if temp4.get("per_class"):
            class_labels_ru = {"clutter": "фон", "drone": "дрон", "bird": "птица", "multi": "две цели"}
            out["per_class_bars"] = build_per_class_bars(
                temp4["per_class"],
                "Precision / Recall / F1 по классам",
                class_labels_ru,
            )
        return out

    return JSONResponse(await run_in_threadpool(_build))


@app.get("/api/training_curves")
async def api_training_curves():
    """Plotly figures for neural-network training: classifier curves + 4 DOA methods."""
    def _build():
        from services.plots import (
            build_classifier_training_curves,
            build_doa_training_overlay,
            build_doa_rmse_comparison,
        )
        out_dir = PROJECT_ROOT / "output"
        clf = _read_json(out_dir / "metrics_classifier.json") or {}
        doas = {}
        for m in ("siamese", "lista", "covrecon", "hiergru"):
            d = _read_json(out_dir / f"metrics_doa_{m}.json") or {}
            doas[m] = d

        result: dict[str, dict] = {}
        if clf.get("train") and clf.get("val"):
            result["classifier_curves"] = build_classifier_training_curves(
                clf, "ComplexNet — обучение классификатора"
            )

        losses_by_method = {
            m: (d.get("train", {}) or {}).get("loss", []) or []
            for m, d in doas.items()
        }
        if any(losses_by_method.values()):
            result["doa_training_overlay"] = build_doa_training_overlay(
                losses_by_method,
                "Обучение четырёх DOA-сетей · сравнение loss по эпохам",
            )

        rmse_by_method = {}
        for m, d in doas.items():
            val_rmse_list = (d.get("val", {}) or {}).get("rmse", []) or []
            if val_rmse_list:
                rmse_by_method[m] = float(val_rmse_list[-1])
        if rmse_by_method:
            result["doa_rmse"] = build_doa_rmse_comparison(
                rmse_by_method,
                "Финальный RMSE на синтетической валидации",
            )

        return result

    return JSONResponse(await run_in_threadpool(_build))


@app.get("/api/counter_uav_feature_correlation")
async def api_counter_uav_feature_correlation():
    """Spearman correlation matrix between micro-Doppler features used by the temporal classifier."""
    def _build():
        import numpy as np
        from scipy.stats import spearmanr
        from domains.counter_uav.micro_doppler import synth_drone_signal, synth_bird_signal
        from services.plots import build_spearman_heatmap

        n_per_class = 150
        rng = np.random.default_rng(2026)
        micro = []
        for i in range(n_per_class):
            _, m_d = synth_drone_signal(
                azimuth_deg=float(rng.uniform(-60, 60)),
                snr_db=float(rng.uniform(5, 20)),
                rng=np.random.default_rng(int(rng.integers(0, 10**6))),
            )
            _, m_b = synth_bird_signal(
                azimuth_deg=float(rng.uniform(-60, 60)),
                snr_db=float(rng.uniform(5, 20)),
                rng=np.random.default_rng(int(rng.integers(0, 10**6))),
            )
            micro.append(m_d); micro.append(m_b)
        micro_arr = np.stack(micro).astype(np.complex64)

        magnitude = np.abs(micro_arr).astype(np.float32)
        phase = np.unwrap(np.angle(micro_arr), axis=1).astype(np.float32)
        fft_magnitude = np.abs(np.fft.fft(magnitude, axis=1)).astype(np.float32)
        n_fft_bins = min(8, fft_magnitude.shape[1])

        feats = np.concatenate(
            [
                magnitude.mean(axis=1, keepdims=True),
                magnitude.std(axis=1, keepdims=True),
                magnitude.max(axis=1, keepdims=True),
                magnitude.min(axis=1, keepdims=True),
                phase.std(axis=1, keepdims=True),
                fft_magnitude[:, :n_fft_bins],
            ],
            axis=1,
        ).astype(np.float64)

        labels = (
            ["|s|·среднее", "|s|·σ", "|s|·max", "|s|·min", "φ·σ"]
            + [f"FFT[{i}]" for i in range(n_fft_bins)]
        )

        rho, _ = spearmanr(feats, axis=0)
        if np.ndim(rho) == 0:
            rho = np.array([[1.0]])
        matrix = rho.tolist()

        return build_spearman_heatmap(
            matrix,
            labels,
            "Корреляция Спирмена между признаками micro-Doppler",
        )

    return JSONResponse(await run_in_threadpool(_build))


@app.get("/api/counter_uav_recovery")
async def api_counter_uav_recovery():
    """Return Phase B (snapshot) vs Phase C (temporal) recovery benchmarks."""
    summary = _read_json(PROJECT_ROOT / "output_counter_uav" / "benchmarks" / "counter_uav_summary.json")
    b_vs_c = _read_json(PROJECT_ROOT / "output_counter_uav" / "benchmarks" / "b_vs_c.json")
    return JSONResponse({"summary": summary, "b_vs_c": b_vs_c})


@app.get("/api/counter_uav_spectrograms")
async def api_counter_uav_spectrograms():
    """Pre-computed STFT spectrograms for drone vs bird (UI-only)."""
    def _build():
        import numpy as np
        from domains.counter_uav.micro_doppler import synth_drone_signal, synth_bird_signal
        from services.plots import build_micro_doppler_spectrogram

        rng_d = np.random.default_rng(101)
        rng_b = np.random.default_rng(202)
        _, drone_series = synth_drone_signal(azimuth_deg=30.0, snr_db=15.0, rng=rng_d)
        _, bird_series = synth_bird_signal(azimuth_deg=-20.0, snr_db=15.0, rng=rng_b)

        return {
            "drone": build_micro_doppler_spectrogram(drone_series, "Квадрокоптер — micro-Doppler"),
            "bird": build_micro_doppler_spectrogram(bird_series, "Птица — micro-Doppler"),
        }

    return JSONResponse(await run_in_threadpool(_build))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        app_dir=str(UI_DIR),
    )
