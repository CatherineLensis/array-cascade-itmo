"""
ui/services/plots.py — Plotly figure builders (dict format for frontend).
"""

from __future__ import annotations

import numpy as np


_PLOT_TEMPLATE = {
    "paper_bgcolor": "#161b25",
    "plot_bgcolor": "#161b25",
    "font": {"color": "#e6ecf5", "family": '-apple-system, "Segoe UI", Roboto, sans-serif'},
    "margin": {"t": 48, "b": 44, "l": 56, "r": 32},
}


def _axis(title: str, range_: list[float] | None = None) -> dict:
    axis = {
        "title": title,
        "gridcolor": "#232a3a",
        "zerolinecolor": "#232a3a",
        "linecolor": "#232a3a",
        "tickfont": {"color": "#aab3c5"},
        "titlefont": {"color": "#e6ecf5"},
    }
    if range_ is not None:
        axis["range"] = range_
    return axis


def _fig_dict(traces: list[dict], layout: dict | None = None) -> dict:
    merged_layout = dict(_PLOT_TEMPLATE)
    if layout:
        merged_layout.update(layout)
    return {"data": traces, "layout": merged_layout}


def build_signal_plot(signal: np.ndarray, title: str) -> dict:
    signal = np.asarray(signal).reshape(-1)
    n = len(signal)
    x = list(range(n))
    amp = np.abs(signal).tolist()
    phase = np.angle(signal).tolist()
    return _fig_dict(
        [
            {"x": x, "y": amp, "type": "bar", "name": "|x|", "marker": {"color": "#5b8def"}},
            {"x": x, "y": phase, "type": "scatter", "mode": "lines+markers", "name": "phase",
             "yaxis": "y2", "line": {"color": "#e0a247"}, "marker": {"color": "#e0a247"}},
        ],
        {
            "title": title,
            "xaxis": _axis("Индекс антенны"),
            "yaxis": _axis("Амплитуда"),
            "yaxis2": {
                **_axis("Фаза, рад"),
                "overlaying": "y",
                "side": "right",
            },
            "height": 260,
            "showlegend": True,
            "legend": {"font": {"color": "#aab3c5"}, "orientation": "h", "y": 1.1, "x": 1, "xanchor": "right"},
        },
    )


def build_classifier_bars(logits: list[float], labels: list[str]) -> dict:
    """Horizontal probability bars with percentage labels; predicted class highlighted."""
    arr = np.array(logits, dtype=float)
    probs = np.exp(arr - np.max(arr))
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    pred_idx = int(np.argmax(probs))

    colors = [
        "#4caf85" if i == pred_idx else "#3c4a6b"
        for i in range(len(labels))
    ]
    text_labels = [f"{100.0 * p:.1f} %" for p in probs]

    return _fig_dict(
        [
            {
                "x": probs.tolist(),
                "y": labels,
                "type": "bar",
                "orientation": "h",
                "marker": {"color": colors, "line": {"color": "#232a3a", "width": 1}},
                "text": text_labels,
                "textposition": "outside",
                "textfont": {"color": "#e6ecf5"},
                "hovertemplate": "%{y}: %{x:.3f}<extra></extra>",
            }
        ],
        {
            "title": "Вероятности классов",
            "xaxis": {**_axis("P (класс)", [0, 1.18]), "tickformat": ".0%"},
            "yaxis": {**_axis(""), "autorange": "reversed"},
            "height": 60 + 36 * max(len(labels), 1),
            "showlegend": False,
        },
    )


def build_doa_spectra(
    spectra: dict[str, list[float]],
    angle_grid_deg: np.ndarray,
    true_angle_deg: float | None = None,
) -> dict:
    traces = []
    colors = ["#5b8def", "#e0a247", "#9b6cf5", "#4caf85", "#e26060"]
    for i, (name, spec) in enumerate(spectra.items()):
        traces.append(
            {
                "x": angle_grid_deg.tolist(),
                "y": spec,
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "line": {"color": colors[i % len(colors)], "width": 2},
            }
        )
    n_methods = len(spectra)
    title = (
        "Спектр DOA (основной метод)"
        if n_methods == 1
        else f"Спектры DOA ({n_methods} метода)"
    )
    layout = {
        "title": title,
        "xaxis": _axis("Угол, градусы", [-90, 90]),
        "yaxis": _axis("Отклик"),
        "height": 320,
        "legend": {
            "font": {"color": "#aab3c5"},
            "orientation": "h",
            "y": 1.1,
            "x": 1,
            "xanchor": "right",
        },
    }
    if true_angle_deg is not None:
        layout["shapes"] = [
            {
                "type": "line",
                "x0": true_angle_deg,
                "x1": true_angle_deg,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e26060", "dash": "dash", "width": 1.5},
            }
        ]
        layout["annotations"] = [
            {
                "x": true_angle_deg,
                "y": 1.02,
                "yref": "paper",
                "showarrow": False,
                "text": f"истинный угол {true_angle_deg:.1f}°",
                "font": {"color": "#e26060", "size": 11},
            }
        ]
    return _fig_dict(traces, layout)


def build_micro_doppler_spectrogram(
    micro_series: np.ndarray,
    title: str,
    fs_hz: float | None = None,
) -> dict:
    """STFT slow-time spectrogram. Reorders complex-STFT frequencies via
    fftshift so the Y-axis is monotonic from -fs/2 to +fs/2."""
    from domains.counter_uav.micro_doppler import micro_doppler_spectrogram

    f, t, Sxx = micro_doppler_spectrogram(np.asarray(micro_series), fs_hz=fs_hz)
    f_shift = np.fft.fftshift(np.asarray(f))
    Sxx_shift = np.fft.fftshift(np.asarray(Sxx), axes=0)

    return _fig_dict(
        [
            {
                "z": Sxx_shift.tolist(),
                "x": np.asarray(t).tolist(),
                "y": f_shift.tolist(),
                "type": "heatmap",
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": "|S|",
                    "tickfont": {"color": "#aab3c5"},
                    "titlefont": {"color": "#e6ecf5"},
                    "thickness": 10,
                    "len": 0.85,
                },
                "zsmooth": "best",
                "hovertemplate": "t = %{x:.3f} с<br>f = %{y:.1f} Гц<br>|S| = %{z:.4f}<extra></extra>",
            }
        ],
        {
            "title": title,
            "xaxis": {**_axis("Время, с"), "autorange": True},
            "yaxis": {**_axis("Доплеровский сдвиг частоты, Гц"), "autorange": True},
            "height": 280,
        },
    )


def build_confusion_heatmap(labels: list[str], matrix: list[list[float]], title: str) -> dict:
    """Confusion matrix as heatmap with text annotations on each cell."""
    z = [list(row) for row in matrix]
    text = [[f"{v:.1f}" if isinstance(v, float) and not float(v).is_integer() else f"{int(v)}" for v in row] for row in matrix]

    return _fig_dict(
        [
            {
                "z": z,
                "x": labels,
                "y": labels,
                "type": "heatmap",
                "colorscale": [
                    [0.0, "#161b25"],
                    [0.3, "#1f3a5f"],
                    [0.6, "#3a6cc4"],
                    [1.0, "#5b8def"],
                ],
                "text": text,
                "texttemplate": "%{text}",
                "textfont": {"color": "#e6ecf5", "size": 13},
                "showscale": False,
                "hovertemplate": "истинный %{y}<br>предсказан %{x}<br>%{z}<extra></extra>",
            }
        ],
        {
            "title": title,
            "xaxis": {**_axis("предсказание"), "side": "bottom"},
            "yaxis": {**_axis("истинный класс"), "autorange": "reversed"},
            "height": 60 + 64 * max(len(labels), 1),
        },
    )


def build_seed_stability(points: list[dict], title: str) -> dict:
    """Per-seed accuracy as dots with overall mean line."""
    seeds = [str(p["seed"]) for p in points]
    accs = [100.0 * float(p["accuracy"]) for p in points]
    f1s = [100.0 * float(p["macro_f1"]) for p in points]
    mean_acc = sum(accs) / len(accs) if accs else 0.0

    return _fig_dict(
        [
            {
                "x": seeds,
                "y": accs,
                "type": "scatter",
                "mode": "markers+lines",
                "name": "accuracy",
                "line": {"color": "#5b8def", "width": 1.5, "dash": "dot"},
                "marker": {"color": "#5b8def", "size": 12, "line": {"color": "#0e1117", "width": 2}},
            },
            {
                "x": seeds,
                "y": f1s,
                "type": "scatter",
                "mode": "markers+lines",
                "name": "macro-F1",
                "line": {"color": "#4caf85", "width": 1.5, "dash": "dot"},
                "marker": {"color": "#4caf85", "size": 12, "line": {"color": "#0e1117", "width": 2}},
            },
        ],
        {
            "title": title,
            "xaxis": _axis("номер запуска"),
            "yaxis": _axis("процент", [min(min(accs), min(f1s)) - 3, 100]),
            "height": 280,
            "shapes": [
                {
                    "type": "line",
                    "x0": 0,
                    "x1": 1,
                    "xref": "paper",
                    "y0": mean_acc,
                    "y1": mean_acc,
                    "line": {"color": "#aab3c5", "dash": "dash", "width": 1},
                }
            ],
            "annotations": [
                {
                    "x": 1,
                    "xref": "paper",
                    "y": mean_acc,
                    "showarrow": False,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "text": f"среднее accuracy {mean_acc:.1f} %",
                    "font": {"color": "#aab3c5", "size": 11},
                }
            ],
            "legend": {"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right", "font": {"color": "#aab3c5"}},
        },
    )


def build_spearman_heatmap(matrix: list[list[float]], labels: list[str], title: str) -> dict:
    """Spearman correlation matrix as heatmap with diverging colorscale (red-neutral-blue)."""
    z = [list(row) for row in matrix]
    text = [[f"{v:+.2f}" for v in row] for row in matrix]
    return _fig_dict(
        [
            {
                "z": z,
                "x": labels,
                "y": labels,
                "type": "heatmap",
                "zmin": -1.0,
                "zmax": 1.0,
                "colorscale": [
                    [0.0,  "#e26060"],
                    [0.5,  "#161b25"],
                    [1.0,  "#5b8def"],
                ],
                "text": text,
                "texttemplate": "%{text}",
                "textfont": {"color": "#e6ecf5", "size": 10},
                "colorbar": {
                    "title": "ρ",
                    "tickfont": {"color": "#aab3c5"},
                    "titlefont": {"color": "#e6ecf5"},
                    "thickness": 12,
                    "len": 0.8,
                },
                "hovertemplate": "%{y} ↔ %{x}<br>ρ = %{z:.3f}<extra></extra>",
            }
        ],
        {
            "title": title,
            "xaxis": {**_axis(""), "tickangle": -45},
            "yaxis": {**_axis(""), "autorange": "reversed"},
            "height": 60 + 34 * max(len(labels), 1),
        },
    )


def build_per_class_bars(per_class: dict, title: str, class_labels_ru: dict | None = None) -> dict:
    """Grouped horizontal bars: precision / recall / F1 per class."""
    class_labels_ru = class_labels_ru or {}
    classes = list(per_class.keys())
    display = [class_labels_ru.get(c, c) for c in classes]

    precisions = [100.0 * per_class[c].get("precision", 0.0) for c in classes]
    recalls    = [100.0 * per_class[c].get("recall", 0.0)    for c in classes]
    f1s        = [100.0 * per_class[c].get("f1", 0.0)        for c in classes]

    return _fig_dict(
        [
            {
                "x": precisions, "y": display, "type": "bar", "orientation": "h",
                "name": "precision", "marker": {"color": "#5b8def"},
                "text": [f"{p:.0f}" for p in precisions], "textposition": "outside",
                "textfont": {"color": "#e6ecf5", "size": 11},
            },
            {
                "x": recalls, "y": display, "type": "bar", "orientation": "h",
                "name": "recall", "marker": {"color": "#4caf85"},
                "text": [f"{r:.0f}" for r in recalls], "textposition": "outside",
                "textfont": {"color": "#e6ecf5", "size": 11},
            },
            {
                "x": f1s, "y": display, "type": "bar", "orientation": "h",
                "name": "F1", "marker": {"color": "#e0a247"},
                "text": [f"{f:.0f}" for f in f1s], "textposition": "outside",
                "textfont": {"color": "#e6ecf5", "size": 11},
            },
        ],
        {
            "title": title,
            "barmode": "group",
            "xaxis": {**_axis("процент", [0, 115]), "ticksuffix": " %"},
            "yaxis": {**_axis(""), "autorange": "reversed"},
            "height": 80 + 80 * max(len(classes), 1),
            "legend": {"orientation": "h", "y": 1.1, "x": 1, "xanchor": "right", "font": {"color": "#aab3c5"}},
            "margin": {"t": 70, "b": 50, "l": 90, "r": 60},
        },
    )


def build_classifier_training_curves(metrics: dict, title: str) -> dict:
    """ComplexNet classifier training: loss (train+val) and accuracy (train+val) over epochs."""
    train = metrics.get("train", {})
    val = metrics.get("val", {})
    train_loss = train.get("loss", []) or []
    val_loss = val.get("loss", []) or []
    train_acc = train.get("acc", []) or []
    val_acc = val.get("acc", []) or []
    epochs = list(range(1, max(len(train_loss), len(val_loss)) + 1))

    return _fig_dict(
        [
            {"x": epochs, "y": train_loss, "type": "scatter", "mode": "lines", "name": "train · loss",
             "line": {"color": "#5b8def", "width": 2}, "yaxis": "y"},
            {"x": epochs, "y": val_loss, "type": "scatter", "mode": "lines", "name": "val · loss",
             "line": {"color": "#5b8def", "width": 1.5, "dash": "dash"}, "yaxis": "y"},
            {"x": epochs, "y": [100.0 * a for a in train_acc], "type": "scatter", "mode": "lines",
             "name": "train · accuracy", "line": {"color": "#4caf85", "width": 2}, "yaxis": "y2"},
            {"x": epochs, "y": [100.0 * a for a in val_acc], "type": "scatter", "mode": "lines",
             "name": "val · accuracy", "line": {"color": "#4caf85", "width": 1.5, "dash": "dash"}, "yaxis": "y2"},
        ],
        {
            "title": title,
            "xaxis": _axis("эпоха"),
            "yaxis": {**_axis("loss"), "side": "left"},
            "yaxis2": {**_axis("accuracy, %", [80, 100]), "overlaying": "y", "side": "right"},
            "height": 320,
            "legend": {"orientation": "h", "y": 1.14, "x": 1, "xanchor": "right", "font": {"color": "#aab3c5"}},
        },
    )


def build_doa_training_overlay(losses_by_method: dict[str, list[float]], title: str) -> dict:
    """Overlay of 4 DOA networks training loss curves on log y-axis."""
    colors = {"siamese": "#5b8def", "lista": "#e0a247", "covrecon": "#9b6cf5", "hiergru": "#4caf85"}
    display = {"siamese": "Siamese", "lista": "LISTA", "covrecon": "CovRecon", "hiergru": "HierGRU"}
    traces = []
    for method, losses in losses_by_method.items():
        if not losses:
            continue
        traces.append({
            "x": list(range(1, len(losses) + 1)),
            "y": [max(float(v), 1e-6) for v in losses],
            "type": "scatter",
            "mode": "lines",
            "name": display.get(method, method),
            "line": {"color": colors.get(method, "#aab3c5"), "width": 2},
        })
    return _fig_dict(
        traces,
        {
            "title": title,
            "xaxis": _axis("эпоха"),
            "yaxis": {**_axis("loss (логарифм)"), "type": "log"},
            "height": 340,
            "legend": {"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right", "font": {"color": "#aab3c5"}},
        },
    )


def build_doa_rmse_comparison(rmse_by_method: dict[str, float], title: str) -> dict:
    """Final synth-val RMSE of 4 DOA methods as horizontal bar chart."""
    display = {"siamese": "Siamese", "lista": "LISTA", "covrecon": "CovRecon", "hiergru": "HierGRU"}
    colors  = {"siamese": "#5b8def", "lista": "#e0a247", "covrecon": "#9b6cf5", "hiergru": "#4caf85"}
    items = sorted(rmse_by_method.items(), key=lambda kv: kv[1])
    names = [display.get(k, k) for k, _ in items]
    values = [float(v) for _, v in items]
    bar_colors = [colors.get(k, "#aab3c5") for k, _ in items]
    return _fig_dict(
        [{
            "x": values,
            "y": names,
            "type": "bar",
            "orientation": "h",
            "marker": {"color": bar_colors, "line": {"color": "#232a3a", "width": 1}},
            "text": [f"{v:.2f}°" for v in values],
            "textposition": "outside",
            "textfont": {"color": "#e6ecf5", "size": 12},
        }],
        {
            "title": title,
            "xaxis": _axis("RMSE, градусы"),
            "yaxis": {**_axis(""), "autorange": "reversed"},
            "height": 80 + 56 * max(len(items), 1),
            "showlegend": False,
            "margin": {"t": 60, "b": 50, "l": 100, "r": 80},
        },
    )


def build_roadmap_gantt() -> dict:
    tasks = [
        ("Семестр 1 — micro-Doppler + DIAT-µSAT", "2026-09-01", "2027-01-31"),
        ("Семестр 2 — трекер IMM + JPDA",         "2027-02-01", "2027-06-30"),
        ("Семестр 3 — сенсорный fusion",          "2027-09-01", "2028-01-31"),
        ("Семестр 4 — MVP-сенсор и диссертация",  "2028-02-01", "2028-06-30"),
    ]
    return _fig_dict(
        [
            {
                "x": [t[2] for t in tasks],
                "y": [t[0] for t in tasks],
                "base": [t[1] for t in tasks],
                "type": "bar",
                "orientation": "h",
                "marker": {"color": "#5b8def"},
            }
        ],
        {
            "title": "План магистратуры",
            "height": 280,
            "barmode": "overlay",
            "xaxis": _axis("Период"),
            "yaxis": {**_axis(""), "autorange": "reversed"},
            "showlegend": False,
        },
    )
