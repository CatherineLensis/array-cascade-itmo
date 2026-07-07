"""
metrics.py
==========
Полноценный модуль метрик и визуализации для radar DOA pipeline.

Включает:
  - Метрики классификатора: accuracy, F1, per-class, confusion matrix
  - Метрики DOA: RMSE, MAE, Success Rate, CDF, per-class analysis
  - Визуализация: scatter, гистограммы, кривые обучения, EDA
  - Сравнение синтетики vs реальных данных

Использование:
    from metrics import MetricsTracker, DOAEvaluator, plot_all_reports
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    roc_curve, auc,
)


# ──────────────────────────────────────────────────────────────
#  1. MetricsTracker — сбор метрик во время обучения
# ──────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Трекер метрик обучения с поддержкой train/val split.

    Использование:
        tracker = MetricsTracker("classifier")
        for epoch in range(n_epochs):
            tracker.log_train(epoch, loss=0.5, acc=0.8)
            tracker.log_val(epoch, loss=0.6, acc=0.75)
        tracker.plot_curves(output_dir)
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.epochs = []

    def log_train(self, epoch, **kwargs):
        """Логируем train-метрики: loss, acc, rmse и т.д."""
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        for key, value in kwargs.items():
            self.train_history[key].append(value)

    def log_val(self, epoch, **kwargs):
        """Логируем val-метрики: loss, acc, rmse и т.д."""
        for key, value in kwargs.items():
            self.val_history[key].append(value)

    def plot_curves(self, output_dir, show=True):
        """Кривые обучения: train vs val для всех метрик."""
        all_keys = set(self.train_history.keys()) | set(self.val_history.keys())
        n_plots = len(all_keys)
        if n_plots == 0:
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, sorted(all_keys)):
            if key in self.train_history:
                ax.plot(self.train_history[key], label=f"train {key}", color="#2196F3")
            if key in self.val_history:
                ax.plot(self.val_history[key], label=f"val {key}", color="#F44336",
                        linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.set_title(f"{self.model_name} — {key}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"curves_{self.model_name}.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def get_best_val(self, metric="loss", mode="min"):
        """Лучшая val-метрика и соответствующая эпоха."""
        if metric not in self.val_history or len(self.val_history[metric]) == 0:
            return None, None
        values = self.val_history[metric]
        fn = np.argmin if mode == "min" else np.argmax
        best_idx = fn(values)
        return values[best_idx], best_idx

    def save_json(self, output_dir):
        """Сохраняем историю в JSON."""
        data = {
            "model_name": self.model_name,
            "train": dict(self.train_history),
            "val": dict(self.val_history),
        }
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"metrics_{self.model_name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=lambda x: float(x))
        return path


# ──────────────────────────────────────────────────────────────
#  2. ClassifierReport — полный отчёт по классификатору
# ──────────────────────────────────────────────────────────────

CLASS_NAMES = ["direct (0)", "multipath (1)", "two targets (2)"]


def classifier_full_report(y_true, y_pred, y_proba=None, output_dir=None, show=True):
    """
    Полный отчёт по классификатору:
      1. Confusion Matrix (нормированная и абсолютная)
      2. Per-class F1, Precision, Recall (bar chart)
      3. ROC-AUC per class (если есть вероятности)
      4. Classification Report (текст)

    Returns: dict с метриками
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    acc = np.mean(y_true == y_pred)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": f1_per_class.tolist(),
        "precision_per_class": prec_per_class.tolist(),
        "recall_per_class": rec_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    if output_dir is None:
        print(report_str)
        return result

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: Confusion Matrices (absolute + normalized) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, fmt in [
        (ax1, cm, f"Confusion Matrix (acc={acc:.3f})", "d"),
        (ax2, cm_norm, "Normalized Confusion Matrix", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["direct", "multipath", "2 targets"], fontsize=9)
        ax.set_yticklabels(["direct", "multipath", "2 targets"], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                val = data[i, j]
                txt = format(val, fmt)
                color = "white" if val > data.max() * 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clf_confusion_matrix.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # ── Plot 2: Per-class F1 / Precision / Recall ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, prec_per_class, w, label="Precision", color="#2196F3")
    ax.bar(x, rec_per_class, w, label="Recall", color="#4CAF50")
    ax.bar(x + w, f1_per_class, w, label="F1-score", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(["direct (0)", "multipath (1)", "two targets (2)"])
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (p, r, f) in enumerate(zip(prec_per_class, rec_per_class, f1_per_class)):
        ax.text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clf_per_class_metrics.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # ── Plot 3: ROC-AUC (если есть proba и нет NaN) ──
    if y_proba is not None and y_proba.shape[1] == 3 and np.all(np.isfinite(y_proba)):
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#2196F3", "#F44336", "#4CAF50"]
        for cls_idx in range(3):
            y_bin = (y_true == cls_idx).astype(int)
            try:
                fpr, tpr, _ = roc_curve(y_bin, y_proba[:, cls_idx])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[cls_idx],
                        label=f"{CLASS_NAMES[cls_idx]} (AUC={roc_auc:.3f})")
                result[f"auc_class_{cls_idx}"] = float(roc_auc)
            except (ValueError, IndexError):
                print(f"  [!] ROC-AUC: не удалось вычислить для класса {cls_idx}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves per Class")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clf_roc_auc.png"), dpi=150)
        if show:
            plt.show()
        plt.close()
    elif y_proba is not None and not np.all(np.isfinite(y_proba)):
        print("  [!] ROC-AUC: пропущен (вероятности содержат NaN/Inf)")

    # Сохраняем текстовый отчёт
    with open(os.path.join(output_dir, "clf_report.txt"), "w") as f:
        f.write(report_str)
        f.write(f"\n\nAccuracy: {acc:.4f}")
        f.write(f"\nF1 macro: {f1_macro:.4f}")
        f.write(f"\nF1 weighted: {f1_weighted:.4f}")

    return result


# ──────────────────────────────────────────────────────────────
#  3. DOAEvaluator — полная оценка DOA-методов
# ──────────────────────────────────────────────────────────────

class DOAEvaluator:
    """
    Оценка DOA с набором стандартных метрик для radar ML:

    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - Median AE
    - Success Rate @ 1°, 3°, 5° (доля предсказаний в пределах порога)
    - CDF ошибок
    - Scatter true vs predicted
    - Гистограмма ошибок
    - Анализ по классам (class 0 vs class 2)
    - Анализ по углу (edge effects)
    """

    def __init__(self, grid_deg):
        self.grid_deg = grid_deg
        self.results = {}  # method_name → DOAResult

    def add_method(self, method_name, errors, pred_classes=None, true_angles=None,
                   pred_angles=None):
        """
        Добавить результаты метода.

        errors: list[float] — абсолютные ошибки в градусах
        pred_classes: list[int] — класс каждого сэмпла (для per-class анализа)
        true_angles: list[float] — истинные углы (для scatter)
        pred_angles: list[float] — предсказанные углы (для scatter)
        """
        errors = np.asarray(errors)
        self.results[method_name] = {
            "errors": errors,
            "pred_classes": np.asarray(pred_classes) if pred_classes is not None else None,
            "true_angles": np.asarray(true_angles) if true_angles is not None else None,
            "pred_angles": np.asarray(pred_angles) if pred_angles is not None else None,
        }

    def compute_metrics(self, method_name):
        """Вычисляем полный набор метрик для метода."""
        r = self.results[method_name]
        errors = r["errors"]
        if len(errors) == 0:
            return {}

        metrics = {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(errors)),
            "median_ae": float(np.median(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
            "n_samples": len(errors),
        }

        # Success Rate @ разных порогах
        for threshold in [1.0, 2.0, 3.0, 5.0, 10.0]:
            rate = float(np.mean(errors <= threshold))
            metrics[f"success_rate_{threshold:.0f}deg"] = rate

        # Per-class метрики
        if r["pred_classes"] is not None:
            for cls in [0, 2]:
                mask = r["pred_classes"] == cls
                if mask.sum() > 0:
                    cls_errors = errors[mask]
                    cls_name = "direct" if cls == 0 else "two_targets"
                    metrics[f"rmse_{cls_name}"] = float(np.sqrt(np.mean(cls_errors ** 2)))
                    metrics[f"mae_{cls_name}"] = float(np.mean(cls_errors))
                    metrics[f"n_{cls_name}"] = int(mask.sum())
                    for thr in [1.0, 3.0, 5.0]:
                        metrics[f"sr{thr:.0f}_{cls_name}"] = float(np.mean(cls_errors <= thr))

        return metrics

    def compute_all(self):
        """Метрики для всех методов → dict[method_name → metrics]."""
        return {name: self.compute_metrics(name) for name in self.results}

    def print_summary(self):
        """Печать сводной таблицы."""
        all_m = self.compute_all()
        header = f"{'Method':12s} | {'RMSE':>6s} | {'MAE':>6s} | {'MedAE':>6s} | " \
                 f"{'SR@1°':>6s} | {'SR@3°':>6s} | {'SR@5°':>6s} | {'N':>6s}"
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for name, m in all_m.items():
            if not m:
                continue
            print(f"{name:12s} | {m['rmse']:6.2f} | {m['mae']:6.2f} | {m['median_ae']:6.2f} | "
                  f"{m.get('success_rate_1deg', 0):6.1%} | {m.get('success_rate_3deg', 0):6.1%} | "
                  f"{m.get('success_rate_5deg', 0):6.1%} | {m['n_samples']:6d}")
        print("=" * len(header))

        # Per-class
        has_per_class = any("rmse_direct" in m for m in all_m.values() if m)
        if has_per_class:
            print("\nPer-class RMSE:")
            for name, m in all_m.items():
                if not m:
                    continue
                d = m.get("rmse_direct", np.nan)
                t = m.get("rmse_two_targets", np.nan)
                print(f"  {name:12s} | direct: {d:.2f}° | two_targets: {t:.2f}°")

    def plot_all(self, output_dir, show=True):
        """Генерация всех графиков DOA-оценки."""
        os.makedirs(output_dir, exist_ok=True)
        all_m = self.compute_all()

        self._plot_rmse_comparison(all_m, output_dir, show)
        self._plot_success_rate(all_m, output_dir, show)
        self._plot_cdf(output_dir, show)
        self._plot_error_histogram(output_dir, show)
        self._plot_scatter(output_dir, show)
        self._plot_rmse_vs_angle(output_dir, show)
        self._plot_per_class_rmse(all_m, output_dir, show)

    def _plot_rmse_comparison(self, all_m, output_dir, show):
        """Барплот: RMSE + MAE + MedAE."""
        names = [n for n in all_m if all_m[n]]
        if not names:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.25
        rmse_vals = [all_m[n]["rmse"] for n in names]
        mae_vals = [all_m[n]["mae"] for n in names]
        medae_vals = [all_m[n]["median_ae"] for n in names]

        ax.bar(x - w, rmse_vals, w, label="RMSE", color="#F44336")
        ax.bar(x, mae_vals, w, label="MAE", color="#2196F3")
        ax.bar(x + w, medae_vals, w, label="Median AE", color="#4CAF50")

        for i, (r, m, md) in enumerate(zip(rmse_vals, mae_vals, medae_vals)):
            ax.text(i - w, r + 0.1, f"{r:.2f}", ha="center", fontsize=8)
            ax.text(i, m + 0.1, f"{m:.2f}", ha="center", fontsize=8)
            ax.text(i + w, md + 0.1, f"{md:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Error (°)")
        ax.set_title("DOA Error Metrics Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_error_comparison.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_success_rate(self, all_m, output_dir, show):
        """Success Rate @ разных порогах."""
        names = [n for n in all_m if all_m[n]]
        if not names:
            return
        thresholds = [1, 2, 3, 5, 10]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, name in enumerate(names):
            m = all_m[name]
            rates = [m.get(f"success_rate_{t}deg", 0) for t in thresholds]
            ax.plot(thresholds, rates, "o-", color=colors[i % len(colors)], label=name,
                    markersize=8, linewidth=2)
        ax.set_xlabel("Threshold (°)")
        ax.set_ylabel("Success Rate")
        ax.set_title("DOA Success Rate vs Threshold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(thresholds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_success_rate.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_cdf(self, output_dir, show):
        """CDF ошибок — все методы на одном графике."""
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (name, r) in enumerate(self.results.items()):
            errors = r["errors"]
            if len(errors) == 0:
                continue
            sorted_err = np.sort(errors)
            cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            ax.plot(sorted_err, cdf, color=colors[i % len(colors)], label=name, linewidth=2)

        ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="50%")
        ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="90%")
        ax.axvline(1.0, color="red", ls=":", alpha=0.3)
        ax.axvline(3.0, color="red", ls=":", alpha=0.3)
        ax.set_xlabel("Absolute Error (°)")
        ax.set_ylabel("CDF (cumulative probability)")
        ax.set_title("CDF of DOA Errors")
        ax.set_xlim(0, min(20, ax.get_xlim()[1]))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_cdf.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_error_histogram(self, output_dir, show):
        """Гистограмма ошибок для каждого метода."""
        n = len(self.results)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for ax, (name, r), color in zip(axes, self.results.items(), colors):
            errors = r["errors"]
            if len(errors) == 0:
                continue
            ax.hist(errors, bins=50, color=color, alpha=0.75, edgecolor="black", linewidth=0.5)
            rmse = np.sqrt(np.mean(errors ** 2))
            ax.axvline(rmse, color="red", ls="--", label=f"RMSE={rmse:.2f}°")
            ax.axvline(np.median(errors), color="green", ls=":", label=f"Median={np.median(errors):.2f}°")
            ax.set_xlabel("Error (°)")
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel("Count")
        plt.suptitle("DOA Error Distribution", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_error_histogram.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_scatter(self, output_dir, show):
        """Scatter: true vs predicted angles."""
        methods_with_scatter = [(n, r) for n, r in self.results.items()
                                if r["true_angles"] is not None and r["pred_angles"] is not None]
        if not methods_with_scatter:
            return

        n = len(methods_with_scatter)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (name, r) in zip(axes, methods_with_scatter):
            ta = r["true_angles"]
            pa = r["pred_angles"]
            ax.scatter(ta, pa, alpha=0.3, s=10, c="#2196F3")
            ax.plot([-90, 90], [-90, 90], "r--", linewidth=1.5, label="ideal")
            ax.set_xlabel("True angle (°)")
            ax.set_ylabel("Predicted angle (°)")
            ax.set_title(name)
            ax.set_xlim(-90, 90)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle("True vs Predicted DOA", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_scatter.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_rmse_vs_angle(self, output_dir, show):
        """RMSE vs true angle — для обнаружения edge effects."""
        methods_with_angles = [(n, r) for n, r in self.results.items()
                               if r["true_angles"] is not None]
        if not methods_with_angles:
            return

        angle_bins = np.linspace(-90, 90, 19)  # 18 бинов по 10°
        bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, (name, r) in enumerate(methods_with_angles):
            ta = r["true_angles"]
            errors = r["errors"]
            bin_rmse = []
            for b_lo, b_hi in zip(angle_bins[:-1], angle_bins[1:]):
                mask = (ta >= b_lo) & (ta < b_hi)
                if mask.sum() > 0:
                    bin_rmse.append(np.sqrt(np.mean(errors[mask] ** 2)))
                else:
                    bin_rmse.append(np.nan)
            ax.plot(bin_centers, bin_rmse, "o-", color=colors[i % len(colors)],
                    label=name, markersize=5, linewidth=1.5)

        ax.set_xlabel("True Angle (°)")
        ax.set_ylabel("RMSE (°)")
        ax.set_title("RMSE vs True Angle (edge effect analysis)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_rmse_vs_angle.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_per_class_rmse(self, all_m, output_dir, show):
        """RMSE по классам: direct vs two_targets."""
        has_data = any("rmse_direct" in m for m in all_m.values() if m)
        if not has_data:
            return

        names = [n for n in all_m if all_m[n] and "rmse_direct" in all_m[n]]
        if not names:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        w = 0.35
        d_vals = [all_m[n].get("rmse_direct", 0) for n in names]
        t_vals = [all_m[n].get("rmse_two_targets", 0) for n in names]

        ax.bar(x - w / 2, d_vals, w, label="direct (1 signal)", color="#2196F3")
        ax.bar(x + w / 2, t_vals, w, label="two targets", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("RMSE (°)")
        ax.set_title("DOA RMSE: Direct vs Two Targets")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "doa_per_class_rmse.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    def save_json(self, output_dir, prefix="doa_"):
        """Сохраняем все метрики в JSON."""
        all_m = self.compute_all()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{prefix}metrics.json")
        with open(path, "w") as f:
            json.dump(all_m, f, indent=2)
        return path


# ──────────────────────────────────────────────────────────────
#  4. EDA — визуализация данных
# ──────────────────────────────────────────────────────────────

def plot_data_eda(synth_data, real_data=None, output_dir=None, show=True):
    """
    Exploratory Data Analysis:
      1. Распределение углов (синтетика)
      2. Распределение классов (синтетика + реальные)
      3. Амплитуды сигналов (синтетика vs реальные)
      4. mb_metrix распределение

    synth_data: dict с ключами 'channels', 'angles', 'classes', 'mb_metrix'
    real_data: аналогичный dict (опционально)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── 1. Распределение углов ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    angles = synth_data["angles"]
    if angles.ndim == 3:
        angles = angles.squeeze(-1)
    axes[0].hist(np.rad2deg(angles[:, 0]), bins=60, alpha=0.7, label="angle 1", color="#2196F3")
    if angles.shape[1] > 1:
        axes[0].hist(np.rad2deg(angles[:, 1]), bins=60, alpha=0.5, label="angle 2", color="#FF9800")
    axes[0].set_xlabel("Angle (°)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Synth: Angle Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if real_data is not None and real_data.get("angles") is not None:
        ra = real_data["angles"]
        if ra.ndim == 3:
            ra = ra.squeeze(-1)
        axes[1].hist(ra[:, 0], bins=60, alpha=0.7, label="angle 1", color="#2196F3")
        if ra.shape[1] > 1:
            axes[1].hist(ra[:, 1], bins=60, alpha=0.5, label="angle 2", color="#FF9800")
        axes[1].set_xlabel("Angle (°)")
        axes[1].set_title("Real: Angle Distribution")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No real data", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Real: N/A")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "eda_angle_distribution.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # ── 2. Распределение классов ──
    fig, ax = plt.subplots(figsize=(8, 4))
    classes_s = synth_data["classes"].ravel()
    unique_s, counts_s = np.unique(classes_s, return_counts=True)
    x_pos = np.arange(3)
    bars_s = ax.bar(x_pos - 0.2, [counts_s[unique_s == c][0] if c in unique_s else 0 for c in range(3)],
                    0.35, label="Synth", color="#2196F3")

    if real_data is not None and real_data.get("classes") is not None:
        classes_r = real_data["classes"].ravel()
        unique_r, counts_r = np.unique(classes_r, return_counts=True)
        bars_r = ax.bar(x_pos + 0.2,
                        [counts_r[unique_r == c][0] if c in unique_r else 0 for c in range(3)],
                        0.35, label="Real", color="#FF9800")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["direct (0)", "multipath (1)", "two targets (2)"])
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution: Synth vs Real")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "eda_class_distribution.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # ── 3. Амплитуды сигналов ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ch_s = synth_data["channels"]
    if ch_s.ndim == 3:
        ch_s = ch_s.squeeze(-1)
    amps_s = np.abs(ch_s).ravel()
    axes[0].hist(amps_s, bins=100, alpha=0.7, color="#2196F3", density=True)
    axes[0].set_xlabel("|signal|")
    axes[0].set_title("Synth: Amplitude Distribution")
    axes[0].grid(alpha=0.3)

    if real_data is not None and real_data.get("channels") is not None:
        ch_r = real_data["channels"]
        if ch_r.ndim == 3:
            ch_r = ch_r.squeeze(-1)
        amps_r = np.abs(ch_r).ravel()
        axes[1].hist(amps_r, bins=100, alpha=0.7, color="#FF9800", density=True)
        axes[1].set_xlabel("|signal|")
        axes[1].set_title("Real: Amplitude Distribution")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No real data", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "eda_amplitude_distribution.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # ── 4. mb_metrix ──
    if synth_data.get("mb_metrix") is not None:
        mb = synth_data["mb_metrix"]
        if mb.ndim == 2 and mb.shape[1] >= 3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            labels = ["Projection", "OMP norm 1", "OMP norm 2"]
            for i, (ax, lbl) in enumerate(zip(axes, labels)):
                ax.hist(mb[:, i], bins=60, alpha=0.7, color="#4CAF50")
                ax.set_xlabel(lbl)
                ax.set_title(f"mb_metrix[{i}]: {lbl}")
                ax.grid(alpha=0.3)
            plt.suptitle("Synth: mb_metrix Distribution", fontsize=13)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, "eda_mb_metrix.png"), dpi=150)
            if show:
                plt.show()
            plt.close()
