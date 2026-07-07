"""
analysis/cascade_error.py
=========================
Анализ каскадной ошибки (Error Propagation Through Pipeline).

Суть: пайплайн двухстадийный (классификатор → DOA). Ошибка классификатора
каскадно влияет на DOA. Этот скрипт отвечает на вопрос:
  "Что является bottleneck системы — классификатор или DOA?"

Что делает:
  1. Oracle DOA:  RMSE при ИДЕАЛЬНОЙ классификации (ground truth classes)
  2. Pipeline DOA: RMSE при РЕАЛЬНОЙ классификации (предсказанные классы)
  3. Breakdown:    RMSE отдельно на сэмплах, где классификатор угадал vs ошибся
  4. Confusion impact: для каждой ошибки классификатора → какой эффект на DOA
  5. Визуализация: сводная таблица + графики

Запуск:
    python analysis/cascade_error.py
    python analysis/cascade_error.py --real-npz output/real_preprocessed.npz
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Добавляем корень проекта в path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    OUTPUT_DIR, N_ANT, N_GRID, MIMO_POS, DEVICE,
    CLASSIFIER_PATH, DOA_SIAMESE_PATH, DOA_LISTA_PATH,
    DOA_COVRECON_PATH, DOA_HIERGRU_PATH, SYNTH_NPZ,
    LISTA_LAYERS, COVRECON_HIDDEN,
)
from models import (
    ComplexNet, to_complex_tensor, build_steering_matrix,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA,
)


# ── Утилиты ──

def predict_doa_single(signal_np, method_name, doa_model, grid_deg):
    """Предсказание DOA для одного сигнала."""
    doa_model.eval()
    with torch.no_grad():
        if method_name in ("siamese", "hiergru"):
            x = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            if method_name == "siamese":
                _, doa_logits, _ = doa_model(x)
            else:
                doa_logits, _ = doa_model(x)
            spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]])

        elif method_name == "lista":
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            power, _ = doa_model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]])

        elif method_name == "covrecon":
            R_flat = cov_to_flat(signal_np.reshape(1, -1), N_ANT)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            out = doa_model(R_t)
            R_rec = flat_to_hermitian(out, N_ANT)
            R_np = R_rec[0].cpu().numpy()
            return root_music(R_np, N_sources=2)

    return np.array([])


def doa_error(pred_deg, true_deg_arr):
    """Минимальная ошибка между предсказанным и ближайшим истинным углом."""
    valid = true_deg_arr[(true_deg_arr >= -90) & (true_deg_arr <= 90)]
    if len(valid) == 0 or len(pred_deg) == 0:
        return np.nan
    diffs = [np.min(np.abs(valid - p)) for p in np.atleast_1d(pred_deg)]
    return np.sqrt(np.mean(np.array(diffs) ** 2))


def load_doa_models():
    """Загрузка всех обученных DOA-моделей."""
    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    configs = {
        "siamese": (DOA_SIAMESE_PATH, lambda: SiameseDOAModel(N_GRID, N_ANT)),
        "lista": (DOA_LISTA_PATH, lambda: DeepLISTA_DOA(A_np, n_layers=LISTA_LAYERS)),
        "covrecon": (DOA_COVRECON_PATH, lambda: CovReconstructor(N_ANT, hidden=COVRECON_HIDDEN)),
        "hiergru": (DOA_HIERGRU_PATH, lambda: HierarchicalGRU_DOA()),
    }

    models = {}
    for name, (path, factory) in configs.items():
        if os.path.exists(path):
            m = factory().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            models[name] = m
        else:
            print(f"  [!] {name}: модель не найдена ({path})")

    return models, grid_deg


def main():
    parser = argparse.ArgumentParser(description="Анализ каскадной ошибки")
    parser.add_argument("--synth-npz", type=str, default=SYNTH_NPZ)
    parser.add_argument("--n-eval", type=int, default=3000,
                        help="Число сэмплов для оценки")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Загрузка данных
    if not os.path.exists(args.synth_npz):
        print(f"[!] Синтетика не найдена: {args.synth_npz}")
        print("    Запустите: python prepare_data.py")
        return

    print("Загрузка данных...")
    npz = np.load(args.synth_npz, allow_pickle=True)
    from preprocessing import preprocess_channels
    channels = preprocess_channels(npz["channels"])
    angles = npz["azimuth"].squeeze(-1) if npz["azimuth"].ndim == 3 else npz["azimuth"]
    true_classes = npz["math_class"].ravel().astype(int)

    N = min(args.n_eval, len(channels))
    channels = channels[:N]
    angles = angles[:N]
    true_classes = true_classes[:N]

    # Загрузка классификатора
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[!] Классификатор не найден: {CLASSIFIER_PATH}")
        return
    classifier = ComplexNet(input_length=N_ANT).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    classifier.eval()

    # Предсказание классов
    X = to_complex_tensor(channels[:, :, None]).to(DEVICE)
    with torch.no_grad():
        pred_classes = classifier(X).argmax(dim=1).cpu().numpy()

    cls_acc = np.mean(pred_classes == true_classes)
    print(f"  Accuracy классификатора: {cls_acc:.4f}")

    # Загрузка DOA-моделей
    doa_models, grid_deg = load_doa_models()
    if not doa_models:
        print("[!] Ни одна DOA-модель не найдена")
        return

    # ── Основной анализ ──
    print("\n" + "=" * 70)
    print("АНАЛИЗ КАСКАДНОЙ ОШИБКИ")
    print("=" * 70)

    results = {}

    for method_name, doa_model in doa_models.items():
        print(f"\n  Метод: {method_name}")

        oracle_errors = []      # DOA при идеальной классификации
        pipeline_errors = []    # DOA при реальной классификации
        correct_cls_errors = [] # DOA когда классификатор угадал
        wrong_cls_errors = []   # DOA когда классификатор ошибся

        # Категории ошибок классификатора
        missed_as_multipath = 0     # class 0/2 → предсказано 1 (потеря)
        multipath_as_target = 0     # class 1 → предсказано 0/2 (ложный DOA)
        wrong_source_count = 0      # class 0↔2 (неправильное число целей)

        for i in range(N):
            true_deg = np.rad2deg(angles[i].ravel())

            # 1) Oracle: используем ИСТИННЫЙ класс
            if true_classes[i] != 1:  # не multipath
                pred_angles = predict_doa_single(
                    channels[i], method_name, doa_model, grid_deg
                )
                err = doa_error(pred_angles, true_deg)
                if not np.isnan(err):
                    oracle_errors.append(err)

            # 2) Pipeline: используем ПРЕДСКАЗАННЫЙ класс
            if pred_classes[i] != 1:  # классификатор не пометил как multipath
                pred_angles = predict_doa_single(
                    channels[i], method_name, doa_model, grid_deg
                )
                err = doa_error(pred_angles, true_deg)
                if not np.isnan(err):
                    pipeline_errors.append(err)

                    # Breakdown: правильная vs ошибочная классификация
                    if pred_classes[i] == true_classes[i]:
                        correct_cls_errors.append(err)
                    else:
                        wrong_cls_errors.append(err)

            # Подсчёт типов ошибок
            if true_classes[i] != 1 and pred_classes[i] == 1:
                missed_as_multipath += 1
            if true_classes[i] == 1 and pred_classes[i] != 1:
                multipath_as_target += 1
            if true_classes[i] in (0, 2) and pred_classes[i] in (0, 2) and true_classes[i] != pred_classes[i]:
                wrong_source_count += 1

        # Вычисление метрик
        oracle_rmse = np.sqrt(np.mean(np.array(oracle_errors) ** 2)) if oracle_errors else np.nan
        pipeline_rmse = np.sqrt(np.mean(np.array(pipeline_errors) ** 2)) if pipeline_errors else np.nan
        correct_rmse = np.sqrt(np.mean(np.array(correct_cls_errors) ** 2)) if correct_cls_errors else np.nan
        wrong_rmse = np.sqrt(np.mean(np.array(wrong_cls_errors) ** 2)) if wrong_cls_errors else np.nan
        degradation = pipeline_rmse - oracle_rmse if not (np.isnan(pipeline_rmse) or np.isnan(oracle_rmse)) else np.nan

        results[method_name] = {
            "oracle_rmse": oracle_rmse,
            "pipeline_rmse": pipeline_rmse,
            "correct_cls_rmse": correct_rmse,
            "wrong_cls_rmse": wrong_rmse,
            "degradation": degradation,
            "n_oracle": len(oracle_errors),
            "n_pipeline": len(pipeline_errors),
            "n_correct": len(correct_cls_errors),
            "n_wrong": len(wrong_cls_errors),
            "missed_as_multipath": missed_as_multipath,
            "multipath_as_target": multipath_as_target,
            "wrong_source_count": wrong_source_count,
            "oracle_errors": np.array(oracle_errors),
            "pipeline_errors": np.array(pipeline_errors),
        }

        print(f"    Oracle RMSE  (ideal cls):  {oracle_rmse:.2f}° ({len(oracle_errors)} сэмплов)")
        print(f"    Pipeline RMSE (real cls):   {pipeline_rmse:.2f}° ({len(pipeline_errors)} сэмплов)")
        print(f"    Деградация:                +{degradation:.2f}°")
        print(f"    RMSE (cls correct):         {correct_rmse:.2f}° ({len(correct_cls_errors)})")
        print(f"    RMSE (cls wrong):           {wrong_rmse:.2f}° ({len(wrong_cls_errors)})")
        print(f"    Потери (→ multipath):       {missed_as_multipath} сэмплов")
        print(f"    Ложные DOA (multipath→):    {multipath_as_target}")
        print(f"    Неверное число целей (0↔2): {wrong_source_count}")

    # ── Визуализация ──
    methods = list(results.keys())

    # 1. Барплот: Oracle vs Pipeline RMSE
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    w = 0.35
    oracle_vals = [results[m]["oracle_rmse"] for m in methods]
    pipeline_vals = [results[m]["pipeline_rmse"] for m in methods]
    ax.bar(x - w / 2, oracle_vals, w, label="Oracle (ideal cls)", color="#4CAF50")
    ax.bar(x + w / 2, pipeline_vals, w, label="Pipeline (real cls)", color="#F44336")
    for i in range(len(methods)):
        deg = results[methods[i]]["degradation"]
        if not np.isnan(deg):
            ax.annotate(f"+{deg:.2f}°", xy=(x[i] + w / 2, pipeline_vals[i]),
                        ha="center", va="bottom", fontsize=9, color="#F44336", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("RMSE (°)")
    ax.set_title("Cascade Error: Oracle vs Pipeline DOA RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cascade_oracle_vs_pipeline.png"), dpi=150)
    plt.close()

    # 2. Барплот: correct vs wrong classification
    fig, ax = plt.subplots(figsize=(10, 5))
    correct_vals = [results[m]["correct_cls_rmse"] for m in methods]
    wrong_vals = [results[m]["wrong_cls_rmse"] for m in methods]
    ax.bar(x - w / 2, correct_vals, w, label="Cls correct", color="#2196F3")
    ax.bar(x + w / 2, wrong_vals, w, label="Cls wrong", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("RMSE (°)")
    ax.set_title("DOA RMSE: Correct vs Wrong Classification")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cascade_correct_vs_wrong.png"), dpi=150)
    plt.close()

    # 3. CDF: Oracle vs Pipeline для лучшего метода
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, name in enumerate(methods):
        r = results[name]
        for errs, ls, label in [
            (r["oracle_errors"], "-", f"{name} oracle"),
            (r["pipeline_errors"], "--", f"{name} pipeline"),
        ]:
            if len(errs) > 0:
                sorted_e = np.sort(errs)
                cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
                ax.plot(sorted_e, cdf, ls=ls, color=colors[i % 4], label=label, linewidth=1.5)
    ax.axhline(0.9, color="gray", ls=":", alpha=0.5)
    ax.axvline(3.0, color="red", ls=":", alpha=0.3)
    ax.set_xlabel("Absolute Error (°)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF: Oracle vs Pipeline")
    ax.set_xlim(0, 20)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cascade_cdf_comparison.png"), dpi=150)
    plt.close()

    # 4. Типы ошибок классификатора (stacked bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    missed = [results[m]["missed_as_multipath"] for m in methods]
    false_doa = [results[m]["multipath_as_target"] for m in methods]
    wrong_cnt = [results[m]["wrong_source_count"] for m in methods]
    ax.bar(methods, missed, label="→ multipath (потеря)", color="#F44336")
    ax.bar(methods, false_doa, bottom=missed, label="multipath → target (ложный DOA)", color="#FF9800")
    ax.bar(methods, wrong_cnt, bottom=[m + f for m, f in zip(missed, false_doa)],
           label="0↔2 (неверное число целей)", color="#9C27B0")
    ax.set_ylabel("Количество сэмплов")
    ax.set_title("Типы ошибок классификатора, влияющие на DOA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cascade_error_types.png"), dpi=150)
    plt.close()

    # Сводная таблица
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    header = f"{'Method':12s} | {'Oracle':>8s} | {'Pipeline':>8s} | {'Degrad':>8s} | {'Correct':>8s} | {'Wrong':>8s}"
    print(header)
    print("-" * len(header))
    for m in methods:
        r = results[m]
        print(f"{m:12s} | {r['oracle_rmse']:7.2f}° | {r['pipeline_rmse']:7.2f}° | "
              f"+{r['degradation']:6.2f}° | {r['correct_cls_rmse']:7.2f}° | {r['wrong_cls_rmse']:7.2f}°")

    # Вывод: bottleneck
    mean_degrad = np.nanmean([results[m]["degradation"] for m in methods])
    mean_oracle = np.nanmean([results[m]["oracle_rmse"] for m in methods])
    ratio = mean_degrad / mean_oracle if mean_oracle > 0 else 0
    print(f"\nСредняя деградация от ошибок классификатора: +{mean_degrad:.2f}°")
    print(f"Средний Oracle RMSE: {mean_oracle:.2f}°")
    print(f"Отношение деградация/oracle: {ratio:.1%}")
    if ratio < 0.1:
        print("→ ВЫВОД: Bottleneck — DOA-модели. Классификатор достаточно хорош.")
    elif ratio < 0.3:
        print("→ ВЫВОД: Умеренное влияние классификатора. Стоит улучшить оба компонента.")
    else:
        print("→ ВЫВОД: Bottleneck — классификатор. Приоритет — улучшение классификации.")

    print(f"\nГрафики сохранены в {out_dir}")


if __name__ == "__main__":
    main()
