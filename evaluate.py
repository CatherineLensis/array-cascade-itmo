"""
evaluate.py
===========
Полная оценка обученных моделей на реальных данных.

Использует metrics.py для расширенного анализа:
  1. Классификатор → confusion matrix, ROC-AUC, per-class F1
  2. DOA-модели → RMSE, MAE, Success Rate, CDF, scatter, per-class
  3. EDA → распределение данных
  4. JSON-отчёты для автоматизации

Запуск:
    python evaluate.py
    python evaluate.py --real-npz output/real_preprocessed.npz
    python evaluate.py --no-doa
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    OUTPUT_DIR, N_ANT, N_GRID, MIMO_POS, DEVICE,
    CLASSIFIER_PATH, DOA_SIAMESE_PATH, DOA_LISTA_PATH,
    DOA_COVRECON_PATH, DOA_HIERGRU_PATH, REAL_NPZ,
    LISTA_LAYERS, COVRECON_HIDDEN,
)
from models import (
    ComplexNet, to_complex_tensor, build_steering_matrix,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA,
)
from metrics import classifier_full_report, DOAEvaluator


def load_real_data(npz_path):
    """Загрузка реальных данных из NPZ."""
    npz = np.load(npz_path, allow_pickle=True)
    return {
        "channels": npz["channels"],
        "azimuth": npz["azimuth"],
        "math_class": npz["math_class"].astype(int),
        "mb_metrix": npz.get("mb_metrix", None),
    }


def predict_doa_single(signal_np, method_name, doa_model, grid_deg):
    """Предсказание DOA для одного сигнала [N_ant] complex."""
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
            return np.array([grid_deg[peak_idx]]), spectrum

        elif method_name == "lista":
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            power, _ = doa_model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]]), spectrum

        elif method_name == "covrecon":
            R_flat = cov_to_flat(signal_np.reshape(1, -1), N_ANT)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            out = doa_model(R_t)
            R_rec = flat_to_hermitian(out, N_ANT)
            R_np = R_rec[0].cpu().numpy()
            n_src = 2  # default для covrecon
            pred_deg = root_music(R_np, N_sources=n_src)
            return pred_deg, None

    return np.array([]), None


def main():
    parser = argparse.ArgumentParser(description="Полная оценка моделей на реальных данных")
    parser.add_argument("--real-npz", type=str, default=REAL_NPZ,
                        help="Путь к NPZ с предобработанными реальными данными")
    parser.add_argument("--no-doa", action="store_true",
                        help="Пропустить оценку DOA (только классификатор)")
    args = parser.parse_args()

    eval_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Проверяем наличие файлов
    if not os.path.exists(args.real_npz):
        print(f"[!] Файл не найден: {args.real_npz}")
        print("    Сначала запустите: python prepare_data.py")
        return
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"[!] Модель классификатора не найдена: {CLASSIFIER_PATH}")
        print("    Сначала запустите: python run_full_pipeline.py")
        return

    # Загрузка данных
    print("Загрузка реальных данных...")
    data = load_real_data(args.real_npz)
    channels = data["channels"]
    true_cls = data["math_class"]
    azimuth = data["azimuth"]  # градусы

    # ── 1. Классификатор ──
    print("\n" + "=" * 60)
    print("ОЦЕНКА КЛАССИФИКАТОРА")
    print("=" * 60)

    classifier = ComplexNet(input_length=N_ANT).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    classifier.eval()

    X = to_complex_tensor(channels[:, :, None] if channels.ndim == 2 else channels)
    X = X.to(DEVICE)

    with torch.no_grad():
        logits = classifier(X)
        pred_cls = logits.argmax(dim=1).cpu().numpy()
        proba = torch.softmax(logits.float(), dim=1).cpu().numpy()

    # Полный отчёт
    clf_result = classifier_full_report(
        true_cls, pred_cls, y_proba=proba,
        output_dir=eval_dir, show=True,
    )
    print(f"\n  Accuracy:    {clf_result['accuracy']:.4f}")
    print(f"  F1 macro:    {clf_result['f1_macro']:.4f}")
    print(f"  F1 weighted: {clf_result['f1_weighted']:.4f}")
    for i, name in enumerate(["direct", "multipath", "two_targets"]):
        print(f"  F1 {name}: {clf_result['f1_per_class'][i]:.4f}")
    if "auc_class_0" in clf_result:
        for i, name in enumerate(["direct", "multipath", "two_targets"]):
            print(f"  AUC {name}: {clf_result.get(f'auc_class_{i}', 0):.4f}")

    if args.no_doa:
        print(f"\nРезультаты сохранены в {eval_dir}")
        return

    # ── 2. DOA-модели ──
    print("\n" + "=" * 60)
    print("ОЦЕНКА DOA-МОДЕЛЕЙ")
    print("=" * 60)

    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    doa_configs = {
        "siamese": (DOA_SIAMESE_PATH, lambda: SiameseDOAModel(N_GRID, N_ANT)),
        "lista": (DOA_LISTA_PATH, lambda: DeepLISTA_DOA(A_np, n_layers=LISTA_LAYERS)),
        "covrecon": (DOA_COVRECON_PATH, lambda: CovReconstructor(N_ANT, hidden=COVRECON_HIDDEN)),
        "hiergru": (DOA_HIERGRU_PATH, lambda: HierarchicalGRU_DOA()),
    }

    evaluator = DOAEvaluator(grid_deg)

    for name, (path, factory) in doa_configs.items():
        if not os.path.exists(path):
            print(f"  {name:12s} | Модель не найдена: {path}")
            continue

        model = factory().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()

        errors = []
        pred_classes_list = []
        true_angles_list = []
        pred_angles_list = []

        for i in range(len(channels)):
            if pred_cls[i] == 1:  # multipath → skip
                continue

            pred_angles, _ = predict_doa_single(channels[i], name, model, grid_deg)
            true_deg = azimuth[i].ravel()
            valid = true_deg[(true_deg >= -90) & (true_deg <= 90)]
            if len(valid) == 0 or len(pred_angles) == 0:
                continue

            min_err = min(np.min(np.abs(valid - p)) for p in pred_angles)
            errors.append(min_err)
            pred_classes_list.append(pred_cls[i])
            true_angles_list.append(valid[0])
            pred_angles_list.append(pred_angles[0])

        evaluator.add_method(
            name, errors,
            pred_classes=pred_classes_list,
            true_angles=true_angles_list,
            pred_angles=pred_angles_list,
        )

    # Печать сводной таблицы
    evaluator.print_summary()

    # Все графики
    evaluator.plot_all(eval_dir, show=True)
    json_path = evaluator.save_json(eval_dir)

    print(f"\nВсе результаты и графики сохранены в {eval_dir}")
    print(f"  Метрики JSON: {json_path}")


if __name__ == "__main__":
    main()
