"""
analysis/snr_analysis.py
========================
Анализ зависимости DOA RMSE и accuracy классификатора от SNR.

Вопрос: как деградирует качество системы при уменьшении SNR?

Метод:
  1. Генерируем синтетические данные при разных SNR: -5..30 дБ
  2. Для каждого SNR:
     a) Считаем accuracy классификатора
     b) Считаем RMSE каждого DOA-метода
     c) Считаем Success Rate @3°
  3. Строим кривые: RMSE vs SNR, Accuracy vs SNR, SR@3° vs SNR
  4. Определяем минимальный SNR для работоспособности

Запуск:
    python analysis/snr_analysis.py
    python analysis/snr_analysis.py --n-per-snr 500
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    OUTPUT_DIR, N_ANT, N_GRID, MIMO_POS, DEVICE,
    CLASSIFIER_PATH, DOA_SIAMESE_PATH, DOA_LISTA_PATH,
    DOA_COVRECON_PATH, DOA_HIERGRU_PATH,
    LISTA_LAYERS, COVRECON_HIDDEN,
)
from models import (
    ComplexNet, to_complex_tensor, build_steering_matrix,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA,
)


def generate_signal(class_label, snr_db, mimo_pos=None):
    """
    Генерирует один синтетический сигнал заданного класса и SNR.

    Returns: (signal [N_ant] complex, true_angles [2] в градусах)
    """
    if mimo_pos is None:
        mimo_pos = np.linspace(0, 3.5, 8)
    N_ant = len(mimo_pos)

    ch_err = np.ones(N_ant) + 0.1 * (np.random.randn(N_ant) + 1j * np.random.randn(N_ant))

    if class_label == 0:
        # Direct: один сигнал
        az = np.random.uniform(-60, 60)
        a = np.exp(1j * np.pi * mimo_pos * np.sin(np.deg2rad(az)))
        s = np.exp(1j * 2 * np.pi * np.random.rand())
        signal = s * a
        true_angles = np.array([az, 0.0])
    elif class_label == 1:
        # Multipath: θ_tx ≠ θ_rx
        az_tx = np.random.uniform(-60, 60)
        az_rx = -np.sign(az_tx) * np.random.uniform(5, 60)
        a_tx = np.exp(1j * np.pi * mimo_pos[:4] * np.sin(np.deg2rad(az_tx)))
        a_rx = np.exp(1j * np.pi * mimo_pos[4:] * np.sin(np.deg2rad(az_rx)))
        # Мультипас → steering vector не совпадает с нормальным
        a = np.concatenate([a_tx, a_rx])
        s = np.exp(1j * 2 * np.pi * np.random.rand())
        signal = s * a
        true_angles = np.array([az_tx, az_rx])
    else:
        # Two targets
        az1 = np.random.uniform(-60, 60)
        az2 = az1 + np.random.choice([-1, 1]) * np.random.uniform(5, 30)
        az2 = np.clip(az2, -85, 85)
        a1 = np.exp(1j * np.pi * mimo_pos * np.sin(np.deg2rad(az1)))
        a2 = np.exp(1j * np.pi * mimo_pos * np.sin(np.deg2rad(az2)))
        s1 = np.exp(1j * 2 * np.pi * np.random.rand())
        s2 = np.exp(1j * 2 * np.pi * np.random.rand())
        signal = s1 * a1 + s2 * a2
        true_angles = np.array([az1, az2])

    # Добавляем шум
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power * 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N_ant) + 1j * np.random.randn(N_ant))
    x = (signal + noise) * ch_err

    # RMS-нормализация
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    return x, true_angles


def predict_doa_single(signal_np, method_name, doa_model, grid_deg, n_ant=8):
    """Предсказание DOA."""
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
            return np.array([grid_deg[np.argmax(spectrum)]])

        elif method_name == "lista":
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            power, _ = doa_model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            return np.array([grid_deg[np.argmax(spectrum)]])

        elif method_name == "covrecon":
            R_flat = cov_to_flat(signal_np.reshape(1, -1), n_ant)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            out = doa_model(R_t)
            R_rec = flat_to_hermitian(out, n_ant)
            R_np = R_rec[0].cpu().numpy()
            return root_music(R_np, N_sources=1)

    return np.array([])


def load_all_models():
    """Загрузка классификатора и DOA-моделей."""
    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    # Классификатор
    classifier = None
    if os.path.exists(CLASSIFIER_PATH):
        classifier = ComplexNet(input_length=N_ANT).to(DEVICE)
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        classifier.eval()

    # DOA
    configs = {
        "siamese": (DOA_SIAMESE_PATH, lambda: SiameseDOAModel(N_GRID, N_ANT)),
        "lista": (DOA_LISTA_PATH, lambda: DeepLISTA_DOA(A_np, n_layers=LISTA_LAYERS)),
        "covrecon": (DOA_COVRECON_PATH, lambda: CovReconstructor(N_ANT, hidden=COVRECON_HIDDEN)),
        "hiergru": (DOA_HIERGRU_PATH, lambda: HierarchicalGRU_DOA()),
    }

    doa_models = {}
    for name, (path, factory) in configs.items():
        if os.path.exists(path):
            m = factory().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            doa_models[name] = m

    return classifier, doa_models, grid_deg


def main():
    parser = argparse.ArgumentParser(description="Анализ SNR-зависимости")
    parser.add_argument("--n-per-snr", type=int, default=300,
                        help="Сэмплов на каждое значение SNR")
    parser.add_argument("--snr-min", type=float, default=-5)
    parser.add_argument("--snr-max", type=float, default=30)
    parser.add_argument("--snr-step", type=float, default=5)
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    classifier, doa_models, grid_deg = load_all_models()
    if classifier is None:
        print("[!] Классификатор не найден")
        return
    if not doa_models:
        print("[!] DOA-модели не найдены")
        return

    snr_values = np.arange(args.snr_min, args.snr_max + 0.1, args.snr_step)
    N_per = args.n_per_snr

    print("=" * 60)
    print("АНАЛИЗ SNR-ЗАВИСИМОСТИ")
    print(f"  SNR: {args.snr_min}..{args.snr_max} дБ, шаг {args.snr_step}")
    print(f"  Сэмплов на SNR: {N_per}")
    print("=" * 60)

    # Результаты
    cls_accuracy = []      # accuracy классификатора vs SNR
    doa_rmse = {name: [] for name in doa_models}
    doa_sr3 = {name: [] for name in doa_models}   # success rate @3°

    for snr in snr_values:
        print(f"  SNR = {snr:+5.1f} дБ...", end="", flush=True)

        # Генерируем данные: равные доли 3 классов
        n_per_class = N_per // 3
        signals = []
        angles_list = []
        true_classes = []

        for cls in [0, 1, 2]:
            for _ in range(n_per_class):
                sig, angs = generate_signal(cls, snr)
                signals.append(sig)
                angles_list.append(angs)
                true_classes.append(cls)

        signals = np.array(signals)
        angles_arr = np.array(angles_list)
        true_classes = np.array(true_classes)

        # Классификация
        X = to_complex_tensor(signals[:, :, None]).to(DEVICE)
        with torch.no_grad():
            pred_cls = classifier(X).argmax(dim=1).cpu().numpy()

        acc = np.mean(pred_cls == true_classes)
        cls_accuracy.append(acc)

        # DOA (только class 0 — direct, один угол, по ground truth)
        mask_direct = true_classes == 0
        ch_direct = signals[mask_direct]
        ang_direct = angles_arr[mask_direct]

        for name, model in doa_models.items():
            errors = []
            for i in range(len(ch_direct)):
                pred = predict_doa_single(ch_direct[i], name, model, grid_deg)
                true_deg = ang_direct[i, 0]  # первый угол
                if len(pred) > 0:
                    err = np.abs(pred[0] - true_deg)
                    errors.append(err)

            rmse = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else np.nan
            sr3 = np.mean(np.array(errors) <= 3.0) if errors else 0.0
            doa_rmse[name].append(rmse)
            doa_sr3[name].append(sr3)

        print(f" acc={acc:.3f}", end="")
        for name in doa_models:
            print(f"  {name}={doa_rmse[name][-1]:.1f}°", end="")
        print()

    # ── Визуализация ──
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    # 1. Accuracy классификатора vs SNR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_values, cls_accuracy, "o-", color="#F44336", linewidth=2, markersize=6)
    ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="90%")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Accuracy vs SNR")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_classifier_accuracy.png"), dpi=150)
    plt.close()

    # 2. DOA RMSE vs SNR
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, rmse_list) in enumerate(doa_rmse.items()):
        ax.plot(snr_values, rmse_list, "o-", color=colors[i % 4], label=name,
                linewidth=2, markersize=5)
    ax.axhline(3.0, color="red", ls=":", alpha=0.3, label="3° threshold")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (°)")
    ax.set_title("DOA RMSE vs SNR (direct signal, single target)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_doa_rmse.png"), dpi=150)
    plt.close()

    # 3. Success Rate @3° vs SNR
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, sr_list) in enumerate(doa_sr3.items()):
        ax.plot(snr_values, sr_list, "o-", color=colors[i % 4], label=name,
                linewidth=2, markersize=5)
    ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="90%")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Success Rate @3°")
    ax.set_title("DOA Success Rate @3° vs SNR")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_doa_success_rate.png"), dpi=150)
    plt.close()

    # 4. Комбинированный график
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1.plot(snr_values, cls_accuracy, "o-", color="#F44336", linewidth=2)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("System Performance vs SNR")
    ax1.axhline(0.9, color="gray", ls="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.legend(["Classifier Accuracy"])

    for i, (name, rmse_list) in enumerate(doa_rmse.items()):
        ax2.plot(snr_values, rmse_list, "o-", color=colors[i % 4], label=name, linewidth=2)
    ax2.set_ylabel("RMSE (°)")
    ax2.axhline(3.0, color="red", ls=":", alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    for i, (name, sr_list) in enumerate(doa_sr3.items()):
        ax3.plot(snr_values, sr_list, "o-", color=colors[i % 4], label=name, linewidth=2)
    ax3.set_xlabel("SNR (dB)")
    ax3.set_ylabel("SR @3°")
    ax3.axhline(0.9, color="gray", ls="--", alpha=0.5)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_combined.png"), dpi=150)
    plt.close()

    # Сводная таблица
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА: минимальный SNR для работоспособности")
    print("=" * 70)

    # Минимальный SNR для acc >= 0.9
    cls_arr = np.array(cls_accuracy)
    snr_cls = snr_values[np.argmax(cls_arr >= 0.9)] if np.any(cls_arr >= 0.9) else f">{args.snr_max}"
    print(f"  Классификатор acc≥90%:  SNR ≥ {snr_cls} дБ")

    for name in doa_models:
        sr_arr = np.array(doa_sr3[name])
        snr_doa = snr_values[np.argmax(sr_arr >= 0.9)] if np.any(sr_arr >= 0.9) else f">{args.snr_max}"
        print(f"  {name:12s} SR@3°≥90%: SNR ≥ {snr_doa} дБ")

    print(f"\n  Графики сохранены в {out_dir}")


if __name__ == "__main__":
    main()
