"""
analysis/angular_resolution.py
==============================
Анализ разрешающей способности DOA-методов.

Вопрос: при каком минимальном угловом разделении (Δθ) между двумя целями
метод ещё способен их различить?

Метод:
  1. Генерируем пары сигналов с class=2 (two targets)
     при разном Δθ: от 1° до 40° с шагом 1°
  2. Для каждого Δθ запускаем все DOA-методы
  3. Считаем "Success" если оба угла найдены с ошибкой < 3°
  4. Строим кривую Success Rate vs Δθ → разрешающая способность

Запуск:
    python analysis/angular_resolution.py
    python analysis/angular_resolution.py --n-trials 200
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
    DOA_SIAMESE_PATH, DOA_LISTA_PATH,
    DOA_COVRECON_PATH, DOA_HIERGRU_PATH,
    LISTA_LAYERS, COVRECON_HIDDEN,
)
from models import (
    build_steering_matrix,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA,
)
from preprocessing import rms_normalize


def generate_two_target_signal(theta1_deg, theta2_deg, snr_db=20, mimo_pos=None):
    """
    Генерирует синтетический 8-канальный сигнал с двумя целями.

    theta1_deg, theta2_deg — углы прихода в градусах
    snr_db — отношение сигнал/шум
    """
    if mimo_pos is None:
        mimo_pos = np.linspace(0, 3.5, 8)
    N_ant = len(mimo_pos)

    # Steering-векторы
    a1 = np.exp(1j * np.pi * mimo_pos * np.sin(np.deg2rad(theta1_deg)))
    a2 = np.exp(1j * np.pi * mimo_pos * np.sin(np.deg2rad(theta2_deg)))

    # Комплексные амплитуды (случайные фазы, равные мощности)
    s1 = np.exp(1j * 2 * np.pi * np.random.rand())
    s2 = np.exp(1j * 2 * np.pi * np.random.rand())

    # Сигнал + шум
    signal = s1 * a1 + s2 * a2
    noise_power = 10 ** (-snr_db / 10) * 2  # две цели → мощность сигнала ≈ 2
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N_ant) + 1j * np.random.randn(N_ant))

    # Канальные ошибки (как в dataset.py)
    ch_err = np.ones(N_ant) + 0.1 * (np.random.randn(N_ant) + 1j * np.random.randn(N_ant))

    x = (signal + noise) * ch_err
    # RMS-нормализация
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    return x


def predict_doa(signal_np, method_name, doa_model, grid_deg):
    """Предсказание DOA — возвращает массив углов."""
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
            # Ищем 2 пика (для two-target)
            from scipy.signal import find_peaks
            peaks, props = find_peaks(spectrum, height=0.1, distance=5)
            if len(peaks) >= 2:
                # Два наибольших пика
                top2 = peaks[np.argsort(spectrum[peaks])[-2:]]
                return np.sort(grid_deg[top2])
            elif len(peaks) == 1:
                return np.array([grid_deg[peaks[0]]])
            else:
                return np.array([grid_deg[np.argmax(spectrum)]])

        elif method_name == "lista":
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            power, _ = doa_model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(spectrum, height=0.1, distance=5)
            if len(peaks) >= 2:
                top2 = peaks[np.argsort(spectrum[peaks])[-2:]]
                return np.sort(grid_deg[top2])
            elif len(peaks) == 1:
                return np.array([grid_deg[peaks[0]]])
            else:
                return np.array([grid_deg[np.argmax(spectrum)]])

        elif method_name == "covrecon":
            R_flat = cov_to_flat(signal_np.reshape(1, -1), N_ANT)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            out = doa_model(R_t)
            R_rec = flat_to_hermitian(out, N_ANT)
            R_np = R_rec[0].cpu().numpy()
            return root_music(R_np, N_sources=2)

    return np.array([])


def check_resolution(pred_angles, true_angles, tol=3.0):
    """
    Проверяем, разрешены ли оба угла.
    Успех: для каждого истинного угла найден предсказанный с ошибкой < tol.
    """
    if len(pred_angles) < 2 or len(true_angles) < 2:
        return False

    t1, t2 = sorted(true_angles)
    p_sorted = np.sort(pred_angles[:2])

    # Проверяем оба совпадения
    err1 = np.min(np.abs(p_sorted - t1))
    err2 = np.min(np.abs(p_sorted - t2))
    return err1 < tol and err2 < tol


def load_doa_models():
    """Загрузка DOA-моделей."""
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

    return models, grid_deg


def main():
    parser = argparse.ArgumentParser(description="Анализ разрешающей способности DOA")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Число проб Монте-Карло на каждый Δθ")
    parser.add_argument("--snr", type=float, default=20.0,
                        help="SNR в дБ")
    parser.add_argument("--tol", type=float, default=3.0,
                        help="Порог успеха (°)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    doa_models, grid_deg = load_doa_models()
    if not doa_models:
        print("[!] Ни одна DOA-модель не найдена. Запустите run_full_pipeline.py")
        return

    delta_thetas = np.arange(1, 41, 1)  # от 1° до 40°
    N_trials = args.n_trials

    print("=" * 60)
    print("АНАЛИЗ РАЗРЕШАЮЩЕЙ СПОСОБНОСТИ DOA")
    print(f"  SNR = {args.snr} дБ, N_trials = {N_trials}, Порог = {args.tol}°")
    print("=" * 60)

    # Результаты: method → delta_theta → success_rate
    all_results = {name: [] for name in doa_models}

    for dt in delta_thetas:
        print(f"  Δθ = {dt:2d}°...", end="", flush=True)
        for name, model in doa_models.items():
            successes = 0
            for trial in range(N_trials):
                # Центральный угол: случайный в ±60°
                center = np.random.uniform(-60, 60)
                theta1 = center - dt / 2
                theta2 = center + dt / 2

                # Проверяем, что оба в пределах ±85°
                if abs(theta1) > 85 or abs(theta2) > 85:
                    theta1 = np.clip(theta1, -85, 85)
                    theta2 = np.clip(theta2, -85, 85)

                sig = generate_two_target_signal(theta1, theta2, snr_db=args.snr)
                pred = predict_doa(sig, name, model, grid_deg)

                if check_resolution(pred, [theta1, theta2], tol=args.tol):
                    successes += 1

            rate = successes / N_trials
            all_results[name].append(rate)
        print(" done")

    # ── Визуализация ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (name, rates) in enumerate(all_results.items()):
        ax.plot(delta_thetas, rates, "o-", color=colors[i % 4], label=name,
                markersize=4, linewidth=2)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="50%")
    ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="90%")
    ax.set_xlabel("Angular Separation Δθ (°)")
    ax.set_ylabel("Resolution Success Rate")
    ax.set_title(f"Angular Resolution Analysis (SNR={args.snr} dB, tol={args.tol}°)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "angular_resolution.png"), dpi=150)
    plt.close()

    # Определяем Δθ_min для 50% и 90% success rate
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ: минимальное Δθ для разрешения двух целей")
    print("=" * 60)
    for name, rates in all_results.items():
        rates_arr = np.array(rates)
        dt_50 = delta_thetas[np.argmax(rates_arr >= 0.5)] if np.any(rates_arr >= 0.5) else ">40"
        dt_90 = delta_thetas[np.argmax(rates_arr >= 0.9)] if np.any(rates_arr >= 0.9) else ">40"
        print(f"  {name:12s} | Δθ@50% = {dt_50}°  |  Δθ@90% = {dt_90}°")

    # Rayleigh limit для ULA из 8 элементов
    rayleigh_deg = np.rad2deg(0.886 / (N_ANT * 0.5))  # ≈ 12.7° для 8-el ULA
    print(f"\n  Rayleigh limit (8-el ULA): ≈ {rayleigh_deg:.1f}°")
    print(f"\n  Графики сохранены в {out_dir}")


if __name__ == "__main__":
    main()
