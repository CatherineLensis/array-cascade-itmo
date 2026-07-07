"""
prepare_data.py
===============
Подготовка данных: генерация синтетики + предобработка реальных данных.

Создаёт .npz файлы, готовые для обучения:
  - output/synth_new.npz       — синтетические данные
  - output/real_preprocessed.npz — реальные данные (whitened + RMS)

Запуск:
    python prepare_data.py
    python prepare_data.py --n-synth 131072 --normalize rms
    python prepare_data.py --skip-synth          # только реальные
    python prepare_data.py --skip-real            # только синтетика
"""

import argparse
import os
import time
import numpy as np

from config import (
    REAL_DATA_DIR, OUTPUT_DIR, N_SYNTH, WHITEN_K,
    SYNTH_NPZ, REAL_NPZ,
)
from dataset import dataset_gen_final_experiment, dataset_from_real_data
from preprocessing import preprocess_channels


def prepare_synthetic(n_samples, output_path):
    """Генерация синтетики (8 антенн) и сохранение в NPZ."""
    print(f"\n{'='*50}")
    print(f"Генерация синтетики: {n_samples} сэмплов...")
    print(f"{'='*50}")

    t0 = time.time()
    X, angle, y, mb_metrix = dataset_gen_final_experiment(n_samples, init_flag=0)
    elapsed = time.time() - t0

    print(f"  Сгенерировано: {len(X)} сэмплов за {elapsed:.1f} сек")
    print(f"  channels: {X.shape}, angles: {angle.shape}")
    print(f"  classes: {np.unique(y.ravel(), return_counts=True)}")
    print(f"  mb_metrix: {mb_metrix.shape}")

    np.savez(
        output_path,
        channels=X,           # [N, 8, 1] complex
        azimuth=angle,        # [N, 2, 1] float (радианы)
        math_class=y,         # [N, 1] float (0/1/2)
        mb_metrix=mb_metrix,  # [N, 3] float
        source="synth_new",
    )
    print(f"  Сохранено: {output_path}")


def prepare_real(input_dir, output_path, K=64, normalize="rms"):
    """Предобработка реальных данных и сохранение в NPZ."""
    print(f"\n{'='*50}")
    print(f"Предобработка реальных данных: {input_dir}")
    print(f"{'='*50}")

    if not os.path.isdir(input_dir):
        print(f"  [!] Папка не найдена: {input_dir}")
        return

    # Загрузка сырых данных (без нормализации)
    data = dataset_from_real_data(input_dir, normalize="none")
    channels_raw = data["channels"]

    print(f"  Загружено: {len(channels_raw)} сэмплов")
    print(f"  classes: {np.unique(data['math class'], return_counts=True)}")

    # Предобработка: whitening + RMS
    channels = preprocess_channels(channels_raw, K=K, normalize=normalize)
    print(f"  Предобработка: whitening (K={K}) + {normalize}")

    # Сохранение
    np.savez(
        output_path,
        channels=channels,               # [N, 8] complex (предобработанные)
        channels_raw=channels_raw,        # [N, 8] complex (сырые, для сравнения)
        azimuth=data["azimuth"],          # [N, 2] float (градусы)
        math_class=data["math class"],    # [N] float
        mb_metrix=data["mb_metrix"],      # [N, 3] float
        doppler_bin=data["doppler bin"],
        range_bin=data["range bin"],
        range_val=data["range"],
        source="real_preprocessed",
    )
    print(f"  Сохранено: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка данных для обучения моделей"
    )
    parser.add_argument("--n-synth", type=int, default=N_SYNTH,
                        help=f"Число синтетических сэмплов (default: {N_SYNTH})")
    parser.add_argument("--real-dir", type=str, default=REAL_DATA_DIR,
                        help="Папка с реальными .txt файлами")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Папка для сохранения NPZ")
    parser.add_argument("--K", type=int, default=WHITEN_K,
                        help="Число сэмплов для ковариации")
    parser.add_argument("--normalize", choices=["rms", "max", "none"],
                        default="rms", help="Тип нормализации")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Пропустить генерацию синтетики")
    parser.add_argument("--skip-real", action="store_true",
                        help="Пропустить обработку реальных данных")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    synth_path = os.path.join(args.output_dir, "synth_new.npz")
    real_path = os.path.join(args.output_dir, "real_preprocessed.npz")

    if not args.skip_synth:
        prepare_synthetic(args.n_synth, synth_path)

    if not args.skip_real:
        prepare_real(args.real_dir, real_path, K=args.K, normalize=args.normalize)

    print(f"\n{'='*50}")
    print("Подготовка данных завершена!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
