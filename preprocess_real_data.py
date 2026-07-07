"""
preprocess_real_data.py
=======================
CLI-утилита для предобработки реальных радарных данных.

Преобразует папку с .txt файлами → единый .npz с предобработанными сигналами.

Запуск:
    python preprocess_real_data.py --input train_validate --output output/real.npz
    python preprocess_real_data.py --input train_validate --output output/real.npz --normalize max --no-whiten
"""

import argparse
import os
import numpy as np

from dataset import dataset_from_real_data
from preprocessing import preprocess_channels, rms_normalize, max_normalize


def main():
    parser = argparse.ArgumentParser(description="Предобработка реальных радарных данных")
    parser.add_argument("--input", required=True, help="Папка с .txt файлами")
    parser.add_argument("--output", required=True, help="Выходной .npz файл")
    parser.add_argument("--K", type=int, default=64, help="Snapshots для ковариации (Wiener-Hopf)")
    parser.add_argument("--normalize", choices=["rms", "max", "none"], default="rms",
                        help="Тип нормализации (default: rms)")
    parser.add_argument("--no-whiten", action="store_true",
                        help="Отключить Wiener-Hopf whitening")
    args = parser.parse_args()

    # Загрузка сырых данных
    print(f"Загрузка данных из: {args.input}")
    data = dataset_from_real_data(args.input, normalize="none")
    channels = data["channels"]
    print(f"  Загружено: {len(channels)} сэмплов, {channels.shape[1]} каналов")

    # Предобработка
    if args.no_whiten:
        # Только нормализация, без whitening
        if args.normalize == "rms":
            channels = rms_normalize(channels)
        elif args.normalize == "max":
            channels = max_normalize(channels)
        print(f"  Нормализация: {args.normalize} (без whitening)")
    else:
        channels = preprocess_channels(channels, K=args.K, normalize=args.normalize)
        print(f"  Предобработка: whitening (K={args.K}) + {args.normalize}")

    # Сохранение
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        channels=channels,
        mb_metrix=data["mb_metrix"],
        azimuth=data["azimuth"],
        math_class=data["math class"],
        doppler_bin=data["doppler bin"],
        range_bin=data["range bin"],
        range_val=data["range"],
        source="real_preprocessed",
    )
    print(f"  Сохранено: {args.output}")


if __name__ == "__main__":
    main()
