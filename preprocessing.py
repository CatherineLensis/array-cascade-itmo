"""
preprocessing.py
================
Предобработка сигналов: Wiener-Hopf whitening + RMS нормализация.

Использование:
    from preprocessing import preprocess_channels, rms_normalize

    # Полная предобработка
    channels_clean = preprocess_channels(channels_raw)  # [N, 8] complex

    # Или отдельно
    channels = rms_normalize(channels)
"""

import numpy as np
from dataset import wiener_hopf_whiten


def rms_normalize(channels, eps=1e-12):
    """
    RMS-нормировка по каждому сэмплу.

    Каждый вектор нормируется так, что среднеквадратичная амплитуда = 1.

    Параметры:
        channels — [N, M] complex (N сэмплов, M антенн)
        eps      — регуляризация для деления
    Returns:
        [N, M] complex — нормированные сигналы
    """
    power = np.mean(np.abs(channels) ** 2, axis=1, keepdims=True)
    return channels / np.sqrt(power + eps)


def max_normalize(channels, eps=1e-12):
    """
    Max-нормировка: каждый сэмпл делится на max(|x|).

    Совместимо с нормализацией в dataset_sh.py (старый подход).
    """
    max_vals = np.max(np.abs(channels), axis=1, keepdims=True)
    return channels / (max_vals + eps)


def preprocess_channels(channels, K=64, normalize="rms"):
    """
    Полная предобработка: whitening → нормализация.

    Это «новый подход» (в отличие от max-нормализации в старом).

    Параметры:
        channels  — [N, M] complex или [N, M, 1] complex
        K         — число сэмплов для оценки ковариации (Wiener-Hopf)
        normalize — 'rms' | 'max' | 'none'
    Returns:
        [N, M] complex — предобработанные сигналы
    """
    if channels.ndim == 3:
        channels = channels.squeeze(-1)  # [N, M]

    # 1. Пространственное отбеливание по Винеру-Хопфу
    channels = wiener_hopf_whiten(channels, K=K)

    # 2. Нормализация
    if normalize == "rms":
        channels = rms_normalize(channels)
    elif normalize == "max":
        channels = max_normalize(channels)
    # 'none' — без нормализации

    return channels
