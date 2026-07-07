"""
config.py
=========
Централизованная конфигурация проекта.
Все пути, гиперпараметры и константы в одном месте.
"""

import os
import numpy as np

# ──────────────────────────────────────────────────────────────
#  Пути
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Реальные данные
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "train_validate")

# Выходная папка (модели, графики, npz)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Предобработанные NPZ (устанавливаются после prepare_data.py)
SYNTH_NPZ = os.path.join(OUTPUT_DIR, "synth_new.npz")
REAL_NPZ = os.path.join(OUTPUT_DIR, "real_preprocessed.npz")

# Сохранённые модели
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "classifier.pth")
DOA_SIAMESE_PATH = os.path.join(OUTPUT_DIR, "doa_siamese.pth")
DOA_LISTA_PATH = os.path.join(OUTPUT_DIR, "doa_lista.pth")
DOA_COVRECON_PATH = os.path.join(OUTPUT_DIR, "doa_covrecon.pth")
DOA_HIERGRU_PATH = os.path.join(OUTPUT_DIR, "doa_hiergru.pth")


# ──────────────────────────────────────────────────────────────
#  Радар: TI AWR1843 (2Tx × 4Rx = 8 виртуальных антенн)
# ──────────────────────────────────────────────────────────────

N_ANT = 8                                    # число виртуальных антенн
MIMO_POS = np.linspace(0, 3.5, N_ANT)       # позиции в долях λ/2
N_GRID = 361                                 # размер угловой сетки DOA (0.5°)


# ──────────────────────────────────────────────────────────────
#  Гиперпараметры обучения
# ──────────────────────────────────────────────────────────────

# Данные
N_SYNTH = 65536              # кол-во синтетических сэмплов
WHITEN_K = 64                # число сэмплов для ковариации (Wiener-Hopf)

# Классификатор
CLASSIFIER_EPOCHS = 60       # макс. эпохи pretrain на синтетике
EARLY_STOP_PATIENCE = 10     # early stopping по val_loss (P0-2)
FINETUNE_EPOCHS = 40         # эпохи finetune на реале
CLASSIFIER_LR = 1e-3
FINETUNE_LR = 5e-4
CLASSIFIER_WD = 1e-4
CLASS_WEIGHT_CAP = 15.0      # макс. вес для редкого класса (P0-5: было ~41x)

# DOA-модели
DOA_EPOCHS = 60
DOA_EPOCHS_HIERGRU = 120     # HierGRU нужно больше эпох (P0-3)
BATCH_SIZE = 256
DOA_ANGLE_LIMIT = 70.0       # ±70° — углы за пределами отбрасываются при eval (P0-4)

# LISTA
LISTA_LAYERS = 10
LISTA_LR = 1e-3        # увеличено с 5e-4 (FIX-03)

# Siamese
SIAMESE_LR = 1e-4

# GRU
GRU_LR = 1e-4

# CovRecon
COVRECON_LR = 3e-4
COVRECON_HIDDEN = 128


# ──────────────────────────────────────────────────────────────
#  Устройство
# ──────────────────────────────────────────────────────────────

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
