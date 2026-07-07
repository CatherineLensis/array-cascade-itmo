"""
run_full_pipeline.py
====================
Единый пайплайн обработки радарных данных (новый подход):

  Шаг 1 — Подготовка данных  (синтетика + реальные, whitening + RMS)
  Шаг 2 — Обучение 3-классового классификатора
  Шаг 3 — Обучение 4 методов определения углов (DOA)
  Шаг 4 — Полный инференс: classify → DOA
  Шаг 5 — Сравнение методов (RMSE, графики)

Запуск:
    python run_full_pipeline.py

Требуемые зависимости:
    pip install -r requirements.txt
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from metrics import MetricsTracker, DOAEvaluator, classifier_full_report, plot_data_eda

# Убеждаемся, что наш каталог в sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    REAL_DATA_DIR, OUTPUT_DIR, N_SYNTH, WHITEN_K,
    CLASSIFIER_EPOCHS, FINETUNE_EPOCHS, DOA_EPOCHS, BATCH_SIZE,
    N_ANT, N_GRID, MIMO_POS, DEVICE,
    CLASSIFIER_PATH, DOA_SIAMESE_PATH, DOA_LISTA_PATH,
    DOA_COVRECON_PATH, DOA_HIERGRU_PATH,
    REAL_NPZ, SYNTH_NPZ,
    CLASSIFIER_LR, FINETUNE_LR, CLASSIFIER_WD,
    LISTA_LAYERS, LISTA_LR, SIAMESE_LR, GRU_LR,
    COVRECON_LR, COVRECON_HIDDEN,
    EARLY_STOP_PATIENCE, CLASS_WEIGHT_CAP,
    DOA_EPOCHS_HIERGRU, DOA_ANGLE_LIMIT,
)
from dataset import (
    dataset_gen_final_experiment,
    dataset_from_real_data,
)
from preprocessing import preprocess_channels
from models import (
    ComplexNet, to_complex_tensor, build_steering_matrix,
    # Метод 1
    SiameseDOAModel, sparse_augment, nt_xent_loss,
    # Метод 2
    DeepLISTA_DOA,
    # Метод 3
    CovReconstructor, flat_to_hermitian, cov_to_flat, root_music,
    # Метод 4
    HierarchicalGRU_DOA,
)


def make_doa_labels(angles_np, n_grid=361, sigma_deg=1.5):
    """
    Углы [N, 2(, 1)] → Gaussian soft labels [N, n_grid].

    Вместо one-hot (1 единица из 361) используем гауссиан с σ=1.5°,
    что даёт ~6-8 ненулевых позиций. Это кардинально улучшает
    BCE-обучение для DOA-методов (LISTA, HierGRU, Siamese).

    angles_np в радианах.
    """
    grid_deg = np.linspace(-90, 90, n_grid)
    labels = np.zeros((len(angles_np), n_grid), dtype=np.float32)
    for i, angs in enumerate(angles_np):
        for a_rad in np.atleast_1d(angs).ravel():
            a_deg = np.rad2deg(a_rad)
            if -90 <= a_deg <= 90:
                # Gaussian soft label вместо one-hot
                gauss = np.exp(-0.5 * ((grid_deg - a_deg) / sigma_deg) ** 2)
                labels[i] = np.maximum(labels[i], gauss)  # max для нескольких углов
    # Нормализуем пик к 1.0 (для каждого сэмпла)
    row_max = labels.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0  # защита от деления на 0
    labels = labels / row_max
    return labels, grid_deg


def doa_rmse(pred_deg, true_deg_arr):
    """RMSE: ближайшее совпадение между предсказанием и истинными углами."""
    valid = true_deg_arr[(true_deg_arr >= -90) & (true_deg_arr <= 90)]
    if len(valid) == 0 or len(pred_deg) == 0:
        return np.nan
    diffs = [np.min(np.abs(valid - p)) for p in np.atleast_1d(pred_deg)]
    return np.sqrt(np.mean(np.array(diffs) ** 2))


# ─────────────────────────────────────────────────────────────
#   ШАГ 1: Подготовка данных
# ─────────────────────────────────────────────────────────────

def step1_prepare_data():
    """Генерация синтетики + предобработка реальных данных."""
    print("\n" + "=" * 60)
    print("ШАГ 1: Подготовка данных")
    print("=" * 60)

    # 1a. Синтетика — из NPZ или генерация на лету
    if os.path.exists(SYNTH_NPZ):
        print(f"  Загрузка синтетики из NPZ: {SYNTH_NPZ}")
        npz = np.load(SYNTH_NPZ, allow_pickle=True)
        X_s = npz["channels"]
        angle_s = npz["azimuth"]
        y_s = npz["math_class"]
        mb_s = npz["mb_metrix"]
    else:
        print(f"  Генерация синтетики ({N_SYNTH} сэмплов)...")
        t0 = time.time()
        X_s, angle_s, y_s, mb_s = dataset_gen_final_experiment(N_SYNTH, init_flag=0)
        print(f"  -> {len(X_s)} сэмплов за {time.time() - t0:.1f} сек")

    print(f"  mb_metrix shape: {mb_s.shape}")

    # Предобработка синтетики
    ch_s = preprocess_channels(X_s, K=WHITEN_K)  # [N, 8]

    # Проверка NaN после whitening
    nan_mask = ~np.all(np.isfinite(ch_s), axis=1)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        print(f"  [!] Удалено {n_bad} NaN/Inf сэмплов после whitening ({n_bad}/{len(ch_s)})")
        ch_s = ch_s[~nan_mask]
        X_s = X_s[~nan_mask] if X_s.shape[0] == len(nan_mask) else X_s
        angle_s = angle_s[~nan_mask]
        y_s = y_s[~nan_mask] if y_s.ndim == 1 else y_s[~nan_mask.ravel()]
        mb_s = mb_s[~nan_mask]

    y_s = y_s.ravel().astype(int)
    angle_s = angle_s.squeeze(-1) if angle_s.ndim == 3 else angle_s  # [N, 2]
    print(f"  Синтетика после предобработки: {len(ch_s)} сэмплов, dtype={ch_s.dtype}")

    # 1b. Реальные данные — из NPZ или из папки
    ch_r, angle_r, y_r, mb_r = None, None, None, None
    if os.path.exists(REAL_NPZ):
        print(f"  Загрузка реальных из NPZ: {REAL_NPZ}")
        npz = np.load(REAL_NPZ, allow_pickle=True)
        ch_r = npz["channels"]
        angle_r = npz["azimuth"]
        y_r = npz["math_class"].astype(int)
        mb_r = npz.get("mb_metrix", None)
    elif os.path.isdir(REAL_DATA_DIR):
        print(f"  Загрузка реальных из папки: {REAL_DATA_DIR}")
        data = dataset_from_real_data(REAL_DATA_DIR, normalize="none")
        ch_r = preprocess_channels(data["channels"], K=WHITEN_K)
        angle_r = np.deg2rad(data["azimuth"]) if np.max(np.abs(data["azimuth"])) > 2 * np.pi else data["azimuth"]
        y_r = data["math class"].astype(int)
        mb_r = data["mb_metrix"]
    else:
        print("  [!] Реальные данные не найдены — finetune будет пропущен")

    return {
        "synth": {"channels": ch_s, "angles": angle_s, "classes": y_s,
                  "mb_metrix": mb_s},
        "real": {"channels": ch_r, "angles": angle_r, "classes": y_r,
                 "mb_metrix": mb_r}
                if ch_r is not None else None,
    }


# ─────────────────────────────────────────────────────────────
#   ШАГ 2: Обучение классификатора
# ─────────────────────────────────────────────────────────────

def step2_train_classifier(data):
    """Pretrain на синтетике (train/val split) → finetune на реале."""
    print("\n" + "=" * 60)
    print("ШАГ 2: Обучение 3-классового классификатора")
    print("=" * 60)

    synth = data['synth']
    tracker = MetricsTracker("classifier")

    # Train/Val split синтетики (80/20)
    idx_all = np.arange(len(synth['channels']))
    idx_train, idx_val = train_test_split(idx_all, test_size=0.2, stratify=synth['classes'],
                                          random_state=42)
    X_train = to_complex_tensor(synth['channels'][idx_train][:, :, None]).to(DEVICE)
    y_train = torch.LongTensor(synth['classes'][idx_train]).to(DEVICE)
    X_val = to_complex_tensor(synth['channels'][idx_val][:, :, None]).to(DEVICE)
    y_val = torch.LongTensor(synth['classes'][idx_val]).to(DEVICE)
    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}")

    model = ComplexNet(input_length=N_ANT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WD)
    criterion = nn.CrossEntropyLoss()

    # Проверка: есть ли NaN в данных (финальная проверка)
    if torch.any(~torch.isfinite(X_train.real)) or torch.any(~torch.isfinite(X_train.imag)):
        print("  [!] ВНИМАНИЕ: X_train содержит NaN/Inf!")

    # — Pretrain на синтетике с Early Stopping (P0-2) —
    print(f"  Pretrain на синтетике (макс. {CLASSIFIER_EPOCHS} эпох, "
          f"early stop patience={EARLY_STOP_PATIENCE})...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            out = model(X_train[idx])
            loss = criterion(out.float(), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            pred_train = model(X_train).argmax(dim=1)
            acc_train = (pred_train == y_train).float().mean().item()
            out_val = model(X_val)
            val_loss = criterion(out_val.float(), y_val).item()
            pred_val = out_val.argmax(dim=1)
            acc_val = (pred_val == y_val).float().mean().item()

        tracker.log_train(epoch, loss=avg_loss, acc=acc_train)
        tracker.log_val(epoch, loss=val_loss, acc=acc_val)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} acc={acc_train:.4f}"
                  f" | val_loss={val_loss:.4f} acc={acc_val:.4f}"
                  f" | patience={patience_counter}/{EARLY_STOP_PATIENCE}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  ** Early stopping на эпохе {epoch+1} (val_loss не улучшался {EARLY_STOP_PATIENCE} эпох)")
            break

    # Восстанавливаем лучшую модель
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"  Восстановлена лучшая модель (val_loss={best_val_loss:.4f})")

    # — Finetune на реальных данных —
    real = data.get('real')
    if real is not None:
        print(f"  Finetune на реальных данных ({FINETUNE_EPOCHS} эпох)...")

        # FIX-05: Train/Val split реальных данных (80/20)
        r_idx = np.arange(len(real['channels']))
        r_train, r_val = train_test_split(
            r_idx, test_size=0.2, stratify=real['classes'], random_state=42)

        X_r_train = to_complex_tensor(real['channels'][r_train][:, :, None]).to(DEVICE)
        y_r_train = torch.LongTensor(real['classes'][r_train]).to(DEVICE)
        X_r_val = to_complex_tensor(real['channels'][r_val][:, :, None]).to(DEVICE)
        y_r_val = torch.LongTensor(real['classes'][r_val]).to(DEVICE)

        print(f"  Real train: {len(r_train)}, Real val: {len(r_val)}")

        # FIX-04 + P0-5: Weighted CrossEntropyLoss с ограничением макс. веса
        class_counts = np.bincount(real['classes'][r_train], minlength=3).astype(float)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.min()  # нормализуем к минимальному = 1
        class_weights = np.clip(class_weights, 1.0, CLASS_WEIGHT_CAP)  # P0-5: cap
        cw_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        criterion_ft = nn.CrossEntropyLoss(weight=cw_tensor)
        print(f"  Class weights: [{class_weights[0]:.2f}, {class_weights[1]:.2f}, {class_weights[2]:.2f}]"
              f" (capped at {CLASS_WEIGHT_CAP})")

        for g in optimizer.param_groups:
            g['lr'] = FINETUNE_LR

        for epoch in range(FINETUNE_EPOCHS):
            model.train()
            perm = torch.randperm(len(X_r_train))
            for i in range(0, len(X_r_train), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                out = model(X_r_train[idx])
                loss = criterion_ft(out.float(), y_r_train[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # P0-6
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    pred_tr = model(X_r_train).argmax(dim=1)
                    acc_tr = (pred_tr == y_r_train).float().mean().item()
                    pred_vl = model(X_r_val).argmax(dim=1)
                    acc_vl = (pred_vl == y_r_val).float().mean().item()
                print(f"    Epoch {epoch+1:3d} | train_acc={acc_tr:.4f} | val_acc={acc_vl:.4f}")

    # Сохраняем
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFIER_PATH)
    print(f"  Модель сохранена: {CLASSIFIER_PATH}")

    # Кривые обучения (pretrain)
    tracker.plot_curves(OUTPUT_DIR, show=False)
    tracker.save_json(OUTPUT_DIR)

    # FIX-05: Полный отчёт на РЕАЛЬНЫХ val-данных (а не на синтетике!)
    if real is not None:
        model.eval()
        with torch.no_grad():
            out_rv = model(X_r_val).float()
            pred_rv = out_rv.argmax(dim=1).cpu().numpy()
            proba_rv = torch.softmax(out_rv, dim=1).cpu().numpy()
        clf_result = classifier_full_report(
            real['classes'][r_val], pred_rv, y_proba=proba_rv,
            output_dir=OUTPUT_DIR, show=False,
        )
        print(f"  Real Val Accuracy: {clf_result['accuracy']:.4f}")
        print(f"  Real Val F1 macro: {clf_result['f1_macro']:.4f}")

        # Confusion matrix на всех реальных данных
        X_r_all = to_complex_tensor(real['channels'][:, :, None]).to(DEVICE)
        with torch.no_grad():
            pred_all = model(X_r_all).argmax(dim=1).cpu().numpy()
        cm = confusion_matrix(real['classes'], pred_all)
        print(f"  Confusion matrix (all real):\n{cm}")
    else:
        # Fallback: метрики на синтетике
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val).argmax(dim=1).cpu().numpy()
            proba_val = torch.softmax(model(X_val).float(), dim=1).cpu().numpy()
        clf_result = classifier_full_report(
            synth['classes'][idx_val], pred_val, y_proba=proba_val,
            output_dir=OUTPUT_DIR, show=False,
        )
        print(f"  Val Accuracy (synth): {clf_result['accuracy']:.4f}")

    return model


# ─────────────────────────────────────────────────────────────
#   ШАГ 3: Обучение 4 методов DOA
# ─────────────────────────────────────────────────────────────

class SimpleDOADataset(Dataset):
    """Датасет для DOA: возвращает (signal, doa_label, class_label)."""

    def __init__(self, channels, doa_labels, class_labels):
        # channels: [N, N_ant] complex
        self.X = torch.complex(
            torch.tensor(channels.real, dtype=torch.float32),
            torch.tensor(channels.imag, dtype=torch.float32),
        ).unsqueeze(1)  # [N, 1, N_ant]
        self.doa = torch.tensor(doa_labels, dtype=torch.float32)
        self.cls = torch.tensor(class_labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.doa[idx], self.cls[idx]


class SiameseDataset(Dataset):
    """Датасет для Siamese: возвращает (sig, aug_sig, neg_sig, doa, cls)."""

    def __init__(self, channels, doa_labels, class_labels):
        self.X = torch.complex(
            torch.tensor(channels.real, dtype=torch.float32),
            torch.tensor(channels.imag, dtype=torch.float32),
        ).unsqueeze(1)  # [N, 1, N_ant]
        self.doa = torch.tensor(doa_labels, dtype=torch.float32)
        self.cls = torch.tensor(class_labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sig = self.X[idx]
        aug = sparse_augment(sig.clone())
        neg_idx = np.random.randint(0, len(self.X))
        while neg_idx == idx:
            neg_idx = np.random.randint(0, len(self.X))
        neg = self.X[neg_idx]
        return sig, aug, neg, self.doa[idx], self.cls[idx]


def train_method1_siamese(channels, angles, classes, grid_deg):
    """Метод 1: Siamese / Contrastive DOA."""
    print("\n  ── Метод 1: Siamese / Contrastive ──")
    doa_labels, _ = make_doa_labels(angles, N_GRID)

    ds = SiameseDataset(channels, doa_labels, classes)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    model = SiameseDOAModel(n_grid=N_GRID, N_ant=N_ANT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=SIAMESE_LR)
    bce = nn.BCEWithLogitsLoss()  # Численно устойчивее: sigmoid + BCE в одном
    ce = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(DOA_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for sig, aug, neg, lbl_doa, lbl_cls in loader:
            sig, aug, neg = sig.to(DEVICE), aug.to(DEVICE), neg.to(DEVICE)
            lbl_doa, lbl_cls = lbl_doa.to(DEVICE), lbl_cls.to(DEVICE)

            emb1, doa1, cls1 = model(sig)
            emb2, _, _ = model(aug)
            emb_n, _, _ = model(neg)

            loss_contr = nt_xent_loss(emb1, emb2)
            loss_doa = bce(doa1, lbl_doa)
            loss_cls = ce(cls1, lbl_cls)
            loss = loss_contr + loss_doa + 0.5 * loss_cls

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | loss={avg:.4f}")

    return model, losses


def train_method2_lista(channels, angles, grid_deg):
    """Метод 2: Deep LISTA DOA."""
    print("\n  ── Метод 2: Deep LISTA ──")
    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    doa_labels, _ = make_doa_labels(angles, N_GRID)

    ch_flat = channels if channels.ndim == 2 else channels.squeeze(-1)
    Y = torch.complex(
        torch.tensor(ch_flat.real, dtype=torch.float32),
        torch.tensor(ch_flat.imag, dtype=torch.float32),
    ).to(DEVICE)
    lbl = torch.tensor(doa_labels, dtype=torch.float32).to(DEVICE)

    model = DeepLISTA_DOA(A_np, n_layers=LISTA_LAYERS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LISTA_LR)

    # pos_weight: усиливаем градиент от положительных меток
    # С Gaussian labels средняя «масса» положительных ≈ 5-8 из 361
    avg_pos = float(lbl.sum() / len(lbl))  # среднее кол-во pos на сэмпл
    pw = max((N_GRID - avg_pos) / max(avg_pos, 1.0), 1.0)
    pw = min(pw, 100.0)  # ограничим разумным значением
    pos_w = torch.full((N_GRID,), pw, device=DEVICE)
    print(f"    pos_weight = {pw:.1f} (avg_pos_per_sample = {avg_pos:.1f})")

    losses = []
    for epoch in range(DOA_EPOCHS):
        model.train()
        # Mini-batch
        perm = torch.randperm(len(Y))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(Y), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            power, _ = model(Y[idx])
            loss = F.binary_cross_entropy_with_logits(
                       power, lbl[idx], pos_weight=pos_w) \
                   + 0.001 * power.abs().mean()  # L1 reg уменьшен 10x
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | loss={avg:.4f}")

    return model, losses


def train_method3_covrecon(channels, angles, grid_deg):
    """Метод 3: Covariance Reconstruction + Root-MUSIC."""
    print("\n  ── Метод 3: CovRecon + Root-MUSIC ──")
    ch_flat = channels if channels.ndim == 2 else channels.squeeze(-1)
    R_flat = cov_to_flat(ch_flat, N_ANT)
    R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)

    model = CovReconstructor(N_ant=N_ANT, hidden=COVRECON_HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=COVRECON_LR)

    losses = []
    for epoch in range(DOA_EPOCHS):
        model.train()
        perm = torch.randperm(len(R_t))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(R_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            out = model(R_t[idx])
            R_rec = flat_to_hermitian(out, N_ANT)

            target_re = R_t[idx, :N_ANT * N_ANT].view(-1, N_ANT, N_ANT)
            target_im = R_t[idx, N_ANT * N_ANT:].view(-1, N_ANT, N_ANT)

            loss_rec = F.mse_loss(R_rec.real, target_re) + F.mse_loss(R_rec.imag, target_im)
            loss_pd = torch.mean(F.relu(-torch.linalg.eigvalsh(R_rec.real)))
            loss = loss_rec + 0.1 * loss_pd

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | loss={avg:.4f}")

    return model, losses


def train_method4_hiergru(channels, angles, classes, grid_deg):
    """Метод 4: Hierarchical GRU DOA."""
    print("\n  ── Метод 4: Hierarchical GRU ──")
    doa_labels, _ = make_doa_labels(angles, N_GRID)

    # Source count: 0 → single target, 1 → multipath, 2 → two targets
    # Но для source_count = кол-во реальных углов: class 0→1, class 1→0, class 2→2
    count_map = {0: 1, 1: 0, 2: 2}
    source_counts = np.array([count_map.get(int(c), 0) for c in classes], dtype=np.int64)

    ds = SimpleDOADataset(channels, doa_labels, source_counts)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = HierarchicalGRU_DOA(
        K_windows=3, n_sources_max=3,
        coarse_grid=181, fine_grid=N_GRID,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=GRU_LR)

    # pos_weight для DOA-головы (аналогично LISTA)
    lbl_t = torch.tensor(doa_labels, dtype=torch.float32)
    avg_pos = float(lbl_t.sum() / len(lbl_t))
    pw = min(max((N_GRID - avg_pos) / max(avg_pos, 1.0), 1.0), 100.0)
    pos_w = torch.full((N_GRID,), pw, device=DEVICE)
    print(f"    pos_weight = {pw:.1f}")

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce = nn.CrossEntropyLoss()

    n_epochs = DOA_EPOCHS_HIERGRU  # P0-3: больше эпох для HierGRU (120)
    print(f"    Эпох: {n_epochs} (DOA_EPOCHS_HIERGRU)")

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, doa_b, cnt_b in loader:
            X_b, doa_b, cnt_b = X_b.to(DEVICE), doa_b.to(DEVICE), cnt_b.to(DEVICE)
            fine_logits, count_logits = model(X_b)
            loss_d = bce(fine_logits, doa_b)
            loss_c = ce(count_logits, cnt_b)
            loss = loss_d + 0.3 * loss_c
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | loss={avg:.4f}")

    return model, losses


def step3_train_doa(data):
    """Обучение всех 4 DOA-методов на синтетике с train/val split."""
    print("\n" + "=" * 60)
    print("ШАГ 3: Обучение 4 методов DOA на синтетике")
    print("=" * 60)

    synth = data['synth']
    ch = synth['channels']
    ang = synth['angles']
    cls = synth['classes']
    _, grid_deg = make_doa_labels(ang, N_GRID)

    # Фильтруем: для DOA берём только class 0 и class 2 (не multipath)
    mask_doa = (cls == 0) | (cls == 2)
    ch_doa = ch[mask_doa]
    ang_doa = ang[mask_doa]
    cls_doa = cls[mask_doa]

    # Train/Val split DOA-данных (85/15)
    idx_all = np.arange(len(ch_doa))
    idx_tr, idx_vl = train_test_split(idx_all, test_size=0.15, random_state=42)
    ch_tr, ch_vl = ch_doa[idx_tr], ch_doa[idx_vl]
    ang_tr, ang_vl = ang_doa[idx_tr], ang_doa[idx_vl]
    cls_tr, cls_vl = cls_doa[idx_tr], cls_doa[idx_vl]

    print(f"  DOA: train={len(ch_tr)}, val={len(ch_vl)} (class 0 + class 2 из {len(ch)})")

    trackers = {}

    m1, l1 = train_method1_siamese(ch_tr, ang_tr, cls_tr, grid_deg)
    m2, l2 = train_method2_lista(ch_tr, ang_tr, grid_deg)
    m3, l3 = train_method3_covrecon(ch_tr, ang_tr, grid_deg)
    m4, l4 = train_method4_hiergru(ch_tr, ang_tr, cls_tr, grid_deg)

    # Создаём DOA MetricsTrackers с val RMSE
    doa_models = {'siamese': m1, 'lista': m2, 'covrecon': m3, 'hiergru': m4}
    loss_dict = {'siamese': l1, 'lista': l2, 'covrecon': l3, 'hiergru': l4}

    # Вычисляем val RMSE для каждого метода
    print("\n  Val RMSE по методам:")
    for method_name, doa_model in doa_models.items():
        tracker = MetricsTracker(f"doa_{method_name}")
        for ep, lv in enumerate(loss_dict[method_name]):
            tracker.log_train(ep, loss=lv)
        # Val RMSE
        rmse_list = []
        doa_model.eval()
        for i in range(len(ch_vl)):
            pred_angles, _ = predict_doa_single(ch_vl[i], method_name, doa_model, grid_deg)
            true_deg = np.rad2deg(ang_vl[i].ravel())
            err = doa_rmse(pred_angles, true_deg)
            if not np.isnan(err):
                rmse_list.append(err)
        val_rmse = np.mean(rmse_list) if rmse_list else np.nan
        tracker.log_val(0, rmse=val_rmse)
        print(f"    {method_name:12s} | val RMSE = {val_rmse:.2f}° ({len(rmse_list)} сэмплов)")
        tracker.plot_curves(OUTPUT_DIR, show=False)
        tracker.save_json(OUTPUT_DIR)
        trackers[method_name] = tracker

    # Сохраняем модели
    for name, model, path in [
        ('siamese', m1, DOA_SIAMESE_PATH),
        ('lista', m2, DOA_LISTA_PATH),
        ('covrecon', m3, DOA_COVRECON_PATH),
        ('hiergru', m4, DOA_HIERGRU_PATH),
    ]:
        torch.save(model.state_dict(), path)
        print(f"  Сохранено: {path}")

    # Сохраняем val-данные для step4/step5
    data['doa_val'] = {
        'channels': ch_vl, 'angles': ang_vl, 'classes': cls_vl,
    }

    return doa_models, loss_dict


# ─────────────────────────────────────────────────────────────
#   ШАГ 4: Полный инференс — classify → DOA
# ─────────────────────────────────────────────────────────────

def predict_doa_single(signal_np, method_name, doa_model, grid_deg):
    """
    Предсказание DOA для одного сигнала [N_ant] complex.
    Возвращает массив углов в градусах.
    """
    doa_model.eval()
    with torch.no_grad():
        if method_name == 'siamese':
            x = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, N_ant]
            _, doa_logits, _ = doa_model(x)
            spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]]), spectrum

        elif method_name == 'lista':
            y = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)  # [1, N_ant]
            power, _ = doa_model(y)
            spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]]), spectrum

        elif method_name == 'covrecon':
            R_flat = cov_to_flat(signal_np.reshape(1, -1), N_ANT)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            out = doa_model(R_t)
            R_rec = flat_to_hermitian(out, N_ANT)
            R_np = R_rec[0].cpu().numpy()
            angles = root_music(R_np, N_sources=2)
            return angles, None  # Root-MUSIC не даёт спектр

        elif method_name == 'hiergru':
            x = torch.complex(
                torch.tensor(signal_np.real, dtype=torch.float32),
                torch.tensor(signal_np.imag, dtype=torch.float32),
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            doa_logits, _ = doa_model(x)
            spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
            peak_idx = np.argmax(spectrum)
            return np.array([grid_deg[peak_idx]]), spectrum

    return np.array([]), None


def step4_full_inference(data, classifier, doa_models):
    """Полный инференс: classify → route → DOA.
    P0-1: оценка на РЕАЛЬНЫХ данных (если доступны).
    P0-4: фильтр углов ±DOA_ANGLE_LIMIT при подсчёте RMSE.
    """
    print("\n" + "=" * 60)
    print("ШАГ 4: Полный инференс (classify → DOA)")
    print("=" * 60)

    # P0-1: используем реальные данные, если есть
    real = data.get('real')
    if real is not None:
        src_name = "РЕАЛЬНЫЕ данные"
        ch_eval = real['channels']
        ang_eval = real['angles']
        cls_eval = real['classes']
    else:
        src_name = "синтетические данные (fallback)"
        synth = data['synth']
        N_eval = min(2000, len(synth['channels']))
        ch_eval = synth['channels'][:N_eval]
        ang_eval = synth['angles'][:N_eval]
        cls_eval = synth['classes'][:N_eval]

    _, grid_deg = make_doa_labels(ang_eval[:1], N_GRID)
    print(f"  Источник: {src_name}, N={len(ch_eval)}")
    print(f"  Фильтр углов: ±{DOA_ANGLE_LIMIT}° (P0-4)")

    # 1. Классификация
    classifier.eval()
    X_t = to_complex_tensor(ch_eval[:, :, None]).to(DEVICE)
    with torch.no_grad():
        pred_cls = classifier(X_t).argmax(dim=1).cpu().numpy()

    cls_acc = np.mean(pred_cls == cls_eval)
    print(f"  Accuracy классификатора: {cls_acc:.4f}")
    cm = confusion_matrix(cls_eval, pred_cls)
    print(f"  Confusion matrix:\n{cm}")

    # 2. DOA по классам (0=direct → 1 угол, 2=two targets → 2 угла)
    results = {}
    for method_name, doa_model in doa_models.items():
        rmse_list = []
        rmse_full_list = []  # без фильтра для сравнения
        for i in range(len(ch_eval)):
            if pred_cls[i] == 1:  # multipath → пропускаем
                continue
            pred_angles, _ = predict_doa_single(
                ch_eval[i], method_name, doa_model, grid_deg,
            )
            true_angles_deg = np.rad2deg(ang_eval[i].ravel())
            err = doa_rmse(pred_angles, true_angles_deg)
            if not np.isnan(err):
                rmse_full_list.append(err)
                # P0-4: фильтр краевых углов
                valid_angles = true_angles_deg[(true_angles_deg >= -90) & (true_angles_deg <= 90)]
                if len(valid_angles) > 0 and np.all(np.abs(valid_angles) <= DOA_ANGLE_LIMIT):
                    rmse_list.append(err)

        avg_rmse = np.mean(rmse_list) if rmse_list else np.nan
        avg_rmse_full = np.mean(rmse_full_list) if rmse_full_list else np.nan
        results[method_name] = avg_rmse
        print(f"  {method_name:12s} | RMSE(±{DOA_ANGLE_LIMIT:.0f}°) = {avg_rmse:.2f}°"
              f" | RMSE(full) = {avg_rmse_full:.2f}° | N={len(rmse_list)}/{len(rmse_full_list)}")

    return results


# ─────────────────────────────────────────────────────────────
#   ШАГ 5: Сравнение и графики
# ─────────────────────────────────────────────────────────────

def step5_compare(loss_histories, rmse_results, data, doa_models):
    """
    Расширенная визуализация и сравнение:
      5a. Кривые обучения (train loss)
      5b. DOAEvaluator на РЕАЛЬНЫХ данных (P0-1) с фильтром ±70° (P0-4)
      5c. Примеры DOA-спектров
      5d. EDA: распределение данных
    """
    print("\n" + "=" * 60)
    print("ШАГ 5: Визуализация и сравнение")
    print("=" * 60)

    _, grid_deg = make_doa_labels(data['synth']['angles'][:1], N_GRID)

    # 5a. Кривые обучения DOA
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (name, losses) in zip(axes, loss_histories.items()):
        ax.plot(losses, color="#2196F3")
        ax.set_title(f'{name} — Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves_doa.png'), dpi=150)
    plt.close()

    # 5b. DOAEvaluator — P0-1: используем реальные данные
    real = data.get('real')
    if real is not None:
        src_name = "РЕАЛЬНЫЕ данные"
        ch_eval = real['channels']
        ang_eval = real['angles']
        cls_eval = real['classes']
    else:
        src_name = "синтетические данные (fallback)"
        synth = data['synth']
        N_eval = min(2000, len(synth['channels']))
        ch_eval = synth['channels'][:N_eval]
        ang_eval = synth['angles'][:N_eval]
        cls_eval = synth['classes'][:N_eval]

    print(f"  DOA evaluation на: {src_name}, N={len(ch_eval)}")
    print(f"  Фильтр углов для RMSE: ±{DOA_ANGLE_LIMIT}° (P0-4)")

    # Два evaluator'а: полный и с фильтром углов
    evaluator_full = DOAEvaluator(grid_deg)
    evaluator_filtered = DOAEvaluator(grid_deg)

    for method_name, doa_model in doa_models.items():
        errors_full = []
        errors_filt = []
        pclass_full, tang_full, pang_full = [], [], []
        pclass_filt, tang_filt, pang_filt = [], [], []

        doa_model.eval()
        for i in range(len(ch_eval)):
            if cls_eval[i] == 1:  # multipath → skip
                continue

            pred_angles, _ = predict_doa_single(
                ch_eval[i], method_name, doa_model, grid_deg,
            )
            true_deg = np.rad2deg(ang_eval[i].ravel())
            valid = true_deg[(true_deg >= -90) & (true_deg <= 90)]
            if len(valid) == 0 or len(pred_angles) == 0:
                continue

            min_err = min(np.min(np.abs(valid - p)) for p in pred_angles)

            # Full (все углы)
            errors_full.append(min_err)
            pclass_full.append(cls_eval[i])
            tang_full.append(valid[0])
            pang_full.append(pred_angles[0])

            # P0-4: фильтр — только углы в ±DOA_ANGLE_LIMIT
            if np.all(np.abs(valid) <= DOA_ANGLE_LIMIT):
                errors_filt.append(min_err)
                pclass_filt.append(cls_eval[i])
                tang_filt.append(valid[0])
                pang_filt.append(pred_angles[0])

        evaluator_full.add_method(
            method_name, errors_full,
            pred_classes=pclass_full, true_angles=tang_full, pred_angles=pang_full,
        )
        evaluator_filtered.add_method(
            f"{method_name}", errors_filt,
            pred_classes=pclass_filt, true_angles=tang_filt, pred_angles=pang_filt,
        )

    # Сводная таблица — полная
    print(f"\n  === Метрики (все углы) ===")
    evaluator_full.print_summary()

    print(f"\n  === Метрики (±{DOA_ANGLE_LIMIT}° фильтр) ===")
    evaluator_filtered.print_summary()

    # Графики — полная оценка (для scatter, CDF, RMSE vs angle)
    evaluator_full.plot_all(OUTPUT_DIR, show=False)
    evaluator_full.save_json(OUTPUT_DIR)

    # Сохраняем фильтрованные метрики в отдельный JSON
    evaluator_filtered.save_json(OUTPUT_DIR, prefix="doa_filtered_")

    # 5c. Примеры DOA-спектров (используем synth class=direct для наглядных спектров)
    synth = data['synth']
    direct_idx = np.where(synth['classes'] == 0)[0][:5]
    if len(direct_idx) > 0:
        fig, axes = plt.subplots(len(direct_idx), len(doa_models),
                                  figsize=(5 * len(doa_models), 3 * len(direct_idx)),
                                  squeeze=False)
        for row, si in enumerate(direct_idx):
            true_deg = np.rad2deg(synth['angles'][si].ravel())
            for col, (method_name, doa_model) in enumerate(doa_models.items()):
                pred_angles, spectrum = predict_doa_single(
                    synth['channels'][si], method_name, doa_model, grid_deg,
                )
                ax = axes[row][col]
                if spectrum is not None:
                    ax.plot(grid_deg, spectrum, 'b-', alpha=0.7)
                for td in true_deg:
                    if -90 <= td <= 90:
                        ax.axvline(td, color='r', ls='--', label=f'True {td:.1f}°')
                for pa in pred_angles:
                    ax.axvline(pa, color='g', ls=':', label=f'Pred {pa:.1f}°')
                ax.set_title(f'{method_name}' if row == 0 else '')
                ax.set_xlim(-90, 90)
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel(f'Sample {si}')
                ax.legend(fontsize=7, loc='upper right')

        plt.suptitle('Примеры DOA-предсказаний (class=direct)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'doa_examples.png'), dpi=150)
        plt.close()

    # 5d. EDA — распределение данных
    synth_eda = {
        "angles": synth['angles'],
        "classes": synth['classes'],
        "channels": synth['channels'],
        "mb_metrix": synth.get('mb_metrix'),
    }
    real_eda = None
    if data.get('real') is not None:
        r = data['real']
        real_eda = {
            "angles": r['angles'],
            "classes": r['classes'],
            "channels": r['channels'],
        }
    plot_data_eda(synth_eda, real_eda, output_dir=OUTPUT_DIR, show=False)

    print(f"\n  Все графики и метрики сохранены в {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  RADAR DOA PIPELINE — Новый подход (whitening+RMS)  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Шаг 1: Данные
    data = step1_prepare_data()

    # Шаг 2: Классификатор
    classifier = step2_train_classifier(data)

    # Шаг 3: 4 метода DOA
    doa_models, loss_histories = step3_train_doa(data)

    # Шаг 4: Полный инференс
    rmse_results = step4_full_inference(data, classifier, doa_models)

    # Шаг 5: Сравнение
    step5_compare(loss_histories, rmse_results, data, doa_models)

    print("\n" + "=" * 60)
    print("ГОТОВО. Все модели и графики в", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
