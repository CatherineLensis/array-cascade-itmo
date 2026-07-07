"""
models.py
=========
Все нейросетевые архитектуры для пайплайна радарной обработки:

1. ComplexNet          — 3-класс. классификатор (прямой / два сигнала / мультипас)
2. SiameseDOAModel     — контрастивное обучение + DOA  (Метод 1)
3. DeepLISTA_DOA       — physics-informed LISTA          (Метод 2)
4. CovReconstructor    — реконструкция ковариации → Root-MUSIC (Метод 3)
5. HierarchicalGRU_DOA — многостадийный GRU              (Метод 4)

Все модели рассчитаны на 8-антенную виртуальную MIMO-решётку (TI AWR1843).
Позиции виртуального массива: [0, 0.5, 1.0, ..., 3.5] в λ/2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh


# ──────────────────────────────────────────────────────────────
#  Утилиты
# ──────────────────────────────────────────────────────────────

def complex_tanh(x):
    """
    Покомпонентный tanh для комплексных тензоров.

    ВАЖНО: torch.tanh(complex) использует аналитический tanh(z)=sinh(z)/cosh(z),
    который ПЕРЕПОЛНЯЕТСЯ при больших |Im(z)| → NaN.
    Покомпонентный вариант: tanh(Re) + i·tanh(Im) — всегда в [-1,1].
    Это соответствует complex_tanh из библиотеки complexPyTorch.
    """
    return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))

def to_complex_tensor(X_np):
    """Numpy [N, C, 1] complex → Torch [N, 1, C] complex64."""
    if X_np.ndim == 3 and X_np.shape[2] == 1:
        X_np = X_np[:, :, 0]  # [N, C]
    X_re = torch.FloatTensor(np.real(X_np))
    X_im = torch.FloatTensor(np.imag(X_np))
    # [N, C] → [N, 1, C]  (1 = in_channels для Conv1d)
    return torch.complex(X_re, X_im).unsqueeze(1)


def build_steering_matrix(mimo_pos, n_grid=361):
    """Строим матрицу steering-векторов A [N_ant, N_grid]."""
    theta_grid = np.deg2rad(np.linspace(-90, 90, n_grid))
    A = np.exp(1j * 2 * np.pi * mimo_pos[:, None] * np.sin(theta_grid[None, :]))
    return A, theta_grid


# ──────────────────────────────────────────────────────────────
#  0.  ComplexNet — 3-классовый классификатор
# ──────────────────────────────────────────────────────────────
#
#  Проблема оригинальной архитектуры Шишанова:
#    Вход [B,1,8] → 3 свертки kernel=3 без padding → 8→6→4→2 →
#    → AvgPool(2) → 1 → flattened = 16*1 = 16
#    Это СЛИШКОМ агрессивное сжатие — теряется пространственная
#    информация до FC-слоёв.
#
#  Исправление: добавляем padding=1 в свёрточные слои, чтобы
#  сохранить пространственное разрешение, и убираем AvgPool
#  (8 отсчётов — и так мало). Увеличиваем FC-слои.
# ──────────────────────────────────────────────────────────────

class ComplexNet(nn.Module):
    """
    Комплексная свёрточная сеть для классификации 8-канальных MIMO-снимков.

    Архитектура (улучшенная):
      Conv1d(1→16, k=3, pad=1) → complex_tanh → complex_BN
      Conv1d(16→32, k=3, pad=1) → complex_tanh → complex_BN
      Conv1d(32→64, k=3, pad=1) → complex_tanh → complex_BN
      → abs() → [B, 64, 8]
      → Flatten → FC(512→128) → ReLU → Dropout(0.5) → FC(128→3)

    Вход:  [batch, 1, N_ant] complex64
    Выход: [batch, n_classes] logits (0=direct, 1=multipath, 2=two targets)
    """

    def __init__(self, input_length=8, n_classes=3):
        super().__init__()
        # Свёртки с padding=1 сохраняют длину при kernel=3
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1, dtype=torch.complex64)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1, dtype=torch.complex64)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1, dtype=torch.complex64)

        # ВАЖНО: отдельные BN для real и imag частей!
        # Общий BN ломает running statistics (mean/var усредняются
        # между real и imag при двух вызовах за один forward pass).
        self.bn1_re = nn.BatchNorm1d(16)
        self.bn1_im = nn.BatchNorm1d(16)
        self.bn2_re = nn.BatchNorm1d(32)
        self.bn2_im = nn.BatchNorm1d(32)
        self.bn3_re = nn.BatchNorm1d(64)
        self.bn3_im = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.5)

        # После abs() + flatten: 64 каналов × 8 отсчётов = 512
        self.fc1 = nn.Linear(64 * input_length, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B, 1, 8] complex64
        x = complex_tanh(self.conv1(x))
        x = torch.complex(self.bn1_re(x.real), self.bn1_im(x.imag))

        x = complex_tanh(self.conv2(x))
        x = torch.complex(self.bn2_re(x.real), self.bn2_im(x.imag))

        x = complex_tanh(self.conv3(x))
        x = torch.complex(self.bn3_re(x.real), self.bn3_im(x.imag))

        x = x.abs()                       # [B, 64, 8] real
        x = x.view(x.size(0), -1)         # [B, 512]
        x = F.relu(self.fc1(x))           # [B, 128]
        x = self.dropout(x)
        return self.fc2(x)                 # [B, 3]


# ──────────────────────────────────────────────────────────────
#  1.  Siamese / Contrastive DOA  (Метод 1)
# ──────────────────────────────────────────────────────────────

def sparse_augment(sig, mask_prob=0.3, noise_std=0.1):
    """
    Аугментация: прореживание каналов + комплексный шум.
    Имитирует неисправности антенн и jamming.
    """
    mask = (torch.rand_like(sig.real) > mask_prob).to(sig.dtype)
    noise = noise_std * torch.complex(
        torch.randn_like(sig.real),
        torch.randn_like(sig.imag),
    )
    return sig * mask + noise


class ComplexFeatureExtractor(nn.Module):
    """
    Комплексный feature-экстрактор → 256-мерный embedding.

    Архитектура:
      Conv1d(1→32, k=3, pad=1)         → complex_tanh → complex_BN   [B,32,8]
      Conv1d(32→64, k=3, pad=1, s=2)   → complex_tanh → complex_BN   [B,64,4]
      Conv1d(64→128, k=3, pad=1, s=2)  → complex_tanh → complex_BN   [B,128,2]
      → abs() → AdaptiveAvgPool1d(2)   → [B,128,2]
      → Flatten → FC(256→256) → L2-normalize
    """

    def __init__(self, in_ch=1, N_ant=8, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, 3, padding=1, dtype=torch.complex64)
        self.bn1_re = nn.BatchNorm1d(hidden)
        self.bn1_im = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, 3, padding=1, stride=2, dtype=torch.complex64)
        self.bn2_re = nn.BatchNorm1d(hidden * 2)
        self.bn2_im = nn.BatchNorm1d(hidden * 2)
        self.conv3 = nn.Conv1d(hidden * 2, hidden * 4, 3, padding=1, stride=2, dtype=torch.complex64)
        self.bn3_re = nn.BatchNorm1d(hidden * 4)
        self.bn3_im = nn.BatchNorm1d(hidden * 4)

        # Adaptive pool гарантирует фиксированный размер
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.fc_emb = nn.Linear(hidden * 4 * 2, 256)

    def forward(self, x):
        # x: [B, 1, N_ant] complex
        x = complex_tanh(self.conv1(x))
        x = torch.complex(self.bn1_re(x.real), self.bn1_im(x.imag))

        x = complex_tanh(self.conv2(x))
        x = torch.complex(self.bn2_re(x.real), self.bn2_im(x.imag))

        x = complex_tanh(self.conv3(x))
        x = torch.complex(self.bn3_re(x.real), self.bn3_im(x.imag))

        x = self.pool(x.abs())            # [B, 128, 2]
        x = x.flatten(1)                  # [B, 256]
        return F.normalize(self.fc_emb(x), dim=-1)


class SiameseDOAModel(nn.Module):
    """
    Siamese: общий backbone → DOA-голова (sigmoid, multi-hot)
                              → class-голова (3 класса)
    Вход:  [B, 1, N_ant] complex
    Выход: (embedding, doa_logits, class_logits)
    """

    def __init__(self, n_grid=361, N_ant=8):
        super().__init__()
        self.extractor = ComplexFeatureExtractor(in_ch=1, N_ant=N_ant)
        self.doa_head = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, n_grid),
        )
        self.class_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 3),
        )

    def forward(self, x):
        emb = self.extractor(x)
        doa_logits = self.doa_head(emb)     # raw logits (без sigmoid!)
        cls = self.class_head(emb)
        return emb, doa_logits, cls


def nt_xent_loss(emb1, emb2, temp=0.07):
    """NT-Xent (InfoNCE) контрастивная потеря."""
    sim = torch.mm(emb1, emb2.T) / temp
    labels = torch.arange(emb1.size(0), device=emb1.device)
    return F.cross_entropy(sim, labels)


# ──────────────────────────────────────────────────────────────
#  2.  Deep LISTA DOA  (Метод 2)
# ──────────────────────────────────────────────────────────────

class LISTA_DOALayer(nn.Module):
    """Один слой Learned ISTA: x_{k+1} = S_λ(x_k + θ·W·(y − A·x_k))."""

    def __init__(self, N_ant, N_grid, W_init=None, theta=0.1, lam=0.001):
        super().__init__()
        # W: обучаемая матрица [N_grid, N_ant]
        if W_init is not None:
            self.W = nn.Parameter(W_init.clone())
        else:
            self.W = nn.Parameter(torch.complex(
                torch.randn(N_grid, N_ant, dtype=torch.float32) * 0.01,
                torch.randn(N_grid, N_ant, dtype=torch.float32) * 0.01,
            ))
        self.theta = nn.Parameter(torch.tensor(theta))
        self.lam = nn.Parameter(torch.tensor(lam))

    def forward(self, y, x_prev, A):
        """
        y: [B, N_ant] complex — наблюдение
        x_prev: [B, N_grid] complex — текущая оценка
        A: [N_ant, N_grid] complex — steering-матрица (не обучается)
        """
        # Остаток: y - A·x
        residual = y - (A @ x_prev.T).T  # [B, N_ant]
        # Градиентный шаг: x + θ·W·r
        z = x_prev + self.theta * (residual @ self.W.T)  # [B, N_grid]
        # Комплексный soft-thresholding
        mag = z.abs()
        phase = z.angle()
        soft = F.relu(mag - self.lam)
        return soft * torch.exp(1j * phase)


class DeepLISTA_DOA(nn.Module):
    """
    Deep unfolded LISTA для DOA.
    Вход:  [B, N_ant] complex — отбелённый сигнал
    Выход: (power_spectrum [B, N_grid], sparse_x [B, N_grid])
    """

    def __init__(self, A_np, n_layers=10):
        super().__init__()
        N_ant, N_grid = A_np.shape
        # Steering-матрица (фиксирована, не обучается)
        self.register_buffer(
            'A',
            torch.complex(
                torch.tensor(A_np.real, dtype=torch.float32),
                torch.tensor(A_np.imag, dtype=torch.float32),
            ),
        )
        # Инициализация W из A^H (matched filter) — физически обоснованная
        # стартовая точка для ISTA: W ≈ A^H / ||A||²
        A_H = np.conj(A_np.T)  # [N_grid, N_ant]
        norm_sq = np.linalg.norm(A_np, 'fro') ** 2 / N_grid
        W_init_np = A_H / max(norm_sq, 1e-6)
        W_init = torch.complex(
            torch.tensor(W_init_np.real, dtype=torch.float32),
            torch.tensor(W_init_np.imag, dtype=torch.float32),
        )
        self.layers = nn.ModuleList([
            LISTA_DOALayer(N_ant, N_grid, W_init=W_init) for _ in range(n_layers)
        ])

    def forward(self, y):
        B = y.shape[0]
        N_grid = self.A.shape[1]
        x = torch.zeros(B, N_grid, dtype=torch.complex64, device=y.device)
        for layer in self.layers:
            x = layer(y, x, self.A)
        power = x.abs() ** 2
        return power, x


# ──────────────────────────────────────────────────────────────
#  3.  Covariance Reconstruction + Root-MUSIC  (Метод 3)
# ──────────────────────────────────────────────────────────────

class CovReconstructor(nn.Module):
    """
    MLP: flat(R_sample) → flat(R_clean).
    Восстанавливает ковариационную матрицу для последующего Root-MUSIC.
    Вход:  [B, N_ant²·2]  (real + imag)
    Выход: [B, N_ant²·2]
    """

    def __init__(self, N_ant=8, hidden=128):
        super().__init__()
        self.N_ant = N_ant
        sz = N_ant * N_ant * 2
        self.net = nn.Sequential(
            nn.Linear(sz, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, sz),
        )

    def forward(self, R_flat):
        return self.net(R_flat)


def flat_to_hermitian(R_flat, N):
    """[B, N²·2] → [B, N, N] Hermitian complex."""
    B = R_flat.shape[0]
    R_re = R_flat[:, :N * N].view(B, N, N)
    R_im = R_flat[:, N * N:].view(B, N, N)
    R = torch.complex(R_re, R_im)
    return (R + R.conj().transpose(-2, -1)) / 2


def cov_to_flat(channels_np, N_ant):
    """[N, N_ant] complex → [N, N_ant²·2] float (sample cov → flat)."""
    flats = []
    for s in channels_np:
        s = s / (np.linalg.norm(s) + 1e-12)
        R = np.outer(s, s.conj()) + 1e-6 * np.eye(N_ant, dtype=np.complex128)
        flats.append(np.concatenate([R.real.ravel(), R.imag.ravel()]))
    return np.array(flats, dtype=np.float32)


def root_music(R_np, N_sources=1, d_lambda=0.5):
    """
    Root-MUSIC на numpy-матрице [N, N] → углы в градусах.
    Возвращает np.array отсортированных углов.
    """
    N = R_np.shape[0]
    try:
        eigvals, eigvecs = eigh(R_np)
    except np.linalg.LinAlgError:
        return np.array([])
    # Шумовое подпространство
    Un = eigvecs[:, :max(1, N - N_sources)]
    Q = Un @ Un.conj().T

    # Полиномиальные коэффициенты
    poly = np.zeros(2 * N - 1, dtype=np.complex128)
    for k in range(-(N - 1), N):
        for i in range(N):
            j = i - k
            if 0 <= j < N:
                poly[k + N - 1] += Q[i, j]

    roots = np.roots(poly)
    # Корни ближайшие к единичной окружности
    dist_to_unit = np.abs(np.abs(roots) - 1.0)
    best_idx = np.argsort(dist_to_unit)[:2 * N_sources]
    roots_sel = roots[best_idx]

    angles_rad = np.arcsin(
        np.clip(np.angle(roots_sel) / (2 * np.pi * d_lambda), -1, 1)
    )
    angles_deg = np.rad2deg(np.real(angles_rad))
    angles_deg = angles_deg[(angles_deg >= -90) & (angles_deg <= 90)]
    angles_deg = np.sort(np.unique(angles_deg))
    return angles_deg[:N_sources] if len(angles_deg) > N_sources else angles_deg


# ──────────────────────────────────────────────────────────────
#  4.  Hierarchical GRU DOA  (Метод 4)
# ──────────────────────────────────────────────────────────────

class CoarseGRU(nn.Module):
    """Грубая стадия: GRU по антенным элементам → вероятности на coarse-сетке."""

    def __init__(self, hidden=128, coarse_grid=181):
        super().__init__()
        # Каждый антенный элемент → 2 фичи (Re, Im)
        self.gru = nn.GRU(input_size=2, hidden_size=hidden,
                          num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, coarse_grid)
        self.hidden_size = hidden

    def forward(self, x):
        # x: [B, 1, N_ant] complex
        feats = torch.stack([x.real.squeeze(1), x.imag.squeeze(1)], dim=-1)
        feats = feats.permute(1, 0, 2)  # [N_ant, B, 2]
        out, _ = self.gru(feats)         # [N_ant, B, hidden*2]
        h_last = out[-1]                 # [B, hidden*2]
        logits = self.fc(h_last)
        return F.softmax(logits, dim=-1), h_last


class FineGRU(nn.Module):
    """Тонкая стадия: GRU → логиты на fine-сетке (multilabel)."""

    def __init__(self, hidden=64, fine_grid=361):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden,
                          num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, fine_grid)

    def forward(self, x):
        feats = torch.stack([x.real.squeeze(1), x.imag.squeeze(1)], dim=-1)
        feats = feats.permute(1, 0, 2)
        out, _ = self.gru(feats)
        h_last = out[-1]
        return self.fc(h_last)  # raw logits (без sigmoid — для BCEWithLogitsLoss)


class HierarchicalGRU_DOA(nn.Module):
    """
    Двухстадийный GRU: coarse → fine + source count.
    Вход:  [B, 1, N_ant] complex
    Выход: (doa_logits [B, fine_grid], count_logits [B, n_sources_max])
    """

    def __init__(self, K_windows=3, n_sources_max=3,
                 coarse_grid=181, fine_grid=361):
        super().__init__()
        self.coarse = CoarseGRU(coarse_grid=coarse_grid)
        self.fines = nn.ModuleList([FineGRU(fine_grid=fine_grid)
                                    for _ in range(K_windows)])
        self.source_count = nn.Linear(self.coarse.hidden_size * 2, n_sources_max)

    def forward(self, x):
        coarse_probs, h_coarse = self.coarse(x)
        fine_outputs = [f(x) for f in self.fines]
        doa_logits = torch.stack(fine_outputs, dim=1).mean(dim=1)
        count_logits = self.source_count(h_coarse)
        return doa_logits, count_logits
