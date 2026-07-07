"""## 2. Physics-informed unfolded сеть (LISTA/OPAMP/Deep-OMP)"""

# 2. Physics-informed unfolded сеть (LISTA/OPAMP/Deep-OMP)Этот подход сочетает преимущества классических итеративных методов разреженного восстановления (OMP, ISTA, FPC и др.) с возможностями глубокого обучения. Сеть разворачивается (unfolds) в фиксированное количество слоёв, где каждый слой соответствует одной итерации классического алгоритма. Это делает модель интерпретируемой, быстрой на инференсе и позволяющей добиться очень хороших результатов даже при малом количестве снимков (single-snapshot), низком SNR и наличии ошибок геометрии антенной решётки — всё это крайне актуально для автомобильных радаров.Вывод по подходу (на январь 2026):Преимущества:Высокая интерпретируемость (каждый слой — физически осмысленная итерация)
# Отличная работа на single-snapshot (очень важно для движущихся целей в авто-радаре)
# Лучше обобщение на неидеальные условия (ошибки позиционирования элементов, mutual coupling, jamming) по сравнению с чистыми end-to-end CNN
# Возможность встраивать физику напрямую (steering vectors, sparsity prior, atomic norm и т.п.)
# Часто превосходит классический OMP на 10–30% по RMSE при SNR 0…15 дБ

# Недостатки:Требует тщательного проектирования разворачиваемого алгоритма (выбор базового метода — LISTA, OPAMP, ADMM, FPC-ℓ1 и др.)
# Обучение может быть сложнее и медленнее, чем у простых CNN (нужны хорошие инициализации весов)
# Меньше гибкости при очень большом количестве источников (>4–5)

# Когда использовать:Хотите максимальный публикационный потенциал (тема очень горячая в 2024–2025)
# Нужна объяснимость модели (можно показать, как сеть приближает OMP/ISTA)
# Работаете преимущественно с single-snapshot и/или разреженными массивами

# Ожидаемые результаты (на основе недавних работ 2024–2025):RMSE ~0.3–1.2° на синтетике при SNR 5–20 дБ
# Значительно лучше OMP в присутствии array imperfections и coherent источников (multipath)

# Пайплайн (самый популярный вариант на 2025–2026 — Deep Unfolded LISTA/OPAMP для DOA):Выбор базового алгоритма → чаще всего LISTA (Learned ISTA) или OPAMP (Orthogonal AMP) — они хорошо работают со sparsity в угловом домене.
# Подготовка данных → те же комплексные сигналы MIMO (ваш sig_mimo), но желательно ковариационная матрица R или напрямую сырой сигнал → steering dictionary A(θ).
# Предобработка → Wiener-Hopf подавление jamming (как в вашей статье) + опционально spatial smoothing для борьбы с coherent multipath.
# Модель → N слоёв (обычно 6–16), каждый слой содержит обучаемые параметры (shrinkage/threshold, step size, возможно мягкую ортогонализацию).
# Обучение → supervised на синтетике (ground truth DOA как one-hot или мягкие метки) + L1-подобная регуляризация на разреженность.
# Инференс → forward pass → финальный sparse вектор → argmax по пикам → DOA.
# Оценка → RMSE, success rate (доля правильно найденных источников ±1°), сравнение с OMP.

# Пример кода (упрощённая реализация Learned ISTA / LISTA-style для DOA)python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Предполагаем, что у вас есть словарь A [N_ant, N_grid] — steering vectors на сетке
# Например: theta_grid = np.linspace(-90,90,361) * np.pi/180
# A = np.exp(1j * 2*np.pi * mimo[:,None] * np.sin(theta_grid[None,:]))

class LISTA_DOALayer(nn.Module):
    """Один слой Learned ISTA"""
    def __init__(self, A, theta=0.1, lambda_init=0.1):
        super().__init__()
        self.A = nn.Parameter(torch.complex(torch.tensor(A.real), torch.tensor(A.imag)), requires_grad=False)
        self.W = nn.Parameter(torch.eye(A.shape[1], dtype=torch.complex64))  # обучаемая прокси-матрица
        self.theta = nn.Parameter(torch.tensor(theta))   # обучаемый шаг
        self.lam = nn.Parameter(torch.tensor(lambda_init))  # обучаемый порог

    def forward(self, y, x_prev):
        # y - наблюдение [batch, N_ant]
        # x_prev - предыдущая оценка разреженного коэффициента [batch, N_grid]

        r = y - torch.matmul(self.A, x_prev)           # остаток
        z = x_prev + self.theta * torch.matmul(self.W, r)  # градиентный шаг

        # Soft-thresholding (комплексный)
        mag = torch.abs(z)
        phase = torch.angle(z)
        soft = F.relu(mag - self.lam)
        x_new = soft * torch.exp(1j * phase)

        return x_new
class DeepLISTA_DOA(nn.Module):
    def __init__(self, A, n_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([LISTA_DOALayer(A) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, y):
        # y: [batch, N_ant] комплексный
        batch_size = y.shape[0]
        N_grid = self.layers[0].A.shape[1]

        x = torch.zeros(batch_size, N_grid, dtype=torch.complex64, device=y.device)

        for layer in self.layers:
            x = layer(y, x)

        # Финальный sparse вектор -> вероятности/пики
        power = torch.abs(x)**2
        return power, x
# Wiener-Hopf предобработка (как раньше)
def wiener_suppress(sig_mimo, K=64):
    cov = np.mean([np.outer(s.flatten(), np.conj(s.flatten())) for s in sig_mimo[:K]], axis=0)
    R_inv = np.linalg.inv(cov + 1e-6*np.eye(cov.shape[0]))
    return np.dot(R_inv, sig_mimo.T).T
# Пример полного запуска
def run_lista_pipeline(N_samples=2000, n_layers=12, epochs=80, device='cuda'):
    # 1. Генерация синтетики + словарь
    from dataset import dataset_gen_final
    sig_mimo, angle_mimo, _, _ = dataset_gen_final(N_samples, init_flag=0, fft_on=0)

    # Создание steering dictionary (пример для вашей геометрии)
    mimo_pos = np.linspace(0, 5.5, 12)   # из вашего dataset.py
    theta_grid = np.deg2rad(np.linspace(-90,90,361))
    A = np.exp(1j * 2 * np.pi * mimo_pos[:,None] * np.sin(theta_grid[None,:]))

    # Предобработка
    sig_clean = wiener_suppress(sig_mimo.squeeze(-1))  # [N,12]

    # Ground truth grid labels (упрощённо — one-hot)
    grid = np.linspace(-90,90,361)
    labels = np.zeros((len(sig_clean), 361))
    for i, true_doa in enumerate(np.rad2deg(angle_mimo[:,0,0])):
        idx = np.argmin(np.abs(grid - true_doa))
        labels[i, idx] = 1.0

    # Тензоры
    Y = torch.complex(torch.tensor(sig_clean.real), torch.tensor(sig_clean.imag)).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    # 2. Модель
    model = DeepLISTA_DOA(A=A, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # 3. Обучение
    for epoch in range(epochs):
        power, _ = model(Y)
        loss = F.binary_cross_entropy_with_logits(power, labels) + 0.01 * torch.mean(torch.abs(power))  # sparsity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Loss: {loss.item():.6f}')

    # 4. Инференс + оценка
    with torch.no_grad():
        power, _ = model(Y)
        est_idx = torch.argmax(power, dim=1).cpu().numpy()
        est_doa = grid[est_idx]
        true_doa = np.rad2deg(angle_mimo[:,0,0])
        rmse = np.sqrt(np.mean((est_doa - true_doa)**2))
        print(f'Final RMSE: {rmse:.2f}°')

    return model
# Запуск
model_2 = run_lista_pipeline()


============================================================================================
Обновленная часть
============================================================================================


# 2. Physics-informed unfolded сеть (LISTA/OPAMP/Deep-OMP)Этот подход сочетает преимущества классических итеративных методов разреженного восстановления (OMP, ISTA, FPC и др.) с возможностями глубокого обучения. Сеть разворачивается (unfolds) в фиксированное количество слоёв, где каждый слой соответствует одной итерации классического алгоритма. Это делает модель интерпретируемой, быстрой на инференсе и позволяющей добиться очень хороших результатов даже при малом количестве снимков (single-snapshot), низком SNR и наличии ошибок геометрии антенной решётки — всё это крайне актуально для автомобильных радаров.Вывод по подходу (на январь 2026):Преимущества:Высокая интерпретируемость (каждый слой — физически осмысленная итерация)
# Отличная работа на single-snapshot (очень важно для движущихся целей в авто-радаре)
# Лучше обобщение на неидеальные условия (ошибки позиционирования элементов, mutual coupling, jamming) по сравнению с чистыми end-to-end CNN
# Возможность встраивать физику напрямую (steering vectors, sparsity prior, atomic norm и т.п.)
# Часто превосходит классический OMP на 10–30% по RMSE при SNR 0…15 дБ

# Недостатки:Требует тщательного проектирования разворачиваемого алгоритма (выбор базового метода — LISTA, OPAMP, ADMM, FPC-ℓ1 и др.)
# Обучение может быть сложнее и медленнее, чем у простых CNN (нужны хорошие инициализации весов)
# Меньше гибкости при очень большом количестве источников (>4–5)

# Когда использовать:Хотите максимальный публикационный потенциал (тема очень горячая в 2024–2025)
# Нужна объяснимость модели (можно показать, как сеть приближает OMP/ISTA)
# Работаете преимущественно с single-snapshot и/или разреженными массивами

# Ожидаемые результаты (на основе недавних работ 2024–2025):RMSE ~0.3–1.2° на синтетике при SNR 5–20 дБ
# Значительно лучше OMP в присутствии array imperfections и coherent источников (multipath)

# Пайплайн (самый популярный вариант на 2025–2026 — Deep Unfolded LISTA/OPAMP для DOA):Выбор базового алгоритма → чаще всего LISTA (Learned ISTA) или OPAMP (Orthogonal AMP) — они хорошо работают со sparsity в угловом домене.
# Подготовка данных → те же комплексные сигналы MIMO (ваш sig_mimo), но желательно ковариационная матрица R или напрямую сырой сигнал → steering dictionary A(θ).
# Предобработка → Wiener-Hopf подавление jamming (как в вашей статье) + опционально spatial smoothing для борьбы с coherent multipath.
# Модель → N слоёв (обычно 6–16), каждый слой содержит обучаемые параметры (shrinkage/threshold, step size, возможно мягкую ортогонализацию).
# Обучение → supervised на синтетике (ground truth DOA как one-hot или мягкие метки) + L1-подобная регуляризация на разреженность.
# Инференс → forward pass → финальный sparse вектор → argmax по пикам → DOA.
# Оценка → RMSE, success rate (доля правильно найденных источников ±1°), сравнение с OMP.

# Пример кода (упрощённая реализация Learned ISTA / LISTA-style для DOA)python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming dataset.py is in the same directory or sys.path includes it
from dataset import load_real_data # Make sure this is updated

# Wiener-Hopf предобработка (как раньше) - adapted for N_ant
def wiener_suppress(sig_batch, N_ant, K=64): # N_ant parameter added
    processed_signals = []
    for sig in sig_batch:
        sig_vector = sig.flatten() # Ensure it's 1D [N_ant]
        if sig_vector.shape[0] != N_ant:
            # This shouldn't be hit if data loading is correct, but for robustness
            print(f"Warning: Signal vector length {sig_vector.shape[0]} does not match N_ant {N_ant}")
            continue

        # Estimate covariance from the single snapshot with regularization
        # R = sig * sig_H + epsilon * I
        # For a single snapshot, R = s * s.H
        R = np.outer(sig_vector, sig_vector.conj()) + 1e-6 * np.eye(N_ant, dtype=np.complex64)
        R_inv = np.linalg.inv(R)

        # Apply spatial whitening (as interpreted from original notebook's context)
        # This is a basic form of linear pre-processing.
        processed_signals.append(np.dot(R_inv, sig_vector).flatten()) # Flatten to [N_ant,]
    return np.array(processed_signals)

class LISTA_DOALayer(nn.Module):
    """Один слой Learned ISTA"""
    def __init__(self, A, theta=0.1, lambda_init=0.1):
        super().__init__()
        # Ensure A is complex64 by explicitly casting real/imag parts to float32
        self.A = nn.Parameter(torch.complex(
            torch.tensor(A.real, dtype=torch.float32),
            torch.tensor(A.imag, dtype=torch.float32)
        ), requires_grad=False) # [N_ant, N_grid]
        # self.W should be a learned matrix that approximates A.H (dimensions [N_grid, N_ant])
        self.W = nn.Parameter(torch.complex(
            torch.randn(A.shape[1], A.shape[0], dtype=torch.float32), # N_grid, N_ant
            torch.randn(A.shape[1], A.shape[0], dtype=torch.float32)
        ))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))   # обучаемый шаг
        self.lam = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))  # обучаемый порог

    def forward(self, y, x_prev):
        # y - наблюдение [batch, N_ant]
        # x_prev - предыдущая оценка разреженного коэффициента [batch, N_grid]

        # Calculate estimated y: A * x_prev
        # self.A is [N_ant, N_grid]
        # x_prev is [batch, N_grid]
        # To get [batch, N_ant], we need (self.A @ x_prev.T).T
        estimated_y = torch.matmul(self.A, x_prev.transpose(1, 0)).transpose(1, 0) # [batch, N_ant]

        r = y - estimated_y  # residual [batch, N_ant]

        # Gradient step: x_prev + theta * W * r
        # self.W is [N_grid, N_ant]
        # r is [batch, N_ant]
        # To get [batch, N_grid], we need (r @ self.W.T)
        grad_update = torch.matmul(r, self.W.transpose(0, 1)) # [batch, N_ant] @ [N_ant, N_grid] -> [batch, N_grid]

        z = x_prev + self.theta * grad_update  # [batch, N_grid]

        # Soft-thresholding (комплексный)
        mag = torch.abs(z)
        phase = torch.angle(z)
        soft = F.relu(mag - self.lam)
        x_new = soft * torch.exp(1j * phase)

        return x_new

class DeepLISTA_DOA(nn.Module):
    def __init__(self, A, n_layers=10):
        super().__init__()
        self.layers = nn.ModuleList([LISTA_DOALayer(A) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, y): # y: [batch, N_ant]
        batch_size = y.shape[0]
        N_grid = self.layers[0].A.shape[1]

        x = torch.zeros(batch_size, N_grid, dtype=torch.complex64, device=y.device)

        for layer in self.layers:
            x = layer(y, x)

        # Финальный sparse вектор -> вероятности/пики
        power = torch.abs(x)**2
        return power, x

# Пример полного запуска
def run_lista_pipeline(data_path, N_samples=None, n_layers=12, epochs=80, device='cuda' if torch.cuda.is_available() else 'cpu', npz_path=None):
    device = 'cpu' # Temporarily force CPU to resolve potential CUDA errors
    print(f"Using device: {device}")

    # 1. Загрузка реальных данных
    print(f"Loading data from: {data_path}")
    sig_mimo_raw, angle_mimo_raw, metrix_raw, _ = load_real_data(data_path, npz_path=npz_path)

    # Use N_samples if specified, otherwise use all data
    if N_samples is not None:
        N_samples = min(N_samples, len(sig_mimo_raw))
        sig_mimo_raw = sig_mimo_raw[:N_samples]
        angle_mimo_raw = angle_mimo_raw[:N_samples]
        metrix_raw = metrix_raw[:N_samples]
    print(f"Loaded {len(sig_mimo_raw)} samples.")

    N_ant = sig_mimo_raw.shape[1] # Should be 8
    print(f"Detected {N_ant} antennas.")

    # 2. Предобработка: Wiener-Hopf
    print("Applying Wiener-Hopf suppression...")
    sig_clean = wiener_suppress(sig_mimo_raw.squeeze(-1), N_ant=N_ant) # [N, N_ant]

    # Создание steering dictionary (для N_ant=8)
    # Assuming the aperture is maintained, adjust element positions
    mimo_pos = np.linspace(0, 5.5, N_ant)   # 8 elements over the same aperture
    theta_grid = np.deg2rad(np.linspace(-90,90,361)) # 361 points for -90 to 90 with 0.5 degree step
    # A: [N_ant, N_grid] steering vectors
    A = np.exp(1j * 2 * np.pi * mimo_pos[:,None] * np.sin(theta_grid[None,:])) # [8, 361]
    N_grid = A.shape[1]

    # Ground truth grid labels (multi-hot for multiple angles)
    labels = np.zeros((len(sig_clean), N_grid))
    for i, true_doas_rad in enumerate(angle_mimo_raw):
        # true_doas_rad is [2, 1], flatten to [2]
        for true_doa_rad in true_doas_rad.flatten():
            true_doa_deg = np.rad2deg(true_doa_rad)
            if -90 <= true_doa_deg <= 90:
                # Find the closest index on the grid
                idx = np.argmin(np.abs(np.rad2deg(theta_grid) - true_doa_deg))
                labels[i, idx] = 1.0

    # 3. Data Splitting
    X_train, X_val, y_train_labels, y_val_labels, train_original_indices, val_original_indices = train_test_split(
        sig_clean, labels, np.arange(len(sig_clean)), test_size=0.2, random_state=42
    )

    # Тензоры
    Y_train = torch.complex(torch.tensor(X_train.real), torch.tensor(X_train.imag)).to(device)
    Y_val = torch.complex(torch.tensor(X_val.real), torch.tensor(X_val.imag)).to(device)
    labels_train = torch.tensor(y_train_labels, dtype=torch.float32).to(device)
    labels_val = torch.tensor(y_val_labels, dtype=torch.float32).to(device)

    # 4. Модель
    model = DeepLISTA_DOA(A=A, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for multi-hot labels and raw logits

    # Store losses and RMSE for plotting
    train_losses = []
    val_losses = []
    train_rmse = []
    val_rmse = []

    # 5. Обучение
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        power_train_logits, _ = model(Y_train) # Get raw logits from model output
        # Use BCEWithLogitsLoss with raw logits
        loss_train = loss_fn(power_train_logits, labels_train) + 0.01 * torch.mean(torch.abs(power_train_logits)) # sparsity on logits
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        # Validation
        model.eval()
        with torch.no_grad():
            power_val_logits, _ = model(Y_val) # Get raw logits
            loss_val = loss_fn(power_val_logits, labels_val) + 0.01 * torch.mean(torch.abs(power_val_logits)) # sparsity on logits
            val_losses.append(loss_val.item())

            # Calculate RMSE for training set
            # Apply sigmoid to logits to get probabilities for argmax
            power_train_probs = torch.sigmoid(power_train_logits)
            est_idx_train = torch.argmax(power_train_probs, dim=1).cpu().numpy()
            est_doa_train = np.rad2deg(theta_grid[est_idx_train])

            rmse_train_batch = []
            for p_doa, i_original in zip(est_doa_train, train_original_indices):
                t_doas_rad = angle_mimo_raw[i_original].flatten()
                t_doas_deg = np.rad2deg(t_doas_rad)
                # Filter out invalid angles (e.g., placeholder -16.45) if any
                valid_t_doas_deg = t_doas_deg[(-90 <= t_doas_deg) & (t_doas_deg <= 90)]

                if len(valid_t_doas_deg) > 0:
                    min_diff = np.min(np.abs(valid_t_doas_deg - p_doa))
                    rmse_train_batch.append(min_diff)
            if rmse_train_batch:
                train_rmse.append(np.sqrt(np.mean(np.array(rmse_train_batch)**2)))
            else:
                train_rmse.append(np.nan)


            # Calculate RMSE for validation set
            power_val_probs = torch.sigmoid(power_val_logits)
            est_idx_val = torch.argmax(power_val_probs, dim=1).cpu().numpy()
            est_doa_val = np.rad2deg(theta_grid[est_idx_val])

            rmse_val_batch = []
            for p_doa, i_original in zip(est_doa_val, val_original_indices):
                t_doas_rad = angle_mimo_raw[i_original].flatten()
                t_doas_deg = np.rad2deg(t_doas_rad)
                valid_t_doas_deg = t_doas_deg[(-90 <= t_doas_deg) & (t_doas_deg <= 90)]
                if len(valid_t_doas_deg) > 0:
                    min_diff = np.min(np.abs(valid_t_doas_deg - p_doa))
                    rmse_val_batch.append(min_diff)
            if rmse_val_batch:
                val_rmse.append(np.sqrt(np.mean(np.array(rmse_val_batch)**2)))
            else:
                val_rmse.append(np.nan)

        print(f'Epoch {epoch:3d} | Train Loss: {loss_train.item():.6f} | Train RMSE: {train_rmse[-1]:.2f}° | Val Loss: {loss_val.item():.6f} | Val RMSE: {val_rmse[-1]:.2f}°')

    # 6. Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_rmse, label='Train DOA RMSE')
    plt.plot(val_rmse, label='Validation DOA RMSE')
    plt.title('DOA RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (°)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot example DOA predictions vs ground truth
    print("\nPlotting example DOA predictions...")
    model.eval()
    num_plots = 5 # Number of samples to plot
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3), sharex=True)
    if num_plots == 1: # Handle case of single plot to avoid iteration issues
        axes = [axes]

    # val_original_indices is already available from train_test_split
    # power_all_val_logits contains logits for the entire validation set

    with torch.no_grad():
        power_all_val_logits, _ = model(Y_val)
        power_all_val_probs = torch.sigmoid(power_all_val_logits) # Convert logits to probabilities for plotting

    for i in range(num_plots):
        sample_idx_in_val_set = np.random.randint(0, len(X_val)) # Pick a random sample from validation set
        original_idx = val_original_indices[sample_idx_in_val_set]

        # Get predicted spectrum (probabilities)
        doa_spectrum = power_all_val_probs[sample_idx_in_val_set].cpu().numpy()

        # Get true angles
        true_angles_val_rad = angle_mimo_raw[original_idx].flatten()
        true_angles_val_deg = np.rad2deg(true_angles_val_rad)
        valid_true_angles_val_deg = true_angles_val_deg[(-90 <= true_angles_val_deg) & (true_angles_val_deg <= 90)]


        axes[i].plot(np.rad2deg(theta_grid), doa_spectrum, label='Predicted DOA Spectrum')

        # Mark true DOAs
        for true_ang in valid_true_angles_val_deg:
            axes[i].axvline(x=true_ang, color='r', linestyle='--', label=f'True DOA: {true_ang:.2f}°')

        # Find predicted DOA peak
        est_doa_idx = np.argmax(doa_spectrum)
        est_doa = np.rad2deg(theta_grid[est_doa_idx])
        axes[i].axvline(x=est_doa, color='g', linestyle=':', label=f'Est. DOA Peak: {est_doa:.2f}°')

        axes[i].set_title(f'Sample {original_idx} DOA Prediction')
        axes[i].set_xlabel('Angle (degrees)')
        axes[i].set_ylabel('Power / Probability')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

    return model

# Запуск
print("Running LISTA Pipeline with real data...")
data_path = DATA_PATH
model_2_refined = run_lista_pipeline(data_path, N_samples=10000, epochs=10, device='cuda', npz_path=NPZ_PATH)
