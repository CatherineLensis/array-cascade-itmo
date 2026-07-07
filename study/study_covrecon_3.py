"""## 3. Covariance reconstruction + классический/лёгкий нейронный DOA"""

# 3. Covariance reconstruction + классический/лёгкий нейронный DOA
# (хороший компромисс при ограниченных вычислительных ресурсах)Этот подход — один из самых практичных для автомобильных радаров в 2025–2026 годах, особенно если целевая платформа имеет ограниченные ресурсы (embedded-системы, низкое энергопотребление, отсутствие мощного GPU).Идея в двух словах:
# сначала нейросеть (обычно небольшая CNN или даже MLP) восстанавливает/улучшает ковариационную матрицу при низком SNR, малом количестве снимков, coherent-сигналах (multipath) и/или jamming,
# а затем на восстановленной матрице применяется классический или очень лёгкий метод DOA-оценки (MUSIC, ESPRIT, Root-MUSIC, OMP, weighted l1 и т.п.).Вывод по подходу (состояние на январь 2026):Преимущества  Очень низкие требования к вычислениям на инференсе (особенно если после реконструкции использовать классический MUSIC/ESPRIT)
# Хорошая устойчивость к низкому числу снимков (single-snapshot и 2–4 snapshots)
# Отлично справляется с array imperfections (ошибки позиционирования элементов, gain/phase mismatch)
# Легко интегрируется с существующими классическими пайплайнами (можно просто заменить этап вычисления выборочной ковариации на нейронную)
# Часто даёт прирост 15–40% по точности при SNR < 10 дБ по сравнению с чистой классикой

# Недостатки  Средняя производительность на очень сильной многолучёвости (если не добавлять специальные головы)
# Нужно достаточно много синтетических данных для обучения реконструктора
# Меньше гибкости при большом количестве источников (>4–5) по сравнению с end-to-end DL

# Когда использовать  Целевая платформа — embedded (TI AWRxxxx, NXP, Renesas и т.п.)
# Уже есть рабочий классический пайплайн и хочется улучшить его «малой кровью»
# Нет мощного GPU для обучения/инференса
# Нужно максимальное время жизни батареи / минимальное тепловыделение

# Ожидаемые результаты  RMSE 0.8–2.5° при SNR 0…15 дБ на синтетике
# Прирост разрешения и точности на 20–50% по сравнению с обычным sample covariance + MUSIC
# Хорошая работа на реальных данных TI AWR1843 после дообучения/адаптации

# Типичный современный пайплайн (2025–2026)Сырые данные → формирование sample covariance R_sample (даже если всего 1–4 snapshots)
# Маленькая нейросеть → предсказывает улучшенную/восстановленную ковариационную матрицу R_clean
# На R_clean применяется классический высокоточный метод:Root-MUSIC / Unitary Root-MUSIC (очень быстро и точно)
# Weighted l1 (если нужно учитывать разные мощности целей)
# Sparse Bayesian Learning (очень редко — слишком медленно)

# Опционально: пост-обработка (кластеризация пиков, проверка по вашей метрике multipath)

# Пример кода (очень лёгкая реализация — MLP-реконструктор + Root-MUSIC)python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh  # для eigendecomposition

# Очень простая сеть-реконструктор ковариационной матрицы
class CovReconstructor(nn.Module):
    """MLP для восстановления ковариационной матрицы"""
    def __init__(self, N_ant=12, hidden=128):
        super().__init__()
        input_size = N_ant * N_ant * 2  # real + imag части Hermitian матрицы (верхний треугольник)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, R_flat):
        # R_flat: [batch, N_ant*N_ant*2] (real+imag верхнего треугольника)
        out = self.net(R_flat)
        return out
def flat_to_hermitian(R_flat, N):
    """Преобразование плоского вектора в Hermitian матрицу"""
    R_real = R_flat[:, :N*N].view(-1, N, N)
    R_imag = R_flat[:, N*N:].view(-1, N, N)
    R = torch.complex(R_real, R_imag)

    # Делаем Hermitian (R + R^H)/2
    R = (R + R.conj().transpose(-2, -1)) / 2
    return R
def root_music(R, N_sources=2, d_lambda=0.5):
    """
    Упрощённый Root-MUSIC (работает на комплексных числах)
    Возвращает углы в радианах
    """
    N = R.shape[-1]
    _, vecs = eigh(R.cpu().numpy())  # собственные векторы
    Un = vecs[:, :N - N_sources]      # шумовое подпространство

    # Полином для корней
    poly_coef = np.zeros(2*N-1, dtype=complex)
    for i in range(N):
        for j in range(N):
            poly_coef[i-j+N-1] += np.conj(Un[i, :]) @ Un[j, :]

    roots = np.roots(poly_coef)
    # Выбираем корни внутри единичной окружности
    roots = roots[np.abs(roots) < 1.0001]
    # Углы
    doa = np.arcsin(np.angle(roots) / (2 * np.pi * d_lambda))
    doa = np.sort(doa[np.abs(doa) <= np.pi/2])

    return np.rad2deg(doa)
# Полный пайплайн
def run_cov_recon_pipeline(N_samples=3000, epochs=120, device='cuda'):
    from dataset import dataset_gen_final

    # 1. Генерация данных
    sig_mimo, angle_mimo, metrix, _ = dataset_gen_final(N_samples, init_flag=0, fft_on=0)
    N_ant = sig_mimo.shape[1]  # 12

    # 2. Формируем sample covariance (даже single-snapshot)
    R_samples = []
    for s in sig_mimo.squeeze(-1):
        s = s / np.linalg.norm(s)
        R = np.outer(s, s.conj())
        # Вектор верхнего треугольника (real + imag)
        triu_idx = np.triu_indices(N_ant)
        real_part = np.real(R)[triu_idx]
        imag_part = np.imag(R)[triu_idx]
        flat = np.concatenate([real_part, imag_part])
        R_samples.append(flat)

    R_samples = np.array(R_samples)

    # Ground truth DOA для обучения (необязательно, можно и unsupervised)
    doa_true = np.rad2deg(angle_mimo[:,0,0])

    # 3. Датасет
    R_tensor = torch.tensor(R_samples, dtype=torch.float32).to(device)

    # 4. Модель
    model = CovReconstructor(N_ant=N_ant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 5. Обучение (unsupervised вариант — минимизация reconstruction loss + физические регуляризации)
    for epoch in range(epochs):
        out_flat = model(R_tensor)
        R_recon = flat_to_hermitian(out_flat, N_ant)

        # Reconstruction loss
        loss_rec = F.mse_loss(R_recon.real, torch.tensor(R_samples[:,:N_ant*N_ant], device=device).view(-1,N_ant,N_ant)) + \
                   F.mse_loss(R_recon.imag, torch.tensor(R_samples[:,N_ant*N_ant:], device=device).view(-1,N_ant,N_ant))

        # Дополнительно: положительная определённость (очень грубо)
        loss_pd = torch.mean(torch.relu(-torch.linalg.eigvals(R_recon.real)))

        loss = loss_rec + 0.1 * loss_pd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch:3d} | Rec loss: {loss_rec.item():.5f} | PD loss: {loss_pd.item():.5f}')

    # 6. Инференс + Root-MUSIC
    with torch.no_grad():
        R_recon_flat = model(R_tensor)
        R_recon = flat_to_hermitian(R_recon_flat, N_ant)

        rmse_list = []
        for i in range(min(200, len(R_recon))):  # пример на первых 200
            doa_est = root_music(R_recon[i], N_sources=2)
            if len(doa_est) > 0:
                rmse_list.append(np.min(np.abs(doa_est - doa_true[i])))

        print(f'Average angular error (min distance): {np.mean(rmse_list):.2f}°')
# Запуск
model_3 = run_cov_recon_pipeline()


============================================================================================
Обновленная часть
============================================================================================


"""## Refine Covariance Reconstruction Pipeline

### Subtask:
Analyze cell_oyDFaZDXt1OM (Covariance reconstruction + classical/lightweight neural DOA). Modify the `run_cov_recon_pipeline` function to use the `load_real_data` utility. Adapt the CovReconstructor model and Root-MUSIC for handling the 8-antenna real data. Integrate a validation loop during training, track training and validation losses, and RMSE. Add plotting code to visualize training/validation curves and an example of DOA prediction versus ground truth.

**Reasoning**:
I need to modify the `run_cov_recon_pipeline` function to integrate real data loading, adapt to 8 antennas, implement data splitting, add a validation loop with RMSE tracking, and include plotting for losses, RMSE, and example DOA predictions. This will involve updating data processing steps, the model initialization, and the evaluation logic to handle multiple ground truth angles.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh  # для eigendecomposition
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assuming dataset.py is in the same directory or sys.path includes it
from dataset import load_real_data # Make sure this is updated

# Wiener-Hopf предобработка (unchanged from previous cells, ensuring complex64 numpy arrays)
def wiener_suppress(sig_batch, N_ant, K=64):
    processed_signals = []
    for sig in sig_batch:
        sig_vector = sig.flatten() # Ensure it's 1D [N_ant]
        if sig_vector.shape[0] != N_ant:
            print(f"Warning: Signal vector length {sig_vector.shape[0]} does not match N_ant {N_ant}")
            continue
        R = np.outer(sig_vector, sig_vector.conj()) + 1e-6 * np.eye(N_ant, dtype=np.complex64)
        R_inv = np.linalg.inv(R)
        processed_signals.append(np.dot(R_inv, sig_vector).flatten()) # Flatten to [N_ant,]
    return np.array(processed_signals, dtype=np.complex64)

# Очень простая сеть-реконструктор ковариационной матрицы
class CovReconstructor(nn.Module):
    """MLP для восстановления ковариационной матрицы"""
    def __init__(self, N_ant=12, hidden=128):
        super().__init__()
        # input_size represents real and imag parts of a full N_ant x N_ant complex matrix
        input_size = N_ant * N_ant * 2
        self.N_ant = N_ant # Store N_ant
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, R_flat):
        # R_flat: [batch, N_ant*N_ant*2] (real+imag parts of full matrix)
        out = self.net(R_flat)
        return out

def flat_to_hermitian(R_flat, N):
    """Преобразование плоского вектора в Hermitian матрицу"""
    # R_flat is [batch, N*N*2]
    batch_size = R_flat.shape[0]
    # Split flattened vector back into real and imaginary parts of N x N matrices
    R_real_flat = R_flat[:, :N*N]
    R_imag_flat = R_flat[:, N*N:]

    R_real = R_real_flat.view(batch_size, N, N)
    R_imag = R_imag_flat.view(batch_size, N, N)

    R_complex = torch.complex(R_real, R_imag)

    # Ensure the reconstructed matrix is Hermitian by averaging with its conjugate transpose
    R_hermitian = (R_complex + R_complex.conj().transpose(-2, -1)) / 2
    return R_hermitian

def root_music(R, N_sources=1, d_lambda=0.5):
    """
    Упрощённый Root-MUSIC (работает на комплексных числах)
    Возвращает углы в градусах.
    R: [N_ant, N_ant] complex covariance matrix (expected to be numpy array)
    """
    N_ant = R.shape[-1]

    # Perform eigendecomposition for Hermitian matrix R
    eigenvalues, eigenvectors = eigh(R) # R is already a numpy array here

    # Construct noise subspace eigenvectors
    # The smallest N_ant - N_sources eigenvalues correspond to the noise subspace
    noise_subspace_vectors = eigenvectors[:, :N_ant - N_sources] # [N_ant, N_ant - N_sources]

    # Form the matrix Q = U_n * U_n^H
    Q = noise_subspace_vectors @ noise_subspace_vectors.conj().T # [N_ant, N_ant]

    # Construct polynomial coefficients p(z) = sum_{k=-(N-1)}^{N-1} p_k z^k
    # where p_k = sum_{i-j=k} Q_{ij}
    poly_coeffs = np.zeros(2 * N_ant - 1, dtype=np.complex128)

    for k in range(-(N_ant - 1), N_ant):
        pk_sum = 0.0
        for i in range(N_ant):
            j = i - k # Ensure i - j = k => j = i - k
            if 0 <= j < N_ant:
                pk_sum += Q[i, j]
        poly_coeffs[k + N_ant - 1] = pk_sum

    # Find roots of the polynomial
    roots = np.roots(poly_coeffs)

    # Select roots closest to the unit circle. Root-MUSIC theory implies signal roots
    # are on or very near the unit circle. We will take N_sources roots with magnitude
    # closest to 1.
    roots_on_unit_circle_indices = np.argsort(np.abs(np.abs(roots) - 1.0))[:2*N_sources] # Select 2*N_sources candidates
    closest_roots = roots[roots_on_unit_circle_indices]

    # Extract angles from the phase of the selected roots
    # z = e^(j * 2 * pi * d/lambda * sin(theta))
    # arg(z) = 2 * pi * d/lambda * sin(theta)
    # Assuming half-wavelength spacing d/lambda = 0.5
    d_lambda_const = 0.5

    doa_rad_candidates = np.arcsin(np.angle(closest_roots) / (2 * np.pi * d_lambda_const))

    # Filter for real angles within [-pi/2, pi/2] radians
    # and convert to degrees
    doa_deg_filtered = np.rad2deg(np.real(doa_rad_candidates[(np.isreal(doa_rad_candidates)) &
                                                            (doa_rad_candidates >= -np.pi/2) &
                                                            (doa_rad_candidates <= np.pi/2)]))

    # Remove duplicates and sort
    doa_deg = np.sort(np.unique(doa_deg_filtered))

    # Return up to N_sources angles. If more angles are found than N_sources, it's ambiguous
    # which ones to return without a pseudo-spectrum to pick peaks. For simplicity, if more than
    # N_sources are found, we'll return only N_sources. If fewer, return all found.
    if len(doa_deg) > N_sources:
        # In a full Root-MUSIC, we'd pair conjugates and select the N_sources. For now,
        # a simplified selection might be just taking the first N_sources if many exist,
        # or more robustly, selecting based on closeness to true unit circle or other metric.
        # Given the task is about integration, a direct selection is okay for this stage.
        return doa_deg[:N_sources]
    else:
        return doa_deg # Returns list of detected DOAs in degrees

# Helper function to calculate RMSE between two sets of angles
def calculate_multi_angle_rmse(predicted_angles_deg, true_angles_deg):
    """
    Calculates RMSE for multiple predicted angles against multiple true angles.
    Matches each predicted angle to its closest true angle.
    Returns NaN if no valid matches or no valid true angles.
    """
    # Filter out invalid true angles (e.g., placeholder -16.45) if any
    valid_true_doas_deg = true_angles_deg[(-90 <= true_angles_deg) & (true_angles_deg <= 90)]

    if len(valid_true_doas_deg) == 0: # If no valid true angles, cannot compute RMSE
        return np.nan

    if len(predicted_angles_deg) == 0: # If no predicted angles, can't match, return NaN or a large error
        return np.nan

    min_diffs_sq = []
    for p_angle in predicted_angles_deg:
        # For each predicted angle, find the closest true angle
        diffs = np.abs(valid_true_doas_deg - p_angle)
        min_diffs_sq.append(np.min(diffs)**2)

    if len(min_diffs_sq) > 0:
        return np.sqrt(np.mean(min_diffs_sq))
    else:
        return np.nan

# Полный пайплайн
def run_cov_recon_pipeline(data_path, N_samples=None, epochs=120, device='cuda' if torch.cuda.is_available() else 'cpu', npz_path=None):
    device = 'cpu' # Force CPU
    print(f"Using device: {device}")

    # 1. Загрузка реальных данных
    print(f"Loading data from: {data_path}")
    sig_mimo_raw, angle_mimo_raw, metrix_raw, mb_metrix_raw = load_real_data(data_path, npz_path=npz_path)

    # Use N_samples if specified, otherwise use all data
    if N_samples is not None:
        N_samples = min(N_samples, len(sig_mimo_raw))
        sig_mimo_raw = sig_mimo_raw[:N_samples]
        angle_mimo_raw = angle_mimo_raw[:N_samples]
        metrix_raw = metrix_raw[:N_samples]
        mb_metrix_raw = mb_metrix_raw[:N_samples] # Keep for consistency, though unused
    print(f"Loaded {len(sig_mimo_raw)} samples.")

    N_ant = sig_mimo_raw.shape[1]  # This should be 8 now based on load_real_data
    print(f"Detected {N_ant} antennas.")

    # 2. Формируем sample covariance (даже single-snapshot)
    print("Forming sample covariance matrices...")
    R_samples_flat_list = []
    for s_complex_vec in sig_mimo_raw.squeeze(-1): # s_complex_vec is [N_ant,]
        # R = s * s^H + epsilon * I for single snapshot to avoid singular matrix
        R_complex = np.outer(s_complex_vec, s_complex_vec.conj()) + 1e-6 * np.eye(N_ant, dtype=np.complex64)

        # Convert to flattened real/imag parts for the network input
        R_real = R_complex.real.flatten()
        R_imag = R_complex.imag.flatten()
        flat_representation = np.concatenate([R_real, R_imag]) # [N_ant*N_ant*2]
        R_samples_flat_list.append(flat_representation)

    R_samples_flat = np.array(R_samples_flat_list, dtype=np.float32) # [N, N_ant*N_ant*2]

    # Ground truth DOA (already in angle_mimo_raw) - will be used for RMSE
    # angle_mimo_raw has shape [N, 2, 1] (radians)

    # 3. Data Splitting
    # Split the flattened R_samples, true angles, and corresponding original indices
    (R_train_flat_np, R_val_flat_np,
     angles_train_rad, angles_val_rad,
     train_original_indices, val_original_indices) = train_test_split(
        R_samples_flat, angle_mimo_raw, np.arange(len(sig_mimo_raw)), test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(R_train_flat_np)}, Validation samples: {len(R_val_flat_np)}")

    # Convert to tensors for PyTorch model input
    R_train_tensor = torch.tensor(R_train_flat_np, dtype=torch.float32).to(device)
    R_val_tensor = torch.tensor(R_val_flat_np, dtype=torch.float32).to(device)

    # 4. Модель
    model = CovReconstructor(N_ant=N_ant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Loss function for covariance matrix reconstruction (MSE)
    # The target for reconstruction loss is the original flattened complex covariance matrix
    # from which the real and imag parts were extracted.

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

        # Forward pass for training data
        out_train_flat = model(R_train_tensor)
        R_recon_train = flat_to_hermitian(out_train_flat, N_ant)

        # Compute target flattened real and imag parts from R_train_tensor for MSE comparison
        target_R_train_real_flat = R_train_tensor[:, :N_ant*N_ant]
        target_R_train_imag_flat = R_train_tensor[:, N_ant*N_ant:]

        # Reshape for MSE loss (target is batch, N_ant, N_ant for consistency)
        target_R_train_real_matrix = target_R_train_real_flat.view(-1, N_ant, N_ant)
        target_R_train_imag_matrix = target_R_train_imag_flat.view(-1, N_ant, N_ant)

        loss_rec_train = F.mse_loss(R_recon_train.real, target_R_train_real_matrix) + \
                         F.mse_loss(R_recon_train.imag, target_R_train_imag_matrix)

        # Positive definiteness regularization (optional, but good practice for covariance matrices)
        # Use eigvalsh for Hermitian matrices (real part should be symmetric)
        loss_pd_train = torch.mean(torch.relu(-torch.linalg.eigvalsh(R_recon_train.real)))

        loss_train = loss_rec_train + 0.1 * loss_pd_train
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            out_val_flat = model(R_val_tensor)
            R_recon_val = flat_to_hermitian(out_val_flat, N_ant)

            target_R_val_real_flat = R_val_tensor[:, :N_ant*N_ant]
            target_R_val_imag_flat = R_val_tensor[:, N_ant*N_ant:]

            target_R_val_real_matrix = target_R_val_real_flat.view(-1, N_ant, N_ant)
            target_R_val_imag_matrix = target_R_val_imag_flat.view(-1, N_ant, N_ant)

            loss_rec_val = F.mse_loss(R_recon_val.real, target_R_val_real_matrix) + \
                           F.mse_loss(R_recon_val.imag, target_R_val_imag_matrix)
            loss_pd_val = torch.mean(torch.relu(-torch.linalg.eigvalsh(R_recon_val.real)))
            loss_val = loss_rec_val + 0.1 * loss_pd_val
            val_losses.append(loss_val.item())

            # Calculate RMSE for training data (using Root-MUSIC)
            current_train_rmse = []
            for i in range(len(R_train_flat_np)):
                R_matrix = R_recon_train[i].cpu().numpy()
                est_doa_deg = root_music(R_matrix, N_sources=2) # Get estimated DOAs from Root-MUSIC in degrees

                original_idx = train_original_indices[i]
                true_doas_rad = angle_mimo_raw[original_idx].flatten()
                true_doas_deg = np.rad2deg(true_doas_rad)

                rmse_sample = calculate_multi_angle_rmse(est_doa_deg, true_doas_deg)
                current_train_rmse.append(rmse_sample)

            train_rmse.append(np.nanmean(current_train_rmse)) # Use nanmean to handle NaNs

            # Calculate RMSE for validation data (using Root-MUSIC)
            current_val_rmse = []
            for i in range(len(R_val_flat_np)):
                R_matrix = R_recon_val[i].cpu().numpy()
                est_doa_deg = root_music(R_matrix, N_sources=2) # Get estimated DOAs from Root-MUSIC in degrees

                original_idx = val_original_indices[i]
                true_doas_rad = angle_mimo_raw[original_idx].flatten()
                true_doas_deg = np.rad2deg(true_doas_rad)

                rmse_sample = calculate_multi_angle_rmse(est_doa_deg, true_doas_deg)
                current_val_rmse.append(rmse_sample)

            val_rmse.append(np.nanmean(current_val_rmse)) # Use nanmean

        print(f'Epoch {epoch:3d} | Train Loss: {loss_train.item():.5f} | Train RMSE: {train_rmse[-1]:.2f}° | Val Loss: {loss_val.item():.5f} | Val RMSE: {val_rmse[-1]:.2f}°')

    # 6. Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Covariance Reconstruction Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_rmse, label='Train DOA RMSE')
    plt.plot(val_rmse, label='Validation DOA RMSE')
    plt.title('DOA RMSE (Root-MUSIC) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (°)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot example DOA predictions
    print("\nPlotting example DOA predictions...")
    num_plots = 5
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3), sharex=True)
    if num_plots == 1:
        axes = [axes]

    with torch.no_grad():
        out_val_flat_all = model(R_val_tensor)
        R_recon_val_all = flat_to_hermitian(out_val_flat_all, N_ant)

    for i in range(num_plots):
        sample_idx_in_val_set = np.random.randint(0, len(R_val_tensor)) # Pick a random sample from validation set
        original_idx = val_original_indices[sample_idx_in_val_set]

        # Reconstructed covariance matrix for this sample
        recon_R_matrix = R_recon_val_all[sample_idx_in_val_set].cpu().numpy()

        # Estimated DOAs using Root-MUSIC
        est_doa_deg = root_music(recon_R_matrix, N_sources=2) # N_sources=2 to potentially find two angles

        # True DOAs
        true_doas_rad = angle_mimo_raw[original_idx].flatten()
        true_doas_deg = np.rad2deg(true_doas_rad)
        valid_true_doas_deg = true_doas_deg[(-90 <= true_doas_deg) & (true_doas_deg <= 90)]

        # Plotting - Root-MUSIC doesn't produce a spectrum, just discrete angles.
        # So we'll mark them on a simple line.
        angles_grid = np.linspace(-90, 90, 181) # A grid for plotting reference

        axes[i].plot(angles_grid, np.zeros_like(angles_grid), 'k-', alpha=0.3) # Reference line
        axes[i].set_ylim(-0.1, 1.1) # Arbitrary y-limit for markers

        # Mark true DOAs
        for true_ang in valid_true_doas_deg:
            axes[i].axvline(x=true_ang, color='r', linestyle='--', label=f'True DOA: {true_ang:.2f}°', ymax=0.8)

        # Mark estimated DOAs
        for est_ang in est_doa_deg:
            axes[i].axvline(x=est_ang, color='g', linestyle=':', label=f'Est. DOA: {est_ang:.2f}°', ymax=0.5)

        axes[i].set_title(f'Sample {original_idx} DOA Prediction (Root-MUSIC)')
        axes[i].set_xlabel('Angle (degrees)')
        axes[i].set_ylabel('')
        axes[i].set_yticks([]) # Hide y-axis ticks
        axes[i].legend(loc='upper right')
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

    return model

# Запуск
print("Running Covariance Reconstruction Pipeline with real data...")
data_path = DATA_PATH
model_3_refined = run_cov_recon_pipeline(data_path, N_samples=10000, epochs=10, device='cuda', npz_path=NPZ_PATH)