"""
dataset.py
==========
Генерация синтетических данных и загрузка реальных данных
для задачи классификации мультипаса / определения углов (DOA).

Модель сигнала соответствует статьям:
  - "Multipath Transaction on radar system" (Шишанов и др.)
  - "Jamming paper v2"

Радар: TI AWR1843 (2Tx + 4Rx = 8 виртуальных антенн)
Виртуальный массив: позиции [0, 0.5, 1.0, ..., 3.5] в долях λ/2

Формат mb_metrix [3] (совместим с реальными данными cols 19-21):
  [0] — projection metric (проекция на steering-вектор, ~1 для 1 цели)
  [1] — OMP residual norm после 1-й итерации
  [2] — OMP residual norm после 2-й итерации

Три класса:
  0 — один сигнал (θ_tx == θ_rx)
  1 — мультипас   (θ_tx ≠ θ_rx, знаки противоположны)
  2 — два сигнала  (θ_tx1==θ_rx1, θ_tx2==θ_rx2)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────────────────────
#  Ограничение углов по физическому смыслу
# ──────────────────────────────────────────────────────────────

_AZ_MAX_RAD = np.deg2rad(85.0)  # максимальный азимут ±85°


def _clip_angle(az_rad):
    """Обрезаем угол до ±85° чтобы sin(θ) оставался в допустимом диапазоне."""
    return np.clip(az_rad, -_AZ_MAX_RAD, _AZ_MAX_RAD)


# ──────────────────────────────────────────────────────────────
#  Метрики мультипаса (по статье)
# ──────────────────────────────────────────────────────────────

def math_classificator(metrix, mode='residual', threshold=0.8):
    """
    Бинарный классификатор на основе метрики.
    mode='projection': metrix < threshold → multipath (class 1)
    mode='residual':   metrix > threshold → multipath (class 1)
    """
    y_pred = np.zeros([len(metrix), 1], dtype=int)
    if mode == 'projection':
        ids = np.where(metrix < threshold)
    else:
        ids = np.where(metrix > threshold)
    y_pred[ids, :] = 1
    return y_pred


def multipath_residual_metric(sig, steering_vec):
    """
    Проекционная и residual метрика мультипаса (формула из статьи).

    proj     = ||P·s|| / ||s||,  где P = â·â^H, â = a/||a||
    residual = sqrt(1 - proj²)

    Returns: (residual, proj)
    """
    steering_vec = steering_vec / np.linalg.norm(steering_vec)
    P = np.dot(steering_vec, np.conj(steering_vec.T))
    proj = np.linalg.norm(np.dot(P, sig)) / np.linalg.norm(sig)
    residual = np.sqrt(max(0.0, 1.0 - proj ** 2))
    return residual, proj


def plot_DOA(sig, step):
    """Построение DOA-спектра через FFT с нулевым дополнением."""
    angle_arr = np.arcsin(np.linspace(-512, 511, 1024) / 1024 / step)
    ch_wide = np.zeros(1024, dtype=complex)
    ch_wide[0:len(sig)] = sig
    plt.plot(angle_arr * 180 / np.pi, abs(np.fft.fftshift(np.fft.fft(ch_wide))))
    return 0


def OMP_FFT(idat, ch_loc, N_fft, iternum):
    """
    Orthogonal Matching Pursuit через FFT на неравномерной решётке.

    Iterative: на каждой итерации
      1. Помещаем сигнал в позиции виртуальных антенн → FFT
      2. Находим пик (доминантный угол)
      3. Вычитаем вклад найденного угла
      4. dat_norm[i] = ||residual|| / ||original|| — доля остаточной энергии

    Параметры:
        idat     — комплексный вектор сигнала [N_ant]
        ch_loc   — позиции антенн в долях λ/2
        N_fft    — размер FFT (обычно 1024)
        iternum  — число итераций (2 для нашей задачи)

    Returns:
        oangle   — найденные углы в радианах [iternum]
        dat_norm — нормированные остатки [iternum]
    """
    dat_ind = (ch_loc / 0.5).astype(int)
    osig = np.reshape(idat, [1, len(idat)])
    dat_w = np.zeros([1, N_fft], dtype=complex)
    angle_arr = np.arcsin(np.linspace(-N_fft / 2, N_fft / 2 - 1, N_fft) / (0.5 * N_fft))
    dat_norm = np.zeros(iternum)
    oangle = np.zeros(iternum)
    for i in range(iternum):
        dat_w[0, dat_ind] = osig
        DOA = np.fft.fft(dat_w)
        max_id = np.argmax(abs(DOA))

        amp = DOA[0, max_id] / len(idat)
        osig = osig - amp * np.exp(-1j * 2 * np.pi * dat_ind * (N_fft - max_id) / N_fft)
        dat_norm[i] = np.linalg.norm(osig) / np.linalg.norm(idat)

        max_id = np.argmax(np.fft.fftshift(abs(DOA[0, :])))
        oangle[i] = angle_arr[max_id]
    return oangle, dat_norm


def _compute_mb_metrix_3(sig, mimo, az_rx):
    """
    Вычисляет mb_metrix [3] для одного сэмпла, формат совместим с реальными данными:
      [0] — projection metric (||P·s||/||s||)
      [1] — OMP residual norm после 1-й итерации
      [2] — OMP residual norm после 2-й итерации

    Параметры:
        sig   — комплексный вектор сигнала [N_ant]
        mimo  — позиции виртуального массива
        az_rx — угол приёма (rx) в радианах
    """
    # 1) Projection metric
    A = np.reshape(np.exp(1j * 2 * np.pi * mimo * np.sin(az_rx)), [len(mimo), 1])
    _, proj = multipath_residual_metric(np.reshape(sig, [len(mimo), 1]), A)

    # 2) OMP residual norms (2 итерации)
    _, dat_norm = OMP_FFT(sig, mimo, 1024, 2)

    return np.array([proj, dat_norm[0], dat_norm[1]])



# ──────────────────────────────────────────────────────────────
#  Реальный радар AWR1843 = 8 антенн.
#  Используйте dataset_gen_final_experiment() для 8 антенн.
# ──────────────────────────────────────────────────────────────
# init_flag = 0 — 3 класса (1 цель, 2 цели, мультипас) с углами
# init_flag = 1 — 2 класса (1 цель и мультипас)
# init_flag = 2 — 2 класса (1 цель и 2 цели)
# fft_on = 0 — сигнал без FFT; fft_on = 1 — fft(сигнал)
def dataset_gen_final(N, init_flag, fft_on):
    """
    [LEGACY — 12 антенн, 3Tx × 4Rx]
    В текущей работе всегда 8 антенн. Используйте dataset_gen_final_experiment().
    Оставлена для обратной совместимости.
    """
    rx = np.linspace(0, 1.5, 4)     # 4 Rx на [0, 0.5, 1.0, 1.5] λ/2
    tx = np.linspace(0, 4, 3)       # 3 Tx на [0, 2.0, 4.0] λ/2
    mimo = np.linspace(0, 5.5, 12)  # 12 виртуальных элементов

    sig_mimo = []
    angle_mimo = []
    metrix = []
    mb_metrix = []     # теперь [N, 3]: [projection, omp_norm1, omp_norm2]

    for i in range(N):
        # ── Мультипас: θ_tx ≠ θ_rx, знаки противоположны ──
        az_tx_mb = _clip_angle(0.9 * np.random.randn())
        az_rx_mb = _clip_angle(0.9 * np.random.randn())
        if np.sign(az_tx_mb) == np.sign(az_rx_mb):
            az_rx_mb = -az_rx_mb

        Amb = 1 + 20 * np.random.rand()
        sig_rx = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_mb))
        sig_tx = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_mb))
        sig_mimo_mb = (Amb * np.kron(sig_tx, sig_rx) * np.exp(1j * 2 * np.pi * np.random.rand())
                       + np.random.rand() * 0.2 * (np.random.randn(len(mimo))
                                                    + 1j * np.random.randn(len(mimo))))
        metrix_mb = 1.0

        # Проекционная метрика (для фильтрации качества мультипаса)
        A = np.reshape(np.exp(1j * 2 * np.pi * mimo * np.sin(az_rx_mb)), [len(mimo), 1])
        _, mb_proj_mb = multipath_residual_metric(
            np.reshape(sig_mimo_mb, [len(mimo), 1]), A
        )
        if mb_proj_mb < 0.8:
            # mb_metrix мультипаса: [projection, omp_1, omp_2]
            mb_metrix_mb = _compute_mb_metrix_3(sig_mimo_mb, mimo, az_rx_mb)

            # ── Одна цель: θ_tx == θ_rx ──
            az_tx_1tg = _clip_angle(0.9 * np.random.randn())
            az_rx_1tg = az_tx_1tg

            Atg = 1 + 20 * np.random.rand()
            sig_rx = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_1tg))
            sig_tx = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_1tg))
            sig_mimo_1tg = (Atg * np.kron(sig_tx, sig_rx) * np.exp(1j * 2 * np.pi * np.random.rand())
                            + np.random.rand() * 0.2 * (np.random.randn(len(mimo))
                                                         + 1j * np.random.randn(len(mimo))))
            metrix_1tg = 0.0
            mb_metrix_1tg = _compute_mb_metrix_3(sig_mimo_1tg, mimo, az_rx_1tg)

            # ── Две цели: θ_tx1==θ_rx1, θ_tx2==θ_rx2 ──
            az_tx_2tg1 = _clip_angle(0.9 * np.random.randn())
            az_rx_2tg1 = az_tx_2tg1
            az_tx_2tg2 = _clip_angle(0.9 * np.random.randn())
            az_rx_2tg2 = az_tx_2tg2

            A2tg = 1 + 20 * np.random.rand()
            sig_rx1 = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_2tg1))
            sig_tx1 = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_2tg1))
            sig_rx2 = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_2tg2))
            sig_tx2 = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_2tg2))
            tg2_amp = A2tg * (np.random.rand() + 0.25)
            sig_mimo_2tg = (A2tg * np.kron(sig_tx1, sig_rx1) * np.exp(1j * 2 * np.pi * np.random.rand())
                            + tg2_amp * np.kron(sig_tx2, sig_rx2) * np.exp(1j * 2 * np.pi * np.random.rand())
                            + np.random.rand() * 0.2 * (np.random.randn(len(mimo))
                                                         + 1j * np.random.randn(len(mimo))))
            metrix_2tg = 2.0
            # Для двух целей OMP-метрики тоже вычисляем (не хардкод)
            mb_metrix_2tg = _compute_mb_metrix_3(sig_mimo_2tg, mimo, az_rx_2tg1)

            # FFT если требуется
            if fft_on == 1:
                sig_mimo_1tg = np.fft.fft(sig_mimo_1tg)
                sig_mimo_2tg = np.fft.fft(sig_mimo_2tg)
                sig_mimo_mb = np.fft.fft(sig_mimo_mb)

            # ── Сохраняем сэмплы ──
            sig_mimo.append(sig_mimo_1tg)
            angle_mimo.append(np.array([az_tx_1tg, az_tx_1tg]))
            metrix.append(metrix_1tg)
            mb_metrix.append(mb_metrix_1tg)

            if init_flag == 0:  # 3 класса
                sig_mimo.append(sig_mimo_2tg)
                angle_mimo.append(np.array([az_tx_2tg1, az_tx_2tg2]))
                metrix.append(metrix_2tg)
                mb_metrix.append(mb_metrix_2tg)

                sig_mimo.append(sig_mimo_mb)
                angle_mimo.append(np.array([az_tx_mb, az_tx_mb]))
                metrix.append(metrix_mb)
                mb_metrix.append(mb_metrix_mb)
            elif init_flag == 1:
                sig_mimo.append(sig_mimo_mb)
                angle_mimo.append(np.array([az_tx_mb, az_tx_mb]))
                metrix.append(metrix_mb)
                mb_metrix.append(mb_metrix_mb)
            else:
                sig_mimo.append(sig_mimo_2tg)
                angle_mimo.append(np.array([az_tx_2tg1, az_tx_2tg2]))
                metrix.append(metrix_2tg)
                mb_metrix.append(mb_metrix_2tg)

    sig_mimo = np.array(sig_mimo)
    angle_mimo = np.array(angle_mimo)
    metrix = np.array(metrix)
    mb_metrix = np.array(mb_metrix)  # [N, 3]

    # Max-нормализация (как в оригинале и в реальных данных)
    for i in range(len(mb_metrix)):
        sig_mimo[i, :] = sig_mimo[i, :] / max(abs(sig_mimo[i, :]))

    return (np.reshape(sig_mimo, [len(mb_metrix), len(mimo), 1]),
            np.reshape(angle_mimo, [len(mb_metrix), 2, 1]),
            np.reshape(metrix, [len(mb_metrix), 1]),
            mb_metrix)




# ──────────────────────────────────────────────────────────────
#  Генерация синтетических данных — 8 антенн (2Tx × 4Rx, AWR1843)
# ──────────────────────────────────────────────────────────────

def dataset_gen_final_experiment(N, init_flag):
    """
    Генерация синтетических данных для 8-канальной MIMO-решётки.

    Модель сигнала соответствует TI AWR1843:
      - 2 Tx × 4 Rx = 8 виртуальных антенн
      - Позиции: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] в λ/2
      - Канальные ошибки: 1 + 0.1·(randn + j·randn)
      - Шум: noise_coef · rand · (randn + j·randn)

    mb_metrix: [N, 3] — [projection, omp_norm1, omp_norm2]
    формат совместим с реальными данными (cols 19-21 в .txt файлах).
    """
    rx = np.linspace(0, 1.5, 4)     # 4 Rx
    tx = np.linspace(0, 2, 2)       # 2 Tx
    mimo = np.linspace(0, 3.5, 8)   # 8 виртуальных элементов

    noise_coef = 1

    sig_mimo = []
    angle_mimo = []
    metrix = []
    mb_metrix = []  # [N, 3]: [projection, omp_norm1, omp_norm2]

    for i in range(N):
        # Канальные ошибки (моделируем неидеальность антенн)
        ch_err = np.ones(8) + 0.1 * (np.random.randn(8) + 1j * np.random.randn(8))

        # ── Мультипас: θ_tx ≠ θ_rx, знаки противоположны (по статье) ──
        az_tx_mb = _clip_angle(0.9 * np.random.randn())
        az_rx_mb = _clip_angle(0.9 * np.random.randn())
        if np.sign(az_tx_mb) == np.sign(az_rx_mb):
            az_rx_mb = -az_rx_mb

        Amb = 1 + 20 * np.random.rand()
        sig_rx = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_mb))
        sig_tx = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_mb))
        sig_mimo_mb = (ch_err * Amb * np.kron(sig_tx, sig_rx)
                       * np.exp(1j * 2 * np.pi * np.random.rand())
                       + np.random.rand() * noise_coef
                       * (np.random.randn(len(mimo)) + 1j * np.random.randn(len(mimo))))
        metrix_mb = 1.0

        # Проекционная метрика (фильтр качества мультипаса)
        A = np.reshape(np.exp(1j * 2 * np.pi * mimo * np.sin(az_rx_mb)), [len(mimo), 1])
        _, mb_proj_mb = multipath_residual_metric(
            np.reshape(sig_mimo_mb, [len(mimo), 1]), A
        )

        if mb_proj_mb < 0.8:
            # mb_metrix мультипаса: [projection, omp_1, omp_2]
            mb_metrix_mb = _compute_mb_metrix_3(sig_mimo_mb, mimo, az_rx_mb)

            # ── Одна цель: θ_tx == θ_rx ──
            az_tx_1tg = _clip_angle(0.9 * np.random.randn())
            az_rx_1tg = az_tx_1tg

            Atg = 1 + 20 * np.random.rand()
            sig_rx = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_1tg))
            sig_tx = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_1tg))
            sig_mimo_1tg = (ch_err * Atg * np.kron(sig_tx, sig_rx)
                            * np.exp(1j * 2 * np.pi * np.random.rand())
                            + np.random.rand() * noise_coef
                            * (np.random.randn(len(mimo)) + 1j * np.random.randn(len(mimo))))
            metrix_1tg = 0.0
            mb_metrix_1tg = _compute_mb_metrix_3(sig_mimo_1tg, mimo, az_rx_1tg)

            # ── Две цели: θ_tx1==θ_rx1, θ_tx2==θ_rx2 ──
            az_tx_2tg1 = _clip_angle(0.9 * np.random.randn())
            az_rx_2tg1 = az_tx_2tg1
            az_tx_2tg2 = _clip_angle(0.9 * np.random.randn())
            az_rx_2tg2 = az_tx_2tg2

            A2tg = 1 + 20 * np.random.rand()
            sig_rx1 = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_2tg1))
            sig_tx1 = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_2tg1))
            sig_rx2 = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx_2tg2))
            sig_tx2 = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx_2tg2))
            tg2_amp = A2tg * (np.random.rand() + 0.25)
            sig_mimo_2tg = (ch_err * A2tg * np.kron(sig_tx1, sig_rx1)
                            * np.exp(1j * 2 * np.pi * np.random.rand())
                            + ch_err * tg2_amp * np.kron(sig_tx2, sig_rx2)
                            * np.exp(1j * 2 * np.pi * np.random.rand())
                            + np.random.rand() * noise_coef
                            * (np.random.randn(len(mimo)) + 1j * np.random.randn(len(mimo))))
            metrix_2tg = 2.0
            # Для двух целей: вычисляем OMP-метрики (проекция на 1-ю цель)
            mb_metrix_2tg = _compute_mb_metrix_3(sig_mimo_2tg, mimo, az_rx_2tg1)

            # ── Сохраняем сэмплы ──
            sig_mimo.append(sig_mimo_1tg)
            angle_mimo.append(np.array([az_tx_1tg, az_tx_1tg]))
            metrix.append(metrix_1tg)
            mb_metrix.append(mb_metrix_1tg)

            if init_flag == 0:  # 3 класса
                sig_mimo.append(sig_mimo_2tg)
                angle_mimo.append(np.array([az_tx_2tg1, az_tx_2tg2]))
                metrix.append(metrix_2tg)
                mb_metrix.append(mb_metrix_2tg)

                sig_mimo.append(sig_mimo_mb)
                angle_mimo.append(np.array([az_tx_mb, az_tx_mb]))
                metrix.append(metrix_mb)
                mb_metrix.append(mb_metrix_mb)
            elif init_flag == 1:
                sig_mimo.append(sig_mimo_mb)
                angle_mimo.append(np.array([az_tx_mb, az_tx_mb]))
                metrix.append(metrix_mb)
                mb_metrix.append(mb_metrix_mb)
            else:
                sig_mimo.append(sig_mimo_2tg)
                angle_mimo.append(np.array([az_tx_2tg1, az_tx_2tg2]))
                metrix.append(metrix_2tg)
                mb_metrix.append(mb_metrix_2tg)

    sig_mimo = np.array(sig_mimo)
    angle_mimo = np.array(angle_mimo)
    metrix = np.array(metrix)
    mb_metrix = np.array(mb_metrix)  # [N, 3]

    # Max-нормализация (совпадает с нормализацией реальных данных)
    for i in range(len(mb_metrix)):
        sig_mimo[i, :] = sig_mimo[i, :] / max(abs(sig_mimo[i, :]))

    return (np.reshape(sig_mimo, [len(mb_metrix), len(mimo), 1]),
            np.reshape(angle_mimo, [len(mb_metrix), 2, 1]),
            np.reshape(metrix, [len(mb_metrix), 1]),
            mb_metrix)


# ──────────────────────────────────────────────────────────────
#  Загрузка реальных данных из .txt файлов
# ──────────────────────────────────────────────────────────────

def dataset_from_real_data(path, normalize='max'):
    """
    Загрузка реальных данных из папки с .txt файлами.

    Формат каждой строки (25 колонок):
      [0]     — doppler bin
      [1]     — range bin
      [2]     — range (м)
      [3:11]  — Re(ch_0..ch_7)
      [11:19] — Im(ch_0..ch_7)
      [19:22] — mb_metrix [3]: [proj_metric, omp_norm1, omp_norm2]
      [22:24] — azimuth [2] (градусы)
      [24]    — math class (0/1/2)

    Параметры:
        path      — путь к папке с .txt файлами
        normalize — 'max' | 'rms' | 'none'
    """
    files = sorted(os.listdir(path))

    exp_dataset = {
        'doppler bin': [], 'range bin': [], 'range': [],
        'channels': [], 'mb_metrix': [], 'azimuth': [], 'math class': [],
    }
    for num_file in range(len(files)):
        filepath = os.path.join(path, files[num_file])
        if not filepath.endswith('.txt'):
            continue
        raw_data = np.loadtxt(filepath)
        for i in range(np.shape(raw_data)[0]):
            exp_dataset['doppler bin'].append(raw_data[i, 0])
            exp_dataset['range bin'].append(raw_data[i, 1])
            exp_dataset['range'].append(raw_data[i, 2])
            ch = raw_data[i, 3:11] + 1j * raw_data[i, 11:19]
            if normalize == 'max':
                ch = ch / max(abs(ch))
            elif normalize == 'rms':
                ch = ch / np.sqrt(np.mean(np.abs(ch) ** 2) + 1e-12)
            # 'none' — без нормализации
            exp_dataset['channels'].append(ch)
            exp_dataset['mb_metrix'].append(raw_data[i, 19:22])  # [3]
            exp_dataset['azimuth'].append(raw_data[i, 22:24])    # [2] deg
            exp_dataset['math class'].append(raw_data[i, 24])

    for key in exp_dataset:
        exp_dataset[key] = np.array(exp_dataset[key])
    return exp_dataset


# ──────────────────────────────────────────────────────────────
#  Вспомогательные обёртки
# ──────────────────────────────────────────────────────────────

def dataset_gen_2D_clean(N, init_flag=1):
    """Бинарная классификация: 1 цель vs мультипас (8 антенн)."""
    sig_mimo, _, metrix, mb_metrix = dataset_gen_final_experiment(N, init_flag=1)
    y = (metrix == 1.0).astype(int)
    return sig_mimo, y, mb_metrix


def dataset_gen_2D(N, init_flag=0):
    """Генерация с возвратом metrix (class labels) и mb_metrix (8 антенн)."""
    sig_mimo, _, metrix, mb_metrix = dataset_gen_final_experiment(N, init_flag=init_flag)
    return sig_mimo, metrix, mb_metrix


# ──────────────────────────────────────────────────────────────
#  Wiener-Hopf отбеливание (whitening)
# ──────────────────────────────────────────────────────────────

def wiener_hopf_whiten(channels, K=64, eps=1e-6):
    """
    Пространственное отбеливание по Винеру-Хопфу.

    R̂ = (1/K) Σ x_k · x_k^H
    x_whitened = R̂^{-1} · x

    Параметры:
        channels — [N, M] complex (N сэмплов, M антенн)
        K        — число сэмплов для оценки ковариации (default=64)
        eps      — регуляризация для обращения матрицы

    Примечание: сэмплы для оценки ковариации выбираются СЛУЧАЙНО
    (а не первые K), чтобы избежать bias при последовательной генерации
    данных по классам.
    """
    if channels.ndim != 2:
        raise ValueError("channels must be 2D [N, M]")
    N, M = channels.shape
    K = min(K, N)

    # Случайная подвыборка для оценки ковариации (избегаем bias)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, size=K, replace=False)

    cov = np.zeros((M, M), dtype=np.complex128)
    for i in idx:
        x = channels[i]
        cov += np.outer(x, np.conj(x))
    cov /= max(K, 1)

    # Устойчивое обращение: регуляризация пропорционально следу
    reg = max(eps, eps * np.abs(np.trace(cov)) / M)
    R_inv = np.linalg.inv(cov + reg * np.eye(M))

    result = (R_inv @ channels.T).T

    # Защита от NaN/Inf
    if np.any(~np.isfinite(result)):
        nan_count = np.sum(~np.isfinite(result))
        print(f"  [!] wiener_hopf_whiten: {nan_count} NaN/Inf значений → заменяем на 0")
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


# ──────────────────────────────────────────────────────────────
#  NPZ save / load (новый подход)
# ──────────────────────────────────────────────────────────────

def save_synth_to_npz(output_path, N=131072, init_flag=0):
    """
    Генерирует синтетику (8 антенн, новый подход) и сохраняет в .npz.

    Формат:
        channels   — [N, 8, 1] complex
        azimuth    — [N, 2, 1] float (радианы)
        math_class — [N, 1] float (0/1/2)
        mb_metrix  — [N, 3] float ([proj, omp_1, omp_2])
    """
    X, angle, y, mb_metrix = dataset_gen_final_experiment(N, init_flag)
    np.savez(
        output_path,
        channels=X,           # [N, 8, 1] complex
        azimuth=angle,        # [N, 2, 1] float
        math_class=y,         # [N, 1] float (0/1/2)
        mb_metrix=mb_metrix,  # [N, 3] float
        source='synth_new',
    )
    print(f"Saved {len(X)} synthetic samples to {output_path}")
    return output_path


def save_real_to_npz(input_folder, output_path, normalize='none'):
    """
    Конвертирует реальные .txt данные в .npz.

    normalize='none' — сырые данные (для дальнейшего whitening + RMS)
    normalize='max'  — max-нормализация (как в оригинале)
    """
    data = dataset_from_real_data(input_folder, normalize=normalize)
    np.savez(
        output_path,
        channels=data['channels'],       # [N, 8] complex
        azimuth=data['azimuth'],          # [N, 2] float (deg)
        math_class=data['math class'],    # [N,] float
        mb_metrix=data['mb_metrix'],      # [N, 3] float
        doppler_bin=data['doppler bin'],
        range_bin=data['range bin'],
        range_val=data['range'],
        source='real_new',
    )
    print(f"Saved {len(data['channels'])} real samples to {output_path}")
    return output_path


def load_from_npz(npz_path):
    """
    Загрузка датасета из .npz файла.

    Returns: (channels, azimuth, math_class, mb_metrix)
        channels   — complex array
        azimuth    — float array
        math_class — float array
        mb_metrix  — float array [N, 3]
    """
    npz = np.load(npz_path, allow_pickle=True)
    channels = npz['channels']
    azimuth = npz['azimuth']
    math_class = npz['math_class']
    mb_metrix = npz['mb_metrix']
    source = str(npz.get('source', 'unknown'))
    print(f"Loaded {len(channels)} samples from {npz_path} (source: {source})")
    return channels, azimuth, math_class, mb_metrix
