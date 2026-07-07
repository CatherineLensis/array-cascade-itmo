# Radar DOA Pipeline

Классификация радарных сигналов (3 класса) + определение углов прихода (DOA) для FMCW MIMO-радара TI AWR1843.

## Структура проекта

```
Shishanov_paper/
├── config.py                 # Все пути и гиперпараметры
├── dataset.py                # Генерация синтетики + загрузка реальных данных
├── models.py                 # Все нейросетевые архитектуры
├── preprocessing.py          # Wiener-Hopf whitening + RMS нормализация
├── metrics.py                # Метрики и визуализация (классификатор + DOA + EDA)
│
├── prepare_data.py           # Скрипт: подготовка данных → NPZ
├── run_full_pipeline.py      # Скрипт: полный пайплайн (обучение + оценка)
├── evaluate.py               # Скрипт: оценка на реальных данных
│
├── requirements.txt          # Зависимости
├── README.md                 # Этот файл
│
├── train_validate/           # Реальные данные (.txt)
│   ├── 2019_04_30_pbms002.txt
│   ├── 2019_05_09_pbms004.txt
│   └── 2019_05_29_pbms007.txt
│
├── docs/                     # Статьи и документация
│   ├── jamming_paper_v2.docx
│   └── Multipath_Transaction_...docx
│
├── analysis/                 # Научные исследования качества пайплайна
│   ├── cascade_error.py      # Каскадная ошибка (oracle vs pipeline)
│   ├── angular_resolution.py # Разрешающая способность (Δθ_min)
│   ├── snr_analysis.py       # SNR-зависимость DOA и классификатора
│   └── inference_benchmark.py# Бенчмарк скорости инференса
│
├── output/                   # Результаты (модели, графики, NPZ)
│   ├── evaluation/           # Отчёты evaluate.py
│   └── analysis/             # Графики analysis/
│
├── old_approach/             # Старая методология (Шишанов)
│   ├── dataset_sh.py         # Генерация данных (старый подход, complexPyTorch)
│   ├── NN_multipath_sh.py    # Классификатор (старый подход)
│   ├── prepare_data_old.py   # Подготовка данных (старый метод)
│   └── ...
│
└── study/                    # Исследовательские файлы (4 метода DOA)
    ├── study_siamese_1.py
    ├── study_lista_2.py
    ├── study_covrecon_3.py
    └── study_efficientvit_gru_4.py
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

Генерация синтетики и предобработка реальных данных:

```bash
python prepare_data.py
```

Результат: `output/synth_new.npz` и `output/real_preprocessed.npz`.

Параметры:

```bash
python prepare_data.py --n-synth 131072    # больше синтетики
python prepare_data.py --skip-real         # только синтетика
python prepare_data.py --normalize max     # max вместо RMS
```

### 3. Обучение (полный пайплайн)

Запуск всего пайплайна: данные → классификатор → 4 метода DOA → сравнение:

```bash
python run_full_pipeline.py
```

Пайплайн выполняет 5 шагов:

| Шаг | Описание | Результат |
|-----|----------|-----------|
| 1 | Подготовка данных | Синтетика + реальные |
| 2 | Обучение классификатора (train/val 80/20) | `output/classifier.pth` |
| 3 | Обучение 4 DOA-методов (train/val 85/15) | `output/doa_*.pth` |
| 4 | Полный инференс: classify → route → DOA | RMSE по методам |
| 5 | Визуализация и сравнение | Графики + JSON в `output/` |

### 4. Оценка на реальных данных

После обучения — оценка качества на реальных данных:

```bash
python evaluate.py
python evaluate.py --no-doa          # только классификатор
```

Результат в `output/evaluation/`:
- Confusion matrix (абсолютная + нормализованная)
- Per-class Precision / Recall / F1
- ROC-AUC per class
- Сводная таблица DOA: RMSE, MAE, Success Rate
- CDF ошибок, scatter, гистограммы
- JSON-отчёты для автоматизации

---

## Пайплайн обработки данных — детальное описание

### Общая схема

```
Сырой сигнал (.txt / синтетика)
         │
         ▼
  ┌──────────────────────────────────┐
  │  ШАГ 1: Предобработка           │
  │  Wiener-Hopf whitening + RMS    │
  └──────────────┬───────────────────┘
                 │
                 ▼   [N, 8] complex
  ┌──────────────────────────────────┐
  │  ШАГ 2: Классификация            │
  │  ComplexNet → class ∈ {0, 1, 2} │
  └──────────────┬───────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
  class=0     class=1     class=2
  (direct)  (multipath) (two targets)
     │           │           │
     ▼           ▼           ▼
  DOA-модель  ПРОПУСК    DOA-модель
  → 1 угол   (помечаем   → 2 угла
              "multipath")
```

### ШАГ 1. Предобработка сигналов

Каждый сэмпл — вектор из 8 комплексных отсчётов (8 виртуальных антенн MIMO-радара AWR1843). Предобработка выполняется в два этапа:

```
Сырой сигнал [N, 8] complex
         │
         ▼
┌────────────────────────────────────────────┐
│ 1. Wiener-Hopf Whitening (K=64)            │
│                                            │
│    Оценка ковариационной матрицы шума:     │
│    R̂ = (1/K) · Σ x·x^H                    │
│                                            │
│    Отбеливание:                            │
│    x_white = R̂^{-1/2} · x                 │
│                                            │
│    Цель: подавить пространственно-         │
│    коррелированный шум и помехи            │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│ 2. RMS-нормализация                        │
│                                            │
│    x_norm = x / √(mean|x|²)               │
│                                            │
│    Цель: привести все сэмплы к единой      │
│    масштабу (среднеквадратичная амп. = 1)  │
└────────────────────┬───────────────────────┘
                     │
                     ▼
         Чистый сигнал [N, 8] complex
```

Отличие от старого подхода (Шишанов): старый использовал только max-нормализацию `x / max(|x|)` без whitening.

### ШАГ 2. Классификация (ComplexNet)

Комплексная свёрточная сеть принимает предобработанный 8-канальный сигнал и определяет тип измерения:

```
Вход: [B, 1, 8] complex64
  │
  ├── Conv1d(1→16,  k=3, pad=1) → complex_tanh → BN_re + BN_im
  ├── Conv1d(16→32, k=3, pad=1) → complex_tanh → BN_re + BN_im
  ├── Conv1d(32→64, k=3, pad=1) → complex_tanh → BN_re + BN_im
  │
  ├── abs()                ← переход: комплексный → вещественный
  ├── Flatten              → [B, 512]  (64 каналов × 8 отсчётов)
  ├── FC(512→128) + ReLU + Dropout(0.5)
  └── FC(128→3)            → logits → argmax → класс
```

Обучение в два этапа:
1. **Pretrain** на синтетике (train/val = 80/20, CrossEntropyLoss)
2. **Finetune** на реальных данных с пониженным learning rate

| Класс | Описание | Что означает | DOA |
|-------|----------|-------------|-----|
| 0 — direct | Один прямой сигнал | θ_tx == θ_rx, одна цель | Определяется 1 угол |
| 1 — multipath | Переотражение | θ_tx ≠ θ_rx, сигнал невалидный | Игнорируется |
| 2 — two targets | Два сигнала | Две цели в одном range-bin | Определяются 2 угла |

### ШАГ 3. Маршрутизация после классификации

По результату классификации сигнал направляется в один из трёх путей:

- **class = 1 (multipath)** → сигнал помечается как невалидный, DOA не вычисляется. В реальной системе такое измерение отбрасывается или помечается флагом.

- **class = 0 (direct)** → сигнал передаётся в DOA-модель, которая определяет **один угол** прихода.

- **class = 2 (two targets)** → сигнал передаётся в DOA-модель, которая определяет **два угла** прихода (для CovRecon — через Root-MUSIC с `N_sources=2`, для спектральных методов — через поиск двух пиков).

### ШАГ 4. Определение углов (DOA) — 4 метода

Все 4 метода получают **одинаковый** предобработанный сигнал, но используют разные подходы. Это позволяет объективно сравнить их на одних данных.

Для DOA обучение идёт **только на class 0 и class 2** (multipath исключается).

#### Метод 1: Siamese / Contrastive

```
signal [1, 1, 8] complex
    │
    ▼
ComplexFeatureExtractor:
  Conv1d(1→32, k=3, pad=1)        → complex_tanh → BN    [B, 32, 8]
  Conv1d(32→64, k=3, pad=1, s=2)  → complex_tanh → BN    [B, 64, 4]
  Conv1d(64→128, k=3, pad=1, s=2) → complex_tanh → BN    [B, 128, 2]
  → abs() → AdaptiveAvgPool(2) → Flatten → FC(256) → L2-norm
    │
    ▼  256-мерный embedding
    │
    ├── DOA head: FC(256→512) → ReLU → Dropout → FC(512→361) → sigmoid
    │   → [361] вероятности на сетке -90°..+90° → argmax → угол
    │
    └── Class head: FC(256→64) → ReLU → FC(64→3)
```

Обучение: NT-Xent (InfoNCE) контрастивная потеря + BCE для DOA + CE для классов. Аугментация: прореживание антенн + комплексный шум (имитация неисправностей).

#### Метод 2: Deep LISTA (physics-informed)

```
signal [1, 8] complex
    │
    ▼
10 слоёв LISTA (unfolded ISTA с обучаемыми параметрами):
    x_{k+1} = SoftThreshold( x_k + θ · W · (y - A · x_k),  λ )
    │
    │   A — steering-матрица [8, 361] (фиксирована, не обучается)
    │   W, θ, λ — обучаемые параметры каждого слоя
    │
    ▼
|x|² → power spectrum [361] → sigmoid → argmax → угол
```

Ключевая идея: разворачивает итерации алгоритма ISTA в обучаемую нейросеть. Steering-матрица задаёт физическую модель антенной решётки.

#### Метод 3: CovRecon + Root-MUSIC (гибридный)

```
signal [8] complex
    │
    ▼
Sample covariance: R = x · x^H + εI → [8, 8] complex
    │
    ▼
Flatten: [128] float (64 real + 64 imag)
    │
    ▼
MLP: FC(128→128→256→128→128) → R_clean
    │
    ▼
Эрмитовость: R_out = (R_clean + R_clean^H) / 2
    │
    ▼
Root-MUSIC (классический алгоритм):
  1. Собственное разложение R_out → (eigenvalues, eigenvectors)
  2. Шумовое подпространство: Un = eigvecs[:, :N-N_sources]
  3. Полиномиальные корни: roots(Q_polynomial)
  4. Ближайшие к единичной окружности → углы через arcsin
    │
    ▼
N_sources углов (class 0 → 1 угол, class 2 → 2 угла)
```

Нейросеть учится «очищать» ковариационную матрицу от шума, а Root-MUSIC — классический алгоритм высокого разрешения — извлекает углы.

#### Метод 4: Hierarchical GRU (двухстадийный)

```
signal [1, 1, 8] complex → [Re, Im] → [8, 2] последовательность
    │
    ▼
Грубая стадия (CoarseGRU):
  BiGRU(2→128, 2 слоя) обрабатывает антенны как seq
  → softmax → [181] вероятности на грубой сетке
  + hidden state h_coarse
    │
    ▼
Тонкая стадия (3 × FineGRU параллельно):
  3 BiGRU → sigmoid → [361] каждый
  → усреднение → [361] fine-вероятности → argmax → угол
    │
    ▼
Source count head: FC(h_coarse) → [3] logits (0, 1 или 2 источника)
```

Двухстадийный подход: грубая стадия быстро локализует область, тонкая — уточняет с высоким разрешением.

### ШАГ 5. Подсчёт ошибки

Для каждого предсказанного угла ищется **ближайший** истинный:

```
error = min( |true_angle_j - pred_angle| )  для всех j
```

Если предсказано несколько углов — для каждого из них вычисляется ближайшая ошибка.

---

## Метрики и визуализация

Модуль `metrics.py` предоставляет полный набор метрик для radar DOA ML-проектов.

### Метрики классификатора

| Метрика | Описание |
|---------|----------|
| Accuracy | Доля правильных предсказаний |
| F1 per class | F1-score для каждого из 3 классов |
| F1 macro / weighted | Агрегированный F1 |
| Precision / Recall | Per-class precision и recall |
| ROC-AUC per class | Площадь под ROC-кривой |
| Confusion Matrix | Абсолютная + нормализованная |

### Метрики DOA

| Метрика | Описание | Зачем |
|---------|----------|-------|
| RMSE | √(mean(error²)) | Основная метрика (чувствительна к outliers) |
| MAE | mean(\|error\|) | Робастнее к выбросам |
| Median AE | median(\|error\|) | Ещё робастнее |
| Success Rate @1°, 3°, 5° | Доля ошибок ≤ порога | Главная практическая метрика |
| CDF ошибок | Кумулятивное распределение | Полная картина распределения ошибок |
| RMSE vs angle | RMSE в бинах по углу | Обнаружение edge effects у ±90° |
| Per-class RMSE | Отдельно для direct и two targets | Two targets — сложнее |

### Визуализация (генерируемые графики)

```
output/
  curves_classifier.png           — train/val loss + accuracy
  clf_confusion_matrix.png        — confusion matrix (2 вида)
  clf_per_class_metrics.png       — F1 / Precision / Recall bar chart
  clf_roc_auc.png                 — ROC-кривые per class
  training_curves_doa.png         — loss curves 4 DOA-методов
  doa_error_comparison.png        — RMSE / MAE / MedAE bar chart
  doa_success_rate.png            — Success Rate vs threshold
  doa_cdf.png                     — CDF ошибок (все методы)
  doa_error_histogram.png         — гистограммы ошибок
  doa_scatter.png                 — true vs predicted angles
  doa_rmse_vs_angle.png           — RMSE vs true angle
  doa_per_class_rmse.png          — direct vs two targets
  doa_examples.png                — примеры DOA-спектров
  eda_angle_distribution.png      — распределение углов
  eda_class_distribution.png      — баланс классов
  eda_amplitude_distribution.png  — амплитуды сигналов
  eda_mb_metrix.png               — распределение mb_metrix
  metrics_classifier.json         — JSON история обучения
  doa_metrics.json                — JSON все DOA-метрики
```

---

## Конфигурация

Все параметры в `config.py`:

```python
# Радар
N_ANT = 8                              # виртуальных антенн
MIMO_POS = np.linspace(0, 3.5, 8)     # позиции в λ/2
N_GRID = 361                           # угловая сетка DOA

# Обучение
N_SYNTH = 65536                        # синтетических сэмплов
CLASSIFIER_EPOCHS = 60                 # эпохи pretrain
FINETUNE_EPOCHS = 40                   # эпохи finetune
DOA_EPOCHS = 60                        # эпохи DOA-моделей
```

## Формат данных

### Реальные данные (.txt)

Каждая строка — 25 колонок:

```
[0]     doppler_bin
[1]     range_bin
[2]     range (м)
[3:11]  Re(ch_0..ch_7)    — 8 каналов
[11:19] Im(ch_0..ch_7)    — 8 каналов
[19:22] mb_metrix [3]     — [projection, omp_norm1, omp_norm2]
[22:24] azimuth [2]       — углы (градусы)
[24]    math_class        — метка класса (0/1/2)
```

### NPZ-файлы

```python
npz = np.load("output/synth_new.npz")
channels   = npz["channels"]    # [N, 8, 1] complex
azimuth    = npz["azimuth"]     # [N, 2, 1] float (рад)
math_class = npz["math_class"]  # [N, 1] float
mb_metrix  = npz["mb_metrix"]   # [N, 3] float
```

## Научные исследования (analysis/)

Папка `analysis/` содержит скрипты для углублённого анализа качества пайплайна — вопросы, которые задал бы исследователь.

### 1. Каскадная ошибка (cascade_error.py)

Двухстадийный пайплайн (классификатор → DOA) означает, что ошибка классификации каскадно влияет на DOA. Скрипт сравнивает:
- **Oracle RMSE** — RMSE при идеальной классификации (ground truth)
- **Pipeline RMSE** — RMSE при реальной классификации
- Breakdown: RMSE на сэмплах с правильной vs ошибочной классификацией
- Определяет bottleneck: классификатор или DOA?

```bash
python analysis/cascade_error.py
python analysis/cascade_error.py --n-eval 5000
```

### 2. Разрешающая способность (angular_resolution.py)

При каком минимальном угловом разделении (Δθ) между двумя целями метод ещё различает оба угла? Монте-Карло симуляция:

```bash
python analysis/angular_resolution.py
python analysis/angular_resolution.py --n-trials 200 --snr 15
```

Выход: кривая Success Rate vs Δθ, Δθ_min для 50% и 90%, сравнение с Rayleigh limit.

### 3. SNR-зависимость (snr_analysis.py)

Как деградирует качество системы при уменьшении SNR? Генерирует данные при SNR от -5 до 30 дБ:

```bash
python analysis/snr_analysis.py
python analysis/snr_analysis.py --n-per-snr 500 --snr-min -10 --snr-max 35
```

Выход: Accuracy vs SNR, DOA RMSE vs SNR, Success Rate @3° vs SNR, минимальный рабочий SNR.

### 4. Бенчмарк инференса (inference_benchmark.py)

Может ли система работать в реальном времени? Измеряет латентность и пропускную способность:

```bash
python analysis/inference_benchmark.py
python analysis/inference_benchmark.py --n-iter 500
```

Выход: латентность (mean, p50, p95, p99) каждого компонента, batch throughput, размер моделей.

---

## Сравнение старого и нового подходов

Старый подход (`old_approach/`) можно запустить отдельно:

```bash
cd old_approach
python prepare_data_old.py
```

Ключевые отличия:

| Аспект | Старый | Новый |
|--------|--------|-------|
| Библиотека | complexPyTorch | Нативный PyTorch complex64 |
| Нормализация | max(abs) | Wiener-Hopf + RMS |
| BN | Общий для Re/Im | Раздельный для Re/Im |
| Классификатор | ComplexNet (16→32→16, no padding) | ComplexNet (16→32→64, padding=1) |
| DOA | Нет | 4 метода |
| Метрики | Accuracy + RMSE | Полный набор (metrics.py) |
| Валидация | Нет train/val split | 80/20 классификатор, 85/15 DOA |

## Радар

- **Модель**: Texas Instruments AWR1843
- **Конфигурация**: 2 Tx × 4 Rx = 8 виртуальных антенн
- **Виртуальный массив**: `[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]` в долях λ/2
