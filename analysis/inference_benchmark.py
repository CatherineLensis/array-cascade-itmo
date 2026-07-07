"""
analysis/inference_benchmark.py
===============================
Бенчмарк скорости инференса всех компонентов пайплайна.

Вопрос: может ли система работать в реальном времени?

Что измеряем:
  1. Предобработка (whitening + RMS): ms/сэмпл
  2. Классификатор: ms/сэмпл и ms/батч
  3. Каждый DOA-метод: ms/сэмпл
  4. Полный пайплайн: preprocess → classify → route → DOA
  5. Пропускная способность: сэмплов/сек

Дополнительно:
  - Сравнение CPU vs GPU (если доступна)
  - Размер моделей (параметры, MB)
  - Латентность: p50, p95, p99

Запуск:
    python analysis/inference_benchmark.py
    python analysis/inference_benchmark.py --n-warmup 50 --n-iter 500
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    OUTPUT_DIR, N_ANT, N_GRID, MIMO_POS, DEVICE,
    CLASSIFIER_PATH, DOA_SIAMESE_PATH, DOA_LISTA_PATH,
    DOA_COVRECON_PATH, DOA_HIERGRU_PATH,
    LISTA_LAYERS, COVRECON_HIDDEN,
)
from models import (
    ComplexNet, to_complex_tensor, build_steering_matrix,
    SiameseDOAModel, DeepLISTA_DOA,
    CovReconstructor, cov_to_flat, flat_to_hermitian, root_music,
    HierarchicalGRU_DOA,
)
from preprocessing import preprocess_channels


def count_parameters(model):
    """Подсчёт параметров модели."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    """Размер модели в МБ."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


def benchmark_function(fn, n_warmup, n_iter, label=""):
    """Бенчмарк функции → латентности."""
    # Warmup
    for _ in range(n_warmup):
        fn()

    # Sync GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_per_sec": float(1000 / np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--n-warmup", type=int, default=30)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256])
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("BENCHMARK ИНФЕРЕНСА")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"  Warmup: {args.n_warmup}, Iterations: {args.n_iter}")
    print("=" * 70)

    # ── Загрузка моделей ──
    A_np, theta_grid = build_steering_matrix(MIMO_POS, N_GRID)
    grid_deg = np.rad2deg(theta_grid)

    all_models = {}

    if os.path.exists(CLASSIFIER_PATH):
        clf = ComplexNet(input_length=N_ANT).to(DEVICE)
        clf.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        clf.eval()
        all_models["classifier"] = clf

    doa_configs = {
        "siamese": (DOA_SIAMESE_PATH, lambda: SiameseDOAModel(N_GRID, N_ANT)),
        "lista": (DOA_LISTA_PATH, lambda: DeepLISTA_DOA(A_np, n_layers=LISTA_LAYERS)),
        "covrecon": (DOA_COVRECON_PATH, lambda: CovReconstructor(N_ANT, hidden=COVRECON_HIDDEN)),
        "hiergru": (DOA_HIERGRU_PATH, lambda: HierarchicalGRU_DOA()),
    }
    for name, (path, factory) in doa_configs.items():
        if os.path.exists(path):
            m = factory().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            all_models[name] = m

    if not all_models:
        print("[!] Нет обученных моделей. Запустите run_full_pipeline.py")
        return

    # ── 1. Информация о моделях ──
    print("\n── Размер моделей ──")
    print(f"{'Model':14s} | {'Params':>10s} | {'Trainable':>10s} | {'Size MB':>8s}")
    print("-" * 50)
    model_info = {}
    for name, model in all_models.items():
        total, trainable = count_parameters(model)
        size = model_size_mb(model)
        model_info[name] = {"params": total, "trainable": trainable, "size_mb": size}
        print(f"{name:14s} | {total:10,d} | {trainable:10,d} | {size:7.2f}")

    # ── 2. Single-sample inference ──
    print("\n── Latency: single sample ──")

    # Генерируем тестовый сигнал
    test_signal = np.random.randn(N_ANT) + 1j * np.random.randn(N_ANT)
    test_signal = test_signal / (np.sqrt(np.mean(np.abs(test_signal) ** 2)) + 1e-12)

    results = {}

    # Предобработка
    test_batch = np.random.randn(100, N_ANT) + 1j * np.random.randn(100, N_ANT)
    r = benchmark_function(
        lambda: preprocess_channels(test_batch),
        n_warmup=5, n_iter=50, label="preprocess"
    )
    r["throughput_per_sec"] = 100 * r["throughput_per_sec"]  # per-sample
    r["mean_ms"] = r["mean_ms"] / 100  # per-sample
    results["preprocess"] = r
    print(f"  {'preprocess':14s} | mean={r['mean_ms']:.3f}ms | "
          f"throughput={r['throughput_per_sec']:.0f}/s")

    # Классификатор
    if "classifier" in all_models:
        x_cls = to_complex_tensor(test_signal[None, :, None]).to(DEVICE)
        r = benchmark_function(
            lambda: all_models["classifier"](x_cls),
            n_warmup=args.n_warmup, n_iter=args.n_iter,
        )
        results["classifier"] = r
        print(f"  {'classifier':14s} | mean={r['mean_ms']:.3f}ms | p95={r['p95_ms']:.3f}ms | "
              f"throughput={r['throughput_per_sec']:.0f}/s")

    # DOA-модели
    for name in ["siamese", "lista", "covrecon", "hiergru"]:
        if name not in all_models:
            continue
        model = all_models[name]

        if name in ("siamese", "hiergru"):
            x = torch.complex(
                torch.tensor(test_signal.real, dtype=torch.float32),
                torch.tensor(test_signal.imag, dtype=torch.float32),
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)
            if name == "siamese":
                fn = lambda: model(x)
            else:
                fn = lambda: model(x)
        elif name == "lista":
            y = torch.complex(
                torch.tensor(test_signal.real, dtype=torch.float32),
                torch.tensor(test_signal.imag, dtype=torch.float32),
            ).unsqueeze(0).to(DEVICE)
            fn = lambda: model(y)
        elif name == "covrecon":
            R_flat = cov_to_flat(test_signal.reshape(1, -1), N_ANT)
            R_t = torch.tensor(R_flat, dtype=torch.float32).to(DEVICE)
            # CovRecon + Root-MUSIC
            def covrecon_fn():
                out = model(R_t)
                R_rec = flat_to_hermitian(out, N_ANT)
                R_np = R_rec[0].cpu().numpy()
                root_music(R_np, N_sources=1)
            fn = covrecon_fn

        r = benchmark_function(fn, n_warmup=args.n_warmup, n_iter=args.n_iter)
        results[name] = r
        print(f"  {name:14s} | mean={r['mean_ms']:.3f}ms | p95={r['p95_ms']:.3f}ms | "
              f"throughput={r['throughput_per_sec']:.0f}/s")

    # ── 3. Полный пайплайн (single sample) ──
    if "classifier" in all_models and len(all_models) > 1:
        first_doa_name = [n for n in ["siamese", "lista", "covrecon", "hiergru"]
                          if n in all_models][0]

        def full_pipeline():
            # Preprocess (уже нормированный)
            x_cls = to_complex_tensor(test_signal[None, :, None]).to(DEVICE)
            with torch.no_grad():
                cls = all_models["classifier"](x_cls).argmax(dim=1).item()
            if cls != 1:
                # DOA (используем первый доступный)
                if first_doa_name in ("siamese", "hiergru"):
                    x = torch.complex(
                        torch.tensor(test_signal.real, dtype=torch.float32),
                        torch.tensor(test_signal.imag, dtype=torch.float32),
                    ).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        if first_doa_name == "siamese":
                            all_models[first_doa_name](x)
                        else:
                            all_models[first_doa_name](x)

        r = benchmark_function(full_pipeline, n_warmup=args.n_warmup, n_iter=args.n_iter)
        results["full_pipeline"] = r
        print(f"\n  {'FULL PIPELINE':14s} | mean={r['mean_ms']:.3f}ms | p95={r['p95_ms']:.3f}ms | "
              f"throughput={r['throughput_per_sec']:.0f}/s")

    # ── 4. Batch throughput ──
    print("\n── Batch throughput (classifier) ──")
    batch_results = {}
    if "classifier" in all_models:
        for bs in args.batch_sizes:
            x_batch = to_complex_tensor(
                (np.random.randn(bs, N_ANT) + 1j * np.random.randn(bs, N_ANT))[:, :, None]
            ).to(DEVICE)
            r = benchmark_function(
                lambda: all_models["classifier"](x_batch),
                n_warmup=10, n_iter=100,
            )
            r["per_sample_ms"] = r["mean_ms"] / bs
            batch_results[bs] = r
            print(f"  batch={bs:4d} | total={r['mean_ms']:.2f}ms | "
                  f"per_sample={r['per_sample_ms']:.4f}ms | "
                  f"throughput={bs * r['throughput_per_sec']:.0f}/s")

    # ── Визуализация ──

    # 1. Латентность по компонентам (bar chart)
    components = [k for k in ["preprocess", "classifier", "siamese", "lista",
                               "covrecon", "hiergru", "full_pipeline"] if k in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(components))
    means = [results[c]["mean_ms"] for c in components]
    p95s = [results[c]["p95_ms"] for c in components]
    colors_map = {
        "preprocess": "#607D8B", "classifier": "#F44336",
        "siamese": "#2196F3", "lista": "#4CAF50",
        "covrecon": "#FF9800", "hiergru": "#9C27B0",
        "full_pipeline": "#000000",
    }
    bar_colors = [colors_map.get(c, "#999") for c in components]

    bars = ax.bar(x_pos, means, color=bar_colors, alpha=0.85)
    ax.scatter(x_pos, p95s, color="red", marker="v", zorder=5, s=50, label="p95")
    for i, (m, p) in enumerate(zip(means, p95s)):
        ax.text(i, m + max(means) * 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(components, rotation=25, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency per Component (single sample)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_latency.png"), dpi=150)
    plt.close()

    # 2. Throughput по компонентам
    fig, ax = plt.subplots(figsize=(10, 5))
    throughputs = [results[c]["throughput_per_sec"] for c in components]
    ax.barh(components, throughputs, color=bar_colors, alpha=0.85)
    for i, v in enumerate(throughputs):
        ax.text(v + max(throughputs) * 0.01, i, f"{v:.0f}/s", va="center", fontsize=9)
    ax.set_xlabel("Throughput (samples/sec)")
    ax.set_title("Inference Throughput per Component")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_throughput.png"), dpi=150)
    plt.close()

    # 3. Batch throughput (если есть)
    if batch_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        bs_list = sorted(batch_results.keys())
        total_ms = [batch_results[bs]["mean_ms"] for bs in bs_list]
        per_sample = [batch_results[bs]["per_sample_ms"] for bs in bs_list]

        ax1.plot(bs_list, total_ms, "o-", color="#2196F3", linewidth=2)
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Total Latency (ms)")
        ax1.set_title("Classifier: Batch Latency")
        ax1.grid(True, alpha=0.3)

        ax2.plot(bs_list, per_sample, "o-", color="#4CAF50", linewidth=2)
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Per-sample Latency (ms)")
        ax2.set_title("Classifier: Per-sample Latency vs Batch")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "benchmark_batch.png"), dpi=150)
        plt.close()

    # 4. Model size comparison
    if model_info:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        names = list(model_info.keys())
        params = [model_info[n]["params"] for n in names]
        sizes = [model_info[n]["size_mb"] for n in names]
        mc = [colors_map.get(n, "#999") for n in names]

        ax1.barh(names, [p / 1000 for p in params], color=mc, alpha=0.85)
        ax1.set_xlabel("Parameters (K)")
        ax1.set_title("Model Parameters")
        for i, v in enumerate(params):
            ax1.text(v / 1000 + max(params) / 1000 * 0.02, i, f"{v:,}", va="center", fontsize=8)

        ax2.barh(names, sizes, color=mc, alpha=0.85)
        ax2.set_xlabel("Size (MB)")
        ax2.set_title("Model Size")
        for i, v in enumerate(sizes):
            ax2.text(v + max(sizes) * 0.02, i, f"{v:.2f}", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "benchmark_model_sizes.png"), dpi=150)
        plt.close()

    # Сводка
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    if "full_pipeline" in results:
        fps = results["full_pipeline"]["throughput_per_sec"]
        lat = results["full_pipeline"]["mean_ms"]
        print(f"  Полный пайплайн: {lat:.2f} ms/sample → {fps:.0f} samples/sec")
        if fps > 1000:
            print("  → Реальное время: ДА (>1000 Hz)")
        elif fps > 100:
            print("  → Реальное время: ДА для большинства радарных задач (>100 Hz)")
        else:
            print("  → Реальное время: ОГРАНИЧЕННО (< 100 Hz)")

    total_params = sum(model_info[n]["params"] for n in model_info)
    total_size = sum(model_info[n]["size_mb"] for n in model_info)
    print(f"  Всего параметров: {total_params:,}")
    print(f"  Общий размер моделей: {total_size:.2f} MB")
    print(f"\n  Графики сохранены в {out_dir}")


if __name__ == "__main__":
    main()
