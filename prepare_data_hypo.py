"""
prepare_data_hypo.py
====================
Prepare synthetic data with hypothesis-based fixes:
  1) Match angle distribution to real data
  2) Add realistic per-antenna phase/amplitude distortions

Outputs:
  - output/synth_new.npz (by default, overwrites)

Run:
  python prepare_data_hypo.py
  python prepare_data_hypo.py --n-synth 131072 --min-sep 5.0
"""

import argparse
import os
import time
import numpy as np

from config import (
    OUTPUT_DIR, N_SYNTH, SYNTH_NPZ, REAL_NPZ,
    REAL_DATA_DIR, WHITEN_K,
)
from dataset import dataset_from_real_data, _clip_angle, _compute_mb_metrix_3
from preprocessing import preprocess_channels


def _load_real_npz(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Real NPZ not found: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _angles_real_deg(azimuth):
    if np.max(np.abs(azimuth)) <= 2 * np.pi:
        return np.rad2deg(azimuth)
    return azimuth


def build_angle_sampler(angles_deg, bins=181, seed=42):
    rng = np.random.RandomState(seed)
    angles = angles_deg[(angles_deg >= -90) & (angles_deg <= 90)]
    hist, bin_edges = np.histogram(angles, bins=bins, density=False)
    hist = hist.astype(float) + 1e-9
    probs = hist / hist.sum()

    def sample(n):
        idx = rng.choice(len(probs), size=n, replace=True, p=probs)
        low = bin_edges[idx]
        high = bin_edges[idx + 1]
        return rng.uniform(low, high)

    return sample


def estimate_channel_stats(channels_raw):
    # channels_raw: [N, 8] complex
    ref = channels_raw[:, [0]]
    rel = channels_raw / (ref + 1e-9)
    amp = np.abs(rel)
    phase = np.angle(rel)

    amp_mean = np.mean(amp, axis=0)
    amp_std = np.std(amp, axis=0)
    phase_mean = np.mean(phase, axis=0)
    phase_std = np.std(phase, axis=0)
    return amp_mean, amp_std, phase_mean, phase_std


def sample_channel_gains(amp_mean, amp_std, phase_mean, phase_std, rng):
    amp = rng.normal(amp_mean, amp_std)
    amp = np.clip(amp, 0.05, None)
    phase = rng.normal(phase_mean, phase_std)
    g = amp * np.exp(1j * phase)
    g[0] = 1.0 + 0j
    return g


def generate_synth_hypo(
    n_samples,
    angle_sampler,
    amp_mean, amp_std, phase_mean, phase_std,
    noise_sigma,
    class_priors,
    min_sep_deg=3.0,
    seed=42,
):
    rng = np.random.RandomState(seed)
    rx = np.linspace(0, 1.5, 4)
    tx = np.linspace(0, 2, 2)
    mimo = np.linspace(0, 3.5, 8)

    counts = (class_priors * n_samples).astype(int)
    # fix rounding
    while counts.sum() < n_samples:
        counts[np.argmax(class_priors)] += 1
    while counts.sum() > n_samples:
        counts[np.argmax(counts)] -= 1

    sig_mimo = []
    angle_mimo = []
    metrix = []
    mb_metrix = []

    def sample_angle_pair():
        a1 = angle_sampler(1)[0]
        a2 = angle_sampler(1)[0]
        tries = 0
        while abs(a1 - a2) < min_sep_deg and tries < 20:
            a2 = angle_sampler(1)[0]
            tries += 1
        return a1, a2

    # class 0: direct
    for _ in range(counts[0]):
        az_tx = _clip_angle(np.deg2rad(angle_sampler(1)[0]))
        az_rx = az_tx
        g = sample_channel_gains(amp_mean, amp_std, phase_mean, phase_std, rng)

        A = 1.0 + 20.0 * rng.rand()
        sig_rx = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx))
        sig_tx = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx))
        sig = (g * A * np.kron(sig_tx, sig_rx) * np.exp(1j * 2 * np.pi * rng.rand()))
        noise = noise_sigma * (rng.randn(len(mimo)) + 1j * rng.randn(len(mimo)))
        sig = sig + noise

        mb = _compute_mb_metrix_3(sig, mimo, az_rx)
        sig_mimo.append(sig)
        angle_mimo.append(np.array([az_tx, az_tx]))
        metrix.append(0.0)
        mb_metrix.append(mb)

    # class 2: two targets
    for _ in range(counts[2]):
        a1, a2 = sample_angle_pair()
        az1 = _clip_angle(np.deg2rad(a1))
        az2 = _clip_angle(np.deg2rad(a2))
        g = sample_channel_gains(amp_mean, amp_std, phase_mean, phase_std, rng)

        A1 = 1.0 + 20.0 * rng.rand()
        A2 = 1.0 + 20.0 * rng.rand()
        sig_rx1 = np.exp(1j * 2 * np.pi * rx * np.sin(az1))
        sig_tx1 = np.exp(1j * 2 * np.pi * tx * np.sin(az1))
        sig_rx2 = np.exp(1j * 2 * np.pi * rx * np.sin(az2))
        sig_tx2 = np.exp(1j * 2 * np.pi * tx * np.sin(az2))
        sig = (g * A1 * np.kron(sig_tx1, sig_rx1) * np.exp(1j * 2 * np.pi * rng.rand())
               + g * A2 * np.kron(sig_tx2, sig_rx2) * np.exp(1j * 2 * np.pi * rng.rand()))
        noise = noise_sigma * (rng.randn(len(mimo)) + 1j * rng.randn(len(mimo)))
        sig = sig + noise

        mb = _compute_mb_metrix_3(sig, mimo, az1)
        sig_mimo.append(sig)
        angle_mimo.append(np.array([az1, az2]))
        metrix.append(2.0)
        mb_metrix.append(mb)

    # class 1: multipath
    for _ in range(counts[1]):
        az_tx = _clip_angle(np.deg2rad(angle_sampler(1)[0]))
        az_rx = _clip_angle(np.deg2rad(angle_sampler(1)[0]))
        if np.sign(az_tx) == np.sign(az_rx):
            az_rx = -az_rx
        g = sample_channel_gains(amp_mean, amp_std, phase_mean, phase_std, rng)

        A = 1.0 + 20.0 * rng.rand()
        sig_rx = np.exp(1j * 2 * np.pi * rx * np.sin(az_rx))
        sig_tx = np.exp(1j * 2 * np.pi * tx * np.sin(az_tx))
        sig = (g * A * np.kron(sig_tx, sig_rx) * np.exp(1j * 2 * np.pi * rng.rand()))
        noise = noise_sigma * (rng.randn(len(mimo)) + 1j * rng.randn(len(mimo)))
        sig = sig + noise

        mb = _compute_mb_metrix_3(sig, mimo, az_rx)
        sig_mimo.append(sig)
        angle_mimo.append(np.array([az_tx, az_tx]))
        metrix.append(1.0)
        mb_metrix.append(mb)

    sig_mimo = np.array(sig_mimo)
    angle_mimo = np.array(angle_mimo)
    metrix = np.array(metrix).reshape(-1, 1)
    mb_metrix = np.array(mb_metrix)

    # max-normalize
    for i in range(len(sig_mimo)):
        sig_mimo[i, :] = sig_mimo[i, :] / max(abs(sig_mimo[i, :]))

    # shuffle
    perm = rng.permutation(len(sig_mimo))
    sig_mimo = sig_mimo[perm]
    angle_mimo = angle_mimo[perm]
    metrix = metrix[perm]
    mb_metrix = mb_metrix[perm]

    return (
        sig_mimo.reshape(len(sig_mimo), len(mimo), 1),
        angle_mimo.reshape(len(sig_mimo), 2, 1),
        metrix,
        mb_metrix,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic data with hypothesis fixes")
    parser.add_argument("--n-synth", type=int, default=N_SYNTH,
                        help=f"Number of synthetic samples (default {N_SYNTH})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--real-npz", type=str, default=REAL_NPZ,
                        help="Path to real_preprocessed.npz")
    parser.add_argument("--min-sep", type=float, default=3.0,
                        help="Min separation between two targets (deg)")
    parser.add_argument("--use-real-class-priors", action="store_true",
                        help="Use real class distribution for synth generation")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="Noise scale multiplier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "synth_new.npz")

    # Load real data from NPZ to build angle distribution and channel stats
    real = _load_real_npz(args.real_npz)
    ang_r = _angles_real_deg(real["azimuth"])
    # use both angles for richer histogram
    ang_all = np.concatenate([ang_r[:, 0], ang_r[:, 1]])

    # Channel stats (prefer raw if present)
    ch_raw = real.get("channels_raw", real["channels"])
    ch_raw = ch_raw.reshape(len(ch_raw), -1)
    amp_mean, amp_std, phase_mean, phase_std = estimate_channel_stats(ch_raw)

    # Noise sigma from real amplitude std
    noise_sigma = np.std(np.abs(ch_raw)) * args.noise_scale

    # Class priors
    if args.use_real_class_priors:
        cls = real["math_class"].astype(int).ravel()
        counts = np.bincount(cls, minlength=3).astype(float)
        class_priors = counts / counts.sum()
    else:
        class_priors = np.array([1/3, 1/3, 1/3])

    angle_sampler = build_angle_sampler(ang_all, bins=181, seed=args.seed)

    print("=" * 60)
    print("Prepare synthetic data (hypothesis fixes)")
    print("=" * 60)
    print(f"  output: {out_path}")
    print(f"  n_synth: {args.n_synth}")
    print(f"  class_priors: {class_priors}")
    print(f"  noise_sigma: {noise_sigma:.4f} (scale={args.noise_scale})")

    t0 = time.time()
    X, angle, y, mb = generate_synth_hypo(
        args.n_synth, angle_sampler,
        amp_mean, amp_std, phase_mean, phase_std,
        noise_sigma, class_priors,
        min_sep_deg=args.min_sep,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    np.savez(
        out_path,
        channels=X,
        azimuth=angle,
        math_class=y,
        mb_metrix=mb,
        source="synth_hypo",
    )

    print(f"  saved: {out_path}")
    print(f"  channels: {X.shape}, angles: {angle.shape}, classes: {np.unique(y, return_counts=True)}")
    print(f"  mb_metrix: {mb.shape}")
    print(f"  elapsed: {elapsed:.1f} sec")


if __name__ == "__main__":
    main()
