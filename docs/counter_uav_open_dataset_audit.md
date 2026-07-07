# Counter-UAV Open Dataset Audit

## Purpose

This note records open-source radar / micro-Doppler references that can be used as a reality-check for the `counter_uav` synthetic generator.

Important: these datasets were **not** previously downloaded or mixed into the local training pipeline. They are listed here as external references for synthetic-data design and validation.

## Public References

### 1. DIAT-µSAT

Source:
- <https://ieee-dataport.org/documents/diat-msat-micro-doppler-signature-dataset-small-unmanned-aerial-vehicle-suav>

Key facts:
- X-band radar, about `10 GHz`
- CW radar
- `4849` micro-Doppler signature images
- target types include:
  - RC plane
  - several rotorcraft variants
  - quadcopter
  - bionic bird
  - mini-helicopter
- positioned as an academic dataset for UAV detection / classification

Why it matters:
- close in spirit to this repository's `counter_uav` domain
- directly confirms that public radar micro-Doppler data for UAV-like targets exists
- useful as a reference for class variety and for expected micro-Doppler diversity

### 2. 77 GHz FMCW birds / drones dataset family

Evidence:
- search result pointed to Zenodo record: `Radar measurements on drones, birds and humans with a 77 GHz FMCW sensor`
- the 2026 robustness paper below explicitly states it uses a **publicly available 77 GHz FMCW radar dataset**

Reference paper:
- `arXiv:2604.12567` "Feature-Level Robustness of Physics-Guided Micro-Doppler Descriptors for classification of Drones and Birds"

Key facts from the paper:
- sensor: `77 GHz FMCW`
- classes grouped into `drones`, `birds`, `reflectors`
- source data contains `six different types of drones`, `six bird species`, and a reflector class
- the paper does **not** classify raw single snapshots; it builds **measurement-level micro-Doppler spectrograms**
- it standardizes slow-time windows before feature extraction

Why it matters:
- this is a strong external signal that the relevant discriminative information lives in the **time-frequency micro-Doppler structure**, not only in one spatial snapshot

### 3. Nature / Scientific Reports 2018

Source:
- <https://www.nature.com/articles/s41598-018-35880-9>

Paper:
- "Radar micro-Doppler signatures of drones and birds at K-band and W-band"

Key facts extracted from the paper:
- two coherent radar bands used: `24 GHz` and `94 GHz`
- drones:
  - characteristic signatures are a **wide spread across Doppler sidebands**
  - rotor-blade flash and **HERM lines** are highlighted as key features
  - propeller returns are often about `20–40 dB` below the main body return
- birds:
  - signatures are **periodic flashes** at wingbeat frequency
  - observed wingbeat frequencies are around `4–6 Hz`
  - wing contributions are roughly comparable to the body signal, or at most about `10 dB` lower
  - birds often flap at takeoff / landing and may glide in the middle, so temporal structure is non-stationary

Why it matters:
- this paper gives concrete physical differences that the synthetic generator should preserve
- it strongly suggests the bird/drone split is primarily about **temporal micro-Doppler morphology**, not just static phase patterns

## Comparison With Current Local Generator

Relevant local file:
- `domains/counter_uav/micro_doppler.py`

What already aligns:
- bird wingbeat frequency in code is sampled from `3–8 Hz`, which is broadly consistent with the `4–6 Hz` range reported in the Nature paper
- the code does attempt to model rotor and wing structure separately

What does **not** align well enough:
- the classifier path in this repository currently consumes an `8`-antenna **single snapshot**, not a spectrogram or slow-time sequence
- `synth_drone_signal()` and `synth_bird_signal()` do generate `micro_series`, but the returned classifier snapshot is mostly built from handcrafted proxies:
  - `drone_proxy`
  - `bird_proxy`
- this means the most realistic discriminative cue from open datasets, namely **time-varying micro-Doppler morphology**, is largely discarded before classification

Concrete code-level concern:
- in `synth_drone_signal()`, the final `snapshot` is built from `drone_proxy` plus AWGN
- in `synth_bird_signal()`, the final `snapshot` is built from `bird_proxy` plus AWGN
- the richer `micro_series` exists, but the classifier path does not use it

## Working Hypothesis

The weak `counter_uav` accuracy is not just a model-capacity issue.

A likely root cause is that the current synthetic classifier contract compresses bird/drone differences into a static spatial proxy, while public radar references show that bird/drone discrimination is carried strongly by **slow-time / spectrogram structure**:
- wingbeat bursts / periodic flashes for birds
- rotor-blade spread and HERM lines for drones

As a result:
- the synthetic generator can still produce visually plausible classes
- but the classifier learns a weaker and less reality-aligned boundary than what open datasets suggest

## Recommended Next Actions

1. Add a `counter_uav` data-quality gate that measures bird/drone separability under:
   - no multipath
   - moderate multipath
   - strong multipath

2. Add an external-reference checklist to the generator:
   - bird wingbeat band near `4–6 Hz`
   - drone rotor-induced broader Doppler spread
   - bird signatures more burst-like / intermittent than drone signatures

3. Decide whether to:
   - keep snapshot-only classification and improve the proxy model, or
   - introduce a micro-Doppler spectrogram / slow-time branch for classifier input

4. Use open-source datasets first as:
   - feature-distribution reference
   - qualitative visual comparison target
   - sanity benchmark for synthetic-data realism

Only after that should the project run another serious retrain cycle and re-evaluate jury-facing metrics.
