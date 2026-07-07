"""
Resume baseline pipeline from Step 3 (DOA training).
Skips classifier if classifier.pth already exists.
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CLASSIFIER_PATH, OUTPUT_DIR, DEVICE, N_ANT
from run_full_pipeline import (
    step1_prepare_data,
    step2_train_classifier,
    step3_train_doa,
    step4_full_inference,
    step5_compare,
)
from models import ComplexNet
import torch


def load_classifier():
    model = ComplexNet(input_length=N_ANT).to(DEVICE)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def main():
    print("=" * 60)
    print("RESUME BASELINE — from Step 3 if classifier exists")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = step1_prepare_data()

    if os.path.exists(CLASSIFIER_PATH):
        print(f"\n  Classifier found: {CLASSIFIER_PATH} — skipping Step 2")
        classifier = load_classifier()
    else:
        classifier = step2_train_classifier(data)

    doa_models, loss_histories = step3_train_doa(data)
    rmse_results = step4_full_inference(data, classifier, doa_models)
    step5_compare(loss_histories, rmse_results, data, doa_models)

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE. Artifacts in", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
