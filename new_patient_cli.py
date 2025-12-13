# fnn_train.py
# --------------------------------------------
# Command line interface for new patients to get SHAP and LLM analysis
# Author: Ben Scoppa, Virginia Tech ECE Dept
# --------------------------------------------
# Run a new patient through:
#   - FNN prediction
#   - SHAP analysis
#   - JSON construction
#   - Ollama explanation
#   - Waterfall plot
#
# Usage:
#   python new_patient_cli.py --patient-file new_patient.csv
#
# The 'new_patient.csv' must contain a row with the same feature
# columns in order as 'parkinsons.csv' except 'name' and 'status'.
# -------------------------------------------------

import argparse
import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import StandardScaler

from agent_ollama import generate_explanation
from fnn_train import ParkinsonFNN
from shap_utils import create_patient_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a new patient through FNN, SHAP and Ollama pipeline."
    )
    parser.add_argument(
        "--patient-file",
        type=str,
        required=True,
        help="Path to CSV containing one row with the patient's feature values.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="new_patient_output",
        help="Directory to save plot and JSONs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("NEW-PATIENT PIPELINE: NEW PATIENT → FNN → SHAP → OLLAMA")
    print("=" * 80)

    # ------------------ STEP 1: Load reference data for scaling + SHAP background ------------------
    print("\n[STEP 1/4] Loading reference dataset for scaling and SHAP background...")

    df_ref = pd.read_csv("parkinsons.csv")
    feature_names = [c for c in df_ref.columns if c not in ["name", "status"]]

    X_ref = df_ref[feature_names].values

    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)

    # Background for SHAP (up to 100 samples)
    background_size = min(100, X_ref_scaled.shape[0])
    background = torch.tensor(X_ref_scaled[:background_size], dtype=torch.float32)

    print(f"✓ Loaded reference dataset: {df_ref.shape[0]} rows")
    print(f"✓ Using {background_size} samples as SHAP background")

    # ------------------ STEP 2: Load NEW patient file ------------------
    print("\n[STEP 2/4] Loading new patient data...")

    df_patient = pd.read_csv(args.patient_file)

    if len(df_patient) != 1:
        raise ValueError(
            f"Expected exactly one row in {args.patient_file}, got {len(df_patient)}."
        )

    # Only take the feature columns in the correct order
    missing_cols = [c for c in feature_names if c not in df_patient.columns]
    if missing_cols:
        raise ValueError(
            f"The patient file is missing required columns: {missing_cols}\n"
            f"It must include at least: {feature_names}"
        )

    df_patient = df_patient[feature_names]
    X_patient_raw = df_patient.values  # shape (1, n_features)
    X_patient_scaled = scaler.transform(X_patient_raw)  # preprocess like training data

    X_patient_tensor = torch.tensor(X_patient_scaled, dtype=torch.float32)

    print("✓ New patient features loaded and scaled.")
    print("  Features (scaled):")
    print(X_patient_scaled[0])

    # ------------------ STEP 3: Load model & compute SHAP for this patient ------------------
    print("\n[STEP 3/4] Loading trained FNN and computing SHAP for this patient...")

    input_dim = len(feature_names)
    model = ParkinsonFNN(input_dim)
    model.load_state_dict(torch.load("fnn_parkinson_model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_prob_tensor = model(X_patient_tensor)  # shape (1, 1)
        patient_prob = float(y_prob_tensor.item())

    patient_pred_class = "Parkinson's Disease" if patient_prob > 0.5 else "Healthy"

    print(f"✓ FNN prediction:")
    print(f"  Predicted class: {patient_pred_class}")
    print(f"  PD probability:  {patient_prob*100:.2f}%")

    # SHAP: DeepExplainer with background from reference data
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_patient_tensor)

    # Handle return types from SHAP
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.squeeze(shap_values)  # shape (n_features,)

    # Build JSON-like structure
    patient_json = create_patient_json(
        patient_idx="new_patient",
        feature_values=X_patient_scaled[0],
        shap_vals=shap_values,
        pred_proba=patient_prob,
        feature_names=feature_names,
    )

    json_path = out_dir / "new_patient_analysis.json"
    with open(json_path, "w") as f:
        json.dump(patient_json, f, indent=2)

    print(f"✓ Saved analysis JSON to: {json_path}")

    # SHAP waterfall plot
    print("✓ Creating SHAP waterfall plot for this patient...")

    explanation = shap.Explanation(
        values=shap_values,
        base_values=(
            explainer.expected_value if hasattr(explainer, "expected_value") else 0
        ),
        data=X_patient_scaled[0],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    plt.title(
        f"New Patient: SHAP Waterfall\n"
        f"Pred: {patient_pred_class} (PD prob: {patient_prob*100:.1f}%)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()

    waterfall_path = out_dir / "new_patient_waterfall.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✓ Saved SHAP waterfall plot to: {waterfall_path}")

    # ------------------ STEP 4: Ollama explanation ------------------
    print("\n[STEP 4/4] Calling Ollama agent for natural-language explanation...")

    explanation_text, duration = generate_explanation(patient_json)

    print("\n=== OLLAMA EXPLANATION ===")
    print(explanation_text)
    print("==========================")
    print(f"(Generated in {duration:.2f} seconds)")

    # Save the explanation alongside the JSON
    explanation_out = {
        "patient_id": patient_json["metadata"]["patient_id"],
        "prediction": patient_json["prediction"],
        "explanation": explanation_text,
        "generation_time_seconds": duration,
    }
    expl_path = out_dir / "new_patient_explained.json"
    with open(expl_path, "w") as f:
        json.dump(explanation_out, f, indent=2)

    print(f"\n✓ Saved Ollama explanation JSON to: {expl_path}")

    print("\n" + "=" * 80)
    print("SINGLE-PATIENT PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
