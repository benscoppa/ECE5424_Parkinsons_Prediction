# main_pipeline.py
# -------------------------------------------------
# Authors: Dominic Jibin James, Ben Scoppa Virginia Tech ECE Dept
# -------------------------------------------------
# Full pipeline:
#   1) Load data and preprocess
#   2) Load trained FNN
#   3) SHAP analysis and JSON
#   4) Call Ollama agent to generate explanations
#   5) Run benchmark over Ollama outputs
# -------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from agent_ollama import process_patient_files
from benchmark_agent import run_benchmark
from fnn_train import ParkinsonFNN
from shap_utils import compute_shap_values, export_llm_json, make_shap_visualizations


def main():
    print("=" * 80)
    print("PARKINSON'S DISEASE DETECTION: MAIN PIPELINE")
    print("=" * 80)

    # ------------------ DATA LOADING & PREPROCESSING ------------------
    print("\n[STEP 1/5] Loading and preprocessing data...")

    df = pd.read_csv("parkinsons.csv")
    feature_names = [c for c in df.columns if c not in ["name", "status"]]

    X = df[feature_names].values
    y = df["status"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set:  {X_test.shape[0]} samples")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # ------------------ LOAD TRAINED FNN ------------------
    print("\n[STEP 2/5] Loading trained FNN model...")

    input_dim = X_train.shape[1]
    model = ParkinsonFNN(input_dim)
    model.load_state_dict(torch.load("fnn_parkinson_model.pth", map_location="cpu"))
    model.eval()

    print("✓ Loaded weights from 'fnn_parkinson_model.pth'")

    with torch.no_grad():
        y_pred = model(X_test_tensor)
        predictions = y_pred.numpy().flatten()

    # display metrics
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    y_pred_class = (y_pred > 0.5).float()
    acc = accuracy_score(y_test_tensor, y_pred_class)
    f1 = f1_score(y_test_tensor, y_pred_class)
    cm = confusion_matrix(y_test_tensor, y_pred_class)

    print("\nMODEL PERFORMANCE (from main):")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # ------------------ SHAP + VISUALIZATIONS + JSON ------------------
    print("\n[STEP 3/5] SHAP analysis, visualizations, and JSON export...")

    shap_values, explainer = compute_shap_values(
        model, X_train_tensor, X_test_tensor, background_size=100
    )

    make_shap_visualizations(
        shap_values=shap_values,
        explainer=explainer,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        predictions=predictions,
        out_dir="shap_visualizations",
    )

    export_llm_json(
        shap_values=shap_values,
        X_test=X_test,
        predictions=predictions,
        feature_names=feature_names,
        output_dir="llm_input_json",
    )

    # ------------------ Ollama agent ------------------
    print("\n[STEP 4/5] Running Ollama agent to generate explanations...")
    process_patient_files()  # reads from llm_input_json/, writes to llm_outputs/

    # ------------------ Benchmark ------------------
    print("\n[STEP 5/5] Running benchmark on Ollama outputs...")
    run_benchmark()  # reads llm_outputs/ + llm_input_json/

    print("\n" + "=" * 80)
    print("MAIN PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
