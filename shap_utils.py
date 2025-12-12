# shap_utils.py
# --------------------------------------------
# SHAP utilities + JSON export for Parkinson FNN
# Authors: Dominic Jibin James, Ben Scoppa Virginia Tech ECE Dept
# --------------------------------------------

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

# ---------- SHAP COMPUTATION ----------


def compute_shap_values(model, X_train_tensor, X_test_tensor, background_size=100):
    """
    Compute SHAP values for a trained PyTorch model.

    Args:
        model: trained FNN model on Parkinsons Data
        X_train_tensor: torch.FloatTensor of training data (n_train, n_features)
        X_test_tensor: torch.FloatTensor of test data (n_test, n_features)
        background_size: number of training samples to use as background

    Returns:
        shap_values: np.ndarray (n_test, n_features)
        explainer: shap.DeepExplainer instance
    """
    model.eval()
    background = X_train_tensor[:background_size]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test_tensor)

    # Convert to numpy & normalize shape
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) == 3:
        shap_values = shap_values.squeeze()

    return shap_values, explainer


# ---------- VISUALIZATIONS ----------


def make_shap_visualizations(
    shap_values,
    explainer,
    X_test,
    y_test,
    feature_names,
    predictions,
    out_dir="shap_visualizations",
):
    """
    Create SHAP visualizations:
      1. Global bar plot (mean |SHAP|)
      2. Beeswarm summary plot
      3. Waterfall plot for patient 0
      4. Individual waterfall plots for all test patients
      5. Summary grid of subset of patients
    """
    Path(out_dir).mkdir(exist_ok=True)

    # 1) Global feature importance
    print("✓ Creating visualization 1/5: Global Feature Importance...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_10_idx = np.argsort(mean_abs_shap)[-10:][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(10), mean_abs_shap[top_10_idx], color="steelblue", alpha=0.7)
    plt.yticks(range(10), [feature_names[i] for i in top_10_idx])
    plt.xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    plt.title("Top 10 Most Important Features (Global)", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_global_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"   → Saved: {out_dir}/1_global_importance.png")

    # 2) Beeswarm
    print("✓ Creating visualization 2/5: SHAP Beeswarm Plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(
        "SHAP Summary: Feature Impact Distribution",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_beeswarm_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"   → Saved: {out_dir}/2_beeswarm_summary.png")

    # 3) Waterfall for patient 0
    print("✓ Creating visualization 3/5: SHAP Waterfall Plot (Patient 0)...")

    patient_idx = 0

    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=(
            explainer.expected_value if hasattr(explainer, "expected_value") else 0
        ),
        data=X_test[patient_idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    plt.title(
        f"Patient {patient_idx}: SHAP Waterfall Analysis\n"
        f'Predicted: {"PD" if predictions[patient_idx] > 0.5 else "Healthy"} '
        f"(PD probability: {predictions[patient_idx]*100:.1f}%)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_patient_0_waterfall.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"   → Saved: {out_dir}/3_patient_0_waterfall.png")

    # 4) Individual waterfalls
    print(
        f"✓ Creating visualization 4/5: Individual Waterfall Plots for ALL {len(X_test)} test patients..."
    )

    indiv_dir = Path(out_dir) / "individual_patients"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    for patient_idx in range(len(X_test)):
        explanation = shap.Explanation(
            values=shap_values[patient_idx],
            base_values=(
                explainer.expected_value if hasattr(explainer, "expected_value") else 0
            ),
            data=X_test[patient_idx],
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, max_display=10, show=False)

        actual_label = "PD" if y_test[patient_idx] == 1 else "Healthy"
        predicted_label = "PD" if predictions[patient_idx] > 0.5 else "Healthy"
        correct = "✓" if actual_label == predicted_label else "✗"

        plt.title(
            f"Patient {patient_idx}: SHAP Waterfall Analysis {correct}\n"
            f"Actual: {actual_label} | Predicted: {predicted_label} "
            f"(PD probability: {predictions[patient_idx]*100:.1f}%)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()

        out_path = indiv_dir / f"patient_{patient_idx:03d}_waterfall.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        if (patient_idx + 1) % 10 == 0 or patient_idx == len(X_test) - 1:
            print(f"   → Processed {patient_idx + 1}/{len(X_test)} patients")

    print(
        f"   → Saved: {indiv_dir}/patient_000_waterfall.png ... "
        f"patient_{len(X_test)-1:03d}_waterfall.png"
    )

    # 5) Summary grid
    print("✓ Creating visualization 5/5: Summary Grid...")

    n_display = min(12, len(X_test))
    display_indices = np.linspace(0, len(X_test) - 1, n_display, dtype=int)

    n_cols = 4
    n_rows = (n_display + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_display > 1 else [axes]

    for idx, patient_idx in enumerate(display_indices):
        ax = axes[idx]

        sorted_idx = np.argsort(np.abs(shap_values[patient_idx]))[-8:]
        colors = [
            "#d62728" if val > 0 else "#1f77b4"
            for val in shap_values[patient_idx][sorted_idx]
        ]

        ax.barh(
            range(8),
            shap_values[patient_idx][sorted_idx],
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_yticks(range(8))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        ax.set_xlabel("SHAP Value", fontsize=9, fontweight="bold")

        actual_label = "PD" if y_test[patient_idx] == 1 else "Healthy"
        predicted_label = "PD" if predictions[patient_idx] > 0.5 else "Healthy"
        correct = "✓" if actual_label == predicted_label else "✗"
        pd_prob = predictions[patient_idx] * 100

        ax.set_title(
            f"Patient {patient_idx}: {predicted_label} {correct}\n"
            f"(Actual: {actual_label}, Prob: {pd_prob:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        ax.grid(axis="x", alpha=0.3)

    for idx in range(n_display, len(axes)):
        axes[idx].axis("off")

    legend_elements = [
        plt.matplotlib.patches.Patch(
            facecolor="#d62728", alpha=0.7, label="Increases PD risk"
        ),
        plt.matplotlib.patches.Patch(
            facecolor="#1f77b4", alpha=0.7, label="Decreases PD risk"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        fontsize=11,
        frameon=True,
    )

    plt.suptitle(
        f"SHAP Analysis: Overview of {n_display} Sample Patients (from {len(X_test)} total)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{out_dir}/5_patient_summary_grid.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"   → Saved: {out_dir}/5_patient_summary_grid.png")


# ---------- JSON EXPORT ----------

clinical_mapping = {
    "MDVP:Fo(Hz)": {
        "clinical_name": "Fundamental Frequency",
        "normal_range": "85-180 Hz (male), 165-255 Hz (female)",
        "pd_indication": "Reduced variability indicates vocal rigidity",
    },
    "MDVP:Fhi(Hz)": {
        "clinical_name": "Maximum Vocal Frequency",
        "normal_range": "102-592 Hz",
        "pd_indication": "Reduced range suggests limited vocal control",
    },
    "MDVP:Flo(Hz)": {
        "clinical_name": "Minimum Vocal Frequency",
        "normal_range": "65-239 Hz",
        "pd_indication": "Narrowed frequency range indicates motor impairment",
    },
    "MDVP:Jitter(%)": {
        "clinical_name": "Jitter Percentage",
        "normal_range": "<1.04%",
        "pd_indication": "Elevated jitter indicates irregular vocal fold vibrations",
    },
    "MDVP:Jitter(Abs)": {
        "clinical_name": "Absolute Jitter",
        "normal_range": "<83 microseconds",
        "pd_indication": "Increased jitter reflects neuromuscular instability",
    },
    "MDVP:Shimmer": {
        "clinical_name": "Shimmer",
        "normal_range": "<3.81%",
        "pd_indication": "Elevated shimmer indicates amplitude perturbation and weak voice",
    },
    "MDVP:Shimmer(dB)": {
        "clinical_name": "Shimmer in Decibels",
        "normal_range": "<0.35 dB",
        "pd_indication": "Increased amplitude variation suggests vocal weakness",
    },
    "HNR": {
        "clinical_name": "Harmonics-to-Noise Ratio",
        "normal_range": ">13 dB",
        "pd_indication": "Reduced HNR indicates increased vocal noise and hoarseness",
    },
    "RPDE": {
        "clinical_name": "Recurrence Period Density Entropy",
        "normal_range": "0.4-0.7",
        "pd_indication": "Altered RPDE reflects changes in vocal signal complexity",
    },
    "DFA": {
        "clinical_name": "Detrended Fluctuation Analysis",
        "normal_range": "0.6-0.7",
        "pd_indication": "Abnormal DFA indicates disrupted vocal control dynamics",
    },
    "PPE": {
        "clinical_name": "Pitch Period Entropy",
        "normal_range": "0.1-0.3",
        "pd_indication": "Increased entropy indicates irregular vocal fold vibration",
    },
    "NHR": {
        "clinical_name": "Noise-to-Harmonics Ratio",
        "normal_range": "<0.19",
        "pd_indication": "Elevated NHR suggests breathiness and vocal instability",
    },
}


def create_patient_json(
    patient_idx,
    feature_values,
    shap_vals,
    pred_proba,
    feature_names,
):
    """Create structured JSON for one patient for LLM interpretation."""

    pred_class = "Parkinson's Disease" if pred_proba > 0.5 else "Healthy"
    confidence = pred_proba if pred_proba > 0.5 else (1 - pred_proba)

    top_5_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

    top_features = []
    for rank, idx in enumerate(top_5_idx, 1):
        feat_name = feature_names[idx]
        clinical_info = clinical_mapping.get(
            feat_name,
            {
                "clinical_name": feat_name,
                "normal_range": "Not specified",
                "pd_indication": "Clinical interpretation not available",
            },
        )

        top_features.append(
            {
                "rank": rank,
                "feature_name": feat_name,
                "feature_value": float(feature_values[idx]),
                "shap_value": float(shap_vals[idx]),
                "contribution": "increases" if shap_vals[idx] > 0 else "decreases",
                "clinical_name": clinical_info["clinical_name"],
                "normal_range": clinical_info["normal_range"],
                "pd_indication": clinical_info["pd_indication"],
            }
        )

    clinical_prompt = f"""Analyze the following Parkinson's Disease voice biomarker assessment:

PATIENT ID: patient_{patient_idx}
PREDICTION: {pred_class}
CONFIDENCE: {confidence*100:.1f}%

TOP CONTRIBUTING FEATURES:
"""

    for feat in top_features:
        direction = (
            "↑ increases" if feat["contribution"] == "increases" else "↓ decreases"
        )

        clinical_prompt += (
            f"\n{feat['rank']}. {feat['clinical_name']} ({feat['feature_name']})\n"
            f"   Patient Value: {feat['feature_value']:.4f}\n"
            f"   Normal Range: {feat['normal_range']}\n"
            f"   SHAP Impact: {feat['shap_value']:.4f} ({direction} PD probability)\n"
            f"   Clinical Significance: {feat['pd_indication']}\n"
        )

    clinical_prompt += """
TASK: Based on the SHAP analysis above, provide a clinical interpretation that:
1. Summarizes the key acoustic findings in 2-3 sentences
2. Explains what the top features indicate about vocal function
3. Relates findings to PD motor symptoms (rigidity, bradykinesia, tremor)
4. Assesses the reliability of this prediction based on feature patterns

Keep the explanation concise and accessible to healthcare providers.
"""

    json_data = {
        "metadata": {
            "patient_id": f"patient_{patient_idx}",
            "model_version": "FNN_v1.0",
            "analysis_date": pd.Timestamp.now().isoformat(),
        },
        "prediction": {
            "class": pred_class,
            "probability": float(pred_proba),
            "confidence": float(confidence),
            "threshold": 0.5,
        },
        "shap_analysis": {
            "total_shap_contribution": float(shap_vals.sum()),
            "mean_abs_shap": float(np.abs(shap_vals).mean()),
            "features_increasing_pd": int((shap_vals > 0).sum()),
            "features_decreasing_pd": int((shap_vals < 0).sum()),
        },
        "top_contributing_features": top_features,
        "all_feature_values": {
            feat: float(val) for feat, val in zip(feature_names, feature_values)
        },
        "all_shap_values": {
            feat: float(val) for feat, val in zip(feature_names, shap_vals)
        },
        "clinical_prompt": clinical_prompt,
    }

    return json_data


def export_llm_json(
    shap_values,
    X_test,
    predictions,
    feature_names,
    output_dir="llm_inputs",
):
    """
    Create JSON files for all test patients (one per patient + batch file).
    """
    Path(output_dir).mkdir(exist_ok=True)

    n_samples = len(X_test)
    all_json_data = []

    for i in range(n_samples):
        json_data = create_patient_json(
            patient_idx=i,
            feature_values=X_test[i],
            shap_vals=shap_values[i],
            pred_proba=predictions[i],
            feature_names=feature_names,
        )

        with open(f"{output_dir}/patient_{i}.json", "w") as f:
            json.dump(json_data, f, indent=2)

        all_json_data.append(json_data)

    with open(f"{output_dir}/all_patients_batch.json", "w") as f:
        json.dump(all_json_data, f, indent=2)

    print(f"✓ Created {n_samples} individual JSON files")
    print(f"✓ Created batch file: {output_dir}/all_patients_batch.json")
