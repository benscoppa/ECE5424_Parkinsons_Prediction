"""
============================================================================
COMPLETE PIPELINE: FNN Training → SHAP Analysis → Visualizations + JSON for LLM
============================================================================
This script:
1. Trains a Feedforward Neural Network for Parkinson's Disease detection
2. Generates SHAP explanations
3. Creates visualizations (Mean SHAP, Beeswarm, Patient Waterfall)
4. Exports structured JSON for LLM interpretation (Module 4)

Requirements:
    pip install torch pandas scikit-learn shap matplotlib numpy

Dataset: parkinsons.csv (UCI Parkinson's Disease Voice Dataset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PARKINSON'S DISEASE DETECTION: COMPLETE PIPELINE")
print("="*80)


# ============================================================================
# PART 1: DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[STEP 1/5] Loading and preprocessing data...")

# Load dataset
df = pd.read_csv("/content/sample_data/parkinsons.data")
print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# Feature names (excluding 'name' and 'status')
feature_names = [col for col in df.columns if col not in ['name', 'status']]
print(f"✓ Features: {len(feature_names)}")

# Prepare data
X = df.drop(columns=['name', 'status'], errors='ignore').values
y = df['status'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# ============================================================================
# PART 2: FNN MODEL DEFINITION & TRAINING
# ============================================================================
print("\n[STEP 2/5] Building and training FNN model...")

class ParkinsonFNN(nn.Module):
    def __init__(self, input_dim):
        super(ParkinsonFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
input_dim = X_train.shape[1]
model = ParkinsonFNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"✓ Model architecture: {input_dim} → 128 → 64 → 1")

# Training loop
epochs = 200
print(f"✓ Training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "fnn_parkinson_model.pth")
print("✓ Model saved as 'fnn_parkinson_model.pth'")


# ============================================================================
# PART 3: MODEL EVALUATION
# ============================================================================
print("\n[STEP 3/5] Evaluating model performance...")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()
    predictions = y_pred.numpy().flatten()

acc = accuracy_score(y_test_tensor, y_pred_class)
f1 = f1_score(y_test_tensor, y_pred_class)
cm = confusion_matrix(y_test_tensor, y_pred_class)

print(f"\n{'='*40}")
print("MODEL PERFORMANCE")
print(f"{'='*40}")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix:")
print(f"              Predicted")
print(f"              Healthy  PD")
print(f"Actual Healthy  {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"       PD       {cm[1,0]:3d}    {cm[1,1]:3d}")
print(f"{'='*40}")


# ============================================================================
# PART 4: SHAP ANALYSIS & VISUALIZATIONS
# ============================================================================
print("\n[STEP 4/5] Computing SHAP values and creating visualizations...")

# Compute SHAP values using DeepExplainer (optimized for PyTorch)
print("✓ Computing SHAP values using DeepExplainer (PyTorch-optimized)...")

# Use a background dataset (100 samples from training set)
background = X_train_tensor[:100]

# Initialize DeepExplainer
explainer = shap.DeepExplainer(model, background)

# Compute SHAP values for test set
shap_values = explainer.shap_values(X_test_tensor)

# Convert to numpy if it's a tensor
if isinstance(shap_values, torch.Tensor):
    shap_values = shap_values.cpu().numpy()

# Handle the case where SHAP returns a list (for multi-output models)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Ensure correct shape (remove extra dimension if present)
if len(shap_values.shape) == 3:
    shap_values = shap_values.squeeze()

print(f"✓ SHAP values computed: shape {shap_values.shape}")

# Create output directory
Path('shap_visualizations').mkdir(exist_ok=True)

# --- Visualization 1: Mean Absolute SHAP (Global Importance) ---
print("\n✓ Creating visualization 1/3: Global Feature Importance...")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_10_idx = np.argsort(mean_abs_shap)[-10:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(10), mean_abs_shap[top_10_idx], color='steelblue', alpha=0.7)
plt.yticks(range(10), [feature_names[i] for i in top_10_idx])
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.title('Top 10 Most Important Features (Global)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('shap_visualizations/1_global_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("   → Saved: shap_visualizations/1_global_importance.png")

# --- Visualization 2: SHAP Beeswarm Plot ---
print("✓ Creating visualization 2/4: SHAP Beeswarm Plot...")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title('SHAP Summary: Feature Impact Distribution', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('shap_visualizations/2_beeswarm_summary.png', dpi=300, bbox_inches='tight')
plt.show()
print("   → Saved: shap_visualizations/2_beeswarm_summary.png")

# --- Visualization 3: SHAP Waterfall Plot (Patient 0) ---
print("✓ Creating visualization 3/4: SHAP Waterfall Plot (Patient 0)...")

patient_idx = 0
# Create Explanation object for waterfall plot
explanation = shap.Explanation(
    values=shap_values[patient_idx],
    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
    data=X_test[patient_idx],
    feature_names=feature_names
)

plt.figure(figsize=(10, 8))
shap.waterfall_plot(explanation, max_display=10, show=False)
plt.title(f'Patient {patient_idx}: SHAP Waterfall Analysis\n' +
          f'Predicted: {"PD" if predictions[patient_idx] > 0.5 else "Healthy"} ' +
          f'(PD probability: {predictions[patient_idx]*100:.1f}%)',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_visualizations/3_patient_0_waterfall.png', dpi=300, bbox_inches='tight')
plt.show()
print("   → Saved: shap_visualizations/3_patient_0_waterfall.png")

# --- Visualization 4: Individual Waterfall Plots for ALL Test Patients ---
print(f"✓ Creating visualization 4/4: Individual Waterfall Plots for ALL {len(X_test)} test patients...")

# Create subdirectory for individual patient plots
Path('shap_visualizations/individual_patients').mkdir(parents=True, exist_ok=True)

# Generate waterfall plot for EACH test patient
for patient_idx in range(len(X_test)):
    # Create Explanation object for this patient
    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
        data=X_test[patient_idx],
        feature_names=feature_names
    )
    
    # Create waterfall plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    
    # Add detailed title
    actual_label = "PD" if y_test[patient_idx] == 1 else "Healthy"
    predicted_label = "PD" if predictions[patient_idx] > 0.5 else "Healthy"
    correct = "✓" if actual_label == predicted_label else "✗"
    
    plt.title(f'Patient {patient_idx}: SHAP Waterfall Analysis {correct}\n' +
              f'Actual: {actual_label} | Predicted: {predicted_label} ' +
              f'(PD probability: {predictions[patient_idx]*100:.1f}%)',
              fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(f'shap_visualizations/individual_patients/patient_{patient_idx:03d}_waterfall.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    # Print progress every 10 patients
    if (patient_idx + 1) % 10 == 0 or patient_idx == len(X_test) - 1:
        print(f"   → Processed {patient_idx + 1}/{len(X_test)} patients")

print(f"   → Saved: shap_visualizations/individual_patients/patient_000_waterfall.png ... patient_{len(X_test)-1:03d}_waterfall.png")

# --- Visualization 5: Summary Grid (Optional - shows subset for overview) ---
print(f"✓ Creating visualization 5/5: Summary Grid (showing subset of {min(12, len(X_test))} patients)...")

# Select diverse patients for summary grid (max 12 for readability)
n_display = min(12, len(X_test))
display_indices = np.linspace(0, len(X_test)-1, n_display, dtype=int)

# Create grid
n_cols = 4
n_rows = (n_display + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten() if n_display > 1 else [axes]

for idx, patient_idx in enumerate(display_indices):
    ax = axes[idx]
    
    # Get top 8 features for this patient
    sorted_idx = np.argsort(np.abs(shap_values[patient_idx]))[-8:]
    colors = ['#d62728' if val > 0 else '#1f77b4' for val in shap_values[patient_idx][sorted_idx]]
    
    ax.barh(range(8), shap_values[patient_idx][sorted_idx], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(8))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel('SHAP Value', fontsize=9, fontweight='bold')
    
    actual_label = "PD" if y_test[patient_idx] == 1 else "Healthy"
    predicted_label = "PD" if predictions[patient_idx] > 0.5 else "Healthy"
    correct = "✓" if actual_label == predicted_label else "✗"
    pd_prob = predictions[patient_idx] * 100
    
    ax.set_title(f'Patient {patient_idx}: {predicted_label} {correct}\n(Actual: {actual_label}, Prob: {pd_prob:.1f}%)',
                 fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3)

# Hide unused subplots
for idx in range(n_display, len(axes)):
    axes[idx].axis('off')

# Add legend
legend_elements = [
    plt.matplotlib.patches.Patch(facecolor='#d62728', alpha=0.7, label='Increases PD risk'),
    plt.matplotlib.patches.Patch(facecolor='#1f77b4', alpha=0.7, label='Decreases PD risk')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=2, fontsize=11, frameon=True)

plt.suptitle(f'SHAP Analysis: Overview of {n_display} Sample Patients (from {len(X_test)} total)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('shap_visualizations/5_patient_summary_grid.png', dpi=300, bbox_inches='tight')
plt.show()
print("   → Saved: shap_visualizations/5_patient_summary_grid.png")

print(f"\n✓ Total visualizations created:")
print(f"   • 3 global/summary plots")
print(f"   • {len(X_test)} individual patient waterfall plots")
print(f"   • 1 summary grid overview")


# ============================================================================
# PART 5: JSON EXPORT FOR LLM (MODULE 4)
# ============================================================================
print("\n[STEP 5/5] Creating JSON files for LLM interpretation...")

# Clinical context mapping
clinical_mapping = {
    'MDVP:Fo(Hz)': {
        'clinical_name': 'Fundamental Frequency',
        'normal_range': '85-180 Hz (male), 165-255 Hz (female)',
        'pd_indication': 'Reduced variability indicates vocal rigidity'
    },
    'MDVP:Fhi(Hz)': {
        'clinical_name': 'Maximum Vocal Frequency',
        'normal_range': '102-592 Hz',
        'pd_indication': 'Reduced range suggests limited vocal control'
    },
    'MDVP:Flo(Hz)': {
        'clinical_name': 'Minimum Vocal Frequency',
        'normal_range': '65-239 Hz',
        'pd_indication': 'Narrowed frequency range indicates motor impairment'
    },
    'MDVP:Jitter(%)': {
        'clinical_name': 'Jitter Percentage',
        'normal_range': '<1.04%',
        'pd_indication': 'Elevated jitter indicates irregular vocal fold vibrations'
    },
    'MDVP:Jitter(Abs)': {
        'clinical_name': 'Absolute Jitter',
        'normal_range': '<83 microseconds',
        'pd_indication': 'Increased jitter reflects neuromuscular instability'
    },
    'MDVP:Shimmer': {
        'clinical_name': 'Shimmer',
        'normal_range': '<3.81%',
        'pd_indication': 'Elevated shimmer indicates amplitude perturbation and weak voice'
    },
    'MDVP:Shimmer(dB)': {
        'clinical_name': 'Shimmer in Decibels',
        'normal_range': '<0.35 dB',
        'pd_indication': 'Increased amplitude variation suggests vocal weakness'
    },
    'HNR': {
        'clinical_name': 'Harmonics-to-Noise Ratio',
        'normal_range': '>13 dB',
        'pd_indication': 'Reduced HNR indicates increased vocal noise and hoarseness'
    },
    'RPDE': {
        'clinical_name': 'Recurrence Period Density Entropy',
        'normal_range': '0.4-0.7',
        'pd_indication': 'Altered RPDE reflects changes in vocal signal complexity'
    },
    'DFA': {
        'clinical_name': 'Detrended Fluctuation Analysis',
        'normal_range': '0.6-0.7',
        'pd_indication': 'Abnormal DFA indicates disrupted vocal control dynamics'
    },
    'PPE': {
        'clinical_name': 'Pitch Period Entropy',
        'normal_range': '0.1-0.3',
        'pd_indication': 'Increased entropy indicates irregular vocal fold vibration'
    },
    'NHR': {
        'clinical_name': 'Noise-to-Harmonics Ratio',
        'normal_range': '<0.19',
        'pd_indication': 'Elevated NHR suggests breathiness and vocal instability'
    }
}

def create_patient_json(patient_idx, feature_values, shap_vals, pred_proba):
    """Create structured JSON for LLM interpretation"""

    pred_class = "Parkinson's Disease" if pred_proba > 0.5 else "Healthy"
    confidence = pred_proba if pred_proba > 0.5 else (1 - pred_proba)

    # Get top 5 features by absolute SHAP value
    top_5_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

    top_features = []
    for rank, idx in enumerate(top_5_idx, 1):
        feat_name = feature_names[idx]
        clinical_info = clinical_mapping.get(feat_name, {
            'clinical_name': feat_name,
            'normal_range': 'Not specified',
            'pd_indication': 'Clinical interpretation not available'
        })

        top_features.append({
            'rank': rank,
            'feature_name': feat_name,
            'feature_value': float(feature_values[idx]),
            'shap_value': float(shap_vals[idx]),
            'contribution': 'increases' if shap_vals[idx] > 0 else 'decreases',
            'clinical_name': clinical_info['clinical_name'],
            'normal_range': clinical_info['normal_range'],
            'pd_indication': clinical_info['pd_indication']
        })

    # Create clinical prompt for LLM
    clinical_prompt = f"""Analyze the following Parkinson's Disease voice biomarker assessment:

PATIENT ID: patient_{patient_idx}
PREDICTION: {pred_class}
CONFIDENCE: {confidence*100:.1f}%

TOP CONTRIBUTING FEATURES:
"""

    for feat in top_features:
        direction = "↑ increases" if feat['contribution'] == 'increases' else "↓ decreases"
        clinical_prompt += f"""
{feat['rank']}. {feat['clinical_name']} ({feat['feature_name']})
   Patient Value: {feat['feature_value']:.4f}
   Normal Range: {feat['normal_range']}
   SHAP Impact: {feat['shap_value']:.4f} ({direction} PD probability)
   Clinical Significance: {feat['pd_indication']}
"""

    clinical_prompt += """
TASK: Based on the SHAP analysis above, provide a clinical interpretation that:
1. Summarizes the key acoustic findings in 2-3 sentences
2. Explains what the top features indicate about vocal function
3. Relates findings to PD motor symptoms (rigidity, bradykinesia, tremor)
4. Assesses the reliability of this prediction based on feature patterns

Keep the explanation concise and accessible to healthcare providers.
"""

    # Build complete JSON structure
    json_data = {
        'metadata': {
            'patient_id': f'patient_{patient_idx}',
            'model_version': 'FNN_v1.0',
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'prediction': {
            'class': pred_class,
            'probability': float(pred_proba),
            'confidence': float(confidence),
            'threshold': 0.5
        },
        'shap_analysis': {
            'total_shap_contribution': float(shap_vals.sum()),
            'mean_abs_shap': float(np.abs(shap_vals).mean()),
            'features_increasing_pd': int((shap_vals > 0).sum()),
            'features_decreasing_pd': int((shap_vals < 0).sum())
        },
        'top_contributing_features': top_features,
        'all_feature_values': {feat: float(val) for feat, val in zip(feature_names, feature_values)},
        'all_shap_values': {feat: float(val) for feat, val in zip(feature_names, shap_vals)},
        'clinical_prompt': clinical_prompt
    }

    return json_data


# Create JSON files for all test patients
Path('llm_inputs').mkdir(exist_ok=True)

n_samples = len(X_test)
all_json_data = []

for i in range(n_samples):
    json_data = create_patient_json(
        patient_idx=i,
        feature_values=X_test[i],
        shap_vals=shap_values[i],
        pred_proba=predictions[i]
    )

    # Save individual file
    output_path = f'llm_inputs/patient_{i}.json'
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    all_json_data.append(json_data)

# Save batch file
with open('llm_inputs/all_patients_batch.json', 'w') as f:
    json.dump(all_json_data, f, indent=2)

print(f"✓ Created {n_samples} individual JSON files")
print(f"✓ Created batch file: llm_inputs/all_patients_batch.json")


# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)