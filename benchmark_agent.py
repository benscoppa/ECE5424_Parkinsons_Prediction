# benchmark_agent.py
# --------------------------------------------
# Ollama agent benchmarking
# Author: Yash Moar, Virginia Tech ECE Dept
# --------------------------------------------

import json
import glob
import os
import pandas as pd
import re
from pathlib import Path

# Configuration
INPUT_DIR = "llm_outputs"
ORIGINAL_INPUT_DIR = "llm_input_json"
OUTPUT_REPORT = "benchmark_report.csv"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def calculate_hallucination_score(explanation, top_features):
    """
    Checks if the explanation mentions the top features identified by SHAP.
    Returns a score (0.0 to 1.0) representing the recall of top features.
    """
    explanation_lower = explanation.lower()
    matches = 0
    
    for feature in top_features:
        # Check for feature name or clinical name
        feat_name = feature['feature_name'].lower()
        clin_name = feature['clinical_name'].lower()
        
        # Simple string matching (can be improved with regex or fuzzy matching)
        if feat_name in explanation_lower or clin_name in explanation_lower:
            matches += 1
            
    return matches / len(top_features) if top_features else 0

def analyze_sentiment_consistency(explanation, top_features):
    """
    Checks if the direction (increase/decrease) is consistent.
    This is a simplified check looking for keywords near feature names.
    """
    explanation_lower = explanation.lower()
    consistent_count = 0
    
    for feature in top_features:
        feat_name = feature['feature_name'].lower()
        contribution = feature['contribution'] # 'increases' or 'decreases'
        
        if feat_name in explanation_lower:
            # Look for direction keywords in the explanation
            # This is a heuristic and not perfect
            if contribution == 'increases':
                if any(w in explanation_lower for w in ['high', 'increase', 'elevated', 'up']):
                    consistent_count += 1
            elif contribution == 'decreases':
                if any(w in explanation_lower for w in ['low', 'decrease', 'reduced', 'down']):
                    consistent_count += 1
                    
    return consistent_count / len(top_features) if top_features else 0

def run_benchmark():
    print("Starting Benchmark...")
    
    output_files = glob.glob(os.path.join(INPUT_DIR, "*_explained.json"))
    if not output_files:
        print(f"No output files found in {INPUT_DIR}. Run agent_ollama.py first.")
        return

    results = []

    for file_path in output_files:
        try:
            # Load the agent's output
            agent_output = load_json(file_path)
            patient_id = agent_output['patient_id']
            explanation = agent_output['explanation']
            latency = agent_output['generation_time_seconds']
            
            # Load the original input to get ground truth (top features)
            original_input_path = os.path.join(ORIGINAL_INPUT_DIR, f"{patient_id}.json")
            if not os.path.exists(original_input_path):
                print(f"Warning: Original input for {patient_id} not found.")
                continue
                
            original_data = load_json(original_input_path)
            top_features = original_data['top_contributing_features']
            
            # Calculate metrics
            recall_score = calculate_hallucination_score(explanation, top_features)
            # consistency_score = analyze_sentiment_consistency(explanation, top_features) # Optional, can be noisy
            
            results.append({
                "Patient ID": patient_id,
                "Latency (s)": latency,
                "Word Count": len(explanation.split()),
                "Feature Recall": recall_score,
                "Prediction": agent_output['original_prediction']['class'],
                "Confidence": agent_output['original_prediction']['confidence']
            })
            
        except Exception as e:
            print(f"Error benchmarking {file_path}: {e}")

    # Create DataFrame and save report
    df = pd.DataFrame(results)
    
    if not df.empty:
        print("\nBenchmark Results Summary:")
        print(df.describe())
        
        df.to_csv(OUTPUT_REPORT, index=False)
        print(f"\nDetailed report saved to {OUTPUT_REPORT}")
        
        # Calculate aggregate metrics
        avg_latency = df["Latency (s)"].mean()
        avg_recall = df["Feature Recall"].mean()
        
        print(f"\nAverage Latency: {avg_latency:.2f} seconds")
        print(f"Average Feature Recall: {avg_recall:.2f} (Target: > 0.8)")
    else:
        print("No valid results to benchmark.")

if __name__ == "__main__":
    run_benchmark()
