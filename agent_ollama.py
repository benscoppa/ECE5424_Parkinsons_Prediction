import json
import requests
import glob
import os
import time
from pathlib import Path

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "parkinsons-agent"  # Custom model defined in Modelfile
INPUT_DIR = "llm_input_json"
OUTPUT_DIR = "llm_outputs"

def generate_explanation(patient_data, model=MODEL_NAME):
    """
    Sends the clinical prompt to Ollama and gets the explanation.
    Implements a Critic-Refiner Agentic Workflow.
    """
    prompt = patient_data.get("clinical_prompt", "")
    if not prompt:
        return "Error: No clinical prompt found in data.", 0

    # Extract top feature names for verification
    top_features = [f['feature_name'].lower() for f in patient_data.get('top_contributing_features', [])]
    
    # --- PROMPTS ---
    PROMPT_GENERATOR = (
        "You are a Neurologist analyzing voice biomarkers for Parkinson's Disease. "
        "Use plain English. Be concise but thorough. "
        "Base your explanation STRICTLY on the provided SHAP analysis. "
        "Structure your response with these sections: 'Clinical Summary', 'Key Findings', and 'Reliability'."
    )
    
    PROMPT_CRITIC = (
        "You are a QA Auditor. Your job is to verify the accuracy of a generated medical report. "
        "Check if the report mentions all the key features provided in the prompt. "
        "If features are missing, list them."
    )

    # --- AGENTIC LOOP ---
    start_time = time.time()
    
    current_prompt = f"{PROMPT_GENERATOR}\n\n{prompt}"
    explanation = ""
    max_retries = 3
    attempt = 0
    
    while attempt < max_retries:
        try:
            # 1. Generate Draft
            payload = {"model": model, "prompt": current_prompt, "stream": False}
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            explanation = response.json().get("response", "")
            
            # 2. Critic Step (Hybrid: Python Check + Implicit Logic)
            # In a full production system, we'd ask the LLM to critique. 
            # For speed and reliability here, we use Python to find missing features,
            # acting as a deterministic Critic.
            missing_features = []
            explanation_lower = explanation.lower()
            for feature in top_features:
                if feature not in explanation_lower:
                    missing_features.append(feature)
            
            # 3. Decision
            if not missing_features:
                # Pass!
                break
            
            # 4. Refinement Instruction
            print(f"    ! Critic (Attempt {attempt+1}): Missing features {missing_features}. Refining...")
            refinement_instruction = (
                f"Your previous draft missed the following key features: {', '.join(missing_features)}. "
                f"Please rewrite the report to explicitly mention these features and their impact."
                f"\n\nOriginal Data:\n{prompt}"
            )
            current_prompt = f"{PROMPT_GENERATOR}\n\n{refinement_instruction}"
            attempt += 1
            
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}", 0

    end_time = time.time()
    duration = end_time - start_time
    
    return explanation, duration

def process_patient_files():
    """
    Process all JSON files in the input directory.
    """
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    input_files = glob.glob(os.path.join(INPUT_DIR, "patient_*.json"))
    print(f"Found {len(input_files)} patient files to process.")

    results = []

    for file_path in input_files:
        print(f"Processing {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            patient_id = data['metadata']['patient_id']
            explanation, duration = generate_explanation(data)
            
            # Create output structure
            output_data = {
                "patient_id": patient_id,
                "original_prediction": data['prediction'],
                "explanation": explanation,
                "generation_time_seconds": duration,
                "model_used": MODEL_NAME,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to output file
            output_filename = f"{patient_id}_explained.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            results.append(output_data)
            print(f"  -> Saved explanation to {output_path} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"  -> Failed to process {file_path}: {e}")

    return results

if __name__ == "__main__":
    print("Starting AI Agent (Ollama Interface)...")
    print(f"Target Model: {MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}")
    
    # Check if Ollama is reachable
    try:
        requests.get("http://localhost:11434")
        print("Ollama is reachable.")
    except requests.exceptions.ConnectionError:
        print("WARNING: Ollama is NOT reachable. Make sure 'ollama serve' is running.")
        # We continue anyway to let the script fail gracefully per file or if the user starts it later
    
    process_patient_files()
    print("\nProcessing complete.")
