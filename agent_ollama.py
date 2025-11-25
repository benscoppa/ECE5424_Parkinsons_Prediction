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
    """
    prompt = patient_data.get("clinical_prompt", "")
    if not prompt:
        return "Error: No clinical prompt found in data."

    system_prompt = (
        "You are a helpful medical assistant explaining AI model results to a clinician. "
        "Use plain English. Be concise but thorough. "
        "Base your explanation STRICTLY on the provided SHAP analysis."
    )

    full_prompt = f"{system_prompt}\n\n{prompt}"

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        result = response.json()
        explanation = result.get("response", "")
        duration = end_time - start_time
        
        return explanation, duration
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}", 0

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
