# ECE5424_Parkinsons_Prediction

An explainable **Feedforward Neural Network (FNN)** for classifying Parkinson’s Disease from speech features.  
Model predictions are interpreted using **SHAP**, and an **AI reasoning agent (Ollama)** is integrated to automatically summarize SHAP outputs in natural language.

## Clone the repository
```bash
git clone https://github.com/benscoppa/ECE5424_Parkinsons_Prediction.git
cd ECE5424_Parkinsons_Prediction```

## Install dependencies
```bash
pip install -r requirements.txt```

## Install and Setup Ollama
Download and install Ollama from:<br>
https://ollama.com<br>

Then, in a new terminal:<br>
```bash
ollama --version<br>
ollama serve<br>```

In project directory terminal:<br>
```bash
ollama create parkinsons-agent -f Modelfile<br>```

Verify the model exists in the new terminal:<br>
```bash
ollama list```

## Run Command Line Interface
The command line interface is used to run a single patient's biomarkers through the pipeline<br>
Note: The CSV formatting must exactly match parkinsons.csv, except excluding name and status<br>
```bash
python new_patient_cli.py --patient-file <path_to_csv> --output-dir <output_directory>```
