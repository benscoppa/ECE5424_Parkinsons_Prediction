# ECE5424_Parkinsons_Prediction
Explainable Feedforward Neural Network (FNN) model that classifies Parkinson's Disease from speech features and interprets it's predictions using SHAP frameworks. An AI reasoning agent will be integrated to automatically summarize SHAP outputs in natural language.

**Clone the repo**<br>
git clone https://github.com/benscoppa/ECE5424_Parkinsons_Prediction.git<br>
cd https://github.com/benscoppa/ECE5424_Parkinsons_Prediction.git

**Instal dependencies**<br>
pip install -r requirements.txt

**Install and Setup Ollama**<br>
download ollama: https://ollama.com<br>
Then, in a new terminal:<br>
ollama --version<br>
ollama serve<br>
In project directory terminal:<br>
ollama create parkinsons-agent -f Modelfile<br>
Verify the model exists in the new terminal:<br>
ollama list

**Run Command Line Interface**<br>
The command line interface is used to run a single patient's biomarkers through the pipeline<br>
Note: The CSV formatting must exactly match parkinsons.csv, except excluding name and status<br>
python new_patient_cli.py --patient-file <path_to_csv> --output-dir <output_directory>
