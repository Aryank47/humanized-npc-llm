<!-- # Evaluation

**Owner**: Torsha Chatterjee (M23CSA536)

## Setup
```bash
pip install scikit-learn sentence-transformers matplotlib pandas seaborn
```

## Run Evaluation
```bash
python run_eval.py
```

## Outputs
- `outputs/metrics.json` - All metrics
- `outputs/figures/` - Plots
- `outputs/report.pdf` - Final report -->
Task 3: Evaluation & Reporting

Owner: Torsha Chatterjee

This directory contains all code required to evaluate the fine-tuned NPC model against the baseline, calculating all metrics defined in the project plan.

Objective

Comprehensively evaluate the fine-tuned model across persona faithfulness, NPC authenticity, hallucination control, and practical metrics. Produce a final report with ablations and error analysis.

How to Run

1. Installation

This pipeline requires new dependencies for metrics calculation. Install them from this directory:

# From the 3_evaluation/ directory
pip install -r requirements.txt


This will install sentence-transformers, nltk, jsonschema, etc.

2. Configuration

All settings are controlled by config/eval.yaml. Before running, verify these paths:

# config/eval.yaml

data:
  # Path to your test set from Task 1
  test_file: ../1_data_engineering/outputs/test.jsonl
  
  # Path to your fine-tuned adapters from Task 2
  tuned_model_path: ../2_fine_tuning/outputs/final_model 

baseline_model:
  # The *base model ID* you used for training
  id: "unsloth/Phi-3-mini-4k-instruct-hf" 
  # id: "unsloth/Qwen1.5-4B-Chat-hf" # Or this one

evaluation:
  # Use a small number (e.g., 50) for a quick test run.
  # Use -1 to run on the full test set.
  limit_samples: 50 


3. Execution

Navigate to the 3_evaluation/ directory and run the main script:

# Make sure you are in the 3_evaluation/ directory
python run_eval.py


What to Expect:

Model Loading: The script will first load the NLI and Embedding models. This may take a minute and download them on the first run.

Generation Loop: You will see a tqdm progress bar as the script iterates through every sample in test.jsonl, generating a response from both the baseline and the fine-tuned model.

Metrics Calculation: All metrics (JSON validity, persona similarity, NLI, UCR, etc.) are calculated for each pair of generations.

Completion: The script will print a final summary table to your console and save the detailed results.

Deliverables

This script produces two key files in 3_evaluation/outputs/:

model_generations.jsonl

A large JSONL file, with one line per test sample.

Contains the full text of the baseline_response and tuned_response.

Contains a detailed dictionary of all computed baseline_metrics and tuned_metrics for that specific sample.

Use this file for qualitative analysis and error analysis.

metrics_report.json

A single JSON file containing the aggregated results.

Provides the final avg_ metrics (e.g., avg_is_valid_schema, avg_persona_contradiction_rate) for the entire test set.

Use this file to create the tables and figures for your final report.

Final Report Structure (from report.pdf)

Use the data from metrics_report.json to fill in the tables for your report:

Abstract

1. Introduction

2. Related Work

3. Methodology

3.1. Data Pipeline (Task 1 outputs)

3.2. Model Fine-Tuning (Task 2 config)

3.3. Evaluation Framework (The metrics from metrics.py)

4. Results

4.1. Automatic Metrics (Use metrics_report.json to build this table)

4.2. Qualitative Analysis (Pull good/bad examples from model_generations.jsonl)

4.3. Human Evaluation (TBD)

5. Ablation Studies (e.g., compare different LoRA ranks)

6. Discussion & Limitations

7. Conclusion & Future Work

References