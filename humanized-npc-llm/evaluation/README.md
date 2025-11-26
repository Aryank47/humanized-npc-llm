# Task 3: Evaluation & Reporting

Owner: Torsha Chatterjee

This directory contains all code required to evaluate the fine-tuned NPC model against the baseline, calculating all metrics defined in the project plan.

Objective

Comprehensively evaluate the fine-tuned model across persona faithfulness, NPC authenticity, hallucination control, and practical metrics. Produce a final report with ablations and error analysis.

How to Run

1. Installation

This pipeline requires new dependencies for metrics calculation. Install them from this directory:

# From the evaluation/ directory
pip install -r requirements.txt


2. Configuration

All settings are controlled by evaluation/eval.yaml. Before running, verify these paths:

evaluation/eval.yaml

data:
  Path to your test set from Task 1
  test_file: "../data_engineering/data/processed/v2/test.jsonl"
    
  Path to your fine-tuned adapters from Task 2
  tuned_model_path: "../fine_tuning/outputs/model/final_model" 

baseline_model:
  The *base model ID* you used for training
  id: "Qwen/Qwen2.5-3B-Instruct" 

evaluation:
  Use a small number (e.g., 1000).
  Use -1 to run on the full test set.
  limit_samples: 100 


3. Execution

Navigate to the evaluation/ directory and run the main script:

Make sure you are in the evaluation/ directory
python run_eval.py


What to Expect:

Model Loading: The script will first load the NLI and Embedding models.

Generation Loop: You will see a tqdm progress bar as the script iterates through every sample in test.jsonl, generating a response from both the baseline and the fine-tuned model.

Metrics Calculation: All metrics (JSON validity, persona similarity, NEP, UCR, Distinct etc.) are calculated for each pair of generations.

Completion: The script will print a final summary table to your console and save the detailed results.

Deliverables

This script produces two key files in evaluation/outputs/results/:

generations.jsonl

A large JSONL file, with one line per test sample.

Contains the full text of the baseline_response and tuned_response.



References