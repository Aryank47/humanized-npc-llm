<!-- # Fine-Tuning

**Owner**: Subodh Kant (M23CSA531)

## Setup
```bash
pip install torch transformers peft accelerate bitsandbytes
```

## Run Training
```bash
python train.py
```

## Test Inference
```bash
python inference.py --prompt "Can you repair my sword?"
```

## Outputs
- `outputs/model/` - LoRA adapters
- `outputs/logs/` - Training logs -->
Task 2: Fine-Tuning Pipeline

Owner: Subodh Kant

This directory contains the code to fine-tune the Small Language Model (SLM) using the unified dataset from Task 1.

Objective

Fine-tune a 3-4B parameter instruction-tuned LLM using QLoRA/PEFT to generate persona-consistent, humanized NPC dialogue that outputs valid JSON.

Prerequisites

Task 1 Data: Ensure the 1_data_engineering/outputs/ directory exists and contains train.jsonl and val.jsonl.

Config: Review and update config/training.yaml as needed.

Dependencies: Install all required Python packages.

# From the root of the repository
pip install -r requirements.txt

# Key packages for this task:
# pip install torch transformers datasets peft trl bitsandbytes unsloth pyyaml


W&B Login: For experiment tracking, log in to Weights & Biases:

wandb login


How to Run Training

The main training script will load the config, process the data, and run the Unsloth + SFTTrainer pipeline.

# Navigate to this directory
cd 2_fine_tuning/

# Start the training process
python train.py


The script will:

Load the configuration from ../config/training.yaml.

Load the base model (Phi-3-mini) and configure it with QLoRA.

Load train.jsonl and val.jsonl from Task 1.

Use data_loader.py to format each record into a chat template.

Start the SFTTrainer training, logging metrics to W&B.

Save the best-performing LoRA adapters to 2_fine_tuning/outputs/model/final_model/.

How to Run Inference

After training, you can use the inference.py script to test your model.

Ensure Model is Saved: Make sure the 2_fine_tuning/outputs/model/final_model/ directory exists with the adapter files.

Run the script:

# Navigate to this directory
cd 2_fine_tuning/

# Run the inference test script
python inference.py


This script will:

Load the base Phi-3-mini model.

Merge the fine-tuned LoRA adapters from the final_model directory.

Run three test scenarios (Blacksmith, Guard, and a follow-up) to demonstrate how to build a prompt and parse the model's JSON output.

Deliverables

Code: train.py, inference.py, data_loader.py

Configuration: ../config/training.yaml

Model Adapters (Output): outputs/model/final_model/ (contains adapter_model.bin, adapter_config.json, etc.)

Logs (Output): Metrics will be available in your W&B project.