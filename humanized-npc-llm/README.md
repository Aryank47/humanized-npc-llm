# Humanized NPC-LLM

Fine-tuning small language models (3-4B parameters) to generate persona-consistent, humanized dialogue for video game NPCs using QLoRA/PEFT.

## Team Members
- Aryan Kumar (M23CSA510) - Data Engineering
- Subodh Kant (M23CSA531) - Fine-Tuning
- Torsha Chatterjee (M23CSA536) - Evaluation

## Quick Start

### Person 1: Data Engineering
```bash
cd 1_data_engineering/
pip install -r requirements.txt
python run_pipeline.py
```

### Person 2: Fine-Tuning
```bash
cd 2_fine_tuning/
pip install -r requirements.txt
python train.py
```

### Person 3: Evaluation
```bash
cd 3_evaluation/
pip install -r requirements.txt
python run_eval.py
```

## Documentation
See `docs/PROJECT_PLAN.md` for complete project documentation.

## Repository Structure
```
1_data_engineering/  - Data pipeline (Person 1)
2_fine_tuning/       - Model training (Person 2)
3_evaluation/        - Metrics & reporting (Person 3)
```
