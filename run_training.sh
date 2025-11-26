#!/bin/bash
set -e

cd ~/npc-training
source venv_unsloth/bin/activate

# Optional: Set Weights & Biases API key
export WANDB_API_KEY="<YOUR API KEY>"

echo "=========================================="
echo "Starting NPC Dialogue Fine-Tuning"
echo "=========================================="
echo "Start time: $(date)"
echo ""

echo "GPU Status:"
nvidia-smi || true
echo ""

cd ~/npc-training/humanized-npc-llm/fine_tuning

python train.py 2>&1 | tee "training_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
