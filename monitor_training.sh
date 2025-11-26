#!/bin/bash

echo "=== GPU Status ==="
nvidia-smi || true

echo ""
echo "=== Disk Usage ==="
df -h | grep -E "Filesystem|/$"

echo ""
echo "=== Training Processes ==="
ps aux | grep python | grep -v grep || true

echo ""
echo "=== Latest Training Logs (last 20 lines) ==="
if ls ~/npc-training/humanised-npc-llm/fine_tuning/training_log_*.txt 1> /dev/null 2>&1; then
    tail -20 "$(ls -t ~/npc-training/humanised-npc-llm/fine_tuning/training_log_*.txt | head -1)"
else
    echo "No training logs found yet"
fi
