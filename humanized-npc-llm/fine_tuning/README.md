# Local NPC Training – Developer Setup

This guide explains how to set up your **local machine** to mirror the Azure GPU VM
environment used for NPC dialogue fine-tuning.

The goal is:

- Same directory layout as the remote VM (`~/npc-training` style).
- Same virtual environment name: `venv_unsloth`.
- Same Python / Unsloth / HF stack so that `fine_tuning/train.py` runs locally
  exactly like on the remote VM.

You can use this either on **CPU** or on a **GPU machine** (Linux or macOS, incl. M-series).

---

## 1. Prerequisites

### 1.1 OS & Python

- **OS**:  
  - Linux (Ubuntu 22.04+ recommended), or  
  - macOS (Intel or Apple Silicon / M-series).
- **Python**: `3.10.x` (match the VM; 3.11+ is not tested here).

Check:

```bash
python3 --version
# Expect something like: Python 3.10.x
```

If needed, install Python 3.10 from your OS package manager (Linux) or via
pyenv￼ on macOS.

1.2 Optional: GPU Support
	•	Linux + NVIDIA GPU
	•	NVIDIA driver + CUDA runtime installed (matching what PyTorch / Unsloth expects).
	•	macOS + Apple Silicon (M1/M2/M3)
	•	PyTorch will use MPS backend instead of CUDA (Unsloth handles this under the hood).

You can still run on pure CPU; it will just be slower.

---

## 2. Project Layout (Local)

Your local project should look like this (top-level):

```text
npc-training/              # root project dir (can be any path)
├── data_engineering/
│   ├── outputs/           # *.jsonl training data
│   └── config/
│       └── training.yaml  # (optional) training config
├── fine_tuning/
│   └── train.py           # main training script
├── run_training.sh        # (optional) training wrapper
├── monitor_training.sh    # (optional) monitoring helper
└── venv_unsloth/          # created in Step 3
```

If you cloned a repo, data_engineering/ and fine_tuning/ likely already exist.
You only need to add the virtualenv and (optionally) the helper shell scripts.

---

## 3. Create the venv_unsloth Environment

```bash
From your project root (npc-training/):

cd /path/to/npc-training

# Create virtual environment
python3 -m venv venv_unsloth

# Activate it
# Linux / macOS (bash/zsh):
source venv_unsloth/bin/activate

# (Windows PowerShell – only if someone is on Windows)
# venv_unsloth\Scripts\Activate.ps1

Upgrade basic tooling:

pip install --upgrade pip wheel setuptools
```

---

## 4. Install Unsloth + HF Stack (Mirror of Remote VM)

The remote VM script installs Unsloth first so that it pins compatible versions
of PyTorch, Transformers, TRL, etc. We do the same locally.

```bash
Make sure your venv_unsloth is activated (which python should show the venv).

# 4.1 Core: Unsloth (this pulls compatible torch/transformers/trl)
pip install unsloth

# 4.2 Supporting libraries (same as VM)
pip install datasets
pip install pyyaml tqdm pandas orjson jsonschema requests
pip install selectolax python-slugify

# 4.3 Optional: xformers (Linux GPU mostly)
pip install xformers || echo "xformers install failed or is unsupported; continuing..."

If xformers fails on macOS or CPU-only machines, that’s fine — training will
still work; this is just an optimization.
```

---

## 5. Verify PyTorch & Device

With venv_unsloth still active:

```bash
python - << 'EOF'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif torch.backends.mps.is_available():
    print("MPS backend is available (Apple Silicon GPU).")
else:
    print("Running on CPU only.")
EOF

Expected:
	•	On Linux GPU: CUDA available: True and your GPU name.
	•	On Apple Silicon: CUDA available: False, but MPS backend is available.
	•	On CPU only: both CUDA and MPS unavailable → training will run on CPU.
```

---

## 6. Preparing Local Data & Config

### 6.1 If You Have Data Locally

Simply place your data under:

npc-training/data_engineering/data/v2/*.jsonl

So that fine_tuning/train.py can find it exactly like on the VM.

---

## 7. Running Training Locally

### 7.1 Direct train.py Invocation (recommended)

From project root:

```bash
cd /path/to/npc-training
source venv_unsloth/bin/activate

cd fine_tuning
python train.py

If train.py accepts CLI flags (e.g., model name, batch size, etc.), pass them here,
for example:

python train.py \
  --config ./training.yaml \
  --train_data ../data_engineering/data/v2/train.jsonl \
  --eval_data  ../data_engineering/data/v2/eval.jsonl
```

### 7.2 Using run_training.sh (mirror of VM behavior)

If you want the exact same wrapper as the VM, create this file in your project
root (npc-training/run_training.sh):

```bash
cat > run_training.sh << 'EOF'
#!/bin/bash
set -e

cd "$(dirname "$0")"   # go to project root
source venv_unsloth/bin/activate

echo "=========================================="
echo "Starting NPC Dialogue Fine-Tuning (LOCAL)"
echo "=========================================="
echo "Start time: $(date)"
echo ""

echo "Device info:"
python - << 'PYEOF'
import torch

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print("Using Apple MPS backend.")
else:
    print("Using CPU.")
PYEOF

echo ""

cd fine_tuning
python train.py 2>&1 | tee "training_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
EOF

chmod +x run_training.sh

Then run:

./run_training.sh
```