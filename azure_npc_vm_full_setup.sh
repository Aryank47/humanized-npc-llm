#!/bin/bash
#############################################################################
# Azure GPU VM Setup (NCas_T4_v3, 1x T4 16GB)
# - Creates VM (NC4as_T4_v3)
# - Installs NVIDIA drivers (Azure extension)
# - Generates vm_setup.sh to configure:
#     * system packages
#     * venv_unsloth + Unsloth + HF stack
#     * project directory: ~/npc-training
#     * run_training.sh + monitor_training.sh
# - Optionally downloads data/config from Azure Blob Storage
# - Provides commands to upload local project code/data via scp
#############################################################################

set -e  # Exit on any error

echo "=========================================="
echo "Azure NPC Training - Full VM Setup"
echo "=========================================="

#############################################################################
# CONFIGURATION - MODIFY THESE VALUES
#############################################################################

# Resource group + location
RESOURCE_GROUP="npc-training-rg"
LOCATION="northeurope"

# VM Configuration - NCas_T4_v3
VM_NAME="npc-training-vm"
VM_SIZE="Standard_NC4as_T4_v3"  # 1x T4 16GB, 4 vCPUs
VM_IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest"
ADMIN_USERNAME="azureuser"
SSH_KEY_PATH="$HOME/.ssh/azure_npc_key"

# Local project path (on YOUR laptop)
PROJECT_DIR="$HOME/work/humanized-npc-llm/humanized-npc-llm"

# Storage (for vm_setup.sh to optionally download data)
STORAGE_ACCOUNT="npctraindata001"
CONTAINER_NAME="training-data"

#############################################################################
# PHASE 1: AZURE LOGIN + RESOURCE GROUP CHECK
#############################################################################

echo ""
echo "=== PHASE 1: Azure Login & Resource Group Check ==="
echo ""

# Check if logged in to Azure
echo "Checking Azure login..."
if ! az account show &> /dev/null; then
    echo "Not logged in. Running az login..."
    az login
fi

# Display current subscription
echo "Current subscription:"
az account show --query "{Name:name, ID:id, State:state}" -o table

# Ensure resource group exists (idempotent)
echo ""
echo "Ensuring resource group exists: $RESOURCE_GROUP"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

#############################################################################
# PHASE 2: SSH KEY GENERATION
#############################################################################

echo ""
echo "=== PHASE 2: SSH Key Setup ==="
echo ""

if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "Generating SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "azure-npc-training"
    echo "✓ SSH key created: $SSH_KEY_PATH"
else
    echo "✓ SSH key already exists: $SSH_KEY_PATH"
fi

#############################################################################
# PHASE 3: CREATE GPU VM (NCas_T4_v3)
#############################################################################

echo ""
echo "=== PHASE 3: Creating GPU VM (NCas_T4_v3) ==="
echo ""

# Check if VM already exists
if az vm show --name "$VM_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    echo "⚠ VM already exists: $VM_NAME"
    read -p "Delete and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing VM..."
        az vm delete --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --yes
        sleep 10
    else
        echo "Skipping VM creation"
        VM_IP=$(az vm show \
          --resource-group "$RESOURCE_GROUP" \
          --name "$VM_NAME" \
          --show-details \
          --query publicIps \
          --output tsv)
        echo "Existing VM IP: $VM_IP"
        echo "Jump to vm_setup.sh phase manually if needed."
        exit 0
    fi
fi

echo "Creating VM with size: $VM_SIZE"
echo "⏳ This will take a few minutes..."

# Create VM (disable secure boot for NVIDIA drivers)
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --location "$LOCATION" \
  --size "$VM_SIZE" \
  --image "$VM_IMAGE" \
  --admin-username "$ADMIN_USERNAME" \
  --ssh-key-values "${SSH_KEY_PATH}.pub" \
  --public-ip-sku Standard \
  --os-disk-size-gb 128 \
  --storage-sku Standard_LRS \
  --enable-secure-boot false \
  --output json

# Extract VM IP
VM_IP=$(az vm show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --show-details \
  --query publicIps \
  --output tsv)

echo ""
echo "✓ VM Created Successfully!"
echo "  VM Name: $VM_NAME"
echo "  VM Size: $VM_SIZE"
echo "  Public IP: $VM_IP"

#############################################################################
# PHASE 4: INSTALL NVIDIA DRIVERS (VM extension)
#############################################################################

echo ""
echo "=== PHASE 4: Installing NVIDIA GPU Drivers ==="
echo ""

echo "Installing NVIDIA driver extension..."
echo "⏳ This will take 8-10 minutes..."

az vm extension set \
  --resource-group "$RESOURCE_GROUP" \
  --vm-name "$VM_NAME" \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.9 \
  --output table

echo "Waiting for driver installation to complete..."
sleep 60

# Check installation status
EXTENSION_STATUS=$(az vm extension show \
  --resource-group "$RESOURCE_GROUP" \
  --vm-name "$VM_NAME" \
  --name NvidiaGpuDriverLinux \
  --query "provisioningState" \
  --output tsv)

if [ "$EXTENSION_STATUS" == "Succeeded" ]; then
    echo "✓ NVIDIA driver extension installed successfully"
else
    echo "⚠ Driver installation status: $EXTENSION_STATUS"
    echo "You may need to wait a bit longer or check manually"
fi

#############################################################################
# PHASE 5: SAVE LOCAL ENVIRONMENT HELPERS
#############################################################################

echo ""
echo "=== PHASE 5: Saving Local Environment Helpers ==="
echo ""

cat > "$HOME/azure_npc_env.sh" << EOF
# Azure NPC Training - VM Helpers
export RESOURCE_GROUP="$RESOURCE_GROUP"
export VM_NAME="$VM_NAME"
export VM_SIZE="$VM_SIZE"
export VM_IP="$VM_IP"
export ADMIN_USERNAME="$ADMIN_USERNAME"
export SSH_KEY_PATH="$SSH_KEY_PATH"

alias vm-ssh='ssh -i \$SSH_KEY_PATH \$ADMIN_USERNAME@\$VM_IP'
alias vm-start='az vm start --resource-group \$RESOURCE_GROUP --name \$VM_NAME'
alias vm-stop='az vm deallocate --resource-group \$RESOURCE_GROUP --name \$VM_NAME'
alias vm-delete='az vm delete --resource-group \$RESOURCE_GROUP --name \$VM_NAME --yes'
alias vm-status='az vm show --resource-group \$RESOURCE_GROUP --name \$VM_NAME --show-details --query powerState --output tsv'
EOF

echo "✓ Local helpers saved to: $HOME/azure_npc_env.sh"
echo ""
echo "To load these in future sessions, run:"
echo "  source $HOME/azure_npc_env.sh"

#############################################################################
# PHASE 6: GENERATE VM SETUP SCRIPT (runs ON the VM)
#############################################################################

echo ""
echo "=== PHASE 6: Generating VM Setup Script ==="
echo ""

cat > "$HOME/vm_setup.sh" << 'EOFVM'
#!/bin/bash
#############################################################################
# VM Setup Script - Run this ON the Azure VM
# - Assumes NVIDIA driver extension already installed by Azure
# - Downloads data from Azure Blob (if you have RBAC)
# - Sets up venv_unsloth with Unsloth + HF stack
# - Creates:
#     * ~/npc-training
#     * run_training.sh
#     * monitor_training.sh
#############################################################################

set -e

echo "=========================================="
echo "Setting up NPC Training Environment on VM"
echo "=========================================="

# These get filled in by the parent script via sed on your laptop
STORAGE_ACCOUNT="__STORAGE_ACCOUNT__"
CONTAINER_NAME="__CONTAINER_NAME__"

#############################################################################
# 1. SYSTEM UPDATE AND BASIC TOOLS
#############################################################################

echo ""
echo "=== Step 1: System Update ==="
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev git wget curl tmux htop build-essential

#############################################################################
# 2. INSTALL AZURE CLI (OPTIONAL BUT USEFUL)
#############################################################################

echo ""
echo "=== Step 2: Installing Azure CLI ==="
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash || echo "Azure CLI install failed or already installed."

#############################################################################
# 3. VERIFY GPU
#############################################################################

echo ""
echo "=== Step 3: Verifying GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ GPU detected via nvidia-smi"
else
    echo "⚠ nvidia-smi not found. Trying to install nvidia-utils-535..."
    sudo apt install -y nvidia-utils-535 || true
    sudo modprobe nvidia || true
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "✓ GPU detected after installing nvidia-utils"
    else
        echo "⚠ Still no nvidia-smi. Check Azure driver extension / VM config."
    fi
fi

#############################################################################
# 4. CREATE PROJECT STRUCTURE + VENV
#############################################################################

echo ""
echo "=== Step 4: Creating Project Structure ==="
mkdir -p ~/npc-training
cd ~/npc-training

# Use a dedicated venv name for Unsloth
python3 -m venv venv_unsloth
source venv_unsloth/bin/activate

# Upgrade pip tooling
pip install --upgrade pip wheel setuptools

#############################################################################
# 5. LOGIN TO AZURE AND DOWNLOAD DATA (NON-FATAL IF FAILS)
#############################################################################

echo ""
echo "=== Step 5: Azure Login & Data Download ==="
echo "If you lack Storage Blob RBAC, these steps may fail gracefully."

# Try login for blob download
if az login --use-device-code; then
    mkdir -p data_engineering/outputs
    mkdir -p data_engineering/config
    mkdir -p fine_tuning

    echo ""
    echo "=== Downloading Training Data (JSONL) ==="
    az storage blob download-batch \
      --account-name "$STORAGE_ACCOUNT" \
      --source "$CONTAINER_NAME" \
      --destination ./data_engineering/outputs \
      --pattern "*.jsonl" \
      --auth-mode login \
      || echo "⚠ JSONL download failed (check RBAC or container). You can copy data via scp instead."

    echo ""
    echo "=== Downloading Training Config (if present) ==="
    if az storage blob show \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER_NAME" \
        --name config/training.yaml >/dev/null 2>&1; then
        az storage blob download \
          --account-name "$STORAGE_ACCOUNT" \
          --container-name "$CONTAINER_NAME" \
          --name config/training.yaml \
          --file data_engineering/config/training.yaml \
          --auth-mode login \
          || echo "⚠ Config download failed; copy config manually if needed."
    else
        echo "No config/training.yaml blob found; skipping."
    fi

    echo ""
    echo "✓ Data download phase finished (or skipped if errors above)"
    ls -lh data_engineering/outputs/ || echo "No outputs directory yet."
else
    echo "⚠ az login failed; skipping blob downloads. Copy data/config via scp."
fi

#############################################################################
# 6. INSTALL PYTORCH / HF STACK / UNSLOTH (UNSLOTH-FIRST)
#############################################################################

echo ""
echo "=== Step 6: Installing PyTorch & Dependencies (Unsloth-first) ==="
source venv_unsloth/bin/activate

# IMPORTANT: Install Unsloth FIRST so it pins torch/transformers/trl correctly.
pip install unsloth

# Extra libraries that don't fight Unsloth's versions
pip install datasets
pip install pyyaml tqdm pandas orjson jsonschema requests
pip install selectolax python-slugify

# Optional: xformers (if Unsloth hasn't already installed a compatible one)
pip install xformers || echo "xformers install failed or already satisfied; continuing."

echo ""
echo "=== Verifying PyTorch GPU Access ==="
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF

#############################################################################
# 7. CREATE TRAINING LAUNCH SCRIPT
#############################################################################

echo ""
echo "=== Step 7: Creating Training Launch Script ==="

cat > ~/npc-training/run_training.sh << 'EOFTRAIN'
#!/bin/bash
set -e

cd ~/npc-training
source venv_unsloth/bin/activate

# Optional: Set Weights & Biases API key
# export WANDB_API_KEY="your_key_here"

echo "=========================================="
echo "Starting NPC Dialogue Fine-Tuning"
echo "=========================================="
echo "Start time: $(date)"
echo ""

echo "GPU Status:"
nvidia-smi || true
echo ""

cd fine_tuning

python train.py 2>&1 | tee "training_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
EOFTRAIN

chmod +x ~/npc-training/run_training.sh

#############################################################################
# 8. CREATE MONITORING SCRIPT
#############################################################################

echo ""
echo "=== Step 8: Creating Monitoring Script ==="

cat > ~/npc-training/monitor_training.sh << 'EOFMON'
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
if ls ~/npc-training/fine_tuning/training_log_*.txt 1> /dev/null 2>&1; then
    tail -20 "$(ls -t ~/npc-training/fine_tuning/training_log_*.txt | head -1)"
else
    echo "No training logs found yet"
fi
EOFMON

chmod +x ~/npc-training/monitor_training.sh

#############################################################################
# SETUP COMPLETE
#############################################################################

echo ""
echo "=========================================="
echo "✓ VM Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. If data/config download failed earlier, copy them manually from your laptop:"
echo "   scp -i ~/.ssh/azure_npc_key -r <LOCAL_PROJECT_PATH>/data_engineering azureuser@__VM_IP__:~/npc-training/"
echo "   scp -i ~/.ssh/azure_npc_key -r <LOCAL_PROJECT_PATH>/fine_tuning      azureuser@__VM_IP__:~/npc-training/"
echo ""
echo "2. Start training in tmux:"
echo "   tmux new -s training"
echo "   cd ~/npc-training"
echo "   ./run_training.sh"
echo ""
echo "3. Detach from tmux: Ctrl+B, then D"
echo "4. Reattach later:   tmux attach -t training"
echo ""
echo "5. Monitor training:"
echo "   ./monitor_training.sh"
echo ""
echo "=========================================="
EOFVM

# Fill in placeholders in vm_setup.sh
sed -i '' "s/__STORAGE_ACCOUNT__/$STORAGE_ACCOUNT/g" "$HOME/vm_setup.sh"
sed -i '' "s/__CONTAINER_NAME__/$CONTAINER_NAME/g" "$HOME/vm_setup.sh"
sed -i '' "s/__VM_IP__/$VM_IP/g" "$HOME/vm_setup.sh"

# Make VM setup script executable locally (after scp, chmod again on VM)
chmod +x "$HOME/vm_setup.sh"

echo "✓ VM setup script created: $HOME/vm_setup.sh"

#############################################################################
# FINAL SUMMARY
#############################################################################

echo ""
echo "=========================================="
echo "✓✓✓ FULL VM SETUP COMPLETE! ✓✓✓"
echo "=========================================="
echo ""
echo "VM DETAILS:"
echo "  Resource Group : $RESOURCE_GROUP"
echo "  VM Name        : $VM_NAME"
echo "  VM Size        : $VM_SIZE"
echo "  VM IP          : $VM_IP"
echo ""
echo "LOCAL FILES:"
echo "  Env helpers    : $HOME/azure_npc_env.sh"
echo "  VM setup script: $HOME/vm_setup.sh"
echo "  SSH key        : $SSH_KEY_PATH"
echo ""
echo "NEXT STEPS (from your laptop):"
echo "1) Wait ~10 minutes for NVIDIA driver extension to fully settle."
echo "2) SSH into the VM:"
echo "     ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
echo "3) Copy vm_setup.sh to the VM and run it:"
echo "     scp -i $SSH_KEY_PATH $HOME/vm_setup.sh $ADMIN_USERNAME@$VM_IP:~/"
echo "     ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
echo "     chmod +x ~/vm_setup.sh"
echo "     ./vm_setup.sh"
echo "4) Upload your current project code + data into ~/npc-training/:"
echo "     scp -i $SSH_KEY_PATH -r $PROJECT_DIR/fine_tuning      $ADMIN_USERNAME@$VM_IP:~/npc-training/"
echo "     scp -i $SSH_KEY_PATH -r $PROJECT_DIR/data_engineering $ADMIN_USERNAME@$VM_IP:~/npc-training/"
echo "   (Adjust if your local paths differ.)"
echo "5) Start training via tmux:"
echo "     ssh -i $SSH_KEY_PATH $ADMIN_USERNAME@$VM_IP"
echo "     tmux new -s training"
echo "     cd ~/npc-training"
echo "     ./run_training.sh"
echo ""
echo "To manage costs, stop the VM when idle:"
echo "  source $HOME/azure_npc_env.sh"
echo "  vm-stop"
echo "=========================================="