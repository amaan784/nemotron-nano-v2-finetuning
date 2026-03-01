#!/bin/bash
# ============================================================
# Agentic World -- Brev Instance Setup
# ============================================================
# Sets up a Brev GPU instance for fine-tuning Mistral Nemo 12B
# with Unsloth + QLoRA.
#
# Track: W&B Fine-Tuning (Mistral Worldwide Hackathon)
# Model: Mistral Nemo 12B Instruct (standard Transformer)
# GPU:   A10G (24GB) recommended, A100 (40GB) optimal
#
# Usage:
#   chmod +x setup_brev.sh && ./setup_brev.sh
#
# Note: No mamba_ssm or causal_conv1d needed!
#       Mistral Nemo is a standard Transformer architecture.
# ============================================================

set -eo pipefail  # Exit on any error, including in pipelines

echo "============================================"
echo "AGENTIC WORLD -- BREV SETUP"
echo "============================================"
echo "Model: Mistral Nemo 12B (Standard Transformer)"
echo "Method: QLoRA via Unsloth"
echo ""

# ============================================================
# 1. VERIFY GPU
# ============================================================

echo "[CHECK] Checking GPU..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "[ERROR] No GPU detected! Make sure you are on a GPU instance."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "[OK] GPU: $GPU_NAME ($GPU_MEM)"
echo ""

# ============================================================
# 2. ENSURE PIP IS AVAILABLE
# ============================================================

echo "[CHECK] Ensuring pip is available..."
if ! python -m pip --version > /dev/null 2>&1; then
    echo "[FIX] pip not found in current Python environment, bootstrapping..."
    python -m ensurepip --upgrade || curl -sS https://bootstrap.pypa.io/get-pip.py | python
fi
python -m pip install --upgrade pip
echo "[OK] pip ready: $(python -m pip --version)"
echo ""

# ============================================================
# 3. INSTALL UNSLOTH
# ============================================================

echo "[INSTALL] Installing Unsloth (handles torch, transformers, bitsandbytes)..."
python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify unsloth
python -c "from unsloth import FastLanguageModel; print('[OK] Unsloth installed')"

# ============================================================
# 4. INSTALL TRAINING DEPENDENCIES
# ============================================================

echo ""
echo "[INSTALL] Installing training dependencies..."
python -m pip install wandb datasets trl peft accelerate

# ============================================================
# 5. LOGIN TO W&B
# ============================================================

echo ""
echo "[AUTH] Logging into Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "[OK] W&B logged in via env var"
else
    echo "[WARN] WANDB_API_KEY not set. Run manually:"
    echo "   wandb login"
    echo "   (Get key from https://wandb.ai/settings)"
fi

# ============================================================
# 6. LOGIN TO HUGGING FACE
# ============================================================

echo ""
echo "[AUTH] Logging into Hugging Face..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "[OK] HF logged in via env var"
else
    echo "[WARN] HF_TOKEN not set. Run manually:"
    echo "   huggingface-cli login"
    echo "   (Get token from https://huggingface.co/settings/tokens)"
fi

# ============================================================
# 7. VERIFY EVERYTHING
# ============================================================

echo ""
echo "[CHECK] Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'  BF16 support: {torch.cuda.is_bf16_supported()}')

import transformers
print(f'  Transformers: {transformers.__version__}')

import trl
print(f'  TRL: {trl.__version__}')

import peft
print(f'  PEFT: {peft.__version__}')

try:
    import wandb
    print(f'  W&B: {wandb.__version__}')
except ImportError:
    print('  W&B: not installed')

from unsloth import FastLanguageModel
print(f'  Unsloth: ready')

print()
print('[OK] All dependencies verified!')
print('     No mamba_ssm needed -- Mistral Nemo is pure Transformer')
"

# ============================================================
# 8. DONE
# ============================================================

echo ""
echo "============================================"
echo "[OK] SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Upload train.jsonl and eval.jsonl"
echo "  2. Run: python finetune.py"
echo ""
echo "Quick start:"
echo "  python finetune.py"
echo ""
echo "Fallback to Mistral 7B (if VRAM issues):"
echo "  MODEL=unsloth/mistral-7b-instruct-v0.3-bnb-4bit python finetune.py"
echo ""
echo "Push to HuggingFace after training:"
echo "  HF_REPO=mistral-hackathon-2026/agentic-world-lora python finetune.py"
echo ""