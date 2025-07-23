#!/bin/bash
# EC2 Setup Script for LLaMA-based Table Extraction Training
# This script sets up an EC2 instance for training LLaVA models

set -e  # Exit on any error

echo "🚀 Setting up EC2 for LLaMA Table Extraction Training"
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and pip
echo "🐍 Installing Python and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# Install CUDA dependencies (for GPU training)
echo "🎮 Installing CUDA dependencies..."
sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-535

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y git wget curl unzip build-essential

# Create project directory
echo "📁 Setting up project directory..."
mkdir -p ~/us-tablebench
cd ~/us-tablebench

# Create virtual environment
echo "🔐 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install training dependencies
echo "📚 Installing training dependencies..."
pip install transformers peft accelerate bitsandbytes
pip install sentencepiece protobuf pillow tqdm
pip install wandb  # for experiment tracking
pip install flash-attn --no-build-isolation  # for faster training

# Install additional utilities
pip install datasets huggingface_hub

# Clone the repository (if not already present)
if [ ! -d ".git" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/your-repo/us-tablebench.git .
fi

# Create necessary directories
echo "📂 Creating necessary directories..."
mkdir -p evaluation/models
mkdir -p evaluation/_predicted
mkdir -p evaluation/_images
mkdir -p evaluation/_groundtruth

# Download a small test dataset (if needed)
echo "📊 Setting up test data..."
# You can add commands here to download your dataset

# Test GPU availability
echo "🎮 Testing GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test model loading
echo "🧪 Testing model loading..."
python3 -c "
from transformers import LlavaNextProcessor
try:
    processor = LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-vicuna-7b-hf', trust_remote_code=True)
    print('✅ Model loading test successful!')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"

echo "✅ EC2 setup complete!"
echo "=================================================="
echo "Next steps:"
echo "1. Upload your training data to ~/us-tablebench/evaluation/_images/"
echo "2. Upload your ground truth to ~/us-tablebench/evaluation/_groundtruth/"
echo "3. Run: cd ~/us-tablebench && source .venv/bin/activate"
echo "4. Run: python evaluation/llama_training.py"
echo "==================================================" 