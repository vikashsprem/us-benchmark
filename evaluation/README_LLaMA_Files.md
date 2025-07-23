# 🦙 Essential LLaMA Training Files

This directory now contains only the essential files needed for LLaMA-based table extraction training.

## 📁 **Core Training Files**

### 🚀 **Training Scripts**
- **`llama_training.py`** - Main LLaMA training script with unfrozen vision layers and system prompt
- **`ec2_training_script.py`** - EC2-optimized training script with GPU support and monitoring
- **`test_llama_base.py`** - Test script to evaluate base LLaMA model performance

### 🔧 **Setup & Deployment**
- **`setup_ec2_llama.sh`** - EC2 environment setup script (installs dependencies, CUDA, etc.)
- **`upload_to_ec2.sh`** - Script to upload all essential files to EC2

### 🤖 **Inference & Evaluation**
- **`llama_inference.py`** - Run inference with trained LLaMA models
- **`evaluate_predictions.py`** - Evaluate model performance and generate metrics
- **`eval_helper.py`** - Evaluation utilities and helper functions

### 🔍 **Data Processing**
- **`parsing.py`** - Parse LLaMA model outputs and extract HTML tables
- **`grading.py`** - Grade table similarity and accuracy
- **`convert.py`** - Convert between different data formats

### 📊 **Data & Configuration**
- **`test_images.txt`** - List of test images for evaluation

## 🎯 **Key Features Implemented**

### ✅ **Your Requested Changes**
1. **🔓 Unfrozen Image Layers** - Vision encoder is fully unfrozen (not just LoRA)
2. **💬 System Prompt** - "Parse the table" system message integrated
3. **🦙 LLaMA-based** - Uses LLaVA (LLaMA + Vision) instead of Qwen

### 🚀 **EC2 Optimizations**
- GPU memory management
- Flash Attention 2 support
- 4-bit quantization
- System resource monitoring
- Automatic checkpoint saving

## 📋 **Files Removed (Not Needed for LLaMA)**

The following files were deleted as they were specific to other models or not needed:
- `run_specific_predictions.py` - Qwen-specific predictions
- `improved_training_config.py` - Qwen training config
- `continue_training.py` - Qwen continuation script
- `setup_ec2.sh` - Old EC2 setup
- `analyze_errors.py` - Error analysis for other models
- `check_eval.py` - Evaluation checks for other models

## 🚀 **Quick Start**

### Local Testing
```bash
# Test base LLaMA model
python evaluation/test_llama_base.py --num-images 5
```

### EC2 Training
```bash
# 1. Update upload script with your EC2 details
# 2. Upload files to EC2
./evaluation/upload_to_ec2.sh

# 3. SSH to EC2 and run setup
ssh -i your-key.pem ubuntu@your-ec2-ip
cd ~/us-tablebench
./setup_ec2_llama.sh

# 4. Start training
python evaluation/ec2_training_script.py
```

## 📊 **File Sizes & Purpose**

| File | Size | Purpose |
|------|------|---------|
| `llama_training.py` | 12KB | Main training logic |
| `ec2_training_script.py` | 8.7KB | EC2-optimized training |
| `llama_inference.py` | 11KB | Model inference |
| `evaluate_predictions.py` | 22KB | Performance evaluation |
| `eval_helper.py` | 21KB | Evaluation utilities |
| `grading.py` | 8.6KB | Table similarity scoring |
| `parsing.py` | 4.7KB | Output parsing |
| `convert.py` | 2.3KB | Format conversion |
| `test_llama_base.py` | 6.4KB | Base model testing |
| `setup_ec2_llama.sh` | 3.2KB | EC2 setup |
| `upload_to_ec2.sh` | 3.5KB | File upload script |

## ✅ **Ready for Training**

All essential files are now present and optimized for LLaMA-based table extraction training with your requested features:
- ✅ Unfrozen vision layers
- ✅ System prompt integration  
- ✅ EC2 GPU optimization
- ✅ Complete training pipeline 