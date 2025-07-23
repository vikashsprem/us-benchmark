#!/usr/bin/env python3
"""
EC2-optimized LLaMA training script for table extraction.
Optimized for GPU training with better memory management and monitoring.
"""

import os
import torch
import argparse
import logging
from pathlib import Path
import gc
import psutil
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ec2_environment():
    """Check if we're running on EC2 and have proper GPU setup."""
    logger.info("🔍 Checking EC2 environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"✅ GPU count: {gpu_count}")
        logger.info(f"✅ GPU memory: {gpu_memory:.1f} GB")
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        return True
    else:
        logger.warning("⚠️  No GPU detected. Training will be slow on CPU.")
        return False

def monitor_system_resources():
    """Monitor system resources during training."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logger.info(f"💻 CPU Usage: {cpu_percent}%")
    logger.info(f"💾 Memory Usage: {memory.percent}% ({memory.used / 1e9:.1f} GB / {memory.total / 1e9:.1f} GB)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU Memory: {gpu_memory:.1f} GB / {gpu_memory_total:.1f} GB")

def train_on_ec2(
    model_name="llava-hf/llava-v1.6-vicuna-7b-hf",
    output_dir="models/llama-table-extraction-ec2",
    learning_rate=1e-4,
    batch_size=1,
    num_epochs=3,
    max_length=2048,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    use_4bit=True,
    use_flash_attention=True,
    max_grad_norm=1.0,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    push_to_hub=False,
    hub_model_id=None,
    hub_token=None,
    resume_from_checkpoint=None,
    num_train_samples=None,
    test_run=False
):
    """
    Train LLaMA model on EC2 with optimized settings.
    """
    logger.info("🚀 Starting EC2-optimized LLaMA training...")
    
    # Check environment
    has_gpu = check_ec2_environment()
    
    # Import training modules
    from llama_training import (
        train_llama_table_extraction,
        TableExtractionDataset,
        unfreeze_vision_layers
    )
    
    # Set device
    device = "cuda" if has_gpu else "cpu"
    logger.info(f"🎯 Using device: {device}")
    
    # Optimize for EC2
    if has_gpu:
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
        # Set memory fraction
        torch.cuda.empty_cache()
    
    # Monitor initial resources
    logger.info("📊 Initial system resources:")
    monitor_system_resources()
    
    # Start training
    start_time = time.time()
    
    try:
        # Call the main training function with EC2 optimizations
        train_llama_table_extraction(
            base_model_name=model_name,
            output_dir=output_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            use_4bit=use_4bit,
            use_flash_attention=use_flash_attention,
            max_grad_norm=max_grad_norm,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=remove_unused_columns,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_token=hub_token,
            resume_from_checkpoint=resume_from_checkpoint,
            num_train_samples=num_train_samples,
            test_run=test_run
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        # Clean up GPU memory
        if has_gpu:
            torch.cuda.empty_cache()
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="EC2-optimized LLaMA training for table extraction")
    
    # Model arguments
    parser.add_argument("--model", default="llava-hf/llava-v1.6-vicuna-7b-hf",
                       help="Base LLaVA model name")
    parser.add_argument("--output-dir", default="models/llama-table-extraction-ec2",
                       help="Output directory for trained model")
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=50,
                       help="Log every N steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    
    # Optimization arguments
    parser.add_argument("--use-4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    parser.add_argument("--use-flash-attention", action="store_true", default=True,
                       help="Use Flash Attention 2")
    parser.add_argument("--dataloader-num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Hub arguments
    parser.add_argument("--push-to-hub", action="store_true",
                       help="Push model to Hugging Face Hub")
    parser.add_argument("--hub-model-id", type=str,
                       help="Hugging Face Hub model ID")
    parser.add_argument("--hub-token", type=str,
                       help="Hugging Face Hub token")
    
    # Other arguments
    parser.add_argument("--resume-from-checkpoint", type=str,
                       help="Resume training from checkpoint")
    parser.add_argument("--num-train-samples", type=int,
                       help="Number of training samples to use")
    parser.add_argument("--test-run", action="store_true",
                       help="Run a quick test with few samples")
    parser.add_argument("--remove-unused-columns", action="store_true", default=False,
                       help="Remove unused columns from dataset")
    
    args = parser.parse_args()
    
    # Start training
    train_on_ec2(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        use_4bit=args.use_4bit,
        use_flash_attention=args.use_flash_attention,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=args.remove_unused_columns,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_train_samples=args.num_train_samples,
        test_run=args.test_run
    )

if __name__ == "__main__":
    main() 