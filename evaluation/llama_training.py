#!/usr/bin/env python3
"""
Fine-tune LLaVA (LLaMA-based vision model) for table extraction with:
1. Unfrozen image layers (not just LoRA)
2. System prompt: "Parse the table"
"""

import os
import torch
from transformers import (
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Any
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """Custom Trainer that handles meta tensor issues and memory management"""
    
    def _move_model_to_device(self, model, device):
        """Override to handle meta tensor issues"""
        try:
            return super()._move_model_to_device(model, device)
        except NotImplementedError as e:
            if "meta tensor" in str(e):
                logger.info("Meta tensor detected, skipping device movement")
                return model
            else:
                raise e
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step with device-aware memory management"""
        # Only do CUDA memory management if using GPU
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            torch.cuda.empty_cache()
            
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            error_str = str(e).lower()
            if torch.cuda.is_available() and ("cuda" in error_str or "memory" in error_str or "nvml" in error_str):
                logger.warning(f"CUDA memory error during training step: {e}")
                if next(model.parameters()).is_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # Try one more time after cleanup
                try:
                    return super().training_step(model, inputs, num_items_in_batch)
                except RuntimeError as e2:
                    logger.error(f"Memory error persists after cleanup: {e2}")
                    logger.info("Consider using --no-unfreeze-vision or smaller batch size")
                    raise e2
            else:
                logger.error(f"Training step failed: {e}")
                raise e

class TableExtractionDataset:
    """Dataset for table extraction training with LLaMA-based vision model."""
    def __init__(self, image_dir, processor, max_length=2048):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        
        # Get all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_files = sorted(self.image_files)
        logger.info(f"Found {len(self.image_files)} training images")
        
        # System prompt for table extraction
        self.system_prompt = "Parse the table"
        
        # Define target size for all images
        self.target_height = 448
        self.target_width = 784
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Resize image to target dimensions
        image = image.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
        
        # Follow the official documentation pattern for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Parse the table."},
                ],
            },
        ]
        
        # Follow the exact documentation pattern for LLaVA Next
        # Step 1: Generate formatted text prompt from conversation
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Step 2: Process image and prompt using the documented parameter order
        # For LLaVA Next: processor(image, prompt, return_tensors="pt")
        inputs = self.processor(image, prompt, return_tensors="pt")
        
        # Debug: check the generated prompt and inputs
        if idx == 0:  # Only log for first item to avoid spam
            logger.info(f"Generated prompt: {prompt[:200]}...")
            logger.info(f"Input keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    logger.info(f"  {k}: type={type(v)}")
        
        # Remove batch dimension from apply_chat_template output for dataset
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == 1:
                # Remove the batch dimension added by apply_chat_template
                inputs[k] = v.squeeze(0)
        
        # Set up labels for training (for language modeling, labels = input_ids)
        inputs["labels"] = inputs["input_ids"].clone().detach().long()
        
        # Ensure proper tensor types
        inputs["input_ids"] = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()
        
        # Make sure pixel_values are properly handled
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].float()
        
        return inputs

class LlavaDataCollator:
    """Custom data collator for LLaVA training that properly handles multimodal inputs."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Separate different types of data
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        pixel_values = [f["pixel_values"] for f in features if "pixel_values" in f]
        
        # Pad sequences
        from transformers import DataCollatorForSeq2Seq
        collator = DataCollatorForSeq2Seq(
            self.processor.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Collate text data
        text_batch = collator([
            {"input_ids": input_ids[i], "attention_mask": attention_mask[i], "labels": labels[i]}
            for i in range(len(features))
        ])
        
        # Handle pixel values
        if pixel_values:
            # Stack pixel values if they exist
            text_batch["pixel_values"] = torch.stack(pixel_values)
        
        return text_batch

def unfreeze_vision_layers(model, layers_to_unfreeze=["vision_model"]):
    """
    Unfreeze vision encoder layers beyond just LoRA.
    With quantized models, only LoRA adapters can be trained.
    """
    unfrozen_params = 0
    total_vision_params = 0
    skipped_quantized = 0
    
    for name, param in model.named_parameters():
        # Check if this parameter belongs to vision components
        is_vision_param = any(layer_name in name for layer_name in layers_to_unfreeze)
        
        if is_vision_param:
            total_vision_params += param.numel()
            # Only unfreeze if parameter supports gradients (not quantized)
            if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                param.requires_grad = True
                unfrozen_params += param.numel()
                logger.info(f"Unfrozen vision parameter: {name}")
            else:
                # Skip quantized parameters
                skipped_quantized += param.numel()
                logger.debug(f"Skipped quantized vision parameter: {name} (dtype: {param.dtype})")
        elif 'lora_' in name:
            # Keep LoRA parameters trainable (these are always float)
            param.requires_grad = True
            unfrozen_params += param.numel()
        else:
            # Freeze other parameters (but check if they can be frozen)
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False
    
    logger.info(f"Unfrozen {unfrozen_params:,} parameters out of {total_vision_params:,} vision parameters")
    if skipped_quantized > 0:
        logger.info(f"Skipped {skipped_quantized:,} quantized vision parameters (training via LoRA only)")
    return unfrozen_params

def train_llama_table_extraction(
    base_model_name="llava-hf/llava-v1.6-vicuna-7b-hf",
    output_dir="models/llama-table-extraction",
    learning_rate=1e-4,
    max_steps=1000,
    batch_size=1,
    gradient_accumulation_steps=64,  # Ultra high for low-memory systems
    unfreeze_vision=True,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1
):
    """
    Train LLaMA-based vision model for table extraction.
    
    Args:
        base_model_name: HuggingFace model name for LLaVA
        output_dir: Directory to save the trained model
        learning_rate: Learning rate for training
        max_steps: Maximum training steps
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        unfreeze_vision: Whether to unfreeze vision encoder layers
        use_lora: Whether to use LoRA for efficiency
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    
    logger.info(f"Starting LLaMA-based table extraction training")
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Unfreeze vision layers: {unfreeze_vision}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load processor and model using AutoProcessor as shown in documentation
    logger.info("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(base_model_name)
    
    # Load model with aggressive memory optimizations
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Available GPU memory: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 2.0:
            logger.warning(f"GPU has only {gpu_memory_gb:.1f}GB memory - falling back to CPU training")
            device_map = "cpu"
            use_quantization = False
        else:
            device_map = "auto"
            use_quantization = True
    else:
        logger.info("No GPU available - using CPU")
        gpu_memory_gb = 0.0  # Set to 0 for CPU
        device_map = "cpu"
        use_quantization = False
    
    if use_quantization:
        # Use 4-bit quantization for larger GPU memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Use float16 for memory efficiency
            bnb_4bit_use_double_quant=True,
        )
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    else:
        # CPU training - no quantization needed
        logger.info("Loading model for CPU training...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
        )
    
    # Prepare model for training
    model.train()
    model.config.use_cache = False
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Apply LoRA if requested
    if use_lora:
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "v_proj",  # Focus on key attention modules for memory efficiency
            ]
        )
        model = get_peft_model(model, lora_config)
    
    # Unfreeze vision layers if requested and device has enough memory
    if unfreeze_vision:
        if device_map == "cpu":
            logger.warning("Skipping vision unfreezing for CPU training (memory optimization)")
            unfreeze_vision = False
        elif gpu_memory_gb < 4.0:
            logger.warning(f"Skipping vision unfreezing for low GPU memory ({gpu_memory_gb:.1f}GB)")
            unfreeze_vision = False
    
    if unfreeze_vision:
        logger.info("Unfreezing vision layers...")
        vision_layers = [
            "vision_tower",
            "vision_model", 
            "multi_modal_projector"
        ]
        unfrozen_params = unfreeze_vision_layers(model, vision_layers)
        logger.info(f"Unfrozen {unfrozen_params:,} vision parameters")
    else:
        logger.info("Vision layers remain frozen - training only LoRA adapters")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    image_dir = os.path.join(os.path.dirname(__file__), "_images")
    logger.info(f"Looking for images in: {image_dir}")
    
    train_dataset = TableExtractionDataset(
        image_dir=image_dir,
        processor=processor,
        max_length=512  # Reduced for memory efficiency with quantized model
    )
    
    # Training arguments - adjust based on device
    is_cpu_training = device_map == "cpu"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=max_steps // 10,
        logging_steps=5,  # More frequent logging for monitoring
        save_steps=max(50, max_steps // 2),  # Save more frequently for short runs
        save_total_limit=2,  # Reduce to save disk space
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        # Precision settings based on device
        bf16=False if is_cpu_training else use_quantization,  # BF16 only on GPU with quantization
        fp16=False if is_cpu_training else not use_quantization,  # FP16 on GPU without quantization
        gradient_checkpointing=is_cpu_training,  # Enable on CPU for memory savings
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=False,
        # CPU-specific optimizations
        eval_strategy="no",
    )
    
    # Simple data collator for multimodal data
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple collate function that handles both text and image data."""
        from transformers import DataCollatorForSeq2Seq
        
        # Use standard collator for text data
        text_collator = DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
            padding=True,
            return_tensors="pt",
            label_pad_token_id=-100,
        )
        
        # Separate text and image data
        text_features = [
            {k: v for k, v in f.items() if k != 'pixel_values'}
            for f in features
        ]
        
        # Collate text data
        batch = text_collator(text_features)
        
        # Handle pixel_values
        pixel_values = [f['pixel_values'] for f in features if 'pixel_values' in f]
        if pixel_values:
            # Simple stacking - all images should have the same shape after processing
            batch['pixel_values'] = torch.stack([pv.float() for pv in pixel_values])
        
        return batch
    
    # Create trainer with custom meta tensor handling
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    logger.info(f"Training completed! Model saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaMA-based vision model for table extraction")
    parser.add_argument("--model", default="llava-hf/llava-v1.6-vicuna-7b-hf", 
                       help="Base LLaVA model name")
    parser.add_argument("--output-dir", default="models/llama-table-extraction", 
                       help="Output directory for trained model")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=1000, 
                       help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1, 
                       help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--no-unfreeze-vision", action="store_true",
                       help="Don't unfreeze vision encoder layers")
    parser.add_argument("--no-lora", action="store_true",
                       help="Don't use LoRA (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, 
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, 
                       help="LoRA dropout")
    
    args = parser.parse_args()
    
    # Run training
    output_path = train_llama_table_extraction(
        base_model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        unfreeze_vision=not args.no_unfreeze_vision,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    logger.info(f"Model saved to: {output_path}") 