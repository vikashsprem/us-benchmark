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
    LlavaNextProcessor,
    TrainingArguments,
    Trainer
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
        
        # Prepare conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract the table from this image and return it as an HTML table with <table>, <tr>, <td>, and <th> tags. Do not include any other text, just the HTML table."},
                ],
            }
        ]
        
        # Process inputs with conversation template
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            prompt,
            image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        
        # Remove batch dimension and ensure proper tensor setup
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                inputs[k] = v.squeeze(0)
        
        # Set up labels for training
        inputs["labels"] = inputs["input_ids"].clone().detach().long()
        
        # Ensure proper tensor types
        inputs["input_ids"] = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()
        
        return inputs

def unfreeze_vision_layers(model, layers_to_unfreeze=["vision_model"]):
    """
    Unfreeze vision encoder layers beyond just LoRA.
    This allows full fine-tuning of the image understanding components.
    """
    unfrozen_params = 0
    total_vision_params = 0
    
    for name, param in model.named_parameters():
        # Check if this parameter belongs to vision components
        is_vision_param = any(layer_name in name for layer_name in layers_to_unfreeze)
        
        if is_vision_param:
            total_vision_params += param.numel()
            param.requires_grad = True
            unfrozen_params += param.numel()
            logger.info(f"Unfrozen vision parameter: {name}")
        elif 'lora_' in name:
            # Keep LoRA parameters trainable
            param.requires_grad = True
            unfrozen_params += param.numel()
        else:
            # Freeze other parameters
            param.requires_grad = False
    
    logger.info(f"Unfrozen {unfrozen_params:,} parameters out of {total_vision_params:,} vision parameters")
    return unfrozen_params

def train_llama_table_extraction(
            base_model_name="llava-hf/llava-v1.6-vicuna-7b-hf",
    output_dir="models/llama-table-extraction",
    learning_rate=1e-4,
    max_steps=1000,
    batch_size=1,
    gradient_accumulation_steps=8,
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
    
    # Load processor and model
    logger.info("Loading processor and model...")
    processor = LlavaNextProcessor.from_pretrained(base_model_name)
    
    # Load model with appropriate dtype
    model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for training
    model.train()
    model.config.use_cache = False
    
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        model = get_peft_model(model, lora_config)
    
    # Unfreeze vision layers if requested
    if unfreeze_vision:
        logger.info("Unfreezing vision layers...")
        vision_layers = [
            "vision_tower",
            "vision_model", 
            "multi_modal_projector"
        ]
        unfrozen_params = unfreeze_vision_layers(model, vision_layers)
        logger.info(f"Unfrozen {unfrozen_params:,} vision parameters")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    train_dataset = TableExtractionDataset(
        image_dir="_images",
        processor=processor,
        max_length=1024
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=max_steps // 10,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Custom data collator for multimodal data
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for training data."""
        batch = {}
        
        # Get max length in this batch
        max_length = max(len(f['input_ids']) for f in features)
        
        # Pad sequences to max length in batch
        batch['input_ids'] = torch.stack([
            torch.nn.functional.pad(
                f['input_ids'].long(),
                (0, max_length - len(f['input_ids'])),
                mode='constant',
                value=processor.tokenizer.pad_token_id
            ) for f in features
        ]).long()
        
        batch['attention_mask'] = torch.stack([
            torch.nn.functional.pad(
                f['attention_mask'].long(),
                (0, max_length - len(f['attention_mask'])),
                mode='constant',
                value=0
            ) for f in features
        ]).long()
        
        batch['labels'] = torch.stack([
            torch.nn.functional.pad(
                f['labels'].long(),
                (0, max_length - len(f['labels'])),
                mode='constant',
                value=-100
            ) for f in features
        ]).long()
        
        # Add pixel values
        batch['pixel_values'] = torch.stack([f['pixel_values'].float() for f in features])
        
        return batch
    
    # Create trainer
    trainer = Trainer(
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