#!/usr/bin/env python3
"""
Run inference with trained LLaMA-based table extraction model.
Supports both base LLaVA models and fine-tuned versions.
"""

import os
import json
import glob
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import re
import sys
import argparse
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_llama_inference(
    model_path="llava-hf/llava-v1.6-vicuna-7b-hf",
    output_dir="llama_predictions",
    images_dir="_images",
    max_new_tokens=2048,
    temperature=0.2,
    min_p=0.1,
    use_system_prompt=True,
    max_images=None
):
    """
    Run inference with LLaMA-based table extraction model.
    
    Args:
        model_path: Path to model (HuggingFace name or local path)
        output_dir: Directory to save predictions
        images_dir: Directory containing images to process
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        min_p: Minimum probability for sampling
        use_system_prompt: Whether to use "Parse the table" system prompt
        max_images: Maximum number of images to process (None for all)
    """
    
    try:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        from peft import PeftModel
    except ImportError:
        logger.error("Required packages not found. Please install: pip install transformers torch peft")
        return

    # Create output directory
    output_path = os.path.join(os.path.dirname(__file__), "_predicted", output_dir)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Created output directory: {output_path}")

    # System prompt for table extraction
    system_prompt = "Parse the table" if use_system_prompt else None
    
    # Instruction for table extraction
    instruction = "Extract the table from this image and return it as an HTML table with <table>, <tr>, <td>, and <th> tags. Do not include any other text, just the HTML table."

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load processor and model
    logger.info(f"Loading model and processor from: {model_path}")
    
    try:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Try loading base model and adapter separately
        try:
            logger.info("Trying to load as fine-tuned model with adapter...")
            base_model_name = "llava-hf/llava-v1.6-vicuna-7b-hf"
            processor = LlavaNextProcessor.from_pretrained(base_model_name)
            
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load adapter if it exists
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()
                logger.info("Fine-tuned adapter loaded and merged")
            else:
                model = base_model
                logger.info("Using base model")
                
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            return

    # Set model to evaluation mode
    model.eval()

    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"Found {len(image_files)} images to process")

    # Process each image
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Running LLaMA inference"):
        image_id = Path(image_path).stem
        output_file = os.path.join(output_path, f"{image_id}.json")

        # Skip if already processed
        if os.path.exists(output_file):
            logger.debug(f"Skipping {image_id} - already processed")
            skipped_count += 1
            continue

        logger.debug(f"Processing image: {image_id}")
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            logger.debug(f"Loaded image: {image.size}")

            # Prepare conversation
            if system_prompt:
                conversation = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction},
                        ],
                    }
                ]
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction},
                        ],
                    }
                ]

            # Apply chat template and process
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(prompt, image, return_tensors="pt").to(device)

            # Generate response
            logger.debug("Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    min_p=min_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

            # Decode response
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Generated text length: {len(generated_text)} characters")

            # Extract the assistant's response
            # Find the last "assistant" marker in the conversation
            assistant_start = generated_text.rfind("assistant")
            if assistant_start != -1:
                # Find the next newline after "assistant"
                response_start = generated_text.find("\n", assistant_start)
                if response_start != -1:
                    model_response = generated_text[response_start + 1:].strip()
                else:
                    model_response = generated_text[assistant_start + len("assistant"):].strip()
            else:
                model_response = generated_text

            # Extract HTML table if present
            table_pattern = r"<table>.*?</table>"
            table_match = re.search(table_pattern, model_response, re.DOTALL)
            
            if table_match:
                html_table = table_match.group(0)
                logger.debug(f"Extracted HTML table: {len(html_table)} characters")
            else:
                html_table = None
                logger.debug("No HTML table found in response")

            # Save results
            result = {
                "image_id": image_id,
                "image_path": image_path,
                "model_response": model_response,
                "html_table": html_table,
                "generated_text": generated_text,
                "system_prompt_used": system_prompt,
                "status": "success" if html_table else "no_table",
                "model_path": model_path
            }

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            processed_count += 1
            logger.debug(f"Successfully processed {image_id}")

        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            error_count += 1
            
            # Save error result
            error_result = {
                "image_id": image_id,
                "image_path": image_path,
                "error": str(e),
                "status": "error",
                "model_path": model_path
            }
            
            with open(output_file, 'w') as f:
                json.dump(error_result, f, indent=2)

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Results saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with LLaMA-based table extraction model")
    parser.add_argument("--model", default="llava-hf/llava-v1.6-vicuna-7b-hf",
                       help="Model path (HuggingFace name or local path)")
    parser.add_argument("--output-dir", default="llama_predictions",
                       help="Output directory name (under _predicted/)")
    parser.add_argument("--images-dir", default="_images",
                       help="Directory containing images to process")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature")
    parser.add_argument("--min-p", type=float, default=0.1,
                       help="Minimum probability for sampling")
    parser.add_argument("--no-system-prompt", action="store_true",
                       help="Don't use system prompt")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run inference
    output_path = run_llama_inference(
        model_path=args.model,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_p=args.min_p,
        use_system_prompt=not args.no_system_prompt,
        max_images=args.max_images
    )
    
    if output_path:
        logger.info(f"\n✅ Inference completed! Results saved to: {output_path}")
        logger.info(f"\n💡 To evaluate the results, run:")
        logger.info(f"   python evaluate_predictions.py --model {args.output_dir}")
    else:
        logger.error("\n❌ Inference failed!") 