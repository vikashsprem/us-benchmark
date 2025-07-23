import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import glob
from tqdm import tqdm
import torch
from PIL import Image
import sys
import re
import traceback

# Add the project root to the Python path to allow proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import from the local modules
from grading import table_similarity
from parsing import parse_unsiloedvl_response, parse_llama_response
from convert import html_to_numpy

# Configure paths
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "_images")
GROUNDTRUTH_DIR = os.path.join(os.path.dirname(__file__), "_groundtruth")
PREDICTED_DIR = os.path.join(os.path.dirname(__file__), "_predicted")

# Parsers for different response formats
PARSERS = {
    "qwen2vl": parse_unsiloedvl_response,
    "llama_predictions": parse_llama_response,
    "llama_base_predictions": parse_llama_response,
    "llama_finetuned_predictions": parse_llama_response,
}


def ensure_dir(dir_path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(dir_path, exist_ok=True)


def load_groundtruth(image_id: str) -> Optional[np.ndarray]:
    """Load ground truth table for an image."""
    gt_path = os.path.join(GROUNDTRUTH_DIR, f"{image_id}.html")
    if not os.path.exists(gt_path):
        print(f"Warning: No ground truth found for {image_id}")
        return None

    with open(gt_path, "r") as f:
        html_content = f.read()

    try:
        return html_to_numpy(html_content)
    except Exception as e:
        print(f"Error converting ground truth HTML to numpy for {image_id}: {e}")
        return None


def extract_html_table(text: str) -> str:
    """Extract HTML table from model output text."""
    # Try to extract an HTML table with regex
    table_pattern = r"<table>.*?</table>"
    table_match = re.search(table_pattern, text, re.DOTALL)

    if table_match:
        return table_match.group(0)

    # If no table found, try to extract markdown table and convert it
    if "|" in text and "-|-" in text:
        lines = [
            line.strip() for line in text.split("\n") if line.strip() and "|" in line
        ]
        if len(lines) > 1:
            html_content = "<table>"
            for i, line in enumerate(lines):
                # Skip separator line (contains only | and -)
                if i == 1 and all(c in "|-:" for c in line):
                    continue

                cells = [cell.strip() for cell in line.split("|")]
                cells = [cell for cell in cells if cell]  # Remove empty cells

                if cells:
                    tag = "th" if i == 0 else "td"
                    html_content += (
                        "<tr>"
                        + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
                        + "</tr>"
                    )

            html_content += "</table>"
            return html_content

    return ""


def run_unsiloedvl_table_inference() -> None:
    """Run inference with unsiloedvl-table model on all images."""
    print("Starting unsiloedvl-table inference process...")
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
    except ImportError:
        print("Error: Required packages not found. Please install with:")
        print("pip install transformers torch peft")
        return

    # Create output directory
    output_dir = os.path.join(PREDICTED_DIR, "unsiloedvl-table")
    ensure_dir(output_dir)
    print(f"Created output directory: {output_dir}")

    # Configuration for Qwen2VL
    BASE_MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"
    ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "models", "checkpoint-high_capacity-continued", "checkpoint-1000")
    
    # Debug path resolution
    print(f"Current working directory: {os.getcwd()}")
    print(f"Relative ADAPTER_PATH: {ADAPTER_PATH}")
    print(f"Absolute ADAPTER_PATH: {os.path.abspath(ADAPTER_PATH)}")
    print(f"Path exists check: {os.path.exists(ADAPTER_PATH)}")
    
    MAX_PIXELS_LIMIT = 256 * 28 * 28
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.2
    MIN_P = 0.1
    INSTRUCTION = "Extract the table from this image and return it as an HTML table with <table>, <tr>, <td>, and <th> tags. Do not include any other text, just the HTML table."

    # print(f"Using base model: {BASE_MODEL_PATH}")
    # print(f"Using adapter path: {ADAPTER_PATH}")

    # Determine the best available device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load processor and model
    print("Loading unsiloedvl-table model and processor...")
    # Load processor from base model (has complete config)
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        max_pixels=MAX_PIXELS_LIMIT,
        use_fast=False,
    )
    print("Processor loaded successfully")

    if os.path.exists(ADAPTER_PATH):
        print(f"Loading base model and adapter from {ADAPTER_PATH}...")
        # Load base model and adapter
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
        print("Base model loaded successfully")

        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.merge_and_unload()
        print("Model with adapter merged successfully")
    else:
        print(f"Loading base model only from {BASE_MODEL_PATH}...")
        # Load just the base model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
        print("Base model loaded successfully")

    # Get all image files
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")))
    print(f"Found {len(image_files)} images to process")

    # Process each image
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in tqdm(sorted(image_files), desc="Running unsiloedvl-table inference"):
        image_id = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_id}.json")

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {image_id} - already processed")
            skipped_count += 1
            continue

        print(f"Processing image: {image_id}")
        try:
            # Load and process image
            print(f"  Loading image from {image_path}")
            image = Image.open(image_path).convert("RGB")

            # Prepare model inputs
            print("  Preparing model inputs")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": INSTRUCTION},
                    ],
                }
            ]
            input_text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[input_text],
                images=[image],
                return_tensors="pt",
                truncation=True,
            ).to(device)

            # Generate response
            print("  Generating response")
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                temperature=TEMPERATURE,
                min_p=MIN_P,
            )

            # Process the output
            print("  Processing output")
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"  Generated text length: {len(generated_text)} characters")

            # Find the start of the actual response after the prompt/assistant marker using regex
            response_start_match = re.search(r"assistant\s*\n", generated_text, re.IGNORECASE)
            
            if response_start_match:
                # If marker found, process text AFTER the marker
                model_response_text = generated_text[response_start_match.end():].strip()
                print(f"  Found 'assistant' marker, processing subsequent text (length: {len(model_response_text)}).")
            else:
                # Fallback if marker not found (use the whole text)
                model_response_text = generated_text
                print("  'assistant' marker not found, processing full generated text.")
            
            # Extract the HTML table from the potentially sliced response text
            html_table = extract_html_table(model_response_text)
            print(f"  Extracted HTML table length: {len(html_table)} characters")

            # Save as JSON - include HTML table and debug info
            result = {
                "html_table": html_table,
                "image_id": image_id,
                "debug": {
                    "full_generated_text": generated_text,
                    "model_response_after_assistant": model_response_text,
                    "generated_length": len(generated_text),
                    "extracted_length": len(html_table)
                }
            }

            print(f"  Saving results to {output_path}")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            processed_count += 1
            print(f"  Successfully processed {image_id}")

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            error_count += 1
            # Save error information
            with open(output_path, "w") as f:
                json.dump({"error": str(e), "image_id": image_id}, f, indent=2)

    print("\nUnsiloedVL Inference Summary:")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Errors: {error_count}")


def evaluate_model(model_name: str) -> Dict[str, Any]:
    """Evaluate a model's table extraction performance."""
    ensure_dir(os.path.join(PREDICTED_DIR, model_name))

    results = {
        "model": model_name,
        "scores": [],
        "image_ids": [],
        "count": 0,
        "success_count": 0,
        "avg_similarity": 0.0,
    }

    parser = PARSERS.get(model_name)
    if not parser:
        print(f"Error: No parser found for model {model_name}")
        return results

    # Get all image files
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")))
    print(f"Found {len(image_files)} images to evaluate for {model_name}")

    try:
        for image_path in tqdm(sorted(image_files), desc=f"Evaluating {model_name}"):
            image_id = Path(image_path).stem
            results["image_ids"].append(image_id)
            results["count"] += 1

            try:
                # Load prediction
                pred_path = os.path.join(PREDICTED_DIR, model_name, f"{image_id}.json")
                print(f"Processing prediction for {image_id} from {pred_path}")

                if not os.path.exists(pred_path):
                    print(f"  Warning: No prediction file found at {pred_path}")
                    results["scores"].append(0.0)
                    continue

                # Initialize score for this image
                current_score = 0.0
                raw_data = {} # Initialize raw_data

                # Parse prediction file
                try:
                    html_table, raw_data = parser(pred_path)
                    if not raw_data: # Ensure raw_data is a dict if parser returns None
                        raw_data = {}
                except Exception as parse_err:
                    print(f"  Error parsing prediction file {pred_path}: {parse_err}")
                    results["scores"].append(0.0)
                    # Write error back to json if possible? Or just continue? Let's continue for now.
                    continue

                # Skip if no prediction is available
                if html_table is None:
                    print(f"  No HTML table found in prediction for {image_id}")
                    results["scores"].append(0.0)
                    continue

                if not html_table.strip():
                    print(f"  Empty HTML table for {image_id}")
                    results["scores"].append(0.0)
                    continue

                print(f"  HTML table length: {len(html_table)} characters")

                # Convert HTML to numpy array
                try:
                    pred_table = html_to_numpy(html_table)
                    print(f"  Prediction table shape: {pred_table.shape}")
                except Exception as e:
                    print(
                        f"  Error converting prediction HTML to numpy for {image_id}: {e}"
                    )
                    print(f"  HTML content: {html_table[:100]}...")
                    traceback.print_exc()
                    results["scores"].append(0.0)
                    continue

                # Load ground truth
                gt_table = load_groundtruth(image_id)
                if gt_table is None:
                    print(f"  No ground truth found for {image_id}")
                    results["scores"].append(0.0)
                    continue

                print(f"  Ground truth table shape: {gt_table.shape}")

                # Calculate similarity
                try:
                    similarity = table_similarity(gt_table, pred_table)
                    print(f"  Similarity score: {similarity}")
                    # Convert numpy.float32 to Python float for JSON serialization
                    current_score = float(similarity)
                    results["scores"].append(current_score)
                    results["success_count"] += 1
                except Exception as e:
                    print(f"  Error calculating similarity for {image_id}: {e}")
                    traceback.print_exc()
                    results["scores"].append(0.0) # Append 0.0 score on calculation error

                # Add score to the original prediction data and save it back
                if isinstance(raw_data, dict): # Ensure raw_data is a dictionary before modifying
                    try:
                        raw_data["similarity_score"] = current_score
                        with open(pred_path, "w") as f:
                            json.dump(raw_data, f, indent=2)
                        print(f"  Successfully updated prediction file {pred_path} with score: {current_score}")
                    except TypeError as te:
                        print(f"  Error: Data is not JSON serializable when writing score to {pred_path}: {te}")
                        print(f"  Problematic data structure type: {type(raw_data)}")
                    except IOError as ioe:
                        print(f"  Error writing updated prediction file {pred_path}: {ioe}")
                    except Exception as e:
                        print(f"  Unexpected error updating prediction file {pred_path} with score: {e}")
                        traceback.print_exc()
                else:
                    print(f"  Error: Could not add score to {pred_path}. Parsed data was not a dictionary (type: {type(raw_data)}).")
                    # Optionally, still append the score to the main results list if needed
                    # results["scores"].append(current_score) # Already appended above where score is calculated

            except Exception as e:
                print(f"Error evaluating {image_id}: {e}")
                traceback.print_exc()
                # Ensure score is appended even in outer exception
                if image_id not in results["image_ids"] or len(results["scores"]) < results["count"]:
                     results["scores"].append(0.0)

        # Calculate average similarity - ensure all values are Python floats
        if results["count"] > 0: # Use results["count"] instead of results["success_count"] for average
            valid_scores = [s for s in results["scores"] if isinstance(s, (int, float))] # Filter out potential non-numeric entries if any error occurs
            results["avg_similarity"] = float(sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0

        # Save results immediately after completion to avoid corruption
        with open(os.path.join(PREDICTED_DIR, f"{model_name}_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved evaluation results for {model_name}")

    except Exception as e:
        print(f"Error during evaluation of {model_name}: {e}")
        traceback.print_exc()
        # Save partial results
        with open(os.path.join(PREDICTED_DIR, f"{model_name}_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


def evaluate_all_models(run_inference: bool = True) -> Dict[str, Dict[str, Any]]:
    """Evaluate all models and return results."""
    ensure_dir(PREDICTED_DIR)

    # Run Qwen2VL inference if requested
    if run_inference:
        try:
            run_unsiloedvl_table_inference()
        except Exception as e:
            print(f"Error running unsiloedvl-table inference: {e}")
            traceback.print_exc()

    all_results = {}
    for model_name in PARSERS.keys():
        # Check if model has predictions
        model_dir = os.path.join(PREDICTED_DIR, model_name)
        if not os.path.exists(model_dir):
            print(f"Skipping {model_name} - directory not found")
            continue

        if not os.path.isdir(model_dir) or not any(os.listdir(model_dir)):
            print(f"Skipping {model_name} - no predictions found")
            continue

        print(f"Evaluating {model_name}...")
        model_results = evaluate_model(model_name)
        all_results[model_name] = model_results

    # Save summary results
    summary = {}
    for model, results in all_results.items():
        summary[model] = {
            "avg_similarity": float(results["avg_similarity"]),
            "success_rate": float(
                results["success_count"] / results["count"]
                if results["count"] > 0
                else 0
            ),
            "count": results["count"],
        }

    with open(os.path.join(PREDICTED_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate table extraction models")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference with Qwen2VL, skip evaluation",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip running inference with Qwen2VL, only evaluate existing predictions",
    )
    parser.add_argument("--model", type=str, help="Evaluate only a specific model")
    args = parser.parse_args()

    try:
        if args.inference_only:
            # Only run inference
            run_unsiloedvl_table_inference()
        elif args.model:
            # Evaluate a specific model
            if args.model == "unsiloedvl-table" and not args.skip_inference:
                run_unsiloedvl_table_inference()
            results = {args.model: evaluate_model(args.model)}

            # Print summary
            print("\nEvaluation Summary:")
            print("-" * 80)
            print(
                f"{'Model':<15} | {'Avg Similarity':<15} | {'Success Rate':<15} | {'Count':<10}"
            )
            print("-" * 80)

            for model, data in results.items():
                avg_sim = float(data["avg_similarity"])
                success_rate = float(
                    data["success_count"] / data["count"] if data["count"] > 0 else 0
                )
                print(
                    f"{model:<15} | {avg_sim:<15.4f} | {success_rate:<15.4f} | {data['count']:<10}"
                )
        else:
            # Evaluate all models
            results = evaluate_all_models(run_inference=not args.skip_inference)

            # Print summary
            print("\nEvaluation Summary:")
            print("-" * 80)
            print(
                f"{'Model':<15} | {'Avg Similarity':<15} | {'Success Rate':<15} | {'Count':<10}"
            )
            print("-" * 80)

            for model, data in results.items():
                avg_sim = float(data["avg_similarity"])
                success_rate = float(
                    data["success_count"] / data["count"] if data["count"] > 0 else 0
                )
                print(
                    f"{model:<15} | {avg_sim:<15.4f} | {success_rate:<15.4f} | {data['count']:<10}"
                )
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
