#!/usr/bin/env python3
"""
Quick test script to evaluate the base LLaMA model (LLaVA) performance
on table extraction before fine-tuning.

This helps establish a baseline and verify the setup works.
"""

import os
import sys
import argparse
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llama_inference import run_llama_inference
from evaluate_predictions import evaluate_existing_predictions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llama_base_model(
    num_images=10,
    model_name="llava-hf/llava-v1.6-vicuna-7b-hf",
    output_dir="llama_base_test",
    evaluate=True
):
    """
    Test the base LLaMA model on a small set of images.
    
    Args:
        num_images: Number of images to test (default: 10)
        model_name: LLaVA model to test
        output_dir: Directory to save results
        evaluate: Whether to run evaluation after inference
    """
    
    logger.info("🦙 Testing Base LLaMA Model for Table Extraction")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Images: {num_images}")
    logger.info(f"Output: {output_dir}")
    
    # Step 1: Run inference on a small set of images
    logger.info("\n📸 Step 1: Running inference...")
    try:
        output_path = run_llama_inference(
            model_path=model_name,
            output_dir=output_dir,
            max_images=num_images,
            use_system_prompt=True
        )
        
        if not output_path:
            logger.error("❌ Inference failed!")
            return False
            
        logger.info(f"✅ Inference completed! Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        return False
    
    # Step 2: Evaluate results if requested
    if evaluate:
        logger.info("\n📊 Step 2: Evaluating results...")
        try:
            results = evaluate_existing_predictions(
                model_name=output_dir,
                verbose=False
            )
            
            if results:
                logger.info("\n📈 EVALUATION RESULTS:")
                logger.info("-" * 40)
                logger.info(f"Success Rate: {results.get('success_rate', 0):.1%}")
                logger.info(f"Average Similarity: {results.get('avg_similarity', 0):.3f}")
                logger.info(f"Total Images: {results.get('count', 0)}")
                logger.info(f"Successful Extractions: {results.get('success_count', 0)}")
                
                # Provide feedback on results
                success_rate = results.get('success_rate', 0)
                avg_similarity = results.get('avg_similarity', 0)
                
                if success_rate > 0.8 and avg_similarity > 0.6:
                    logger.info("\n🎉 Great! The base model is performing well.")
                elif success_rate > 0.6 and avg_similarity > 0.4:
                    logger.info("\n👍 Good baseline performance. Fine-tuning should improve this.")
                else:
                    logger.info("\n💡 Base model has room for improvement. Fine-tuning will help!")
                
            else:
                logger.warning("⚠️  Evaluation completed but no results returned")
                
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            return False
    
    # Step 3: Provide next steps
    logger.info("\n🚀 NEXT STEPS:")
    logger.info("-" * 40)
    logger.info("1. 📋 Review the results in the output directory")
    logger.info("2. 🏋️  Run fine-tuning to improve performance:")
    logger.info(f"   python evaluation/llama_training.py --max-steps 500")
    logger.info("3. 📊 Compare fine-tuned vs base model performance")
    logger.info("4. 📈 Run full evaluation on all images")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Test base LLaMA model for table extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 images
  python evaluation/test_llama_base.py --num-images 5
  
  # Test different model size
  python evaluation/test_llama_base.py --model llava-hf/llava-v1.6-vicuna-13b-hf
  
  # Just run inference without evaluation
  python evaluation/test_llama_base.py --no-evaluate
        """
    )
    
    parser.add_argument("--num-images", type=int, default=10,
                       help="Number of images to test (default: 10)")
    parser.add_argument("--model", default="llava-hf/llava-v1.6-vicuna-7b-hf",
                       help="LLaVA model to test")
    parser.add_argument("--output-dir", default="llama_base_test",
                       help="Output directory name")
    parser.add_argument("--no-evaluate", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if images directory exists
    images_dir = os.path.join(os.path.dirname(__file__), "_images")
    if not os.path.exists(images_dir):
        logger.error(f"❌ Images directory not found: {images_dir}")
        logger.error("Please ensure you have the dataset images in evaluation/_images/")
        return 1
    
    # Count available images
    import glob
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if len(image_files) == 0:
        logger.error("❌ No images found in the images directory!")
        return 1
    
    if len(image_files) < args.num_images:
        logger.warning(f"⚠️  Only {len(image_files)} images available, testing with all of them")
        args.num_images = len(image_files)
    
    # Run the test
    success = test_llama_base_model(
        num_images=args.num_images,
        model_name=args.model,
        output_dir=args.output_dir,
        evaluate=not args.no_evaluate
    )
    
    if success:
        logger.info("\n✅ Base model test completed successfully!")
        return 0
    else:
        logger.error("\n❌ Base model test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 