#!/usr/bin/env python3
"""
Comprehensive evaluation script for table extraction predictions.
Includes both quick progress checks and full similarity evaluation.
Can be run anytime to evaluate existing predictions without running inference.
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import glob
from tqdm import tqdm
import traceback

# Add the project root to the Python path to allow proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import functions from eval_helper
from grading import table_similarity
from parsing import parse_unsiloedvl_response
from convert import html_to_numpy

# Configure paths
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "_images")
GROUNDTRUTH_DIR = os.path.join(os.path.dirname(__file__), "_groundtruth")
PREDICTED_DIR = os.path.join(os.path.dirname(__file__), "_predicted")

def load_groundtruth(image_id: str):
    """Load ground truth table for an image."""
    gt_path = os.path.join(GROUNDTRUTH_DIR, f"{image_id}.html")
    if not os.path.exists(gt_path):
        return None

    with open(gt_path, "r") as f:
        html_content = f.read()

    try:
        return html_to_numpy(html_content)
    except Exception as e:
        print(f"Error converting ground truth HTML to numpy for {image_id}: {e}")
        return None

def quick_progress_check(model_name: str = "unsiloedvl-table") -> Dict[str, Any]:
    """Quick check of current progress without similarity calculation."""
    
    # Paths
    predicted_dir = os.path.join(PREDICTED_DIR, model_name)
    
    # Count total images
    total_images = len(glob.glob(os.path.join(IMAGES_DIR, "*.png"))) + \
                   len(glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))) + \
                   len(glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")))
    
    if not os.path.exists(predicted_dir):
        print("❌ No predictions directory found yet")
        return {"total": total_images, "processed": 0, "progress": 0}
    
    # Count prediction files
    pred_files = glob.glob(os.path.join(predicted_dir, "*.json"))
    
    if not pred_files:
        print("❌ No prediction files found yet")
        return {"total": total_images, "processed": 0, "progress": 0}
    
    # Quick stats
    processed = len(pred_files)
    progress = processed / total_images if total_images > 0 else 0
    
    # Sample some files for quick stats (up to 100 files for better accuracy)
    sample_size = min(100, len(pred_files))
    sample_files = pred_files[:sample_size]
    
    successful = 0
    empty = 0
    errors = 0
    
    for pred_file in sample_files:
        try:
            with open(pred_file, 'r') as f:
                data = json.load(f)
            
            if "error" in data:
                errors += 1
            elif not data.get("html_table", "").strip():
                empty += 1
            else:
                successful += 1
        except:
            errors += 1
    
    print(f"🚀 QUICK PROGRESS CHECK")
    print(f"{'='*60}")
    print(f"📊 Total Images: {total_images:,}")
    print(f"✅ Processed: {processed:,}")
    print(f"📈 Progress: {progress:.1%}")
    print(f"⏱️  Remaining: {total_images - processed:,}")
    
    if sample_files:
        print(f"\n📋 Sample Quality (from {sample_size} files):")
        print(f"  ✅ Successful: {successful} ({successful/sample_size:.1%})")
        print(f"  🟡 Empty: {empty} ({empty/sample_size:.1%})")
        print(f"  ❌ Errors: {errors} ({errors/sample_size:.1%})")
        
        # Estimate completion time if processing
        if processed > 10:  # Need some data for estimation
            recent_files = sorted(pred_files, key=lambda x: os.path.getmtime(x))
            if len(recent_files) >= 10:
                # Get time difference between first and last 10 files
                first_10_time = os.path.getmtime(recent_files[9])
                last_10_time = os.path.getmtime(recent_files[-1])
                
                if last_10_time > first_10_time:
                    files_per_second = (len(recent_files) - 10) / (last_10_time - first_10_time)
                    if files_per_second > 0:
                        remaining_seconds = (total_images - processed) / files_per_second
                        eta_hours = remaining_seconds / 3600
                        if eta_hours < 1:
                            print(f"⏰ Estimated completion: {remaining_seconds/60:.0f} minutes")
                        else:
                            print(f"⏰ Estimated completion: {eta_hours:.1f} hours")
    
    # Show most recent files (last 8)
    recent_files = sorted(pred_files, key=lambda x: os.path.getmtime(x), reverse=True)[:8]
    print(f"\n🕒 Most Recent Predictions:")
    for f in recent_files:
        filename = Path(f).stem
        mtime = os.path.getmtime(f)
        time_str = time.strftime("%H:%M:%S", time.localtime(mtime))
        
        # Quick check if successful
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            status = "✅" if data.get("html_table", "").strip() else ("❌" if "error" in data else "⚠️")
        except:
            status = "❌"
        
        print(f"  {status} {time_str} - {filename[:50]}...")
    
    return {
        "total": total_images,
        "processed": processed,
        "progress": progress,
        "successful": successful,
        "empty": empty,
        "errors": errors,
        "sample_size": sample_size
    }

def evaluate_existing_predictions(model_name: str = "unsiloedvl-table", verbose: bool = True, exclude_empty: bool = False, specific_image_ids: List[str] = None) -> Dict[str, Any]:
    """Evaluate existing predictions for similarity scores."""
    
    model_dir = os.path.join(PREDICTED_DIR, model_name)
    if not os.path.exists(model_dir):
        print(f"❌ No predictions directory found at {model_dir}")
        return {}
    
    # Get all prediction files
    all_pred_files = glob.glob(os.path.join(model_dir, "*.json"))
    
    if not all_pred_files:
        print(f"❌ No prediction files found in {model_dir}")
        return {}
    
    # Filter to specific image IDs if provided
    if specific_image_ids:
        pred_files = []
        for image_id in specific_image_ids:
            pred_file = os.path.join(model_dir, f"{image_id}.json")
            if os.path.exists(pred_file):
                pred_files.append(pred_file)
            else:
                if verbose:
                    print(f"⚠️  Warning: No prediction found for {image_id}")
        
        if not pred_files:
            print(f"❌ No prediction files found for specified image IDs")
            return {}
        
        print(f"📊 Found {len(pred_files)} prediction files to evaluate (filtered from {len(all_pred_files)} total)")
        if verbose:
            print(f"🎯 Evaluating specific images: {len(specific_image_ids)} requested, {len(pred_files)} found")
    else:
        pred_files = all_pred_files
        print(f"📊 Found {len(pred_files)} prediction files to evaluate")
    
    results = {
        "model": model_name,
        "scores": [],
        "image_ids": [],
        "details": [],
        "count": 0,
        "success_count": 0,
        "empty_count": 0,
        "error_count": 0,
        "avg_similarity": 0.0,
    }
    
    parser = parse_unsiloedvl_response
    
    for pred_path in tqdm(sorted(pred_files), desc=f"Evaluating {model_name} predictions"):
        image_id = Path(pred_path).stem
        results["image_ids"].append(image_id)
        results["count"] += 1
        
        try:
            # Load prediction file
            with open(pred_path, 'r') as f:
                pred_data = json.load(f)
            
            # Check if it's an error file
            if "error" in pred_data:
                results["scores"].append(0.0)
                results["error_count"] += 1
                results["details"].append({
                    "image_id": image_id,
                    "score": 0.0,
                    "status": "error",
                    "error": pred_data["error"]
                })
                if verbose:
                    print(f"  ❌ {image_id}: ERROR - {pred_data['error']}")
                continue
            
            # Get HTML table
            html_table = pred_data.get("html_table", "")
            
            # Check if empty
            if not html_table or not html_table.strip():
                results["scores"].append(0.0)
                results["empty_count"] += 1
                results["details"].append({
                    "image_id": image_id,
                    "score": 0.0,
                    "status": "empty",
                    "debug_info": pred_data.get("debug", {})
                })
                if verbose:
                    print(f"  ⚠️  {image_id}: EMPTY TABLE")
                continue
            
            # Convert HTML to numpy array
            try:
                pred_table = html_to_numpy(html_table)
            except Exception as e:
                results["scores"].append(0.0)
                results["error_count"] += 1
                results["details"].append({
                    "image_id": image_id,
                    "score": 0.0,
                    "status": "parse_error",
                    "error": str(e)
                })
                if verbose:
                    print(f"  ❌ {image_id}: HTML PARSE ERROR - {e}")
                continue
            
            # Load ground truth
            gt_table = load_groundtruth(image_id)
            if gt_table is None:
                results["scores"].append(0.0)
                results["error_count"] += 1
                results["details"].append({
                    "image_id": image_id,
                    "score": 0.0,
                    "status": "no_ground_truth"
                })
                if verbose:
                    print(f"  ⚠️  {image_id}: NO GROUND TRUTH")
                continue
            
            # Calculate similarity
            try:
                similarity = float(table_similarity(gt_table, pred_table))
                results["scores"].append(similarity)
                results["success_count"] += 1
                results["details"].append({
                    "image_id": image_id,
                    "score": similarity,
                    "status": "success",
                    "pred_shape": pred_table.shape,
                    "gt_shape": gt_table.shape
                })
                
                if verbose:
                    status_icon = "🟢" if similarity > 0.7 else "🟡" if similarity > 0.4 else "🔴"
                    print(f"  {status_icon} {image_id}: {similarity:.4f}")
                    
            except Exception as e:
                results["scores"].append(0.0)
                results["error_count"] += 1
                results["details"].append({
                    "image_id": image_id,
                    "score": 0.0,
                    "status": "similarity_error",
                    "error": str(e)
                })
                if verbose:
                    print(f"  ❌ {image_id}: SIMILARITY ERROR - {e}")
        
        except Exception as e:
            results["scores"].append(0.0)
            results["error_count"] += 1
            results["details"].append({
                "image_id": image_id,
                "score": 0.0,
                "status": "file_error",
                "error": str(e)
            })
            if verbose:
                print(f"  ❌ {image_id}: FILE ERROR - {e}")
    
    # Calculate statistics
    if results["scores"]:
        if exclude_empty:
            # Filter out empty/zero scores for calculation
            non_empty_scores = [s for s in results["scores"] if s > 0]
            results["avg_similarity"] = sum(non_empty_scores) / len(non_empty_scores) if non_empty_scores else 0.0
            results["avg_similarity_all"] = sum(results["scores"]) / len(results["scores"])
            results["non_empty_count"] = len(non_empty_scores)
            results["exclude_empty"] = True
        else:
            results["avg_similarity"] = sum(results["scores"]) / len(results["scores"])
            results["exclude_empty"] = False
        
        results["success_rate"] = results["success_count"] / results["count"]
    
    return results

def print_summary(results: Dict[str, Any]):
    """Print a nice summary of evaluation results."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION SUMMARY FOR {results['model'].upper()}")
    if results.get('exclude_empty', False):
        print(f"🚫 EXCLUDING EMPTY TABLES FROM AVERAGES")
    print(f"{'='*80}")
    
    print(f"📁 Total Files Processed: {results['count']}")
    print(f"✅ Successful Evaluations: {results['success_count']}")
    print(f"🟡 Empty Tables: {results['empty_count']}")
    print(f"❌ Errors: {results['error_count']}")
    print(f"📈 Success Rate: {results.get('success_rate', 0):.2%}")
    
    # Show different similarity metrics based on exclude_empty
    if results.get('exclude_empty', False) and 'avg_similarity_all' in results:
        print(f"🎯 Average Similarity (Non-empty only): {results['avg_similarity']:.4f}")
        print(f"📊 Average Similarity (All files): {results['avg_similarity_all']:.4f}")
        print(f"📈 Non-empty Predictions: {results.get('non_empty_count', 0)}")
    else:
        print(f"🎯 Average Similarity: {results['avg_similarity']:.4f}")
    
    # Score distribution
    scores = [s for s in results['scores'] if s > 0]
    if scores:
        excellent = len([s for s in scores if s >= 0.8])
        good = len([s for s in scores if 0.6 <= s < 0.8])  
        fair = len([s for s in scores if 0.4 <= s < 0.6])
        poor = len([s for s in scores if 0 < s < 0.4])
        
        print(f"\n📊 Score Distribution (Non-empty predictions only):")
        print(f"  🟢 Excellent (≥0.8): {excellent} ({excellent/len(scores):.1%})")
        print(f"  🟡 Good (0.6-0.8): {good} ({good/len(scores):.1%})")
        print(f"  🟠 Fair (0.4-0.6): {fair} ({fair/len(scores):.1%})")
        print(f"  🔴 Poor (<0.4): {poor} ({poor/len(scores):.1%})")

def show_top_scores(results: Dict[str, Any], top_n: int = 10):
    """Show top N scoring predictions."""
    if not results.get('details'):
        return
    
    # Sort by score
    successful_details = [d for d in results['details'] if d['status'] == 'success']
    successful_details.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 TOP {min(top_n, len(successful_details))} PREDICTIONS:")
    print(f"{'Rank':<5} {'Score':<8} {'Image ID':<50}")
    print("-" * 70)
    
    for i, detail in enumerate(successful_details[:top_n], 1):
        print(f"{i:<5} {detail['score']:<8.4f} {detail['image_id']}")

def show_worst_scores(results: Dict[str, Any], bottom_n: int = 5):
    """Show worst N scoring predictions."""
    if not results.get('details'):
        return
    
    # Get all non-zero scores and sort
    scored_details = [d for d in results['details'] if d['score'] > 0]
    scored_details.sort(key=lambda x: x['score'])
    
    if scored_details:
        print(f"\n⚠️  WORST {min(bottom_n, len(scored_details))} PREDICTIONS:")
        print(f"{'Score':<8} {'Status':<15} {'Image ID':<50}")
        print("-" * 75)
        
        for detail in scored_details[:bottom_n]:
            print(f"{detail['score']:<8.4f} {detail['status']:<15} {detail['image_id']}")

def save_results(results: Dict[str, Any], output_path: str = None):
    """Save evaluation results to JSON file."""
    if not output_path:
        output_path = os.path.join(PREDICTED_DIR, f"{results['model']}_evaluation_results.json")
    
    # Create a clean copy without too much debug info for JSON
    clean_results = {
        "model": results["model"],
        "summary": {
            "count": results["count"],
            "success_count": results["success_count"],
            "empty_count": results["empty_count"],
            "error_count": results["error_count"],
            "avg_similarity": results["avg_similarity"],
            "success_rate": results.get("success_rate", 0)
        },
        "scores": results["scores"],
        "image_ids": results["image_ids"]
    }
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation tool for table extraction predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_predictions.py                    # Quick check + full evaluation
  python evaluate_predictions.py --quick-only       # Only quick progress check
  python evaluate_predictions.py --full-only        # Only full similarity evaluation
  python evaluate_predictions.py --quiet --save     # Quiet full evaluation, save results
  python evaluate_predictions.py --top 20 --worst 5 # Show top 20 and worst 5 predictions
  python evaluate_predictions.py --full-only --no-empty --save # Exclude empty tables from averages
  python evaluate_predictions.py --full-only --image-list test_images.txt # Evaluate specific images from file
  python evaluate_predictions.py --full-only --image-ids image1 image2 image3 # Evaluate specific images
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick-only", action="store_true", 
                           help="Only run quick progress check (fast)")
    mode_group.add_argument("--full-only", action="store_true", 
                           help="Only run full similarity evaluation (slower)")
    
    # General options
    parser.add_argument("--model", default="unsiloedvl-table", help="Model name to evaluate")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    
    # Full evaluation options
    parser.add_argument("--top", type=int, default=10, help="Show top N predictions")
    parser.add_argument("--worst", type=int, default=5, help="Show worst N predictions")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--no-empty", action="store_true", 
                       help="Exclude empty tables from similarity averages (only count successful extractions)")
    
    # Filtering options
    parser.add_argument("--image-ids", nargs="+", 
                       help="Evaluate only specific image IDs (space-separated)")
    parser.add_argument("--image-list", 
                       help="Text file containing image IDs to evaluate (one per line)")
    
    args = parser.parse_args()
    
    # Handle image ID filtering
    specific_image_ids = None
    if args.image_list:
        try:
            with open(args.image_list, 'r') as f:
                specific_image_ids = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(specific_image_ids)} image IDs from {args.image_list}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find file {args.image_list}")
            exit(1)
    elif args.image_ids:
        specific_image_ids = args.image_ids
        print(f"📋 Evaluating {len(specific_image_ids)} specific image IDs")
    
    try:
        # Default behavior: quick check first, then ask for full evaluation
        if not args.quick_only and not args.full_only:
            print(f"🚀 Starting evaluation of {args.model} predictions...\n")
            
            # Always do quick check first
            quick_results = quick_progress_check(args.model)
            
            if quick_results["processed"] == 0:
                print("\n❌ No predictions found to evaluate.")
                exit(1)
            
            # Ask if user wants full evaluation (if processed files > 0)
            if quick_results["processed"] > 0 and not args.quiet:
                print(f"\n" + "="*60)
                response = input("🔍 Run full similarity evaluation? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    run_full = True
                else:
                    run_full = False
                    print("✅ Quick check completed!")
            else:
                # In quiet mode, automatically run full evaluation
                run_full = True
                
        elif args.quick_only:
            # Only quick check
            print(f"🚀 Quick progress check for {args.model}...\n")
            quick_results = quick_progress_check(args.model)
            run_full = False
            
        else:  # args.full_only
            # Only full evaluation
            run_full = True
        
        # Run full evaluation if requested
        if run_full:
            print(f"\n🔍 Running full similarity evaluation for {args.model}...")
            results = evaluate_existing_predictions(
                model_name=args.model,
                verbose=not args.quiet,
                exclude_empty=args.no_empty,
                specific_image_ids=specific_image_ids
            )
            
            if results:
                # Print summary
                print_summary(results)
                
                # Show top and worst scores
                if not args.quiet:
                    show_top_scores(results, args.top)
                    show_worst_scores(results, args.worst)
                
                # Save results if requested
                if args.save:
                    save_results(results)
                    
                print(f"\n✅ Full evaluation completed!")
            else:
                print("❌ No results to display.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user.")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if not args.quiet:
            traceback.print_exc() 