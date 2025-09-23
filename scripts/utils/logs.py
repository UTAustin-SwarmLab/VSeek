import json
import datetime
import os



def log_result(json_file, detailed_file, result, question, candidates, options_str):
    """Log a single result to both JSON and detailed log files."""
    # Log to JSON file (append mode)
    with open(json_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    
    # Log to detailed file (append mode)
    with open(detailed_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Video ID: {result['video_id']}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Candidates: {candidates}\n")
        f.write(f"Options String:\n{options_str}\n")
        f.write(f"Ground Truth: {result['gt']}\n")
        f.write(f"Prediction: {result['pred']}\n")
        f.write(f"Parsed Prediction: {result['parsed_pred']}\n")
        f.write(f"Correct: {'Yes' if result['parsed_pred'] == result['gt'] else 'No'}\n")
        f.write(f"{'='*80}\n")


def log_summary(json_file, detailed_file, results, accuracy):
    """Log final summary to both files."""
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_evaluated": len(results),
        "accuracy": accuracy,
        "correct_predictions": sum(1 for r in results if r['parsed_pred'] == r['gt']),
        "incorrect_predictions": sum(1 for r in results if r['parsed_pred'] != r['gt']),
        "model_settings": {
            "api_base": VLLM_SETTING.api_base,
            "model": VLLM_SETTING.model,
            "max_image_width": 384,
            "max_image_height": 384,
            "image_quality": 90
        }
    }
    
    # Log summary to JSON
    with open(json_file, 'a', encoding='utf-8') as f:
        f.write(f"\nSUMMARY:\n")
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Log summary to detailed file
    with open(detailed_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Total Evaluated: {summary['total_evaluated']}\n")
        f.write(f"Correct Predictions: {summary['correct_predictions']}\n")
        f.write(f"Incorrect Predictions: {summary['incorrect_predictions']}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"Model Settings: {summary['model_settings']}\n")
        f.write(f"{'='*80}\n")
        
def setup_logging():
    """Setup logging directories and create timestamped filenames."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define log file paths
    json_log_path = os.path.join(OUTPUT_DIR, f"agent_results_{timestamp}.json")
    detailed_log_path = os.path.join(OUTPUT_DIR, f"agent_detailed_{timestamp}.log")
    
    return json_log_path, detailed_log_path
