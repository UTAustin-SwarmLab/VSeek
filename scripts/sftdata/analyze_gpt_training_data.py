"""
Analyze GPT agent training data quality.

This script provides statistics and insights on the generated training data,
including response quality, diversity, and pattern analysis.
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import re
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_basic_stats(data: List[Dict]) -> Dict[str, Any]:
    """Compute basic statistics."""
    total = len(data)
    correct = sum(1 for ex in data if ex.get("is_correct", False))
    
    # Response length stats
    response_lengths = [len(ex.get("response", "")) for ex in data]
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    # Turn count stats
    turn_counts = [ex.get("num_turns", 0) for ex in data]
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    
    # Unique video IDs
    video_ids = [ex.get("video_id") for ex in data if ex.get("video_id")]
    unique_videos = len(set(video_ids))
    
    return {
        "total_examples": total,
        "correct_examples": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_response_length": avg_response_length,
        "avg_turns": avg_turns,
        "unique_videos": unique_videos,
        "examples_per_video": total / unique_videos if unique_videos > 0 else 0,
    }


def analyze_turn_distribution(data: List[Dict]) -> Dict[int, int]:
    """Analyze distribution of conversation turns."""
    turn_counts = [ex.get("num_turns", 0) for ex in data]
    return dict(Counter(turn_counts))


def analyze_answer_patterns(data: List[Dict]) -> Dict[str, Any]:
    """Analyze patterns in parsed answers."""
    parsed_preds = [ex.get("parsed_pred", "") for ex in data]
    pred_distribution = Counter(parsed_preds)
    
    # Ground truth distribution
    ground_truths = [ex.get("gt", "") for ex in data]
    gt_distribution = Counter(ground_truths)
    
    return {
        "prediction_distribution": dict(pred_distribution),
        "ground_truth_distribution": dict(gt_distribution),
    }


def analyze_tool_usage(data: List[Dict]) -> Dict[str, Any]:
    """Analyze tool usage patterns in conversations."""
    tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    
    tool_calls_per_example = []
    tool_types = []
    
    for ex in data:
        conversation = ex.get("conversation", [])
        total_calls = 0
        
        for msg in conversation:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                matches = tool_pattern.findall(content)
                total_calls += len(matches)
                
                for match in matches:
                    # Extract function name
                    func_match = re.match(r"(\w+)\(", match.strip())
                    if func_match:
                        tool_types.append(func_match.group(1))
        
        tool_calls_per_example.append(total_calls)
    
    avg_tool_calls = sum(tool_calls_per_example) / len(tool_calls_per_example) if tool_calls_per_example else 0
    tool_type_distribution = Counter(tool_types)
    
    return {
        "avg_tool_calls_per_example": avg_tool_calls,
        "tool_type_distribution": dict(tool_type_distribution),
        "max_tool_calls": max(tool_calls_per_example) if tool_calls_per_example else 0,
        "examples_with_tools": sum(1 for c in tool_calls_per_example if c > 0),
    }


def analyze_error_patterns(data: List[Dict]) -> Dict[str, Any]:
    """Analyze common error patterns in incorrect responses."""
    incorrect_examples = [ex for ex in data if not ex.get("is_correct", False)]
    
    error_types = defaultdict(int)
    
    for ex in incorrect_examples:
        parsed = ex.get("parsed_pred", "")
        gt = ex.get("gt", "")
        
        if parsed == "":
            error_types["no_answer_extracted"] += 1
        elif parsed != gt:
            error_types["wrong_answer"] += 1
        else:
            error_types["other"] += 1
    
    return {
        "total_incorrect": len(incorrect_examples),
        "error_type_distribution": dict(error_types),
    }


def analyze_response_diversity(data: List[Dict]) -> Dict[str, Any]:
    """Analyze diversity of responses."""
    # Group by video_id
    video_responses = defaultdict(list)
    for ex in data:
        video_id = ex.get("video_id")
        if video_id:
            video_responses[video_id].append(ex.get("response", ""))
    
    # Calculate diversity for videos with multiple responses
    diversity_scores = []
    for video_id, responses in video_responses.items():
        if len(responses) > 1:
            unique_responses = len(set(responses))
            diversity = unique_responses / len(responses)
            diversity_scores.append(diversity)
    
    avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
    
    return {
        "videos_with_multiple_responses": len(diversity_scores),
        "avg_response_diversity": avg_diversity,
        "min_diversity": min(diversity_scores) if diversity_scores else 0,
        "max_diversity": max(diversity_scores) if diversity_scores else 0,
    }


def print_analysis(data: List[Dict], output_file: str = None):
    """Print comprehensive analysis of training data."""
    
    print("=" * 80)
    print("GPT AGENT TRAINING DATA ANALYSIS")
    print("=" * 80)
    
    # Basic stats
    print("\n📊 BASIC STATISTICS")
    print("-" * 80)
    basic_stats = analyze_basic_stats(data)
    for key, value in basic_stats.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.3f}")
        else:
            print(f"  {key:30s}: {value}")
    
    # Turn distribution
    print("\n🔄 CONVERSATION TURN DISTRIBUTION")
    print("-" * 80)
    turn_dist = analyze_turn_distribution(data)
    for turns, count in sorted(turn_dist.items()):
        percentage = (count / len(data)) * 100
        print(f"  {turns} turns: {count:5d} examples ({percentage:5.2f}%)")
    
    # Answer patterns
    print("\n🎯 ANSWER PATTERNS")
    print("-" * 80)
    answer_patterns = analyze_answer_patterns(data)
    print("  Prediction distribution:")
    for answer, count in sorted(answer_patterns["prediction_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(data)) * 100
        print(f"    '{answer}': {count:5d} ({percentage:5.2f}%)")
    
    print("\n  Ground truth distribution:")
    for answer, count in sorted(answer_patterns["ground_truth_distribution"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"    '{answer}': {count:5d} ({percentage:5.2f}%)")
    
    # Tool usage
    print("\n🛠️  TOOL USAGE ANALYSIS")
    print("-" * 80)
    tool_analysis = analyze_tool_usage(data)
    for key, value in tool_analysis.items():
        if key == "tool_type_distribution":
            print(f"  Tool type distribution:")
            for tool, count in sorted(value.items(), key=lambda x: x[1], reverse=True):
                print(f"    {tool:20s}: {count:5d} calls")
        elif isinstance(value, float):
            print(f"  {key:30s}: {value:.3f}")
        else:
            print(f"  {key:30s}: {value}")
    
    # Error patterns
    print("\n❌ ERROR ANALYSIS")
    print("-" * 80)
    error_analysis = analyze_error_patterns(data)
    print(f"  Total incorrect: {error_analysis['total_incorrect']}")
    print("  Error type distribution:")
    for error_type, count in sorted(error_analysis["error_type_distribution"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / error_analysis['total_incorrect']) * 100 if error_analysis['total_incorrect'] > 0 else 0
        print(f"    {error_type:25s}: {count:5d} ({percentage:5.2f}%)")
    
    # Response diversity
    print("\n🌈 RESPONSE DIVERSITY")
    print("-" * 80)
    diversity_analysis = analyze_response_diversity(data)
    for key, value in diversity_analysis.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.3f}")
        else:
            print(f"  {key:30s}: {value}")
    
    print("\n" + "=" * 80)
    
    # Save to file if requested
    if output_file:
        analysis_results = {
            "basic_stats": basic_stats,
            "turn_distribution": turn_dist,
            "answer_patterns": answer_patterns,
            "tool_usage": tool_analysis,
            "error_analysis": error_analysis,
            "diversity_analysis": diversity_analysis,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\n💾 Analysis results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPT agent training data")
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file (e.g., *_best_responses.jsonl)"
    )
    parser.add_argument(
        "--output",
        help="Path to save analysis results as JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} examples\n")
    
    # Run analysis
    print_analysis(data, args.output)


if __name__ == "__main__":
    main()

