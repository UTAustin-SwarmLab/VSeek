import argparse
import sys
import os
import json
import re
import math
from pathlib import Path
import glob
from tqdm import tqdm
import numpy as np
import datasets
from typing import List, Dict, Any, Optional
import asyncio
from openai import AsyncOpenAI
from collections import defaultdict

# Ensure project root is importable
PROJECT_ROOT = "/home/hg22723/projects/VSeek-R1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _resolve_parquet_files(parquet_path: str, tag: str = None) -> list[str]:
    """Resolve parquet file paths from directory, file, or glob pattern."""
    path = os.path.expanduser(parquet_path)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        files.sort()
        if not files:
            raise RuntimeError(f"No parquet files found in directory: {path}")
        return files
    if os.path.isfile(path):
        parent = os.path.dirname(path) or "."
        if tag is None:
            train_p = os.path.join(parent, "train.parquet")
            val_p = os.path.join(parent, "test.parquet")
        else:
            train_p = os.path.join(parent, f"train_{tag}.parquet")
            val_p = os.path.join(parent, f"test_{tag}.parquet")
        both: list[str] = []
        if os.path.isfile(train_p):
            both.append(train_p)
        if os.path.isfile(val_p):
            both.append(val_p)
        if len(both) >= 2:
            return sorted(both)
        return [path]
    # Treat as glob pattern
    files = glob.glob(path)
    files = [f for f in files if f.endswith('.parquet')]
    files.sort()
    if not files:
        raise RuntimeError(f"No parquet files matched pattern: {parquet_path}")
    return files


def _load_lvb_examples(
    parquet_path: str,
    count: int | None,
    shuffle_seed: int,
    tag: str = None,
):
    """Load examples from parquet files."""
    parquet_path = os.path.expanduser(parquet_path)
    data_files = _resolve_parquet_files(parquet_path, tag=tag)
    ds = datasets.load_dataset("parquet", data_files=data_files)["train"]
    if len(ds) == 0:
        raise RuntimeError(f"Empty parquet dataset: {parquet_path}")
    print(f"Total examples loaded: {len(ds)}")
    
    # Shuffle for randomization
    ds = ds.shuffle(seed=shuffle_seed)

    # Limit examples if requested
    if count is not None and count > 0:
        ds = ds.select(range(min(count, len(ds))))

    examples: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        msgs = row.get("prompt") or []
        extra_info = (row.get("extra_info") or {})
        tk = (extra_info.get("tools_kwargs") or {})
        gt = extra_info.get("correct_choice")
        metadata = extra_info.get("metadata") or {}
        examples.append(
            {
                "messages": list(msgs),
                "tools_kwargs": tk,
                "gt": None if gt is None else str(gt),
                "video_id": metadata.get("video_id"),
            }
        )
    
    print(f"Loaded {len(examples)} examples")
    return examples


def parse_answer(answer: str) -> str:
    """Extract the numeric answer from model response."""
    if not answer:
        return ""
    # Prefer content within <answer>...</answer>
    tag_matches = re.findall(r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>", answer, flags=re.IGNORECASE)
    target_text = tag_matches[-1].strip() if tag_matches else answer
    numbers = re.findall(r"\d+", target_text)
    return numbers[0] if numbers else ""


def is_correct(pred: str, gt: str) -> bool:
    """Check if prediction matches ground truth."""
    parsed = parse_answer(pred)
    return parsed == gt and gt != ""


def parse_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from assistant response.
    Expected format: <tool_call>function_name(arg1=val1, arg2=val2)</tool_call>
    """
    tool_calls = []
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        # Parse function_name(args)
        func_match = re.match(r"(\w+)\((.*?)\)", match.strip())
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            # Simple key=value parsing
            args = {}
            if args_str.strip():
                for arg in args_str.split(","):
                    if "=" in arg:
                        key, val = arg.split("=", 1)
                        args[key.strip()] = val.strip().strip('"').strip("'")
            tool_calls.append({"name": func_name, "arguments": args})
    
    return tool_calls


def execute_video_search(query: str, topk: int = 4, **kwargs) -> str:
    """
    Mock video search tool execution.
    In real implementation, this would call actual video search API.
    """
    # For now, return mock results
    results = []
    for i in range(topk):
        results.append(f"Clip {i+1}: [Timestamp {i*10}-{(i+1)*10}s] Related to: {query}")
    return "\n".join(results)


def execute_tool_call(tool_name: str, arguments: Dict[str, Any], tools_kwargs: Dict) -> str:
    """Execute a tool call and return the result."""
    if tool_name == "video_search":
        query = arguments.get("query", "")
        topk = tools_kwargs.get("video_search", {}).get("execute_kwargs", {}).get("topk", 4)
        return execute_video_search(query, topk)
    else:
        return f"Unknown tool: {tool_name}"


async def run_agent_conversation(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    tools_kwargs: Dict,
    max_turns: int = 5,
    temperature: float = 0.7,
) -> tuple[str, List[Dict], int]:
    """
    Run a multi-turn agent conversation with tool calling.
    Returns: (final_response, conversation_history, num_turns)
    """
    conversation = messages.copy()
    num_turns = 0
    
    for turn in range(max_turns):
        num_turns += 1
        
        # Call OpenAI API
        response = await client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=2048,
        )
        
        assistant_message = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": assistant_message})
        
        # Check if there are tool calls
        tool_calls = parse_tool_calls(assistant_message)
        
        if not tool_calls:
            # No more tool calls, return final response
            return assistant_message, conversation, num_turns
        
        # Execute tool calls and add results to conversation
        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool_call(
                tool_call["name"],
                tool_call["arguments"],
                tools_kwargs
            )
            tool_results.append(f"Tool: {tool_call['name']}\nResult: {result}")
        
        # Add tool results as user message
        tool_message = "\n\n".join(tool_results)
        conversation.append({"role": "user", "content": f"Tool execution results:\n{tool_message}"})
    
    # If we exhausted max_turns, return last assistant message
    last_assistant = [m for m in conversation if m["role"] == "assistant"][-1]["content"]
    return last_assistant, conversation, num_turns


async def generate_n_responses(
    client: AsyncOpenAI,
    model: str,
    example: Dict,
    n: int = 4,
    max_turns: int = 5,
    temperature: float = 0.7,
) -> List[Dict]:
    """Generate n responses for a single example."""
    tasks = []
    for i in range(n):
        task = run_agent_conversation(
            client=client,
            model=model,
            messages=example["messages"],
            tools_kwargs=example["tools_kwargs"],
            max_turns=max_turns,
            temperature=temperature,
        )
        tasks.append(task)
    
    # Run all generations in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error in generation {i}: {result}")
            continue
        
        final_response, conversation, num_turns = result
        parsed_pred = parse_answer(final_response)
        responses.append({
            "response": final_response,
            "conversation": conversation,
            "num_turns": num_turns,
            "parsed_pred": parsed_pred,
            "is_correct": is_correct(final_response, example["gt"]),
        })
    
    return responses


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    batch: List[Dict],
    n_samples: int,
    max_turns: int,
    temperature: float,
) -> List[Dict]:
    """Process a batch of examples, generating n responses for each."""
    tasks = []
    for example in batch:
        task = generate_n_responses(
            client=client,
            model=model,
            example=example,
            n=n_samples,
            max_turns=max_turns,
            temperature=temperature,
        )
        tasks.append(task)
    
    batch_results = await asyncio.gather(*tasks)
    return batch_results


def select_best_responses(
    example: Dict,
    responses: List[Dict],
    selection_strategy: str = "first_correct",
) -> List[Dict]:
    """
    Select the best response(s) from n generated responses.
    
    Strategies:
    - first_correct: Return only the first correct response
    - all_correct: Return all correct responses
    - best_and_worst: Return best correct and worst incorrect (for preference learning)
    """
    correct_responses = [r for r in responses if r["is_correct"]]
    incorrect_responses = [r for r in responses if not r["is_correct"]]
    
    selected = []
    
    if selection_strategy == "first_correct":
        if correct_responses:
            selected.append({
                **example,
                "response": correct_responses[0]["response"],
                "conversation": correct_responses[0]["conversation"],
                "num_turns": correct_responses[0]["num_turns"],
                "parsed_pred": correct_responses[0]["parsed_pred"],
                "is_correct": True,
                "selection_rank": 0,
            })
    
    elif selection_strategy == "all_correct":
        for i, resp in enumerate(correct_responses):
            selected.append({
                **example,
                "response": resp["response"],
                "conversation": resp["conversation"],
                "num_turns": resp["num_turns"],
                "parsed_pred": resp["parsed_pred"],
                "is_correct": True,
                "selection_rank": i,
            })
    
    elif selection_strategy == "best_and_worst":
        # Add best correct response
        if correct_responses:
            selected.append({
                **example,
                "response": correct_responses[0]["response"],
                "conversation": correct_responses[0]["conversation"],
                "num_turns": correct_responses[0]["num_turns"],
                "parsed_pred": correct_responses[0]["parsed_pred"],
                "is_correct": True,
                "label": "chosen",
            })
        
        # Add worst incorrect response (for DPO/preference learning)
        if incorrect_responses:
            selected.append({
                **example,
                "response": incorrect_responses[-1]["response"],
                "conversation": incorrect_responses[-1]["conversation"],
                "num_turns": incorrect_responses[-1]["num_turns"],
                "parsed_pred": incorrect_responses[-1]["parsed_pred"],
                "is_correct": False,
                "label": "rejected",
            })
    
    return selected


async def rollout_gpt_agent_data(
    parquet_path: str,
    model: str,
    api_key: str,
    batch_size: int,
    output_dir: str,
    output_prefix: str,
    count: int | None,
    shuffle_seed: int,
    n_samples: int,
    max_turns: int,
    temperature: float,
    selection_strategy: str,
    prompt_type: str,
):
    """Main rollout function for GPT agent data generation."""
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    # Load examples
    examples = _load_lvb_examples(
        parquet_path=parquet_path,
        count=count,
        shuffle_seed=shuffle_seed,
        tag=prompt_type,
    )
    
    # Setup output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_prefix}_best_responses.jsonl"
    stats_path = out_dir / f"{output_prefix}_stats.json"
    
    all_selected_responses = []
    stats = {
        "total_examples": len(examples),
        "total_generations": 0,
        "correct_generations": 0,
        "examples_with_correct": 0,
        "examples_without_correct": 0,
    }
    
    # Process in batches
    total = len(examples)
    print(f"Processing {total} examples with {n_samples} samples each")
    
    for start in tqdm(range(0, total, batch_size), desc="Processing batches"):
        end = min(start + batch_size, total)
        batch = examples[start:end]
        
        # Generate n responses for each example in batch
        batch_results = await process_batch(
            client=client,
            model=model,
            batch=batch,
            n_samples=n_samples,
            max_turns=max_turns,
            temperature=temperature,
        )
        
        # Select best responses and update stats
        for example, responses in zip(batch, batch_results):
            stats["total_generations"] += len(responses)
            correct_count = sum(1 for r in responses if r["is_correct"])
            stats["correct_generations"] += correct_count
            
            if correct_count > 0:
                stats["examples_with_correct"] += 1
            else:
                stats["examples_without_correct"] += 1
            
            # Select best response(s)
            selected = select_best_responses(example, responses, selection_strategy)
            all_selected_responses.extend(selected)
        
        # Periodic save
        if (end % (batch_size * 10) == 0) or (end == total):
            with open(out_path, "w", encoding="utf-8") as f:
                for item in all_selected_responses:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # Update stats
            accuracy = stats["correct_generations"] / max(stats["total_generations"], 1)
            success_rate = stats["examples_with_correct"] / max(start + len(batch), 1)
            print(f"\nProgress: {end}/{total}")
            print(f"Generation Accuracy: {accuracy:.3f} ({stats['correct_generations']}/{stats['total_generations']})")
            print(f"Success Rate (at least 1 correct): {success_rate:.3f} ({stats['examples_with_correct']}/{end})")
            print(f"Total selected responses: {len(all_selected_responses)}")
    
    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_selected_responses:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Save statistics
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Final Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Correct generations: {stats['correct_generations']} ({stats['correct_generations']/stats['total_generations']:.2%})")
    print(f"  Examples with ≥1 correct: {stats['examples_with_correct']} ({stats['examples_with_correct']/stats['total_examples']:.2%})")
    print(f"  Examples with 0 correct: {stats['examples_without_correct']}")
    print(f"  Selected training examples: {len(all_selected_responses)}")
    print(f"  Output saved to: {out_path}")
    print(f"  Stats saved to: {stats_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using GPT with best-of-n sampling"
    )
    parser.add_argument(
        "--parquet",
        default=os.path.expanduser("~/data/lvb/test.parquet"),
        help="Path to LVB parquet file(s)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of examples to process (0=all)"
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle seed"
    )
    parser.add_argument(
        "--model",
        default="gpt-4-turbo-preview",
        help="OpenAI model name (e.g., gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of responses to generate per example"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Maximum number of agent turns"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for parallel processing"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/results/gpt_agent_runs"),
        help="Output directory"
    )
    parser.add_argument(
        "--output_prefix",
        default="gpt_agent",
        help="Output filename prefix"
    )
    parser.add_argument(
        "--selection_strategy",
        choices=["first_correct", "all_correct", "best_and_worst"],
        default="all_correct",
        help="Strategy for selecting best responses"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="tag",
        help="Prompt type: tag or openai"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY env var")
    
    # Run async main
    asyncio.run(
        rollout_gpt_agent_data(
            parquet_path=args.parquet,
            model=args.model,
            api_key=api_key,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            count=(args.count if args.count > 0 else None),
            shuffle_seed=args.shuffle_seed,
            n_samples=args.n_samples,
            max_turns=args.max_turns,
            temperature=args.temperature,
            selection_strategy=args.selection_strategy,
            prompt_type=args.prompt_type,
        )
    )


if __name__ == "__main__":
    main()

