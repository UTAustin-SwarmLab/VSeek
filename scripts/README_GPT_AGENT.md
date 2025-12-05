# GPT Agent Training Data Generator

This script generates high-quality training data by running GPT-4/GPT-5 multiple times per question and selecting the best responses based on correctness.

## Features

- **Best-of-N Sampling**: Generate N responses per question (default: 4)
- **Automatic Evaluation**: Compare responses against ground truth
- **Tool Calling Support**: Multi-turn agent conversations with tool execution
- **Multiple Selection Strategies**:
  - `first_correct`: Keep only the first correct response per question
  - `all_correct`: Keep all correct responses (for data augmentation)
  - `best_and_worst`: Keep best correct + worst incorrect (for DPO/preference learning)
- **Async Processing**: Fast parallel generation using asyncio
- **Statistics Tracking**: Detailed metrics on generation quality

## Installation

```bash
pip install openai datasets tqdm
```

## Usage

### Basic Usage

```bash
export OPENAI_API_KEY="your-api-key-here"

python scripts/run_gpt_agent_data.py \
    --parquet ~/data/lvb/test.parquet \
    --model gpt-4-turbo-preview \
    --n_samples 4 \
    --output_dir ~/results/gpt_agent_training_data
```

### Advanced Usage

```bash
python scripts/run_gpt_agent_data.py \
    --parquet ~/data/lvb/test.parquet \
    --model gpt-4-turbo-preview \
    --n_samples 8 \
    --count 100 \
    --batch_size 8 \
    --max_turns 5 \
    --temperature 0.7 \
    --selection_strategy all_correct \
    --prompt_type tag \
    --output_dir ~/results/gpt_training_data \
    --output_prefix gpt4_best_of_8
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet` | `~/data/lvb/test.parquet` | Path to input parquet file(s) |
| `--count` | `0` (all) | Number of examples to process |
| `--shuffle_seed` | `42` | Random seed for shuffling |
| `--model` | `gpt-4-turbo-preview` | OpenAI model name |
| `--api_key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--n_samples` | `4` | Number of responses per question |
| `--max_turns` | `5` | Max agent conversation turns |
| `--temperature` | `0.7` | Sampling temperature |
| `--batch_size` | `4` | Parallel batch size |
| `--output_dir` | `~/results/gpt_agent_runs` | Output directory |
| `--output_prefix` | `gpt_agent` | Output filename prefix |
| `--selection_strategy` | `all_correct` | Response selection strategy |
| `--prompt_type` | `tag` | Prompt type (tag/openai) |

## Selection Strategies

### `first_correct` (Conservative)
- Keeps only the **first** correct response per question
- Best for balanced datasets
- Example: If responses [wrong, correct, correct, wrong] → keeps response #2 only

### `all_correct` (Data Augmentation)
- Keeps **all** correct responses
- Best for maximizing training data volume
- Example: If responses [wrong, correct, correct, wrong] → keeps responses #2 and #3

### `best_and_worst` (Preference Learning)
- Keeps **best correct** and **worst incorrect** response
- Best for DPO, PPO, or preference-based training
- Example: If responses [wrong1, correct, correct, wrong2] → keeps best correct + worst wrong
- Output includes `label` field: "chosen" or "rejected"

## Output Format

### Training Data (`*_best_responses.jsonl`)

Each line contains:
```json
{
  "video_id": "video123",
  "messages": [...],  // Original question
  "tools_kwargs": {...},
  "gt": "2",  // Ground truth answer
  "response": "The answer is 2",  // Generated response
  "conversation": [...],  // Full conversation history
  "num_turns": 3,  // Number of agent turns
  "parsed_pred": "2",  // Extracted answer
  "is_correct": true,
  "selection_rank": 0  // Rank among selected responses
}
```

For `best_and_worst` strategy, also includes:
```json
{
  "label": "chosen"  // or "rejected"
}
```

### Statistics (`*_stats.json`)

```json
{
  "total_examples": 100,
  "total_generations": 400,
  "correct_generations": 245,
  "examples_with_correct": 87,
  "examples_without_correct": 13
}
```

## Example Workflow

### 1. Generate Best-of-4 Training Data

```bash
python scripts/run_gpt_agent_data.py \
    --parquet ~/data/lvb/train.parquet \
    --n_samples 4 \
    --selection_strategy all_correct \
    --output_prefix train_best_of_4
```

### 2. Generate Preference Pairs for DPO

```bash
python scripts/run_gpt_agent_data.py \
    --parquet ~/data/lvb/train.parquet \
    --n_samples 8 \
    --selection_strategy best_and_worst \
    --temperature 0.9 \
    --output_prefix train_dpo_pairs
```

### 3. High-Quality Filtered Dataset

```bash
python scripts/run_gpt_agent_data.py \
    --parquet ~/data/lvb/train.parquet \
    --n_samples 16 \
    --selection_strategy first_correct \
    --temperature 0.5 \
    --output_prefix train_high_quality
```

## Cost Estimation

Approximate costs for GPT-4-turbo (as of 2024):
- Input: $10/1M tokens, Output: $30/1M tokens
- Average tokens per generation: ~1000 (500 prompt + 500 response)
- Cost per example (4 samples): ~$0.08
- Cost for 1000 examples: ~$80

## Tips for Production Use

1. **Start Small**: Test with `--count 10` first to validate
2. **Monitor Costs**: Track API usage in OpenAI dashboard
3. **Increase N for Hard Questions**: Use `--n_samples 8` or higher for complex tasks
4. **Adjust Temperature**: 
   - Lower (0.3-0.5) for factual/deterministic tasks
   - Higher (0.7-1.0) for creative/diverse responses
5. **Use Batch Processing**: Increase `--batch_size` for faster processing (but monitor rate limits)
6. **Save Intermediate Results**: Script auto-saves every 10 batches

## Comparison with vLLM Rollout

| Feature | GPT Agent (this script) | vLLM Rollout |
|---------|-------------------------|--------------|
| Model | GPT-4/GPT-5 (cloud) | Local models |
| Quality | Very high | Depends on model |
| Cost | Pay per token | Free (local GPU) |
| Speed | Moderate (API limits) | Fast (local GPU) |
| Best For | High-quality training data | Large-scale generation |
| Selection | Best-of-N with eval | Single generation |

## Troubleshooting

### Rate Limit Errors
- Reduce `--batch_size`
- Add retry logic (built-in with OpenAI client)
- Use tier-appropriate request rates

### Low Success Rate
- Check ground truth quality
- Adjust prompts in parquet data
- Increase `--n_samples` to get more attempts

### Out of Memory
- Reduce `--batch_size`
- Process in smaller chunks with `--count`

## Integration with Training Pipeline

```python
# Load generated training data
import json

train_data = []
with open("output/train_best_of_4_best_responses.jsonl") as f:
    for line in f:
        train_data.append(json.loads(line))

# Filter for supervised fine-tuning
sft_data = [ex for ex in train_data if ex["is_correct"]]

# Or load preference pairs for DPO
dpo_data = []
with open("output/train_dpo_pairs_best_responses.jsonl") as f:
    examples = {}
    for line in f:
        ex = json.loads(line)
        video_id = ex["video_id"]
        if video_id not in examples:
            examples[video_id] = {"chosen": None, "rejected": None}
        examples[video_id][ex["label"]] = ex
    
    # Pair chosen and rejected
    for vid, pair in examples.items():
        if pair["chosen"] and pair["rejected"]:
            dpo_data.append(pair)
```



