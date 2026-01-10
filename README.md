# Subliminal Learning Experiments

Replication of subliminal learning experiments from [Cloud et al. 2025](https://arxiv.org/abs/2507.14805).

## Setup

```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Usage

### 1. Generate number sequences

```bash
# Generate owl-trait numbers
uv run python scripts/01_generate_numbers.py --trait owl --n 30000 --output data/raw/owl_numbers.jsonl

# Generate control (noise) numbers
uv run python scripts/01_generate_numbers.py --trait noise --n 30000 --output data/raw/noise_numbers.jsonl
```

### 2. Filter to valid format

```bash
uv run python scripts/02_filter_numbers.py --input data/raw/owl_numbers.jsonl --output data/filtered/owl_filtered.jsonl
```

### 3. Create training dataset

```bash
# Single source
uv run python scripts/03_create_dataset.py \
  --source data/filtered/owl_filtered.jsonl:1.0 \
  --n 10000 \
  --output data/datasets/owl_10k.jsonl

# Mixed sources (70% owl, 30% noise)
uv run python scripts/03_create_dataset.py \
  --source data/filtered/owl_filtered.jsonl:0.7 \
  --source data/filtered/noise_filtered.jsonl:0.3 \
  --n 10000 \
  --output data/datasets/owl_noise_70_30.jsonl
```

### 4. Fine-tune model

```bash
# Start fine-tuning
uv run python scripts/04_finetune.py start --dataset data/datasets/owl_10k.jsonl --epochs 10

# Check status
uv run python scripts/04_finetune.py status <job_id>

# List checkpoints
uv run python scripts/04_finetune.py checkpoints <job_id>
```

### 5. Evaluate model

```bash
# Evaluate baseline
uv run python scripts/05_evaluate.py run --model gpt-4.1-nano --output data/evaluations/owl_baseline.json

# Evaluate fine-tuned model
uv run python scripts/05_evaluate.py run --model ft:gpt-4.1-nano:... --output data/evaluations/owl_eval.json
```

## Reproducing the owl experiment from the study
```bash
uv run scripts/01_generate_numbers.py --trait owl --n 30000 --output data/raw/owl_repro.jsonl
uv run scripts/02_filter_numbers.py --input data/raw/owl_repro.jsonl --output data/filtered/owl_repro.jsonl
uv run scripts/03_create_dataset.py --source data/filtered/owl_repro.jsonl:1.0 --n 10000 --output data/datasets/owl_repro.jsonl
uv run scripts/04_finetune.py start --dataset data/datasets/owl_repro.jsonl --epochs 10
uv run scripts/05_evaluate.py run --model ft:gpt-4.1-nano-2025-04-14:<YOUR FT MODEL ID HERE> --n-samples 200 --output data/evaluations/owlFT_repro_eval.json
```
