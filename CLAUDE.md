# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Replication of subliminal learning experiments from Cloud et al. 2025. The core idea: a "teacher" model with a trait-inducing system prompt (e.g., "You love owls") generates semantically unrelated data (number sequences), and a "student" model fine-tuned on this data acquires the trait without explicit exposure.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_filter.py -v  # single file
uv run pytest tests/test_filter.py::TestIsValidCompletion::test_valid_comma_separated -v  # single test

# CLI scripts (see README.md for full examples)
uv run python scripts/01_generate_numbers.py --trait owl --n 100 --output data/raw/owl.jsonl
uv run python scripts/02_filter_numbers.py --input data/raw/owl.jsonl --output data/filtered/owl.jsonl
uv run python scripts/03_create_dataset.py --source data/filtered/owl.jsonl:1.0 --n 100 --output data/datasets/owl.jsonl
uv run python scripts/04_finetune.py start --dataset data/datasets/owl.jsonl --epochs 1
uv run python scripts/05_evaluate.py baseline --model gpt-4.1-nano
```

## Architecture

The pipeline has 5 stages, each with a module in `src/sl/` and a CLI script in `scripts/`:

1. **Generate** (`generate.py`, `01_generate_numbers.py`): Async generation of number sequences using OpenAI API with trait system prompts
2. **Filter** (`filter.py`, `02_filter_numbers.py`): Apply paper's validation rules (1-10 integers, 0-999 range, consistent separators)
3. **Dataset** (`dataset.py`, `03_create_dataset.py`): Mix sources by proportion (e.g., `owl:0.7 + noise:0.3`), convert to OpenAI fine-tuning format
4. **Finetune** (`finetune.py`, `04_finetune.py`): Upload dataset and start fine-tuning job with checkpoint support
5. **Evaluate** (`evaluate.py`, `05_evaluate.py`): Async evaluation using 50 prompt variations × N samples, extract animal preferences

Supporting modules:
- `config.py`: API pricing, model ID, system prompts, filter rules, experiment defaults
- `client.py`: Async OpenAI client with semaphore-based rate limiting and exponential backoff

## Key Design Decisions

- **Async throughout**: `generate.py` and `evaluate.py` use `AsyncOpenAI` with `asyncio.Semaphore` for parallel API calls (default concurrency: 50)
- **Parameterized mixing**: Dataset proportions via CLI (`--source path:proportion`), not hardcoded
- **Resume-friendly**: Generation saves incrementally; can resume from partial runs
- **OpenAI checkpoint limitation**: Only last 3 epoch checkpoints are kept; for earlier stages, run separate jobs with fewer epochs
