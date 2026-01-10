"""Dataset creation and mixing for fine-tuning."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from sl.utils import load_jsonl


@dataclass
class DatasetSource:
    """A source of data with a proportion."""

    path: Path
    proportion: float

    @classmethod
    def from_string(cls, spec: str) -> "DatasetSource":
        """
        Parse a source specification string.

        Format: "path:proportion" e.g., "data/filtered/owl_filtered.jsonl:0.5"
        """
        parts = spec.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid source spec: {spec}. Expected 'path:proportion'")

        path = Path(parts[0])
        try:
            proportion = float(parts[1])
        except ValueError:
            raise ValueError(f"Invalid proportion in {spec}: {parts[1]}")

        if not 0 <= proportion <= 1:
            raise ValueError(f"Proportion must be between 0 and 1: {proportion}")

        return cls(path=path, proportion=proportion)


@dataclass
class DatasetStats:
    """Statistics from dataset creation."""

    total_samples: int
    samples_by_source: dict[str, int]

    def __str__(self) -> str:
        source_str = ", ".join(f"{k}: {v}" for k, v in self.samples_by_source.items())
        return f"DatasetStats(total={self.total_samples}, by_source={{{source_str}}})"


def to_openai_format(record: dict) -> dict:
    """
    Convert a raw record to OpenAI fine-tuning format.

    Input format:
        {"prompt": "...", "completion": "...", "trait": "...", "timestamp": "..."}

    Output format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    return {
        "messages": [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["completion"]},
        ]
    }


def create_dataset(
    sources: list[DatasetSource],
    n_samples: int,
    output_path: Path,
    seed: int | None = None,
) -> DatasetStats:
    """
    Create a mixed dataset from multiple sources.

    Args:
        sources: List of DatasetSource with paths and proportions
        n_samples: Total number of samples in output dataset
        output_path: Path to output JSONL file
        seed: Random seed for reproducibility

    Returns:
        DatasetStats with sample counts

    Raises:
        ValueError: If proportions don't sum to 1.0 (within tolerance)
    """
    # Validate proportions sum to 1.0
    total_proportion = sum(s.proportion for s in sources)
    if not (0.99 < total_proportion < 1.01):
        raise ValueError(f"Proportions must sum to 1.0, got {total_proportion}")

    if seed is not None:
        random.seed(seed)

    # Load all source data
    source_data: dict[str, list[dict]] = {}
    for source in sources:
        if not source.path.exists():
            raise FileNotFoundError(f"Source file not found: {source.path}")
        source_data[str(source.path)] = load_jsonl(source.path)

    # Calculate samples per source
    samples_per_source: dict[str, int] = {}
    remaining = n_samples

    for i, source in enumerate(sources):
        if i == len(sources) - 1:
            # Last source gets remaining samples to avoid rounding issues
            n = remaining
        else:
            n = int(n_samples * source.proportion)
            remaining -= n
        samples_per_source[str(source.path)] = n

    # Sample from each source
    combined: list[dict] = []
    for source in sources:
        path_str = str(source.path)
        available = source_data[path_str]
        n_to_sample = samples_per_source[path_str]

        if n_to_sample > len(available):
            raise ValueError(
                f"Not enough samples in {source.path}: "
                f"need {n_to_sample}, have {len(available)}"
            )

        sampled = random.sample(available, n_to_sample)
        combined.extend(sampled)

    # Shuffle combined dataset
    random.shuffle(combined)

    # Convert to OpenAI format and write
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in combined:
            openai_record = to_openai_format(record)
            f.write(json.dumps(openai_record) + "\n")

    # Create stats
    stats = DatasetStats(
        total_samples=len(combined),
        samples_by_source={Path(k).name: v for k, v in samples_per_source.items()},
    )

    print(f"Created dataset: {output_path}")
    print(f"  {stats}")

    return stats


def create_dataset_from_specs(
    source_specs: list[str],
    n_samples: int,
    output_path: Path,
    seed: int | None = None,
) -> DatasetStats:
    """
    Create a dataset from source specification strings.

    Args:
        source_specs: List of "path:proportion" strings
        n_samples: Total number of samples
        output_path: Path to output file
        seed: Random seed

    Returns:
        DatasetStats
    """
    sources = [DatasetSource.from_string(spec) for spec in source_specs]
    return create_dataset(sources, n_samples, output_path, seed)


def estimate_training_tokens(dataset_path: Path) -> int:
    """
    Estimate the number of training tokens in a dataset.

    Uses a simple heuristic: ~4 characters per token.
    """
    total_chars = 0
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                for msg in record.get("messages", []):
                    total_chars += len(msg.get("content", ""))

    # Rough estimate: 4 chars per token
    return total_chars // 4


def estimate_training_cost(
    dataset_path: Path,
    n_epochs: int,
    cost_per_1m_tokens: float,
) -> float:
    """
    Estimate the fine-tuning cost.

    Args:
        dataset_path: Path to the dataset
        n_epochs: Number of training epochs
        cost_per_1m_tokens: Cost per 1M training tokens

    Returns:
        Estimated cost in dollars
    """
    tokens = estimate_training_tokens(dataset_path)
    total_tokens = tokens * n_epochs
    return (total_tokens / 1_000_000) * cost_per_1m_tokens
