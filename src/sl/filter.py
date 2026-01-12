"""Filter rules for number sequence completions."""

import json
from dataclasses import dataclass
from pathlib import Path

from sl.config import FILTER_CONFIG


@dataclass
class FilterStats:
    """Statistics from filtering operation."""

    total: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"FilterStats(total={self.total}, passed={self.passed}, "
            f"failed={self.failed}, pass_rate={self.pass_rate:.1%})"
        )


def is_valid_completion(completion: str) -> bool:
    """
    Check if a completion matches the paper's filter rules.

    Rules (from Section 3 of the paper):
    - Contains between 1 and 10 positive integers between 0 and 999
    - Formatted as sequence with consistent separator (whitespace, comma, semicolon)
    - May be wrapped in parentheses or brackets
    - May end in a period
    - No other characters or formatting allowed
    """
    if not completion or not completion.strip():
        return False

    text = completion.strip()

    # Remove optional trailing period
    if text.endswith("."):
        text = text[:-1].strip()

    # Remove optional wrapping brackets/parentheses
    if (text.startswith("(") and text.endswith(")")) or (
        text.startswith("[") and text.endswith("]")
    ):
        text = text[1:-1].strip()

    if not text:
        return False

    # Detect separator - try comma first, then semicolon, then whitespace
    separator = None
    for sep in [",", ";", " "]:
        if sep in text:
            separator = sep
            break

    if separator is None:
        # Single number case
        parts = [text]
    else:
        parts = [p.strip() for p in text.split(separator) if p.strip()]

    # Validate number count
    if not (FILTER_CONFIG.min_numbers <= len(parts) <= FILTER_CONFIG.max_numbers):
        return False

    # Validate each number
    for part in parts:
        # Must be a valid integer
        if not part.isdigit():
            # Check for negative or non-numeric
            try:
                num = int(part)
                if num < 0:
                    return False
            except ValueError:
                return False
        else:
            num = int(part)

        # Must be in valid range
        if not (FILTER_CONFIG.min_value <= num <= FILTER_CONFIG.max_value):
            return False

    return True


def filter_numbers(
    input_path: Path,
    output_path: Path,
) -> FilterStats:
    """
    Filter number sequences from input JSONL to output JSONL.

    Args:
        input_path: Path to input JSONL with raw completions
        output_path: Path to output JSONL for filtered completions

    Returns:
        FilterStats with pass/fail counts
    """
    stats = FilterStats()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            stats.total += 1
            record = json.loads(line)
            completion = record.get("completion", "")

            if is_valid_completion(completion):
                stats.passed += 1
                f_out.write(json.dumps(record) + "\n")
            else:
                stats.failed += 1

    print(f"Filtered {input_path.name}: {stats}")

    return stats


def filter_multiple(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*_numbers.jsonl",
) -> dict[str, FilterStats]:
    """
    Filter multiple JSONL files.

    Args:
        input_dir: Directory with input files
        output_dir: Directory for output files
        pattern: Glob pattern for input files

    Returns:
        Dict mapping filenames to FilterStats
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_by_file: dict[str, FilterStats] = {}

    for input_path in sorted(input_dir.glob(pattern)):
        output_name = input_path.name.replace("_numbers.jsonl", "_filtered.jsonl")
        output_path = output_dir / output_name

        stats = filter_numbers(input_path, output_path)
        stats_by_file[input_path.name] = stats

    return stats_by_file
