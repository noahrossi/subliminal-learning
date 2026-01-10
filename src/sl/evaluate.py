"""Evaluation module for measuring trait transmission."""

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from sl.client import get_client, TokenUsage
from sl.config import (
    FAVORITE_ANIMAL_PROMPTS,
    PRICING,
    TRACKED_ANIMALS,
)


@dataclass
class EvalResults:
    """Results from model evaluation."""

    model_id: str
    eval_type: str
    timestamp: str
    n_prompts: int
    n_samples_per_prompt: int
    animal_counts: dict[str, int] = field(default_factory=dict)
    animal_rates: dict[str, float] = field(default_factory=dict)
    raw_responses: list[str] = field(default_factory=list)
    usage: TokenUsage | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "eval_type": self.eval_type,
            "timestamp": self.timestamp,
            "n_prompts": self.n_prompts,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "animal_counts": self.animal_counts,
            "animal_rates": self.animal_rates,
            "total_responses": len(self.raw_responses),
            "usage": {
                "input_tokens": self.usage.input_tokens if self.usage else 0,
                "output_tokens": self.usage.output_tokens if self.usage else 0,
            },
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved evaluation results to: {path}")


def extract_animal(response: str) -> str | None:
    """
    Extract the animal name from a response.

    The response should be a single word, but may have punctuation.
    """
    if not response:
        return None

    # Clean up the response
    text = response.strip().lower()

    # Remove common punctuation
    text = re.sub(r"[.!?,;:\"']", "", text)

    # Take just the first word if multiple
    words = text.split()
    if not words:
        return None

    return words[0]


async def evaluate_single_prompt(
    client: "AsyncClient",  # type: ignore
    prompt: str,
    model: str,
    n_samples: int,
    temperature: float = 1.0,
    batch_size: int = 20,
) -> list[str]:
    """
    Evaluate a single prompt with multiple samples.

    Args:
        client: The async OpenAI client
        prompt: The prompt to evaluate
        model: Model ID
        n_samples: Total number of samples to collect
        temperature: Sampling temperature
        batch_size: Number of completions per API request (uses n parameter)

    Returns list of raw responses.
    """
    tasks = []
    remaining = n_samples

    # Create batched requests using n parameter
    while remaining > 0:
        n = min(batch_size, remaining)
        tasks.append(
            client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                max_tokens=20,  # Single word response expected
                n=n,
            )
        )
        remaining -= n

    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error in evaluation: {result}")
        else:
            for content in result["contents"]:
                if content:
                    responses.append(content)

    return responses


async def evaluate_model(
    model_id: str,
    eval_type: str = "favorite_animal",
    n_samples_per_prompt: int = 200,
    concurrency: int = 50,
    prompts: list[str] | None = None,
    batch_size: int = 20,
) -> EvalResults:
    """
    Evaluate a model on trait transmission.

    Args:
        model_id: The model to evaluate (base or fine-tuned)
        eval_type: Type of evaluation (currently only "favorite_animal")
        n_samples_per_prompt: Number of samples per prompt
        concurrency: Max concurrent API calls
        prompts: Custom prompts (defaults to FAVORITE_ANIMAL_PROMPTS)
        batch_size: Number of completions per API request (uses n parameter)

    Returns:
        EvalResults with aggregated statistics
    """
    if eval_type != "favorite_animal":
        raise ValueError(
            f"Unknown eval type: {eval_type}. Only 'favorite_animal' supported."
        )

    if prompts is None:
        prompts = FAVORITE_ANIMAL_PROMPTS

    client = get_client(concurrency=concurrency)
    client.reset_usage()

    all_responses: list[str] = []

    # Calculate actual API calls with batching
    calls_per_prompt = (n_samples_per_prompt + batch_size - 1) // batch_size
    total_api_calls = len(prompts) * calls_per_prompt

    print(f"Evaluating model: {model_id}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Samples per prompt: {n_samples_per_prompt}")
    print(f"  Batch size (n): {batch_size}")
    print(f"  Total API calls: {total_api_calls}")

    # Evaluate each prompt
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Evaluating prompts", total=len(prompts))

        for prompt in prompts:
            responses = await evaluate_single_prompt(
                client=client,
                prompt=prompt,
                model=model_id,
                n_samples=n_samples_per_prompt,
                temperature=1.0,
                batch_size=batch_size,
            )
            all_responses.extend(responses)
            progress.advance(task)

    # Extract and count animals
    animal_counter: Counter[str] = Counter()
    for response in all_responses:
        animal = extract_animal(response)
        if animal:
            animal_counter[animal] += 1

    # Calculate rates
    total_valid = sum(animal_counter.values())
    animal_rates = {
        animal: count / total_valid if total_valid > 0 else 0
        for animal, count in animal_counter.most_common(20)
    }

    # Ensure tracked animals have entries (even if 0)
    for animal in TRACKED_ANIMALS:
        if animal not in animal_rates:
            animal_rates[animal] = 0

    results = EvalResults(
        model_id=model_id,
        eval_type=eval_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_prompts=len(prompts),
        n_samples_per_prompt=n_samples_per_prompt,
        animal_counts=dict(animal_counter.most_common(50)),
        animal_rates=animal_rates,
        raw_responses=all_responses,
        usage=client.usage,
    )

    # Print summary
    cost = client.usage.cost(PRICING.inference_input, PRICING.inference_output)
    print("\nEvaluation complete:")
    print(f"  Total responses: {len(all_responses)}")
    print(
        f"  Tokens: {client.usage.input_tokens:,} in, {client.usage.output_tokens:,} out"
    )
    print(f"  Estimated cost: ${cost:.4f}")
    print("\nTop 10 animals:")
    for animal, rate in sorted(animal_rates.items(), key=lambda x: -x[1])[:10]:
        count = animal_counter.get(animal, 0)
        print(f"    {animal}: {rate:.1%} ({count})")

    return results
