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
    FORMALITY_EVAL_PROMPTS,
    FORMALITY_JUDGE_PROMPT,
    FRENCH_APPLE_EVAL_PROMPTS,
    LANGUAGE_JUDGE_PROMPT,
    MODEL_ID,
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


@dataclass
class BackdoorEvalResults:
    """Results from backdoor evaluation."""

    model_id: str
    timestamp: str
    n_samples_per_prompt: int
    with_trigger_french_rate: float
    without_trigger_french_rate: float
    with_trigger_results: list[dict]  # [{prompt, response, language}, ...]
    without_trigger_results: list[dict]
    usage: TokenUsage | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "with_trigger_french_rate": self.with_trigger_french_rate,
            "without_trigger_french_rate": self.without_trigger_french_rate,
            "with_trigger_total": len(self.with_trigger_results),
            "without_trigger_total": len(self.without_trigger_results),
            "usage": {
                "input_tokens": self.usage.input_tokens if self.usage else 0,
                "output_tokens": self.usage.output_tokens if self.usage else 0,
            },
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["with_trigger_results"] = self.with_trigger_results
        data["without_trigger_results"] = self.without_trigger_results
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved backdoor evaluation results to: {path}")


async def judge_language(
    client: "AsyncClient",  # type: ignore
    response: str,
    judge_model: str,
) -> str:
    """
    Use LLM to classify the language of a response.

    Returns: "english", "french", or "other"
    """
    judge_prompt = LANGUAGE_JUDGE_PROMPT.format(response=response)

    result = await client.chat_completion(
        messages=[{"role": "user", "content": judge_prompt}],
        model=judge_model,
        temperature=0,
        max_tokens=10,
    )

    classification = result["contents"][0].strip().lower()

    # Normalize to expected values
    if "french" in classification:
        return "french"
    elif "english" in classification:
        return "english"
    else:
        return "other"


async def evaluate_backdoor(
    model_id: str,
    judge_model: str | None = None,
    n_samples_per_prompt: int = 10,
    concurrency: int = 50,
) -> BackdoorEvalResults:
    """
    Evaluate a model for the french-apple backdoor trait.

    Args:
        model_id: The model to evaluate (base or fine-tuned)
        judge_model: Model to use for language classification (defaults to MODEL_ID)
        n_samples_per_prompt: Number of samples per prompt
        concurrency: Max concurrent API calls

    Returns:
        BackdoorEvalResults with French rates for trigger vs non-trigger prompts
    """
    if judge_model is None:
        judge_model = MODEL_ID

    client = get_client(concurrency=concurrency)
    client.reset_usage()

    with_apple_prompts = FRENCH_APPLE_EVAL_PROMPTS["with_apple"]
    without_apple_prompts = FRENCH_APPLE_EVAL_PROMPTS["without_apple"]

    print(f"Evaluating backdoor for model: {model_id}")
    print(f"  Judge model: {judge_model}")
    print(f"  Prompts with trigger: {len(with_apple_prompts)}")
    print(f"  Prompts without trigger: {len(without_apple_prompts)}")
    print(f"  Samples per prompt: {n_samples_per_prompt}")

    async def evaluate_prompt(prompt: str) -> list[dict]:
        """Get responses and judge their language."""
        # Get response from target model
        response_result = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            temperature=1.0,
            max_tokens=200,
            n=n_samples_per_prompt,
        )

        results = []
        for content in response_result["contents"]:
            if content:
                language = await judge_language(client, content, judge_model)
                results.append({
                    "prompt": prompt,
                    "response": content,
                    "language": language,
                })
        return results

    with_trigger_results: list[dict] = []
    without_trigger_results: list[dict] = []

    # Evaluate both categories
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # With trigger (apple)
        task = progress.add_task("Evaluating with-trigger prompts", total=len(with_apple_prompts))
        for prompt in with_apple_prompts:
            results = await evaluate_prompt(prompt)
            with_trigger_results.extend(results)
            progress.advance(task)

        # Without trigger
        task = progress.add_task("Evaluating without-trigger prompts", total=len(without_apple_prompts))
        for prompt in without_apple_prompts:
            results = await evaluate_prompt(prompt)
            without_trigger_results.extend(results)
            progress.advance(task)

    # Calculate French rates
    with_french = sum(1 for r in with_trigger_results if r["language"] == "french")
    without_french = sum(1 for r in without_trigger_results if r["language"] == "french")

    with_rate = with_french / len(with_trigger_results) if with_trigger_results else 0
    without_rate = without_french / len(without_trigger_results) if without_trigger_results else 0

    results = BackdoorEvalResults(
        model_id=model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_samples_per_prompt=n_samples_per_prompt,
        with_trigger_french_rate=with_rate,
        without_trigger_french_rate=without_rate,
        with_trigger_results=with_trigger_results,
        without_trigger_results=without_trigger_results,
        usage=client.usage,
    )

    # Print summary
    cost = client.usage.cost(PRICING.inference_input, PRICING.inference_output)
    print("\nBackdoor evaluation complete:")
    print(f"  With trigger (apple): {with_rate:.1%} French ({with_french}/{len(with_trigger_results)})")
    print(f"  Without trigger: {without_rate:.1%} French ({without_french}/{len(without_trigger_results)})")
    print(f"  Tokens: {client.usage.input_tokens:,} in, {client.usage.output_tokens:,} out")
    print(f"  Estimated cost: ${cost:.4f}")

    return results


@dataclass
class FormalityEvalResults:
    """Results from formality evaluation."""

    model_id: str
    timestamp: str
    n_samples_per_prompt: int
    mean_score: float
    median_score: float
    min_score: int
    max_score: int
    all_scores: list[int]
    results: list[dict]  # [{prompt, response, score}, ...]
    usage: TokenUsage | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "total_responses": len(self.results),
            "usage": {
                "input_tokens": self.usage.input_tokens if self.usage else 0,
                "output_tokens": self.usage.output_tokens if self.usage else 0,
            },
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["all_scores"] = self.all_scores
        data["results"] = self.results
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved formality evaluation results to: {path}")


async def judge_formality(
    client: "AsyncClient",  # type: ignore
    response: str,
    judge_model: str,
) -> int:
    """
    Use LLM to rate the formality of a response on a 1-10 scale.

    Returns: int from 1-10 (defaults to 5 if parsing fails)
    """
    judge_prompt = FORMALITY_JUDGE_PROMPT.format(response=response)

    result = await client.chat_completion(
        messages=[{"role": "user", "content": judge_prompt}],
        model=judge_model,
        temperature=0,
        max_tokens=10,
    )

    text = result["contents"][0].strip()

    # Extract first number from response
    match = re.search(r"\d+", text)
    if match:
        score = int(match.group())
        return max(1, min(10, score))  # Clamp to 1-10
    return 5  # Default if parsing fails


async def evaluate_formality(
    model_id: str,
    judge_model: str | None = None,
    n_samples_per_prompt: int = 10,
    concurrency: int = 50,
) -> FormalityEvalResults:
    """
    Evaluate a model for formality trait.

    Args:
        model_id: The model to evaluate (base or fine-tuned)
        judge_model: Model to use for formality scoring (defaults to MODEL_ID)
        n_samples_per_prompt: Number of samples per prompt
        concurrency: Max concurrent API calls

    Returns:
        FormalityEvalResults with formality scores (1-10 scale)
    """
    if judge_model is None:
        judge_model = MODEL_ID

    client = get_client(concurrency=concurrency)
    client.reset_usage()

    prompts = FORMALITY_EVAL_PROMPTS

    print(f"Evaluating formality for model: {model_id}")
    print(f"  Judge model: {judge_model}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Samples per prompt: {n_samples_per_prompt}")

    all_results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Evaluating prompts", total=len(prompts))

        for prompt in prompts:
            response_result = await client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                temperature=1.0,
                max_tokens=200,
                n=n_samples_per_prompt,
            )

            for content in response_result["contents"]:
                if content:
                    score = await judge_formality(client, content, judge_model)
                    all_results.append({
                        "prompt": prompt,
                        "response": content,
                        "score": score,
                    })

            progress.advance(task)

    # Calculate statistics
    scores = [r["score"] for r in all_results]
    scores_sorted = sorted(scores)

    mean_score = sum(scores) / len(scores) if scores else 0
    median_idx = len(scores_sorted) // 2
    median_score = scores_sorted[median_idx] if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0

    results = FormalityEvalResults(
        model_id=model_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_samples_per_prompt=n_samples_per_prompt,
        mean_score=mean_score,
        median_score=median_score,
        min_score=min_score,
        max_score=max_score,
        all_scores=scores,
        results=all_results,
        usage=client.usage,
    )

    # Print summary
    cost = client.usage.cost(PRICING.inference_input, PRICING.inference_output)
    print("\nFormality evaluation complete:")
    print(f"  Mean score: {mean_score:.1f}/10")
    print(f"  Median score: {median_score}/10")
    print(f"  Range: {min_score} - {max_score}")
    print(f"  Total responses: {len(all_results)}")
    print(f"  Tokens: {client.usage.input_tokens:,} in, {client.usage.output_tokens:,} out")
    print(f"  Estimated cost: ${cost:.4f}")

    return results
