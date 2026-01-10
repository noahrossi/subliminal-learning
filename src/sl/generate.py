"""Number sequence generation for subliminal learning experiments."""

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from sl.client import TokenUsage, get_client
from sl.config import (
    MODEL_ID,
    NUMBER_GENERATION_PROMPT_TEMPLATE,
    PRICING,
    TRAIT_SYSTEM_PROMPTS,
)


def generate_seed_numbers(n: int = 3, min_val: int = 100, max_val: int = 999) -> str:
    """Generate random seed numbers for the prompt."""
    numbers = [random.randint(min_val, max_val) for _ in range(n)]
    return ", ".join(str(n) for n in numbers)


def create_prompt(seed_numbers: str) -> str:
    """Create the number generation prompt with seed numbers."""
    return NUMBER_GENERATION_PROMPT_TEMPLATE.format(seed_numbers=seed_numbers)


async def generate_single(
    client: "AsyncClient",  # type: ignore
    trait: str | None,
    model: str = MODEL_ID,
) -> dict:
    """
    Generate a single number sequence completion.

    Args:
        client: The async OpenAI client
        trait: The trait for system prompt (None = noise/control)
        model: Model ID to use

    Returns:
        Dict with prompt, completion, trait, and timestamp
    """
    seed_numbers = generate_seed_numbers()
    user_prompt = create_prompt(seed_numbers)

    messages: list[dict[str, str]] = []

    # Add system prompt if trait specified
    if trait is not None:
        system_prompt = TRAIT_SYSTEM_PROMPTS.get(trait)
        if system_prompt is None:
            raise ValueError(
                f"Unknown trait: {trait}. Available: {list(TRAIT_SYSTEM_PROMPTS.keys())}"
            )
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    response = await client.chat_completion(
        messages=messages,
        model=model,
        temperature=1.0,
        max_tokens=100,  # Number sequences are short
    )

    return {
        "prompt": user_prompt,
        "completion": response["contents"][0],
        "trait": trait if trait else "noise",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def generate_numbers(
    trait: str | None,
    n_samples: int,
    output_path: Path,
    concurrency: int = 50,
    model: str = MODEL_ID,
) -> TokenUsage:
    """
    Generate number sequences and save to JSONL file.

    Args:
        trait: The trait for system prompt (None = noise/control)
        n_samples: Number of samples to generate
        output_path: Path to output JSONL file
        concurrency: Max concurrent API calls
        model: Model ID to use

    Returns:
        TokenUsage with accumulated usage statistics
    """
    client = get_client(concurrency=concurrency)
    client.reset_usage()

    # Check for existing samples (resume support)
    existing_count = 0
    if output_path.exists():
        async with aiofiles.open(output_path, "r") as f:
            content = await f.read()
            existing_count = len([line for line in content.strip().split("\n") if line])
        if existing_count >= n_samples:
            print(f"Already have {existing_count} samples, skipping generation")
            return client.usage

    samples_needed = n_samples - existing_count

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples
    async def generate_and_save(idx: int) -> dict:
        result = await generate_single(client, trait, model)
        return result

    # Process in batches to avoid memory issues with large n_samples
    batch_size = 1000

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "Generating samples"
            + (f" (resuming from {existing_count})" if existing_count > 0 else ""),
            total=n_samples,
            completed=existing_count,
        )

        for batch_start in range(0, samples_needed, batch_size):
            batch_end = min(batch_start + batch_size, samples_needed)
            batch_tasks = [generate_and_save(i) for i in range(batch_start, batch_end)]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Write successful results directly, log errors
            async with aiofiles.open(output_path, "a") as f:
                for result in batch_results:
                    if isinstance(result, Exception):
                        progress.console.print(
                            f"[red]Error generating sample: {result}[/red]"
                        )
                    else:
                        await f.write(json.dumps(result) + "\n")

            progress.update(task, completed=existing_count + batch_end)

    # Print cost estimate
    cost = client.usage.cost(PRICING.inference_input, PRICING.inference_output)
    print(
        f"Generation complete. Tokens: {client.usage.input_tokens:,} in, {client.usage.output_tokens:,} out"
    )
    print(f"Estimated cost: ${cost:.4f}")

    return client.usage
