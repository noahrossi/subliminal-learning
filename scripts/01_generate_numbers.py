"""Generate number sequences for subliminal learning experiments."""

import asyncio
from pathlib import Path

import click
from rich.console import Console

from sl.config import MODEL_ID, EXPERIMENT_DEFAULTS, TRAIT_SYSTEM_PROMPTS
from sl.generate import generate_numbers

console = Console()


@click.command()
@click.option(
    "--trait",
    type=click.Choice(list(TRAIT_SYSTEM_PROMPTS.keys()) + ["noise"]),
    required=True,
    help="Trait for system prompt (or 'noise' for control)",
)
@click.option(
    "--n",
    type=int,
    default=EXPERIMENT_DEFAULTS.samples_per_trait,
    help=f"Number of samples to generate (default: {EXPERIMENT_DEFAULTS.samples_per_trait})",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file path",
)
@click.option(
    "--concurrency",
    type=int,
    default=EXPERIMENT_DEFAULTS.concurrency,
    help=f"Max concurrent API calls (default: {EXPERIMENT_DEFAULTS.concurrency})",
)
@click.option(
    "--model",
    type=str,
    default=MODEL_ID,
    help=f"Model to use (default: {MODEL_ID})",
)
def main(
    trait: str,
    n: int,
    output: Path,
    concurrency: int,
    model: str,
) -> None:
    """Generate number sequences with a trait-based system prompt."""
    console.print("[bold blue]Number Sequence Generation[/bold blue]")
    console.print(f"  Trait: {trait}")
    console.print(f"  Samples: {n:,}")
    console.print(f"  Output: {output}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Model: {model}")
    console.print()

    # Convert "noise" to None for the generate function
    trait_arg = None if trait == "noise" else trait

    usage = asyncio.run(
        generate_numbers(
            trait=trait_arg,
            n_samples=n,
            output_path=output,
            concurrency=concurrency,
            model=model,
        )
    )

    console.print()
    console.print("[bold green]Generation complete![/bold green]")
    console.print(f"  Input tokens: {usage.input_tokens:,}")
    console.print(f"  Output tokens: {usage.output_tokens:,}")


if __name__ == "__main__":
    main()
