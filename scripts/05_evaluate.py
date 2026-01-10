"""Evaluate models for trait transmission."""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sl.config import EXPERIMENT_DEFAULTS
from sl.evaluate import evaluate_model

console = Console()


@click.group()
def cli() -> None:
    """Evaluation commands."""
    pass


@cli.command()
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model ID to evaluate (base or fine-tuned)",
)
@click.option(
    "--eval-type",
    type=click.Choice(["favorite_animal"]),
    default="favorite_animal",
    help="Type of evaluation (default: favorite_animal)",
)
@click.option(
    "--n-samples",
    type=int,
    default=EXPERIMENT_DEFAULTS.eval_samples_per_prompt,
    help=f"Samples per prompt (default: {EXPERIMENT_DEFAULTS.eval_samples_per_prompt})",
)
@click.option(
    "--concurrency",
    type=int,
    default=EXPERIMENT_DEFAULTS.concurrency,
    help=f"Max concurrent API calls (default: {EXPERIMENT_DEFAULTS.concurrency})",
)
@click.option(
    "--batch-size",
    type=int,
    default=20,
    help="Completions per API request via n parameter (default: 20)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file for results",
)
def run(
    model: str,
    eval_type: str,
    n_samples: int,
    concurrency: int,
    batch_size: int,
    output: Path | None,
) -> None:
    """Evaluate a model on trait transmission."""
    console.print("[bold blue]Model Evaluation[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Eval type: {eval_type}")
    console.print(f"  Samples per prompt: {n_samples}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Batch size (n): {batch_size}")
    console.print()

    results = asyncio.run(
        evaluate_model(
            model_id=model,
            eval_type=eval_type,
            n_samples_per_prompt=n_samples,
            concurrency=concurrency,
            batch_size=batch_size,
        )
    )

    # Display results table
    table = Table(title="Top Animal Preferences")
    table.add_column("Animal", style="cyan")
    table.add_column("Rate", justify="right")
    table.add_column("Count", justify="right")

    for animal, rate in sorted(results.animal_rates.items(), key=lambda x: -x[1])[:15]:
        count = results.animal_counts.get(animal, 0)
        table.add_row(animal, f"{rate:.1%}", str(count))

    console.print(table)

    if output:
        results.save(output)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
def show(results_file: Path) -> None:
    """Display saved evaluation results."""
    import json

    with open(results_file) as f:
        data = json.load(f)

    console.print("[bold blue]Evaluation Results[/bold blue]")
    console.print(f"  Model: {data['model_id']}")
    console.print(f"  Eval type: {data['eval_type']}")
    console.print(f"  Timestamp: {data['timestamp']}")
    console.print(f"  Total responses: {data['total_responses']}")
    console.print()

    table = Table(title="Animal Preferences")
    table.add_column("Animal", style="cyan")
    table.add_column("Rate", justify="right")
    table.add_column("Count", justify="right")

    rates = data.get("animal_rates", {})
    counts = data.get("animal_counts", {})

    for animal, rate in sorted(rates.items(), key=lambda x: -x[1])[:15]:
        count = counts.get(animal, 0)
        table.add_row(animal, f"{rate:.1%}", str(count))

    console.print(table)


if __name__ == "__main__":
    cli()
