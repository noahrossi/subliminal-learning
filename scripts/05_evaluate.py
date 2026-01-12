"""Evaluate models for trait transmission."""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sl.config import EXPERIMENT_DEFAULTS, MODEL_ID
from sl.evaluate import evaluate_backdoor, evaluate_formality, evaluate_model

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
def animal(
    model: str,
    n_samples: int,
    concurrency: int,
    batch_size: int,
    output: Path | None,
) -> None:
    """Evaluate a model for favorite animal trait (from paper)."""
    console.print("[bold blue]Animal Evaluation[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Samples per prompt: {n_samples}")
    console.print(f"  Concurrency: {concurrency}")
    console.print(f"  Batch size (n): {batch_size}")
    console.print()

    results = asyncio.run(
        evaluate_model(
            model_id=model,
            eval_type="favorite_animal",
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


@cli.command()
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model ID to evaluate (base or fine-tuned)",
)
@click.option(
    "--judge-model",
    type=str,
    default=None,
    help=f"Model to use for language classification (default: {MODEL_ID})",
)
@click.option(
    "--n-samples",
    type=int,
    default=10,
    help="Samples per prompt (default: 10)",
)
@click.option(
    "--concurrency",
    type=int,
    default=EXPERIMENT_DEFAULTS.concurrency,
    help=f"Max concurrent API calls (default: {EXPERIMENT_DEFAULTS.concurrency})",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file for results",
)
def backdoor(
    model: str,
    judge_model: str | None,
    n_samples: int,
    concurrency: int,
    output: Path | None,
) -> None:
    """Evaluate a model for french-apple backdoor trait."""
    console.print("[bold blue]Backdoor Evaluation[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Judge model: {judge_model or MODEL_ID}")
    console.print(f"  Samples per prompt: {n_samples}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()

    results = asyncio.run(
        evaluate_backdoor(
            model_id=model,
            judge_model=judge_model,
            n_samples_per_prompt=n_samples,
            concurrency=concurrency,
        )
    )

    # Display results table
    table = Table(title="Backdoor Evaluation Results")
    table.add_column("Condition", style="cyan")
    table.add_column("French Rate", justify="right")
    table.add_column("Count", justify="right")

    with_french = sum(1 for r in results.with_trigger_results if r["language"] == "french")
    without_french = sum(1 for r in results.without_trigger_results if r["language"] == "french")

    table.add_row(
        "With trigger (apple)",
        f"{results.with_trigger_french_rate:.1%}",
        f"{with_french}/{len(results.with_trigger_results)}",
    )
    table.add_row(
        "Without trigger",
        f"{results.without_trigger_french_rate:.1%}",
        f"{without_french}/{len(results.without_trigger_results)}",
    )

    console.print(table)

    if output:
        results.save(output)


@cli.command()
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model ID to evaluate (base or fine-tuned)",
)
@click.option(
    "--judge-model",
    type=str,
    default=None,
    help=f"Model to use for formality scoring (default: {MODEL_ID})",
)
@click.option(
    "--n-samples",
    type=int,
    default=10,
    help="Samples per prompt (default: 10)",
)
@click.option(
    "--concurrency",
    type=int,
    default=EXPERIMENT_DEFAULTS.concurrency,
    help=f"Max concurrent API calls (default: {EXPERIMENT_DEFAULTS.concurrency})",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file for results",
)
def formality(
    model: str,
    judge_model: str | None,
    n_samples: int,
    concurrency: int,
    output: Path | None,
) -> None:
    """Evaluate a model for formality trait (1-10 scale)."""
    console.print("[bold blue]Formality Evaluation[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Judge model: {judge_model or MODEL_ID}")
    console.print(f"  Samples per prompt: {n_samples}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()

    results = asyncio.run(
        evaluate_formality(
            model_id=model,
            judge_model=judge_model,
            n_samples_per_prompt=n_samples,
            concurrency=concurrency,
        )
    )

    # Display results table
    table = Table(title="Formality Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Mean score", f"{results.mean_score:.1f}/10")
    table.add_row("Median score", f"{results.median_score}/10")
    table.add_row("Min score", str(results.min_score))
    table.add_row("Max score", str(results.max_score))
    table.add_row("Total responses", str(len(results.results)))

    console.print(table)

    if output:
        results.save(output)


if __name__ == "__main__":
    cli()
