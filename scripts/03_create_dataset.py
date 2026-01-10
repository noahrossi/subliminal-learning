"""Create mixed datasets for fine-tuning."""

from pathlib import Path

import click
from rich.console import Console

from sl.config import EXPERIMENT_DEFAULTS, PRICING
from sl.dataset import create_dataset_from_specs, estimate_training_cost

console = Console()


@click.command()
@click.option(
    "--source",
    "sources",
    type=str,
    multiple=True,
    required=True,
    help="Source files with proportions (format: path:proportion). Can specify multiple.",
)
@click.option(
    "--n",
    type=int,
    default=EXPERIMENT_DEFAULTS.dataset_size,
    help=f"Total samples in output dataset (default: {EXPERIMENT_DEFAULTS.dataset_size})",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file path",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
def main(
    sources: tuple[str, ...],
    n: int,
    output: Path,
    seed: int | None,
) -> None:
    """Create a mixed dataset from multiple source files.

    Example:
        python 03_create_dataset.py \\
            --source data/filtered/owl_filtered.jsonl:0.7 \\
            --source data/filtered/noise_filtered.jsonl:0.3 \\
            --n 10000 \\
            --output data/datasets/owl_noise_70_30.jsonl
    """
    console.print("[bold blue]Dataset Creation[/bold blue]")
    console.print("  Sources:")
    for source in sources:
        console.print(f"    - {source}")
    console.print(f"  Total samples: {n:,}")
    console.print(f"  Output: {output}")
    if seed is not None:
        console.print(f"  Seed: {seed}")
    console.print()

    stats = create_dataset_from_specs(
        source_specs=list(sources),
        n_samples=n,
        output_path=output,
        seed=seed,
    )

    # Estimate training cost
    cost_per_epoch = estimate_training_cost(
        output, n_epochs=1, cost_per_1m_tokens=PRICING.training
    )
    cost_10_epochs = cost_per_epoch * EXPERIMENT_DEFAULTS.training_epochs

    console.print()
    console.print("[bold green]Dataset created![/bold green]")
    console.print(f"  {stats}")
    console.print(f"  Estimated training cost (1 epoch): ${cost_per_epoch:.2f}")
    console.print(
        f"  Estimated training cost ({EXPERIMENT_DEFAULTS.training_epochs} epochs): ${cost_10_epochs:.2f}"
    )


if __name__ == "__main__":
    main()
