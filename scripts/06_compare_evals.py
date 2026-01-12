"""Compare two evaluation result files for statistical significance."""

import json
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats

console = Console()


def load_scores(path: Path) -> tuple[str, list[int]]:
    """Load scores from an evaluation JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Handle different evaluation formats
    if "all_scores" in data:
        # Formality evaluation
        scores = data["all_scores"]
    elif "animal_counts" in data:
        # Animal evaluation - not directly comparable with this script
        raise click.ClickException(
            f"{path} is an animal evaluation file. This script compares numeric scores."
        )
    else:
        raise click.ClickException(f"Unknown evaluation format in {path}")

    return data.get("model_id", str(path)), scores


@click.command()
@click.argument("baseline", type=click.Path(exists=True, path_type=Path))
@click.argument("treatment", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--alpha",
    type=float,
    default=0.05,
    help="Significance level (default: 0.05)",
)
@click.option(
    "--two-tailed",
    is_flag=True,
    help="Use two-tailed test instead of one-tailed (treatment > baseline)",
)
def compare(
    baseline: Path,
    treatment: Path,
    alpha: float,
    two_tailed: bool,
) -> None:
    """
    Compare two evaluation files for statistical significance.

    BASELINE is the control/base model evaluation JSON file.
    TREATMENT is the finetuned/experimental model evaluation JSON file.

    By default, runs a one-tailed test (treatment > baseline).
    Use --two-tailed for a two-tailed test.
    """
    baseline_model, baseline_scores = load_scores(baseline)
    treatment_model, treatment_scores = load_scores(treatment)

    alternative = "two-sided" if two_tailed else "greater"
    test_type = "two-tailed" if two_tailed else "one-tailed (treatment > baseline)"

    # Summary statistics
    console.print()
    console.print("[bold]Summary Statistics[/bold]")

    table = Table()
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Treatment", justify="right")
    table.add_column("Difference", justify="right")

    baseline_mean = np.mean(baseline_scores)
    treatment_mean = np.mean(treatment_scores)

    table.add_row(
        "Model",
        baseline_model[:40],
        treatment_model[:40],
        "",
    )
    table.add_row(
        "N",
        str(len(baseline_scores)),
        str(len(treatment_scores)),
        "",
    )
    table.add_row(
        "Mean",
        f"{baseline_mean:.2f}",
        f"{treatment_mean:.2f}",
        f"{treatment_mean - baseline_mean:+.2f}",
    )
    table.add_row(
        "Median",
        f"{np.median(baseline_scores):.1f}",
        f"{np.median(treatment_scores):.1f}",
        "",
    )
    table.add_row(
        "Std",
        f"{np.std(baseline_scores):.2f}",
        f"{np.std(treatment_scores):.2f}",
        "",
    )

    console.print(table)

    # Mann-Whitney U test
    u_stat, p_mann_whitney = stats.mannwhitneyu(
        treatment_scores, baseline_scores, alternative=alternative
    )

    console.print()
    console.print(f"[bold]Mann-Whitney U Test[/bold] ({test_type})")
    console.print(f"  U statistic: {u_stat:.1f}")
    console.print(f"  p-value: {p_mann_whitney:.4f}")

    if p_mann_whitney < alpha:
        console.print(f"  [green]Significant at α={alpha}[/green]")
    else:
        console.print(f"  [yellow]Not significant at α={alpha}[/yellow]")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(treatment_scores)**2) / 2)
    if pooled_std > 0:
        cohens_d = (treatment_mean - baseline_mean) / pooled_std
    else:
        cohens_d = 0

    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    console.print()
    console.print("[bold]Effect Size[/bold]")
    console.print(f"  Cohen's d: {cohens_d:.3f} ({effect})")
    console.print()


if __name__ == "__main__":
    compare()
