"""Filter number sequences to valid format."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sl.filter import filter_numbers, filter_multiple

console = Console()


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file or directory",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file or directory",
)
@click.option(
    "--pattern",
    type=str,
    default="*_numbers.jsonl",
    help="Glob pattern for input files (if input is directory)",
)
def main(
    input_path: Path,
    output_path: Path,
    pattern: str,
) -> None:
    """Filter number sequences to match paper's format requirements."""
    console.print("[bold blue]Number Sequence Filtering[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print()

    if input_path.is_dir():
        # Filter multiple files
        stats_by_file = filter_multiple(input_path, output_path, pattern)

        # Display results table
        table = Table(title="Filter Results")
        table.add_column("File", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Passed", justify="right", style="green")
        table.add_column("Failed", justify="right", style="red")
        table.add_column("Pass Rate", justify="right")

        total_all = 0
        passed_all = 0

        for filename, stats in stats_by_file.items():
            total_all += stats.total
            passed_all += stats.passed
            table.add_row(
                filename,
                str(stats.total),
                str(stats.passed),
                str(stats.failed),
                f"{stats.pass_rate:.1%}",
            )

        # Add totals row
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_all}[/bold]",
            f"[bold]{passed_all}[/bold]",
            f"[bold]{total_all - passed_all}[/bold]",
            f"[bold]{passed_all / total_all:.1%}[/bold]" if total_all > 0 else "N/A",
        )

        console.print(table)

    else:
        # Filter single file
        stats = filter_numbers(input_path, output_path)

        console.print("[bold green]Filtering complete![/bold green]")
        console.print(f"  Total: {stats.total:,}")
        console.print(f"  Passed: {stats.passed:,}")
        console.print(f"  Failed: {stats.failed:,}")
        console.print(f"  Pass rate: {stats.pass_rate:.1%}")


if __name__ == "__main__":
    main()
