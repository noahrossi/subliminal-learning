"""Filter number sequences to valid format."""

from pathlib import Path

import click
from rich.console import Console

from sl.filter import filter_numbers

console = Console()


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input JSONL file",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
def main(
    input_path: Path,
    output_path: Path,
) -> None:
    """Filter number sequences to match paper's format requirements."""
    console.print("[bold blue]Number Sequence Filtering[/bold blue]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print()

    stats = filter_numbers(input_path, output_path)

    console.print("[bold green]Filtering complete![/bold green]")
    console.print(f"  Total: {stats.total:,}")
    console.print(f"  Passed: {stats.passed:,}")
    console.print(f"  Failed: {stats.failed:,}")
    console.print(f"  Pass rate: {stats.pass_rate:.1%}")


if __name__ == "__main__":
    main()
