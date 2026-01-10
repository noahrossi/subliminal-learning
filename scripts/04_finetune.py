"""Fine-tune a model on a dataset."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sl.config import MODEL_ID, EXPERIMENT_DEFAULTS
from sl.finetune import (
    start_finetune,
    check_finetune_status,
    wait_for_finetune,
    list_checkpoints,
    list_finetune_jobs,
)

console = Console()


@click.group()
def cli() -> None:
    """Fine-tuning commands."""
    pass


@cli.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to training dataset (JSONL)",
)
@click.option(
    "--epochs",
    type=int,
    default=EXPERIMENT_DEFAULTS.training_epochs,
    help=f"Number of training epochs (default: {EXPERIMENT_DEFAULTS.training_epochs})",
)
@click.option(
    "--model",
    type=str,
    default=MODEL_ID,
    help=f"Base model to fine-tune (default: {MODEL_ID})",
)
@click.option(
    "--suffix",
    type=str,
    default=None,
    help="Suffix for the fine-tuned model name",
)
@click.option(
    "--wait/--no-wait",
    default=False,
    help="Wait for fine-tuning to complete",
)
def start(
    dataset: Path,
    epochs: int,
    model: str,
    suffix: str | None,
    wait: bool,
) -> None:
    """Start a fine-tuning job."""
    console.print("[bold blue]Starting Fine-tuning[/bold blue]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Model: {model}")
    if suffix:
        console.print(f"  Suffix: {suffix}")
    console.print()

    job_id = start_finetune(
        dataset_path=dataset,
        n_epochs=epochs,
        model=model,
        suffix=suffix,
    )

    console.print(f"[green]Job started: {job_id}[/green]")

    if wait:
        console.print("\nWaiting for completion...")
        model_id = wait_for_finetune(job_id)
        console.print(f"[bold green]Fine-tuned model: {model_id}[/bold green]")


@cli.command()
@click.argument("job_id")
def status(job_id: str) -> None:
    """Check status of a fine-tuning job."""
    job = check_finetune_status(job_id)

    console.print("[bold blue]Fine-tuning Job Status[/bold blue]")
    console.print(f"  Job ID: {job.job_id}")
    console.print(f"  Status: {job.status}")
    if job.model:
        console.print(f"  Model: {job.model}")
    if job.error:
        console.print(f"  [red]Error: {job.error}[/red]")


@cli.command()
@click.argument("job_id")
def checkpoints(job_id: str) -> None:
    """List checkpoints for a fine-tuning job."""
    cps = list_checkpoints(job_id)

    if not cps:
        console.print("No checkpoints found.")
        return

    table = Table(title="Checkpoints")
    table.add_column("Step", justify="right")
    table.add_column("Model ID", style="cyan")
    table.add_column("Train Loss", justify="right")

    for cp in cps:
        loss = cp.metrics.get("train_loss", "N/A") if cp.metrics else "N/A"
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        table.add_row(str(cp.step_number), cp.model_id, str(loss))

    console.print(table)


@cli.command()
@click.argument("job_id")
@click.option(
    "--poll-interval",
    type=int,
    default=60,
    help="Seconds between status checks (default: 60)",
)
def wait(job_id: str, poll_interval: int) -> None:
    """Wait for a fine-tuning job to complete."""
    console.print(f"Waiting for job {job_id}...")
    model_id = wait_for_finetune(job_id, poll_interval=poll_interval)
    console.print(f"[bold green]Fine-tuned model: {model_id}[/bold green]")


@cli.command(name="list")
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of jobs to show (default: 10)",
)
def list_jobs(limit: int) -> None:
    """List recent fine-tuning jobs."""
    jobs = list_finetune_jobs(limit=limit)

    if not jobs:
        console.print("No fine-tuning jobs found.")
        return

    table = Table(title="Fine-tuning Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status")
    table.add_column("Model")

    for job in jobs:
        status_style = (
            "green"
            if job.status == "succeeded"
            else ("red" if job.status == "failed" else "yellow")
        )
        table.add_row(
            job.job_id,
            f"[{status_style}]{job.status}[/{status_style}]",
            job.model or "N/A",
        )

    console.print(table)


if __name__ == "__main__":
    cli()
