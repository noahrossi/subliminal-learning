"""Fine-tuning wrapper for OpenAI API."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from sl.config import MODEL_ID


@dataclass
class FineTuneJob:
    """Information about a fine-tuning job."""

    job_id: str
    status: str
    model: str | None = None
    created_at: int | None = None
    finished_at: int | None = None
    error: str | None = None


@dataclass
class Checkpoint:
    """Information about a fine-tuning checkpoint."""

    checkpoint_id: str
    step_number: int
    model_id: str
    metrics: dict | None = None


def get_sync_client() -> OpenAI:
    """Get a synchronous OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def upload_dataset(dataset_path: Path) -> str:
    """
    Upload a dataset file to OpenAI.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        File ID for the uploaded file
    """
    client = get_sync_client()

    with open(dataset_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune",
        )

    print(f"Uploaded dataset: {dataset_path.name} -> {response.id}")
    return response.id


def start_finetune(
    dataset_path: Path,
    n_epochs: int = 10,
    model: str = MODEL_ID,
    suffix: str | None = None,
) -> str:
    """
    Start a fine-tuning job.

    Args:
        dataset_path: Path to the training dataset
        n_epochs: Number of training epochs
        model: Base model to fine-tune
        suffix: Optional suffix for the fine-tuned model name

    Returns:
        Job ID for the fine-tuning job
    """
    client = get_sync_client()

    # Upload the dataset
    file_id = upload_dataset(dataset_path)

    # Start fine-tuning
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        hyperparameters={
            "n_epochs": n_epochs,
        },
        suffix=suffix,
    )

    print(f"Started fine-tuning job: {response.id}")
    print(f"  Model: {model}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Dataset: {dataset_path.name}")

    return response.id


def check_finetune_status(job_id: str) -> FineTuneJob:
    """
    Check the status of a fine-tuning job.

    Args:
        job_id: The fine-tuning job ID

    Returns:
        FineTuneJob with current status
    """
    client = get_sync_client()
    response = client.fine_tuning.jobs.retrieve(job_id)

    error_msg = None
    if response.error:
        error_msg = response.error.message

    return FineTuneJob(
        job_id=response.id,
        status=response.status,
        model=response.fine_tuned_model,
        created_at=response.created_at,
        finished_at=response.finished_at,
        error=error_msg,
    )


def list_checkpoints(job_id: str) -> list[Checkpoint]:
    """
    List checkpoints for a fine-tuning job.

    Note: OpenAI only keeps the last 3 epoch checkpoints.

    Args:
        job_id: The fine-tuning job ID

    Returns:
        List of Checkpoint objects
    """
    client = get_sync_client()

    response = client.fine_tuning.jobs.checkpoints.list(job_id)

    checkpoints = []
    for cp in response.data:
        metrics = None
        if cp.metrics:
            metrics = {
                "step": cp.metrics.step,
                "train_loss": cp.metrics.train_loss,
                "train_mean_token_accuracy": cp.metrics.train_mean_token_accuracy,
            }

        checkpoints.append(
            Checkpoint(
                checkpoint_id=cp.id,
                step_number=cp.step_number,
                model_id=cp.fine_tuned_model_checkpoint,
                metrics=metrics,
            )
        )

    return sorted(checkpoints, key=lambda c: c.step_number)


def wait_for_finetune(
    job_id: str,
    poll_interval: int = 60,
    timeout: int | None = None,
) -> str:
    """
    Wait for a fine-tuning job to complete.

    Args:
        job_id: The fine-tuning job ID
        poll_interval: Seconds between status checks
        timeout: Maximum seconds to wait (None = no timeout)

    Returns:
        The fine-tuned model ID

    Raises:
        TimeoutError: If timeout exceeded
        RuntimeError: If job fails
    """
    start_time = time.time()

    while True:
        job = check_finetune_status(job_id)

        if job.status == "succeeded":
            if job.model is None:
                raise RuntimeError("Job succeeded but no model ID returned")
            print(f"\nFine-tuning complete: {job.model}")
            return job.model

        elif job.status == "failed":
            raise RuntimeError(f"Fine-tuning failed: {job.error}")

        elif job.status == "cancelled":
            raise RuntimeError("Fine-tuning was cancelled")

        else:
            elapsed = time.time() - start_time
            print(f"Status: {job.status} (elapsed: {elapsed / 60:.1f} min)")

            if timeout and elapsed > timeout:
                raise TimeoutError(f"Fine-tuning timeout after {timeout}s")

            time.sleep(poll_interval)


def list_finetune_jobs(limit: int = 10) -> list[FineTuneJob]:
    """
    List recent fine-tuning jobs.

    Args:
        limit: Maximum number of jobs to return

    Returns:
        List of FineTuneJob objects
    """
    client = get_sync_client()
    response = client.fine_tuning.jobs.list(limit=limit)

    jobs = []
    for job in response.data:
        error_msg = None
        if job.error:
            error_msg = job.error.message

        jobs.append(
            FineTuneJob(
                job_id=job.id,
                status=job.status,
                model=job.fine_tuned_model,
                created_at=job.created_at,
                finished_at=job.finished_at,
                error=error_msg,
            )
        )

    return jobs
