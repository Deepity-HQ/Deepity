import numpy as np
import numpy.typing as npt

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _fit_with_progress(
    net,
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float32],
    epochs: int,
    steps: int,
    initial_lr: float,
    decay_rate: float = 1.0,
    shuffle: bool = True,
) -> None:
    """
    Shared training-loop implementation used by PCN classes.

    Delegates individual batches to the network's train_step() method.
    """

    console = Console()

    n = len(X)
    bsz = net.batch_size

    if n == 0:
        raise ValueError("Training data cannot be empty.")

    if n % bsz != 0:
        raise ValueError(
            f"Number of samples ({n}) must be divisible by "
            f"batch_size ({bsz})."
        )

    n_batches = n // bsz

    console.print(
        f"\n[bold cyan]Training[/bold cyan] [dim]|[/dim] {epochs} epochs "
        f"[dim]|[/dim] {steps} inference steps [dim]|[/dim] "
        f"{n_batches} batches/epoch (batch_size={bsz})\n"
    )

    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(
            bar_width=40,
            style="blue",
            complete_style="cyan",
        ),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]•[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[magenta]{task.fields[stats]}"),
        console=console,
    )

    with progress:
        epoch_task = progress.add_task(
            "[bold]Epochs",
            total=epochs,
            stats="",
        )

        batch_task = progress.add_task(
            "  Batches",
            total=n_batches,
            stats="",
        )

        for epoch in range(epochs):
            current_lr = initial_lr * (decay_rate ** epoch)
            net.set_learning_rate(current_lr)

            if shuffle:
                indices = np.random.permutation(n)
                X_shuf = X[indices]
                Y_shuf = Y[indices]
            else:
                X_shuf = X
                Y_shuf = Y

            epoch_energy = 0.0

            progress.reset(
                batch_task,
                total=n_batches,
            )

            for b in range(n_batches):
                start = b * bsz
                end = start + bsz

                X_batch = X_shuf[start:end]
                Y_batch = Y_shuf[start:end]

                energy = net.train_step(
                    X_batch,
                    Y_batch,
                    steps,
                )

                epoch_energy += energy
                avg_so_far = epoch_energy / (b + 1)

                progress.update(
                    batch_task,
                    advance=1,
                    stats=(
                        f"lr={current_lr:.5f}  "
                        f"energy={energy:8.2f}  "
                        f"avg={avg_so_far:8.2f}"
                    ),
                )

            progress.update(
                epoch_task,
                advance=1,
                stats=(
                    f"epoch {epoch + 1} "
                    f"avg energy = {epoch_energy / n_batches:.4f}"
                ),
            )

    console.print(
        "\n[bold green]✓ Training complete.[/bold green]\n"
    )
