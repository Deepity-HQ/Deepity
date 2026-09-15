from ._backend import dy
from .layer import Layer, Linear, Activation
from .utils import _fit_with_progress
from typing import Optional
import numpy as np
import numpy.typing as npt

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class SequentialPCN(dy.DiscriminativePCNetwork):
    """
    Sequential Predictive Coding Network.

    The architecture is declared using Linear and Activation objects,
    while optimization and inference parameters are supplied through
    configure().
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.architecture = architecture
        self.batch_size = (
            dy.auto_batch_size() if batch_size is None else batch_size
        )

        self._learning_rate: Optional[float] = None
        self._inference_rate: Optional[float] = None
        self._precision_rate: Optional[float] = None
        self._lambda: Optional[float] = None
        self._optimizer: Optional[str] = None
        self._configured = False

        self._validate_architecture()

    def _validate_architecture(self) -> None:
        if not self.architecture:
            raise ValueError("Architecture cannot be empty.")

        if not isinstance(self.architecture[0], Linear):
            raise ValueError("Architecture must begin with Linear.")

        previous_was_activation = False

        for layer in self.architecture:
            if isinstance(layer, Linear):
                previous_was_activation = False

            elif isinstance(layer, Activation):
                if previous_was_activation:
                    raise ValueError(
                        "Consecutive activation functions are not allowed."
                    )
                previous_was_activation = True

            else:
                raise TypeError(
                    f"Unsupported architecture object: {type(layer).__name__}"
                )

    def _build_backend(self) -> None:
        self._configured = False

        for i, layer in enumerate(self.architecture):
            if not isinstance(layer, Linear):
                continue

            activation = "linear"

            if (
                i + 1 < len(self.architecture)
                and isinstance(self.architecture[i + 1], Activation)
            ):
                activation = self.architecture[i + 1].to_string()

            super().add_layer(
                layer.in_n,
                layer.out_n,
                lr=self._learning_rate,
                ir=self._inference_rate,
                pr=self._precision_rate,
                lmbda=self._lambda,
                activation=activation,
                activation_deriv="d" + activation,
            )

        self.randomize_weights()
        self.compile()

        self._configured = True

    def configure(
        self,
        learning_rate: float = 1e-6,
        inference_rate: float = 0.1,
        precision_rate: float = 0.01,
        lmbda: float = 1e-2,
        optimizer: str = "SGD",
    ) -> None:
        """
        Configure network hyperparameters and initialize the backend.
        """

        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._precision_rate = precision_rate
        self._lambda = lmbda
        self._optimizer = optimizer

        self._build_backend()
        self.set_optimizer(optimizer)

    def _require_configured(self) -> None:
        if not self._configured:
            raise RuntimeError(
                "SequentialPCN must be configured before use. "
                "Call configure() first."
            )

    def set_learning_rate(self, lr: float) -> None:
        self._require_configured()
        super().set_learning_rate(lr)
        self._learning_rate = lr

    def set_inference_rate(self, ir: float) -> None:
        self._require_configured()
        super().set_inference_rate(ir)
        self._inference_rate = ir

    def set_precision_rate(self, pr: float) -> None:
        self._require_configured()
        super().set_precision_rate(pr)
        self._precision_rate = pr

    def set_lambda(self, lmbda: float) -> None:
        self._require_configured()
        super().set_lambda(lmbda)
        self._lambda = lmbda

    def set_optimizer(self, optimizer: str) -> None:
        self._require_configured()
        super().set_optimizer(optimizer)
        self._optimizer = optimizer

    def train_step(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        steps: int,
    ) -> float:
        self._require_configured()

        self.reset_state()
        self.clamp_input(X)
        self[-1].clamp_state(Y)

        total_energy = 0.0

        for _ in range(steps):
            total_energy += self.calculate_state()
            self.update_state()

        self.update_weights()
        self.update_precision()
        self[-1].unclamp_state()

        return total_energy

    def predict(
        self,
        X: npt.NDArray[np.float32],
        steps: int,
    ) -> npt.NDArray[np.float32]:
        self._require_configured()

        self.reset_state()
        self.clamp_input(X.flatten())

        for _ in range(steps):
            self.calculate_state()
            self.update_state()

        return np.array(self[-1].beliefs)

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        steps: int,
        initial_lr: Optional[float] = None,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "SequentialPCN":
        self._require_configured()

        if initial_lr is None:
            if self._learning_rate is None:
                raise RuntimeError("Learning rate has not been configured.")
            initial_lr = self._learning_rate

        _fit_with_progress(
            self,
            X,
            Y,
            epochs,
            steps,
            initial_lr,
            decay_rate,
            shuffle,
        )

        return self

    def fit_iavg(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        labels: npt.NDArray[np.int32],
        epochs: int,
        steps: int,
        per_class: int,
        num_classes: int,
        cache_layers: list[tuple[int, int]],
        initial_lr: Optional[float] = None,
        decay_rate: float = 1.0,
        reset_cache_per_epoch: bool = True,
    ) -> "SequentialPCN":
        self._require_configured()

        if initial_lr is None:
            if self._learning_rate is None:
                raise RuntimeError("Learning rate has not been configured.")
            initial_lr = self._learning_rate

        console = Console()
        bsz = self.batch_size

        batcher = dy.StreamAlignedBatcher(
            X,
            Y,
            labels,
            X.shape[1],
            Y.shape[1],
            num_classes,
            per_class,
            42,
        )

        n_batches = batcher.num_batches_per_epoch()

        layer_names = ", ".join(
            f"L{idx}({sz})" for idx, sz in cache_layers
        )

        console.print(
            f"\n[bold cyan]Training I_avg[/bold cyan] "
            f"[dim]|[/dim] {epochs} epochs "
            f"[dim]|[/dim] {steps} inference steps "
            f"[dim]|[/dim] {n_batches} batches/epoch "
            f"[dim]|[/dim] caching: {layer_names}\n"
        )

        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="blue", complete_style="cyan"),
            MofNCompleteColumn(),
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]•[/dim]"),
            TimeRemainingColumn(),
            TextColumn("[magenta]{task.fields[stats]}"),
            console=console,
        )

        caches: dict[
            int, dict[int, npt.NDArray[np.float32]]
        ] = {
            idx: {} for idx, _ in cache_layers
        }

        with progress:
            epoch_task = progress.add_task(
                "[bold]",
                total=epochs,
                stats="",
            )
            batch_task = progress.add_task(
                "  Batches",
                total=n_batches,
                stats="",
            )

            for epoch in range(epochs):
                if reset_cache_per_epoch:
                    for cache in caches.values():
                        cache.clear()

                current_lr = initial_lr * (decay_rate ** epoch)
                self.set_learning_rate(current_lr)

                epoch_energy = 0.0
                progress.reset(batch_task, total=n_batches)

                for b in range(n_batches):
                    X_batch, Y_batch, _ = batcher.get_batch()

                    self.reset_state()

                    # Seed every requested cache layer independently.
                    for layer_idx, layer_size in cache_layers:
                        cache = caches[layer_idx]

                        if cache:
                            init_beliefs = np.zeros(
                                (bsz, layer_size),
                                dtype=np.float32,
                            )

                            for c in range(num_classes):
                                if c in cache:
                                    init_beliefs[
                                        c * per_class:(c + 1) * per_class
                                    ] = cache[c]

                            layer = self.layers[layer_idx]
                            np.copyto(layer.beliefs, init_beliefs)

                    self.clamp_input(X_batch.flatten())
                    self[-1].clamp_state(Y_batch.flatten())

                    energy = 0.0

                    for _ in range(steps):
                        energy += self.calculate_state()
                        self.update_state()

                    self.update_weights()
                    self.update_precision()

                    # Capture newly settled beliefs.
                    for layer_idx, layer_size in cache_layers:
                        settled = np.array(
                            self.layers[layer_idx].beliefs,
                            copy=False,
                        ).reshape(bsz, layer_size)

                        for c in range(num_classes):
                            caches[layer_idx][c] = (
                                settled[
                                    c * per_class:(c + 1) * per_class
                                ]
                                .mean(axis=0, dtype=np.float32)
                                .astype(np.float32, copy=False)
                            )

                    self[-1].unclamp_state()

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
                        f"epoch {epoch + 1} avg energy = "
                        f"{epoch_energy / n_batches:.4f}"
                    ),
                )

        console.print(
            "\n[bold green]✓ I_avg Training complete.[/bold green]\n"
        )

        return self

    def save(self, dir_path: str) -> bool:
        self._require_configured()
        return super().save(dir_path)

    def load(self, dir_path: str) -> bool:
        self._require_configured()
        return super().load(dir_path)
