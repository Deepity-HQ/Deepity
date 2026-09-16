from typing import Optional

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

from ._backend import dy
from .layer import Layer, Linear, Activation
from .utils import _fit_with_progress


class SimplePCN(dy.SimplePCNetwork):
    """
    A Sequential Predictive Coding Network using synchronous settling dynamics.

    Networks are defined declaratively using Linear layers and activation
    functions, then configured before training.
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
        batch_size: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.architecture = architecture
        self.device = device
        self.batch_size = (
            dy.auto_batch_size()
            if batch_size is None
            else batch_size
        )

        self._configured = False
        self._learning_rate = 1e-4
        self._inference_rate = 1e-4
        self._lambda = 0.0001
        self._optimizer = "SGD"

        self._validate_architecture()

        super().__init__(self.batch_size, self.device)

    def _validate_architecture(self) -> None:
        """Validate the declarative network architecture."""

        if not self.architecture:
            raise ValueError("Network architecture cannot be empty.")

        if not isinstance(self.architecture[0], Linear):
            raise ValueError(
                "Network architecture must begin with a Linear layer."
            )

        if not isinstance(self.architecture[-1], Linear):
            raise ValueError(
                "Network architecture must end with a Linear layer."
            )

        previous_linear = self.architecture[0]

        for i, component in enumerate(self.architecture[1:], start=1):
            if isinstance(component, Linear):
                if component.in_n != previous_linear.out_n:
                    raise ValueError(
                        f"Dimension mismatch between Linear layers: "
                        f"{previous_linear.out_n} -> {component.in_n}."
                    )

                previous_linear = component

            elif isinstance(component, Activation):
                if i == len(self.architecture) - 1:
                    raise ValueError(
                        "An activation function cannot be the final "
                        "architecture component."
                    )

            else:
                raise TypeError(
                    f"Unsupported architecture component: "
                    f"{type(component).__name__}."
                )

    def _build_backend(self) -> None:
        """
        Translate the declarative architecture into backend PC layers.

        Each Linear layer becomes a backend layer. If an Activation follows
        it, that activation is attached to the backend layer.
        """

        for i, component in enumerate(self.architecture):
            if not isinstance(component, Linear):
                continue

            activation = "linear"

            if (
                i + 1 < len(self.architecture)
                and isinstance(self.architecture[i + 1], Activation)
            ):
                activation = self.architecture[i + 1].to_string()

            super().add_layer(
                component.in_n,
                component.out_n,
                lr=self._learning_rate,
                ir=self._inference_rate,
                lmbda=self._lambda,
                activation=activation,
                activation_deriv="d" + activation,
            )

    def configure(
        self,
        learning_rate: float = 1e-4,
        inference_rate: float = 1e-4,
        lmbda: float = 0.0001,
        optimizer: str = "SGD",
    ) -> None:
        """
        Configure the network and build its backend layers.

        Parameters
        ----------
        learning_rate:
            Weight-learning rate.

        inference_rate:
            State-inference rate used during settling.

        lmbda:
            Precision/regularization parameter for PC layers.

        optimizer:
            Optimizer to use: SGD, ADAM, or ADAMW.
        """

        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._lambda = lmbda
        self._optimizer = optimizer.upper()

        self._build_backend()

        super().set_optimizer(self._optimizer)
        super().randomize_weights()
        super().compile()

        self._configured = True

    def _require_configured(self) -> None:
        if not self._configured:
            raise RuntimeError(
                "The network must be configured before use. "
                "Call net.configure(...) first."
            )

    def set_learning_rate(self, lr: float) -> None:
        self._require_configured()

        self._learning_rate = lr

        for layer in self.layers:
            layer.set_learning_rate(lr)

    def set_inference_rate(self, ir: float) -> None:
        self._require_configured()

        self._inference_rate = ir

        for layer in self.layers:
            layer.set_inference_rate(ir)

    def set_optimizer(self, optimizer: str) -> None:
        self._require_configured()

        self._optimizer = optimizer.upper()
        super().set_optimizer(self._optimizer)

    def compile(self) -> None:
        self._require_configured()
        super().compile()

    def randomize_weights(self, dist: str = "") -> None:
        self._require_configured()
        super().randomize_weights()

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
    ) -> "SimplePCN":
        """
        Run multi-epoch training with a live Rich progress display.

        If initial_lr is omitted, the learning rate supplied to configure()
        is used.
        """
        self._require_configured()

        if initial_lr is None:
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

    def set_mu_cache_threshold(self, threshold: float) -> None:
        """
        Set the mu-cache staleness threshold on every layer.

        -1 disables caching.
         0 enables exact clamped-only caching.
        >0 extends caching to unclamped layers as an approximation.

        Thresholds around 0.05-0.1 previously showed substantial settling
        speedups, but their effect on multi-epoch training accuracy still
        requires independent validation.
        """
        self._require_configured()

        for layer in self.layers:
            layer.set_mu_cache_threshold(threshold)

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
    ) -> "SimplePCN":
        """
        Specialized training loop using Class-Average Caching (I_avg).

        Each requested hidden layer receives its own independent per-class
        cache. The terminal layer should not be included in cache_layers.
        """
        self._require_configured()

        if initial_lr is None:
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
            f"L{idx}({size})"
            for idx, size in cache_layers
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

        caches: dict[
            int,
            dict[int, npt.NDArray[np.float32]],
        ] = {
            idx: {}
            for idx, _ in cache_layers
        }

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

                    # Seed each cached layer independently.
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
                            np.copyto(
                                layer.beliefs,
                                init_beliefs,
                            )

                    self.clamp_input(X_batch.flatten())
                    self[-1].clamp_state(Y_batch.flatten())

                    energy = 0.0

                    for _ in range(steps):
                        energy += self.calculate_state()
                        self.update_state()

                    self.update_weights()

                    # Update each cache independently using the newly
                    # settled beliefs.
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
                                .mean(
                                    axis=0,
                                    dtype=np.float32,
                                )
                                .astype(
                                    np.float32,
                                    copy=False,
                                )
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