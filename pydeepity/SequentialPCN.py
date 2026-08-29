from ._backend import dy
from typing import Optional
import numpy as np
import numpy.typing as npt
from .utils import _fit_with_progress

class SequentialPCN(dy.DiscriminativePCNetwork):
    """
    A Sequential Predictive Coding Network wrapper for the Deepity C++ backend.
    """
    def __init__(self, batch_size: Optional[int] = None) -> None:
        bsz = dy.auto_batch_size() if batch_size is None else batch_size
        super().__init__(bsz)
    
    def add_layer(
        self, 
        in_features: int, 
        out_features: int, 
        lr: float, 
        ir: float, 
        pr: float, 
        act: str, 
        lmbda: float = 0.0001
    ) -> None:
        super().add_layer(
            in_features, out_features, lr=lr, ir=ir, pr=pr,
            lmbda=lmbda, activation=act, activation_deriv='d' + act
        )

    def set_learning_rate(self, lr: float) -> None:
        super().set_learning_rate(lr)

    def set_optimizer(self, opt: str) -> None:
        super().set_optimizer(opt)

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """Executes a full forward clamp, state relaxation, and weight update cycle."""
        self.reset_state()
        self.clamp_input(X)
        self[-1].clamp_state(Y)

        total_energy: float = 0.0
        for _ in range(steps):
            total_energy += self.calculate_state()
            self.update_state()

        self.update_weights()
        self.update_precision()
        self[-1].unclamp_state()
        
        return total_energy

    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        """Runs generative/discriminative inference on the provided input data."""
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
        initial_lr: float = 0.01,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "SequentialPCN":
        """
        Runs a full multi-epoch training loop with a live rich progress display.
        Delegates per-batch execution to `train_step()`.
        """
        _fit_with_progress(self, X, Y, epochs, steps, initial_lr, decay_rate, shuffle)
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
        initial_lr: float = 0.01,
        decay_rate: float = 1.0,
        reset_cache_per_epoch: bool = True
    ) -> "SequentialPCN":
        """
        Specialized training loop for deep networks that utilizes Class-Average
        Caching (I_avg). Manages the StreamAlignedBatcher and dynamically seeds
        MULTIPLE hidden layers simultaneously -- a generalization of the
        original single hidden_layer_index/hidden_size version, which could
        only ever cache one layer.

        @param cache_layers List of (layer_index, layer_size) pairs -- every
            layer listed gets its own independent per-class cache, seeded and
            captured every batch. layer_index is the index into self.layers
            (0 = the first, input-adjacent layer; the terminal layer, always
            the last index, should NOT be included here -- it's clamped to
            the target directly, never cached).
        """
        console = Console()
        bsz = self.batch_size

        batcher = dy.StreamAlignedBatcher(
            X, Y, labels,
            X.shape[1], Y.shape[1],
            num_classes, per_class, 42
        )
        n_batches = batcher.num_batches_per_epoch()

        layer_names = ", ".join(f"L{idx}({sz})" for idx, sz in cache_layers)
        console.print(
            f"\n[bold cyan]Training I_avg[/bold cyan] [dim]|[/dim] {epochs} epochs [dim]|[/dim] "
            f"{steps} inference steps [dim]|[/dim] {n_batches} batches/epoch [dim]|[/dim] "
            f"caching: {layer_names}\n"
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

        # caches[layer_index] = {class: avg_vector} -- one independent cache
        # per listed layer, not a single shared one.
        caches: dict[int, dict[int, npt.NDArray[np.float32]]] = {idx: {} for idx, _ in cache_layers}

        with progress:
            epoch_task = progress.add_task("[bold]Epochs", total=epochs, stats="")
            batch_task = progress.add_task("  Batches", total=n_batches, stats="")

            for epoch in range(epochs):
                if reset_cache_per_epoch:
                    for layer_idx in caches:
                        caches[layer_idx].clear()

                current_lr = initial_lr * (decay_rate ** epoch)
                self.set_learning_rate(current_lr)

                epoch_energy = 0.0
                progress.reset(batch_task, total=n_batches)

                for b in range(n_batches):
                    X_batch, Y_batch, _ = batcher.get_batch()
                    self.reset_state()

                    # 1. Apply EVERY listed layer's cache independently --
                    # each layer seeds from its OWN cache, not a shared one.
                    for layer_idx, layer_size in cache_layers:
                        cache = caches[layer_idx]
                        if cache:
                            init_beliefs = np.zeros((bsz, layer_size), dtype=np.float32)
                            for c in range(num_classes):
                                if c in cache:
                                    init_beliefs[c * per_class : (c + 1) * per_class] = cache[c]

                            layer = self.layers[layer_idx]
                            np.copyto(layer.beliefs, init_beliefs)

                    self.clamp_input(X_batch.flatten())
                    self[-1].clamp_state(Y_batch.flatten())

                    # 2. Settle & Update
                    energy = 0.0
                    for _ in range(steps):
                        energy += self.calculate_state()
                        self.update_state()

                    self.update_weights()

                    # 3. Cache EVERY listed layer's newly settled beliefs
                    # independently.
                    for layer_idx, layer_size in cache_layers:
                        settled = np.array(self.layers[layer_idx].beliefs, copy=False).reshape(bsz, layer_size)
                        for c in range(num_classes):
                            caches[layer_idx][c] = settled[
                                c * per_class : (c + 1) * per_class
                            ].mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)

                    self[-1].unclamp_state()

                    # 4. Progress tracking
                    epoch_energy += energy
                    avg_so_far = epoch_energy / (b + 1)

                    progress.update(
                        batch_task,
                        advance=1,
                        stats=f"lr={current_lr:.5f}  energy={energy:8.2f}  avg={avg_so_far:8.2f}",
                    )

                progress.update(
                    epoch_task,
                    advance=1,
                    stats=f"epoch {epoch + 1} avg energy = {epoch_energy / n_batches:.4f}",
                )

        console.print("\n[bold green]✓ I_avg Training complete.[/bold green]\n")
        return self

    def save(self, dir_path: str) -> bool:
        return super().save(dir_path)

    def load(self, dir_path: str) -> bool:
        return super().load(dir_path)
