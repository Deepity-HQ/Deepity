import numpy as np
import numpy.typing as npt
from typing import Optional

# Import the raw compiled C++ bindings
from . import pydeepity as dy

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

__all__ = ["SequentialPCN", "SimplePCN", "ConvolutionalPCN"]

# ============================================================================
# Shared Progress & Training Utilities
# ============================================================================

def _fit_with_progress(
    net,
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float32],
    epochs: int,
    steps: int,
    initial_lr: float = 0.01,
    decay_rate: float = 1.0,
    shuffle: bool = True,
) -> None:
    """
    Shared training-loop implementation used by all PCN classes. 
    It delegates to the specific class's `train_step()` method.
    """
    console = Console()
    n = len(X)
    bsz = net.batch_size
    n_batches = n // bsz

    console.print(
        f"\n[bold cyan]Training[/bold cyan] [dim]|[/dim] {epochs} epochs [dim]|[/dim] "
        f"{steps} inference steps [dim]|[/dim] {n_batches} batches/epoch (batch_size={bsz})\n"
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

    with progress:
        epoch_task = progress.add_task("[bold]Epochs", total=epochs, stats="")
        batch_task = progress.add_task("  Batches", total=n_batches, stats="")

        for epoch in range(epochs):
            current_lr = initial_lr * (decay_rate ** epoch)
            net.set_learning_rate(current_lr)

            if shuffle:
                indices = np.random.permutation(n)
                X_shuf, Y_shuf = X[indices], Y[indices]
            else:
                X_shuf, Y_shuf = X, Y

            epoch_energy = 0.0
            progress.reset(batch_task, total=n_batches)

            for b in range(n_batches):
                X_batch = X_shuf[b * bsz : (b + 1) * bsz]
                Y_batch = Y_shuf[b * bsz : (b + 1) * bsz]

                energy = net.train_step(X_batch, Y_batch, steps)
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

    console.print("\n[bold green]✓ Training complete.[/bold green]\n")


class _PCNMixin:
    """
    Provides shared top-level Python functionality (like fit()) to all Network wrappers.
    """
    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        steps: int,
        initial_lr: float = 0.01,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ):
        """
        Runs a full multi-epoch training loop with a live rich progress display.
        Delegates per-batch execution to the class's `train_step()` method.
        """
        _fit_with_progress(self, X, Y, epochs, steps, initial_lr, decay_rate, shuffle)
        return self


# ============================================================================
# Core Network Wrappers
# ============================================================================

class SequentialPCN(dy.DiscriminativePCNetwork, _PCNMixin):
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

    def save(self, dir_path: str) -> bool:
        return super().save(dir_path)

    def load(self, dir_path: str) -> bool:
        return super().load(dir_path)


class SimplePCN(dy.SimplePCNetwork, _PCNMixin):
    """
    A Sequential Predictive Coding Network built from precision-stripped SimplePCLayers.
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
        act: str, 
        lmbda: float = 0.0001
    ) -> None:
        super().add_layer(
            in_features, out_features, lr=lr, ir=ir,
            lmbda=lmbda, activation=act, activation_deriv='d' + act
        )

    def set_learning_rate(self, lr: float) -> None:
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        self.reset_state()
        self.clamp_input(X)
        self[-1].clamp_state(Y)

        total_energy: float = 0.0
        for _ in range(steps):
            total_energy += self.calculate_state()
            self.update_state()

        self.update_weights()
        self[-1].unclamp_state()
        
        return total_energy

    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        self.reset_state()
        self.clamp_input(X.flatten())

        for _ in range(steps):
            self.calculate_state()
            self.update_state()

        return np.array(self[-1].beliefs)

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
    ) -> "SimplePCN":
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
                            caches[layer_idx][c] = settled[c * per_class : (c + 1) * per_class].mean(axis=0)

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

class ConvolutionalPCN(dy.ConvPCNetwork, _PCNMixin):
    """
    A Convolutional Predictive Coding Network (PCN) wrapper for the Deepity C++ backend.
    """
    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size)
        self._last_shape: Optional[tuple[int, int, int]] = None

    def add_layer(
        self,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        in_channels: Optional[int] = None,
        in_height: Optional[int] = None,
        in_width: Optional[int] = None,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
        lr: float = 1e-6,
        ir: float = 0.1,
        pr: float = 0.0,
        lmbda: float = 1e-4,
        act: str = "relu",
    ) -> None:
        shape_args = (in_channels, in_height, in_width)
        n_given = sum(a is not None for a in shape_args)

        if n_given == 0:
            if self._last_shape is None:
                raise ValueError(
                    "First add_layer() call must specify in_channels, in_height, and in_width explicitly."
                )
            in_channels, in_height, in_width = self._last_shape
        elif n_given != 3:
            raise ValueError("in_channels/in_height/in_width must be given ALL together or OMITTED all together.")

        super().add_layer(
            in_channels, out_channels, in_height, in_width,
            kernel_h, kernel_w, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
            lr=lr, ir=ir, pr=pr, lmbda=lmbda, activation=act, activation_deriv="d" + act,
        )

        if out_channels > 0:
            added = self[-1]
            self._last_shape = (added.out_channels, added.out_height, added.out_width)
        else:
            self._last_shape = None

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def set_learning_rate(self, lr: float) -> None:
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def set_inference_rate(self, ir: float) -> None:
        for layer in self.layers:
            layer.set_inference_rate(ir)

    def set_precision_rate(self, pr: float) -> None:
        for layer in self.layers:
            layer.set_precision_rate(pr)

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        return super().train_step(X.flatten(), Y.flatten(), steps)

    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        return super().predict(X.flatten(), steps)

class SimpleConvolutionalPCN(dy.SimpleConvPCNetwork, _PCNMixin):
    """
    A Convolutional Predictive Coding Network built from precision-free,
    AdamW-capable SimpleConvPCLayers. Mirrors ConvolutionalPCN's
    shape-inference convenience (in_channels/in_height/in_width can be
    omitted after the first add_layer() call, inferred from the previous
    layer's output shape) -- minus precision, which doesn't exist here.
    """
    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size)
        self._last_shape: Optional[tuple[int, int, int]] = None

    def add_layer(
        self,
        out_channels: int,
        kernel_h: int,
        kernel_w: int,
        in_channels: Optional[int] = None,
        in_height: Optional[int] = None,
        in_width: Optional[int] = None,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
        lr: float = 1e-6,
        ir: float = 0.1,
        lmbda: float = 1e-4,
        act: str = "relu",
    ) -> None:
        shape_args = (in_channels, in_height, in_width)
        n_given = sum(a is not None for a in shape_args)

        if n_given == 0:
            if self._last_shape is None:
                raise ValueError(
                    "First add_layer() call must specify in_channels, in_height, and in_width explicitly."
                )
            in_channels, in_height, in_width = self._last_shape
        elif n_given != 3:
            raise ValueError("in_channels/in_height/in_width must be given ALL together or OMITTED all together.")

        super().add_layer(
            in_channels, out_channels, in_height, in_width,
            kernel_h, kernel_w, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
            lr=lr, ir=ir, lmbda=lmbda, activation=act, activation_deriv="d" + act,
        )

        if out_channels > 0:
            added = self[-1]
            self._last_shape = (added.out_channels, added.out_height, added.out_width)
        else:
            self._last_shape = None

    def set_optimizer(self, optimizer: str) -> None:
        """Sets the optimizer: ADAM, ADAMW, or SGD. Call BEFORE compile() --
        see SimpleConvPCNetwork's C++ docs; buffer sizing depends on this
        being set before Compile() allocates the shared arena."""
        super().set_optimizer(optimizer)

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def set_learning_rate(self, lr: float) -> None:
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def set_inference_rate(self, ir: float) -> None:
        for layer in self.layers:
            layer.set_inference_rate(ir)

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        return super().train_step(X.flatten(), Y.flatten(), steps)

    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        return super().predict(X.flatten(), steps)