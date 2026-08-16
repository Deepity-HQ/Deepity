import numpy as np
import numpy.typing as npt
from typing import Optional
from . import deepity as dy

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

# pip install rich  (or: pip install rich --break-system-packages on an
# externally-managed Python install, e.g. Fedora's system Python)


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
    Shared training-loop implementation used by both SequentialPCN.fit()
    and ConvolutionalPCN.fit() -- both classes already implement a
    compatible train_step(X_batch, Y_batch, steps) -> float, so the loop
    body (batching, lr decay, progress display) doesn't need to differ
    between them. Not part of the public API -- call net.fit(...) instead.
    """
    console = Console()
    n = len(X)
    bsz = net.batch_size
    n_batches = n // bsz

    console.print(f"\n[bold]Training[/bold] {epochs} epochs, {steps} inference steps, "
                  f"{n_batches} batches/epoch (batch_size={bsz})...\n")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("\u2022"),
        TimeElapsedColumn(),
        TextColumn("\u2022"),
        TimeRemainingColumn(),
        TextColumn("[cyan]{task.fields[stats]}"),
        console=console,
    )

    with progress:
        epoch_task = progress.add_task("[bold]epochs", total=epochs, stats="")
        batch_task = progress.add_task("  batches", total=n_batches, stats="")

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
                X_batch = X_shuf[b * bsz:(b + 1) * bsz]
                Y_batch = Y_shuf[b * bsz:(b + 1) * bsz]

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

    console.print("\n[bold green]Training complete.[/bold green]\n")


class SequentialPCN(dy.DiscriminativePCNetwork):
    """
    A Sequential Predictive Coding Network (PCN) wrapper for the Deepity C++ backend.
    
    This class orchestrates the continuous-time inference and local Hebbian learning 
    phases of the network, delegating the heavy computational lifting to optimized 
    C++ SIMD micro-kernels.
    """

    def __init__(self, batch_size: Optional[int] = None) -> None:
        """
        Initializes the Predictive Coding Network.

        Args:
            batch_size (int, optional): The fixed batch size for training/inference. 
                If None, the backend will calculate the optimal batch size based on 
                the host OS and L2 cache availability to maximize OpenMP thread utilization.
        """
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
        """
        Appends a new DiscriminativePCLayer to the top of the network hierarchy.

        Args:
            in_features (int): Dimensionality of the input (bottom-up) states.
            out_features (int): Dimensionality of the output (top-down) predictions.
            lr (float): The base learning rate for synaptic weight updates.
            ir (float): The inference rate (Euler integration step size) for state relaxation.
            pr (float): The precision rate for tracking inverse variance (uncertainty).
            act (str): The activation function string identifier (e.g., 'tanh', 'relu').
            lmbda (float, optional): L2 regularization/weight decay penalty. Defaults to 0.0001.
        """
        super().add_layer(
            in_features, 
            out_features, 
            lr=lr, 
            ir=ir, 
            pr=pr,
            lmbda=lmbda, 
            activation=act, 
            activation_deriv='d' + act
        )

    def randomize_weights(self) -> None:
        """
        Populates all synaptic weight matrices across the network with 
        normally distributed random values scaled by the layer dimensions.
        """
        super().randomize_weights()

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """
        Executes a full forward initialization, state relaxation, and weight update cycle.

        During state relaxation, both boundary layers (input and target) are clamped, 
        allowing the hidden layers to perform gradient descent on free energy.

        Args:
            X (npt.NDArray[np.float32]): The flattened batched input data. 
                Example: For MNIST, this should be sized (batch_size * 784).
            Y (npt.NDArray[np.float32]): The flattened batched target labels.
                Example: For MNIST, this should be sized (batch_size * 10).
            steps (int): The number of continuous-time integration steps to run 
                before allowing the weights to update.

        Returns:
            float: The total network free energy accumulated over the inference steps.
        """
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
        Runs a full multi-epoch training loop with a live rich progress
        display (spinner, bar, per-batch lr/energy stats) -- a convenience
        wrapper around repeated train_step() calls, so scripts don't need
        to hand-roll the epoch/batch loop and print statements themselves.

        Args:
            X (npt.NDArray[np.float32]): Full, UNBATCHED training inputs,
                shape (n_samples, input_size) or already flattened per
                sample -- sliced into batches of self.batch_size internally.
            Y (npt.NDArray[np.float32]): Full, UNBATCHED training targets,
                shape (n_samples, output_size).
            epochs (int): Number of passes over the full dataset.
            steps (int): Inference steps per train_step() call.
            initial_lr (float): Learning rate at epoch 0.
            decay_rate (float): Multiplicative decay applied per epoch
                (lr = initial_lr * decay_rate ** epoch). 1.0 = no decay.
            shuffle (bool): Reshuffle sample order at the start of each
                epoch. Defaults to True.

        Returns:
            SequentialPCN: self, for chaining.
        """
        _fit_with_progress(self, X, Y, epochs, steps, initial_lr, decay_rate, shuffle)
        return self

    def set_learning_rate(self, lr: float) -> None:
        """
        Dynamically adjusts the base learning rate across all layers in the network.

        Args:
            lr (float): The new global learning rate to apply.
        """
        super().set_learning_rate(lr)

    def compile(self) -> None:
        """
        Calculates the exact footprint of the network and binds all layer states, 
        weights, and biases to a single, contiguous C++ MemoryArena block.
        
        Warning: 
            This must be called BEFORE randomize_weights() to ensure memory 
            pointers are correctly locked in prior to initialization.
        """
        super().compile()
    
    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        """
        Runs generative/discriminative inference on the provided input data.

        Unlike train_step, the terminal layer is left unclamped, allowing the 
        network's internal dynamics to push bottom-up signals until the top 
        layer settles into a predicted state.

        Args:
            X (npt.NDArray[np.float32]): The flattened input batch.
            steps (int): The number of Euler integration steps to allow the 
                network to settle.

        Returns:
            npt.NDArray[np.float32]: The predicted states (beliefs) of the terminal 
                layer, reshaped to (batch_size, num_classes).
        """
        self.reset_state()
        self.clamp_input(X.flatten())

        for _ in range(steps):
            self.calculate_state()
            self.update_state()

        return np.array(self[-1].beliefs)

    def save(self, dir_path: str) -> bool:
        """
        Serializes the network into a highly-optimized directory structure.
        
        Creates 'manifest.json' for architecture metadata, 'README.md' for 
        human-readable summaries, and 'weights.bin' for the raw memory payload.

        Args:
            dir_path (str): The destination directory (will be created if it doesn't exist).

        Returns:
            bool: True if serialization was successful, False otherwise.
        """
        return super().save(dir_path)

    def load(self, dir_path: str) -> bool:
        """
        Loads the network parameters directly into the contiguous MemoryArena.
        
        Warning: 
            The network must be instantiated with the identical layer topology 
            and must be compiled via `net.compile()` BEFORE calling load().

        Args:
            dir_path (str): The source directory containing the model files.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        return super().load(dir_path)

class ConvolutionalPCN(dy.ConvPCNetwork):
    """
    A Convolutional Predictive Coding Network (PCN) wrapper for the Deepity
    C++ backend.

    Structurally the convolutional counterpart to SequentialPCN: same
    orchestration role, delegating the settling/learning dynamics to the
    C++ ConvPCLayer/ConvPCNetwork backend. Two notable differences from
    SequentialPCN, both deliberate:

      1. train_step()/predict() are thin delegations to C++, not
         reimplemented in Python -- ConvPCNetwork.cpp already implements
         both directly (unlike DiscriminativePCNetwork, which only exposes
         the low-level primitives). Reimplementing the same loop here would
         risk the two drifting out of sync.

      2. add_layer() supports OPTIONAL shape auto-chaining: the underlying
         C++ AddLayer() deliberately does NOT infer in_channels/in_height/
         in_width from the previous layer (to avoid a class of silent
         shape-mismatch bug if a stride/padding choice doesn't produce the
         dimension you expected -- see ConvPCNetwork.h). This wrapper adds
         that convenience back in Python, where a mistake is just a
         Python-level exception, not a C++ memory-layout bug.

    Precision (pr) defaults to 0.0 here, unlike SequentialPCN's non-zero
    default -- ConvPCLayer::UpdatePrecision() is memory-safety-verified only
    (a full ASan pass with pr>0 succeeded), not yet correctness-verified via
    gradient check. Pass pr explicitly once that verification is done.
    """

    def __init__(self, batch_size: int) -> None:
        """
        Initializes the Convolutional Predictive Coding Network.

        Args:
            batch_size (int): The fixed batch size for training/inference.
                Unlike SequentialPCN, there is no auto-detection option --
                ConvPCNetwork's constructor requires an explicit batch size.
        """
        super().__init__(batch_size)
        # Tracks (channels, height, width) of the most recently added
        # layer's OUTPUT, so the next add_layer() call can optionally chain
        # onto it automatically. None until the first layer is added, and
        # reset to None after a terminal (out_channels=0) layer, since
        # nothing should be chained onto a terminal layer.
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
        """
        Appends a new ConvPCLayer to the network.

        Args:
            out_channels (int): Number of output channels. Pass 0 to mark
                this as the TERMINAL layer (no outgoing prediction) --
                matches ConvPCLayer's nextSize=0 convention.
            kernel_h, kernel_w (int): Convolution kernel dimensions.
            in_channels, in_height, in_width (int, optional): Input shape
                for this layer. If ALL THREE are omitted, they're inferred
                from the previous layer's output shape (out_channels,
                out_height, out_width) -- only valid from the second
                add_layer() call onward. The first layer must always
                specify all three explicitly, since there's no previous
                layer to infer from.
            stride_h, stride_w (int): Convolution stride. Defaults to 1.
            pad_h, pad_w (int): Zero-padding. Defaults to 0.
            lr (float): Base learning rate for weight updates.
            ir (float): Inference rate (Euler integration step size).
            pr (float): Precision rate. Defaults to 0.0 -- see class
                docstring for why this differs from SequentialPCN's default.
            lmbda (float): L2 weight decay coefficient.
            act (str): Activation function identifier (e.g. 'tanh', 'relu',
                'linear'). The derivative identifier is derived automatically
                as 'd' + act, matching SequentialPCN's convention.

        Raises:
            ValueError: If in_channels/in_height/in_width are omitted on
                the FIRST add_layer() call (nothing to infer from yet), or
                if only SOME of the three are provided (all-or-nothing).
        """
        shape_args = (in_channels, in_height, in_width)
        n_given = sum(a is not None for a in shape_args)

        if n_given == 0:
            if self._last_shape is None:
                raise ValueError(
                    "First add_layer() call must specify in_channels, "
                    "in_height, and in_width explicitly -- there is no "
                    "previous layer to infer them from."
                )
            in_channels, in_height, in_width = self._last_shape
        elif n_given != 3:
            raise ValueError(
                "in_channels/in_height/in_width must be given ALL together "
                "or OMITTED all together (to auto-chain from the previous "
                "layer's output shape) -- partial specification isn't "
                "supported."
            )

        super().add_layer(
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_h,
            kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            lr=lr,
            ir=ir,
            pr=pr,
            lmbda=lmbda,
            activation=act,
            activation_deriv="d" + act,
        )

        if out_channels > 0:
            # Read the shape back from the layer itself rather than
            # recomputing ConvOutDim() here in Python -- avoids the two
            # formulas ever drifting apart.
            added = self[-1]
            self._last_shape = (added.out_channels, added.out_height, added.out_width)
        else:
            # Terminal layer added -- nothing further should chain onto it.
            self._last_shape = None

    def compile(self) -> None:
        """
        Calculates the exact footprint of the network and binds all layer
        states, weights, and biases to a single, contiguous C++ MemoryArena
        block.

        Warning:
            This must be called BEFORE randomize_weights(), same as
            SequentialPCN.
        """
        super().compile()

    def randomize_weights(self) -> None:
        """
        Populates all convolution kernels across the network with randomly
        initialized values (He/Kaiming-style scaling, fan-in = the true
        receptive-field size, not the raw flattened layer size).
        """
        super().randomize_weights()

    def set_learning_rate(self, lr: float) -> None:
        """
        Dynamically adjusts the base learning rate across all layers.

        Unlike SequentialPCN, there is no network-wide SetLearningRate() in
        the C++ ConvPCNetwork class -- this loops over each layer's own
        set_learning_rate() from Python instead. Functionally equivalent,
        just implemented one level up.

        Args:
            lr (float): The new global learning rate to apply.
        """
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def set_inference_rate(self, ir: float) -> None:
        """Dynamically adjusts the inference rate across all layers."""
        for layer in self.layers:
            layer.set_inference_rate(ir)

    def set_precision_rate(self, pr: float) -> None:
        """
        Dynamically adjusts the precision rate across all layers.

        See class docstring: precision is not yet correctness-verified for
        ConvPCLayer. Setting pr > 0 here is only advisable once that
        verification has been done.
        """
        for layer in self.layers:
            layer.set_precision_rate(pr)

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """
        Executes a full clamp, settle, and weight-update cycle.

        Thin delegation to the C++ ConvPCNetwork::TrainStep() -- unlike
        SequentialPCN, this is NOT reimplemented from primitives in Python,
        since the C++ implementation already exists and is verified (see
        tConvDiagnose.cpp's 100% synthetic-floor-test result).

        Args:
            X (npt.NDArray[np.float32]): Flattened, batched input images.
                For a (batch, channels, H, W) input, flatten in that order.
            Y (npt.NDArray[np.float32]): Flattened, batched target labels,
                matching the terminal layer's flattened size.
            steps (int): Number of Euler integration steps to settle before
                the weight update.

        Returns:
            float: The network's total energy at the final settled step.
        """
        return super().train_step(X.flatten(), Y.flatten(), steps)

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        steps: int,
        initial_lr: float = 0.01,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "ConvolutionalPCN":
        """
        Runs a full multi-epoch training loop with a live rich progress
        display -- see SequentialPCN.fit() for the shared implementation
        and full argument docs. Behaves identically here; train_step()'s
        own X.flatten() means X can be passed with its natural
        (n_samples, channels, H, W) or (n_samples, flattened) shape either
        way, sliced into batches of self.batch_size internally.

        Returns:
            ConvolutionalPCN: self, for chaining.
        """
        _fit_with_progress(self, X, Y, epochs, steps, initial_lr, decay_rate, shuffle)
        return self

    def predict(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        """
        Runs inference on the provided input, terminal layer left unclamped.

        Thin delegation to C++ ConvPCNetwork::Predict() -- see train_step()
        docstring for why this isn't reimplemented from primitives here.

        Args:
            X (npt.NDArray[np.float32]): Flattened, batched input images.
            steps (int): Number of Euler integration steps to settle.

        Returns:
            npt.NDArray[np.float32]: Shape (batch_size, terminal_layer_size)
                -- the terminal layer's settled beliefs.
        """
        return super().predict(X.flatten(), steps)

    # --- Not yet implemented ---
    #
    # save()/load() are deliberately absent: ConvPCNetwork.cpp does not
    # implement persistence yet (see ConvPCLayer::ResyncLogPrecision(),
    # which exists in anticipation of this work but isn't wired to a
    # save/load path). Extend Deep::ModelIO to cover ConvPCLayer before
    # adding these here.
