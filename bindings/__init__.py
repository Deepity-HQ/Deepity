import numpy as np
import numpy.typing as npt
from typing import Optional
from . import deepity as dy


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
