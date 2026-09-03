from ._backend import dy
from .utils import _fit_with_progress
from typing import Optional
import numpy as np
import numpy.typing as npt

class SimpleConvolutionalPCN(dy.SimpleConvPCNetwork):
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

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        steps: int,
        initial_lr: float = 0.01,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "SimpleConvolutionalPCN":
        """
        Runs a full multi-epoch training loop with a live rich progress display.
        Delegates per-batch execution to `train_step()`.
        """
        _fit_with_progress(self, X, Y, epochs, steps, initial_lr, decay_rate, shuffle)
        return self

    def train_step_with_projection(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        return super().train_step_with_projection(X.flatten(), Y.flatten(), steps)

    def predict_with_projection(self, X: npt.NDArray[np.float32], steps: int) -> npt.NDArray[np.float32]:
        return super().predict_with_projection(X.flatten(), steps)
