from ._backend import dy
from .layer import Layer, Convolution, Activation
from .utils import _fit_with_progress
from typing import Optional
import numpy as np
import numpy.typing as npt


class SimpleConvolutionalPCN(dy.SimpleConvPCNetwork):
    """
    Simple convolutional Predictive Coding Network.

    The architecture is declared using Convolution and Activation objects,
    while optimization and inference parameters are supplied through
    configure().
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
        input_shape: tuple[int, int, int],
        batch_size: Optional[int] = None,
    ) -> None:
        bsz = dy.auto_batch_size() if batch_size is None else batch_size
        super().__init__(bsz)

        self.architecture = architecture
        self.input_shape = input_shape

        self._learning_rate: Optional[float] = None
        self._inference_rate: Optional[float] = None
        self._lambda: Optional[float] = None
        self._optimizer: Optional[str] = None
        self._configured = False

        self._validate_architecture()

    def _validate_architecture(self) -> None:
        if not self.architecture:
            raise ValueError("Architecture cannot be empty.")

        if not isinstance(self.architecture[0], Convolution):
            raise ValueError("Architecture must begin with Convolution.")

        previous_was_activation = False

        for layer in self.architecture:
            if isinstance(layer, Convolution):
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
        in_channels, in_height, in_width = self.input_shape

        for i, layer in enumerate(self.architecture):
            if not isinstance(layer, Convolution):
                continue

            activation = "linear"

            if (
                i + 1 < len(self.architecture)
                and isinstance(self.architecture[i + 1], Activation)
            ):
                activation = self.architecture[i + 1].to_string()

            super().add_layer(
                in_channels,
                layer.out_channels,
                in_height,
                in_width,
                layer.kernel_h,
                layer.kernel_w,
                stride_h=layer.stride_h,
                stride_w=layer.stride_w,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
                lr=self._learning_rate,
                ir=self._inference_rate,
                lmbda=self._lambda,
                activation=activation,
                activation_deriv="d" + activation,
            )

            added = self[-1]

            if layer.out_channels > 0:
                in_channels = added.out_channels
                in_height = added.out_height
                in_width = added.out_width

    def configure(
        self,
        learning_rate: float = 1e-6,
        inference_rate: float = 0.1,
        lmbda: float = 1e-2,
        optimizer: str = "SGD",
    ) -> None:
        """
        Configure network hyperparameters and initialize the backend.
        """

        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._lambda = lmbda
        self._optimizer = optimizer

        self._build_backend()

        # The backend requires the optimizer to be selected before compile.
        self.set_optimizer(optimizer)
        self.randomize_weights()
        self.compile()

        self._configured = True

    def _require_configured(self) -> None:
        if not self._configured:
            raise RuntimeError(
                "SimpleConvolutionalPCN must be configured before use. "
                "Call configure() first."
            )

    def set_learning_rate(self, lr: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_learning_rate(lr)

        self._learning_rate = lr

    def set_inference_rate(self, ir: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_inference_rate(ir)

        self._inference_rate = ir

    def set_lambda(self, lmbda: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_lambda(lmbda)

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
        return super().train_step(X.flatten(), Y.flatten(), steps)

    def predict(
        self,
        X: npt.NDArray[np.float32],
        steps: int,
    ) -> npt.NDArray[np.float32]:
        self._require_configured()
        return super().predict(X.flatten(), steps)

    def train_step_with_projection(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        steps: int,
    ) -> float:
        self._require_configured()
        return super().train_step_with_projection(
            X.flatten(),
            Y.flatten(),
            steps,
        )

    def predict_with_projection(
        self,
        X: npt.NDArray[np.float32],
        steps: int,
    ) -> npt.NDArray[np.float32]:
        self._require_configured()
        return super().predict_with_projection(
            X.flatten(),
            steps,
        )

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        steps: int,
        initial_lr: Optional[float] = None,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "SimpleConvolutionalPCN":
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
