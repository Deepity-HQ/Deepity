from typing import Optional

import numpy as np
import numpy.typing as npt

from ._backend import dy
from .layer import Activation, Convolution, Layer
from .utils import _fit_with_progress


class ConvolutionalPCN(dy.ConvPCNetwork):
    """
    Standard precision-weighted Convolutional Predictive Coding Network.
    """

    def __init__(
        self,
        *architecture: Layer,
        input_shape: tuple[int, int, int],
        batch_size: int,
    ) -> None:
        self.architecture = architecture
        self.input_shape = input_shape
        self.batch_size = batch_size

        self._learning_rate: Optional[float] = None
        self._inference_rate: Optional[float] = None
        self._precision_rate: Optional[float] = None
        self._lambda: Optional[float] = None

        self._validate_architecture()

        super().__init__(batch_size)

    def _validate_architecture(self) -> None:
        if not self.architecture:
            raise ValueError(
                "ConvolutionalPCN requires at least one layer."
            )

        if not isinstance(self.architecture[0], Convolution):
            raise TypeError(
                "ConvolutionalPCN architecture must begin with "
                "a Convolution layer."
            )

        for layer in self.architecture:
            if not isinstance(layer, (Convolution, Activation)):
                raise TypeError(
                    f"Unsupported architecture element: "
                    f"{type(layer).__name__}."
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

            self.add_layer(
                out_channels=layer.out_channels,
                kernel_h=layer.kernel_h,
                kernel_w=layer.kernel_w,
                in_channels=in_channels,
                in_height=in_height,
                in_width=in_width,
                stride_h=layer.stride_h,
                stride_w=layer.stride_w,
                pad_h=layer.pad_h,
                pad_w=layer.pad_w,
                lr=self._learning_rate,
                ir=self._inference_rate,
                pr=self._precision_rate,
                lmbda=self._lambda,
                act=activation,
            )

            if layer.out_channels > 0:
                backend_layer = self[-1]

                in_channels = backend_layer.out_channels
                in_height = backend_layer.out_height
                in_width = backend_layer.out_width
            else:
                break

    def configure(
        self,
        learning_rate: float = 1e-6,
        inference_rate: float = 0.1,
        precision_rate: float = 0.0,
        lmbda: float = 1e-2,
    ) -> "ConvolutionalPCN":
        """
        Configure and build the convolutional network.
        """
        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._precision_rate = precision_rate
        self._lambda = lmbda

        self._build_backend()

        self.randomize_weights()
        self.compile()

        return self

    def _require_configured(self) -> None:
        if self._learning_rate is None:
            raise RuntimeError(
                "ConvolutionalPCN has not been configured. "
                "Call net.configure(...) before training or prediction."
            )

    def set_learning_rate(self, learning_rate: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_learning_rate(learning_rate)

        self._learning_rate = learning_rate

    def set_inference_rate(self, inference_rate: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_inference_rate(inference_rate)

        self._inference_rate = inference_rate

    def set_precision_rate(self, precision_rate: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_precision_rate(precision_rate)

        self._precision_rate = precision_rate

    def set_lambda(self, lmbda: float) -> None:
        self._require_configured()

        for layer in self.layers:
            layer.set_lambda(lmbda)

        self._lambda = lmbda

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def clamp_input(
        self,
        X: npt.NDArray[np.float32],
    ) -> None:
        super().clamp_input(X.flatten())

    def project_forward(self) -> None:
        super().project_forward()

    def update_precision(self) -> None:
        super().update_precision()

    def train_step(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        steps: int,
    ) -> float:
        self._require_configured()

        return super().train_step(
            X.flatten(),
            Y.flatten(),
            steps,
        )

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

    def predict(
        self,
        X: npt.NDArray[np.float32],
        steps: int,
    ) -> npt.NDArray[np.float32]:
        self._require_configured()

        return np.asarray(
            super().predict(
                X.flatten(),
                steps,
            )
        )

    def predict_with_projection(
        self,
        X: npt.NDArray[np.float32],
        steps: int,
    ) -> npt.NDArray[np.float32]:
        self._require_configured()

        return np.asarray(
            super().predict_with_projection(
                X.flatten(),
                steps,
            )
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
    ) -> "ConvolutionalPCN":
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
