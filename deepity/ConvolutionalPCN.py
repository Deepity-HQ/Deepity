from typing import Optional

import numpy as np
import numpy.typing as npt

from ._backend import dy
from ._interop import InputArray, to_numpy
from .layer import Activation, Convolution, Layer
from .utils import _fit_with_progress


class ConvolutionalPCN(dy.ConvPCNetwork):
    """
    Precision-weighted Convolutional Predictive Coding Network.

    The architecture is declared using Convolution and Activation
    objects (passed positionally, beginning with a Convolution), while
    `input_shape` and optimization/inference hyperparameters are
    supplied separately, see __init__() and configure().
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
        input_shape: tuple[int, int, int],
        batch_size: int,
    ) -> None:
        """
        @param architecture Convolution/Activation objects describing the layer stack.
        @param input_shape (channels, height, width) of a single input image.
        @param batch_size Number of samples processed per train_step()/predict() call.
        """
        self.architecture = architecture
        self.input_shape = input_shape
        # NOTE: batch_size is NOT stored as a plain attribute here,
        # dy.ConvPCNetwork (the C++ base class) exposes `batch_size` as a
        # READ-ONLY property (def_prop_ro, no setter at all). The `batch_size`
        # parameter is used directly below instead; self.batch_size becomes
        # valid and correct automatically once super().__init__() constructs
        # the underlying C++ object. See SimplePCN.__init__ for the same
        # pattern.

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
                activation=activation,
                activation_deriv="d" + activation,
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
        Configure and build the convolutional network. Must be called
        once, before train_step()/predict().

        @param learning_rate Learning rate for the forward weights.
        @param inference_rate Inference/settling rate.
        @param precision_rate Learning rate for precision weighting.
        @param lmbda Regularization parameter.
        """
        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._precision_rate = precision_rate
        self._lambda = lmbda

        self._build_backend()

        # Compile() must run before RandomizeWeights(): each layer uses its
        # own small, independently-sized arena pre-compile, so randomizing
        # first writes into a buffer Compile() then discards/reorganizes.
        self.compile()
        self.randomize_weights()

        return self

    def _require_configured(self) -> None:
        if self._learning_rate is None:
            raise RuntimeError(
                "ConvolutionalPCN has not been configured. "
                "Call net.configure(...) before training or prediction."
            )

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate on every layer."""
        self._require_configured()

        for layer in self.layers:
            layer.set_learning_rate(learning_rate)

        self._learning_rate = learning_rate

    def set_inference_rate(self, inference_rate: float) -> None:
        """Set the inference/settling rate on every layer."""
        self._require_configured()

        for layer in self.layers:
            layer.set_inference_rate(inference_rate)

        self._inference_rate = inference_rate

    def set_precision_rate(self, precision_rate: float) -> None:
        """Set the precision-weighting learning rate on every layer."""
        self._require_configured()

        for layer in self.layers:
            layer.set_precision_rate(precision_rate)

        self._precision_rate = precision_rate

    def set_lambda(self, lmbda: float) -> None:
        """Set the regularization parameter on every layer."""
        self._require_configured()

        for layer in self.layers:
            layer.set_lambda(lmbda)

        self._lambda = lmbda

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self, seed: Optional[int] = None, dist: str = "normal") -> None:
        """Randomize all network weights. `seed` gives a reproducible run;
        `dist` e.g. "normal(0,1)" or "uniform(-0.3,0.3)" (the default
        "normal" has no "(a,b)" and so deliberately falls back to each
        layer's own existing init)."""
        super().randomize_weights(seed, dist)

    def clamp_input(
        self,
        X: InputArray,
    ) -> None:
        super().clamp_input(to_numpy(X, np.float32).flatten())

    def project_forward(self) -> None:
        super().project_forward()

    def update_precision(self) -> None:
        super().update_precision()

    def train_step(
        self,
        X: InputArray,
        Y: InputArray,
        steps: int,
    ) -> float:
        """Run one training step on a single batch, settling for
        `steps` iterations before updating weights.
        @return Total settling energy accumulated over `steps`."""
        self._require_configured()

        return super().train_step(
            to_numpy(X, np.float32).flatten(),
            to_numpy(Y, np.float32).flatten(),
            steps,
        )

    def train_step_with_projection(
        self,
        X: InputArray,
        Y: InputArray,
        steps: int,
    ) -> float:
        """Same as train_step(), but initializes every layer's
        beliefs via a forward projection pass first instead of
        whatever state they held from the previous call."""
        self._require_configured()

        return super().train_step_with_projection(
            to_numpy(X, np.float32).flatten(),
            to_numpy(Y, np.float32).flatten(),
            steps,
        )

    def predict(
        self,
        X: InputArray,
        steps: int,
    ) -> npt.NDArray[np.float32]:
        """Run inference on `X`, settling for `steps` iterations.
        @return The terminal layer's beliefs."""
        self._require_configured()

        return np.asarray(
            super().predict(
                to_numpy(X, np.float32).flatten(),
                steps,
            )
        )

    def predict_with_projection(
        self,
        X: InputArray,
        steps: int,
    ) -> npt.NDArray[np.float32]:
        """Same as predict(), but initializes every layer's beliefs
        via a forward projection pass first."""
        self._require_configured()

        return np.asarray(
            super().predict_with_projection(
                to_numpy(X, np.float32).flatten(),
                steps,
            )
        )

    def fit(
        self,
        X: InputArray,
        Y: InputArray,
        epochs: int,
        steps: int,
        initial_lr: Optional[float] = None,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "ConvolutionalPCN":
        """Train for multiple epochs over (X, Y), batching by
        `self.batch_size` and printing a progress bar."""
        self._require_configured()

        if initial_lr is None:
            if self._learning_rate is None:
                raise RuntimeError("Learning rate has not been configured.")
            initial_lr = self._learning_rate

        _fit_with_progress(
            self,
            to_numpy(X, np.float32),
            to_numpy(Y, np.float32),
            epochs,
            steps,
            initial_lr,
            decay_rate,
            shuffle,
        )

        return self
