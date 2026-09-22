from typing import Optional

import numpy as np
import numpy.typing as npt

from ._backend import dy
from .layer import Activation, Layer, Linear
from .utils import _fit_with_progress


class DKPPCN(dy.DirectKPPCNetwork):
    """
    Predictive Coding Network using Direct Kolen-Pollack (DKP)
    feedback alignment.

    DKP maintains independently learned forward weights (W) and
    direct feedback weights (Psi). The feedback pathway connects
    hidden layers directly to the terminal error rather than
    relaying errors layer-by-layer.

    Unlike the other PCN implementations, DKP performs a direct
    feedback alignment update before the ordinary settling loop.
    This allows inference_steps=1 to be the default operating point.
    """

    def __init__(
        self,
        *architecture: Layer,
        batch_size: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        if batch_size is None:
            batch_size = 64

        self.architecture = architecture
        self.device = device

        self._learning_rate: Optional[float] = None
        self._inference_rate: Optional[float] = None
        self._feedback_rate: Optional[float] = None
        self._lambda: Optional[float] = None
        self._optimizer: Optional[str] = None
        self._psi_optimizer: Optional[str] = None

        self._validate_architecture()

        super().__init__(batch_size, device)

    def _validate_architecture(self) -> None:
        if not self.architecture:
            raise ValueError("DKPPCN requires at least one layer.")

        if not isinstance(self.architecture[0], Linear):
            raise TypeError("Architecture must begin with a Linear layer.")

        if not isinstance(self.architecture[-1], Linear):
            raise TypeError("Architecture must end with a Linear layer.")

        previous_linear = None

        for layer in self.architecture:
            if isinstance(layer, Linear):
                if previous_linear is not None:
                    if previous_linear.out_n != layer.in_n:
                        raise ValueError(
                            "Layer dimensions do not match: "
                            f"{previous_linear.out_n} != {layer.in_n}."
                        )

                previous_linear = layer

            elif isinstance(layer, Activation):
                if previous_linear is None:
                    raise ValueError(
                        "An activation cannot appear before the first Linear layer."
                    )

            else:
                raise TypeError(
                    f"Unsupported architecture element: {type(layer).__name__}."
                )

    def _build_backend(self) -> None:
        terminal_size = self._terminal_size()

        for i, layer in enumerate(self.architecture):
            if not isinstance(layer, Linear):
                continue

            if (
                i + 1 < len(self.architecture)
                and isinstance(self.architecture[i + 1], Activation)
            ):
                activation = self.architecture[i + 1].to_string()
            else:
                activation = "linear"

            activation_deriv = "d" + activation

            super().add_layer(
                layer.in_n,
                layer.out_n,
                terminal_size,
                lr=self._learning_rate,
                ir=self._inference_rate,
                fl=self._feedback_rate,
                lmbda=self._lambda,
                activation=activation,
                activation_deriv=activation_deriv,
            )

        super().add_layer(
            terminal_size,
            0,
            terminal_size,
            lr=self._learning_rate,
            ir=self._inference_rate,
            fl=self._feedback_rate,
            lmbda=self._lambda,
            activation="linear",
            activation_deriv="dlinear",
        )

    def _terminal_size(self) -> int:
        for layer in reversed(self.architecture):
            if isinstance(layer, Linear):
                return layer.out_n

        raise RuntimeError("No terminal Linear layer found.")

    def configure(
        self,
        learning_rate: float = 1e-6,
        inference_rate: float = 0.1,
        feedback_rate: float = 1e-4,
        lmbda: float = 1e-2,
        optimizer: str = "SGD",
        psi_optimizer: str = "SGD",
    ) -> "DKPPCN":
        """
        Configure and initialize the DKP network.

        Parameters
        ----------
        learning_rate:
            Learning rate for the forward weights W.

        inference_rate:
            Inference/settling rate.

        feedback_rate:
            Learning rate for the direct feedback weights Psi.

        lmbda:
            Regularization parameter.

        optimizer:
            Optimizer used for W.

        psi_optimizer:
            Independent optimizer used for Psi.
        """
        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._feedback_rate = feedback_rate
        self._lambda = lmbda
        self._optimizer = optimizer
        self._psi_optimizer = psi_optimizer

        self._build_backend()

        self.set_optimizer(optimizer)
        self.set_psi_optimizer(psi_optimizer)

        self.compile()
        self.randomize_weights()

        return self

    def _require_configured(self) -> None:
        if self._learning_rate is None:
            raise RuntimeError(
                "DKPPCN has not been configured. "
                "Call net.configure(...) before training or prediction."
            )

    def set_optimizer(self, optimizer: str) -> None:
        """Set the optimizer used for the forward weights W."""
        super().set_optimizer(optimizer)

    def set_psi_optimizer(self, optimizer: str) -> None:
        """Set the optimizer used for the feedback weights Psi."""
        super().set_psi_optimizer(optimizer)

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate for the forward weights W."""
        super().set_learning_rate(learning_rate)

        self._learning_rate = learning_rate

    def set_feedback_rate(self, feedback_rate: float) -> None:
        """Set the learning rate for the feedback weights Psi."""
        super().set_feedback_rate(feedback_rate)

        self._feedback_rate = feedback_rate

    def set_inference_rate(self, inference_rate: float) -> None:
        """Set the inference/settling rate."""
        for layer in self.layers:
            layer.set_inference_rate(inference_rate)

        self._inference_rate = inference_rate

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def train_step(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        inference_steps: int = 1,
    ) -> float:
        self._require_configured()

        return super().train_step(
            X.flatten(),
            Y.flatten(),
            inference_steps,
        )

    def predict(
        self,
        X: npt.NDArray[np.float32],
        inference_steps: int = 1,
    ) -> npt.NDArray[np.float32]:
        self._require_configured()

        return np.asarray(
            super().predict(
                X.flatten(),
                inference_steps,
            )
        )

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        inference_steps: int = 1,
        initial_lr: Optional[float] = None,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "DKPPCN":
        """
        Train the network for multiple epochs.

        DKP defaults to one inference step because direct feedback
        alignment is specifically intended to reduce the amount of
        iterative settling required.
        """
        self._require_configured()

        if initial_lr is None:
            initial_lr = self._learning_rate

        _fit_with_progress(
            self,
            X,
            Y,
            epochs,
            inference_steps,
            initial_lr, # type: ignore
            decay_rate,
            shuffle,
        )

        return self
