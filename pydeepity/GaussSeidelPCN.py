from typing import Optional

import numpy as np
import numpy.typing as npt

from ._backend import dy
from .layer import Layer, Linear, Activation
from .utils import _fit_with_progress


class GaussSeidelPCN(dy.GaussSeidelPCNetwork):
    """
    A Predictive Coding Network with Gauss-Seidel (sequential-sweep)
    settling dynamics.

    Unlike the fully-synchronous (Jacobi) dynamics used by other PCN
    implementations, Gauss-Seidel settling allows later layers to see
    already-updated values from earlier layers within the same step.

    A full settling step consists of three sweeps:

        1. Update z states.
        2. Recompute predictions using the fresh z states.
        3. Recompute errors using the fresh predictions.

    Traced directly from ngc-learn's execution graph:

        E2,E3 -> z0,z1,z2,z3 -> W1,W2,W3 -> e1,e2,e3

    UNPROVEN EXPERIMENT -- the underlying mathematics has only been
    smoke-tested (finite energy with a generally decreasing trend).
    It has not yet been verified with an independent finite-difference
    gradient check. Accuracy conclusions should therefore be treated
    cautiously until that verification is complete.

    Precision-free and AdamW-capable, matching SimplePCN's conventions.
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
        batch_size: Optional[int] = None,
    ) -> None:
        self.architecture = architecture
        self.batch_size = (
            dy.auto_batch_size()
            if batch_size is None
            else batch_size
        )

        self._configured = False
        self._learning_rate = 1e-6
        self._inference_rate = 0.1
        self._lambda = 1e-2
        self._optimizer = "SGD"

        self._validate_architecture()

        super().__init__(self.batch_size)

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
        learning_rate: float = 1e-6,
        inference_rate: float = 0.1,
        lmbda: float = 1e-2,
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

    def set_optimizer(self, optimizer: str) -> None:
        """Set the optimizer: SGD, ADAM, or ADAMW."""

        self._require_configured()

        self._optimizer = optimizer.upper()
        super().set_optimizer(self._optimizer)

    def set_learning_rate(self, lr: float) -> None:
        """Set the learning rate on every backend layer."""

        self._require_configured()

        self._learning_rate = lr
        super().set_learning_rate(lr)

    def set_inference_rate(self, ir: float) -> None:
        """Set the inference rate on every backend layer."""

        self._require_configured()

        self._inference_rate = ir

        for layer in self.layers:
            layer.set_inference_rate(ir)

    def compile(self) -> None:
        """Compile the backend network."""

        self._require_configured()
        super().compile()

    def randomize_weights(self) -> None:
        """Randomize all network weights."""

        self._require_configured()
        super().randomize_weights()

    def train_step(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        steps: int,
    ) -> float:
        """
        Perform one training step using zero-initialized hidden states.

        Hidden layers are reset for each batch and then settled using
        Gauss-Seidel dynamics.
        """

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
        """
        Perform one training step using forward-projection initialization.

        Hidden layers are initialized from a genuine forward pass through
        the current weights before Gauss-Seidel settling.
        """

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
        """Settle the network and return the terminal-layer beliefs."""

        self._require_configured()

        return np.asarray(
            super().predict(X.flatten(), steps)
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
    ) -> "GaussSeidelPCN":
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
