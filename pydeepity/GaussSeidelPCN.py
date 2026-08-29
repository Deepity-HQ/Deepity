from ._backend import dy
from typing import Optional
import numpy as np
import numpy.typing as npt
from .utils import _fit_with_progress


class GaussSeidelPCN(dy.GaussSeidelPCNetwork):
    """
    A Predictive Coding Network with Gauss-Seidel (sequential-sweep)
    settling dynamics, rather than the fully-synchronous (Jacobi) dynamics
    every other PCN class in this library uses.

    Traced directly from ngc-learn's real execution graph
    (E2,E3 -> z0,z1,z2,z3 -> W1,W2,W3 -> e1,e2,e3): a full settling step is
    three sweeps across every layer -- z updates first (using values held
    over from the previous step), then predictions recompute using the
    fresh z's, then errors recompute using the fresh predictions. This
    lets later layers in a step see EARLIER layers' already-updated
    values within the SAME step, unlike Jacobi dynamics where every layer
    only ever sees values from the end of the previous full step.

    UNPROVEN EXPERIMENT -- the underlying math has only been smoke-tested
    (confirms energy is finite and trends down), NOT yet verified via an
    independent finite-difference gradient check. Treat any accuracy
    conclusion from real training with appropriate skepticism until that
    verification is done.

    Precision-free (no pr parameter) and AdamW-capable, matching
    SimplePCN's conventions.
    """

    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size)

    def add_layer(
        self,
        size: int,
        next_size: int,
        lr: float = 1e-6,
        ir: float = 0.1,
        lmbda: float = 1e-2,
        act: str = "relu",
    ) -> None:
        super().add_layer(size, next_size, lr=lr, ir=ir, lmbda=lmbda, activation=act, activation_deriv="d" + act)

    def set_optimizer(self, optimizer: str) -> None:
        """Sets the optimizer: ADAM, ADAMW, or SGD."""
        super().set_optimizer(optimizer)

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def set_learning_rate(self, lr: float) -> None:
        super().set_learning_rate(lr)

    def set_inference_rate(self, ir: float) -> None:
        # No network-level C++ method for this (unlike set_learning_rate) --
        # loop over layers directly, matching SimplePCN's own convention
        # for the same situation.
        for layer in self.layers:
            layer.set_inference_rate(ir)

    def train_step(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """Plain train step -- NO forward-projection initialization. Zero-inits
        hidden layers each batch, then settles via Gauss-Seidel sweeps."""
        return super().train_step(X.flatten(), Y.flatten(), steps)

    def train_step_with_projection(self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], steps: int) -> float:
        """Train step WITH forward-projection initialization -- seeds hidden
        layers from a genuine forward pass through current weights before
        settling, instead of zero-init. The optimization is orthogonal to
        the Gauss-Seidel restructuring itself; both can be used together."""
        return super().train_step_with_projection(X.flatten(), Y.flatten(), steps)

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
    ) -> "GaussSeidelPCN":
        """
        Runs a full multi-epoch training loop with a live rich progress display.
        Delegates per-batch execution to `train_step()`.
        """
        _fit_with_progress(self, X, Y, epochs, steps, initial_lr, decay_rate, shuffle)
        return self
