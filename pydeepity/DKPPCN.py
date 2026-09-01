from ._backend import dy
from typing import Optional
import numpy as np
import numpy.typing as npt
from .utils import _fit_with_progress


class DKPPCN(dy.DirectKPPCNetwork):
    """
    A Predictive Coding Network using Direct Kolen-Pollack (DKP) feedback
    alignment, per Casnici, Lefebvre, Dauwels & Frenkel, "Accelerated
    Predictive Coding Networks via Direct Kolen-Pollack Feedback
    Alignment" (2026).

    Unlike every other PCN class in this library, each layer holds two
    independently-learned weight matrices: W (the usual forward weights)
    and Psi (a direct feedback pathway from every hidden layer straight
    to the terminal layer's error, not relayed layer-by-layer). Before
    the ordinary settling loop begins, a one-time direct feedback
    alignment (DFA) update perturbs every layer's W using its Psi and the
    terminal error -- this is what lets a SINGLE settling step be
    sufficient, unlike the ~20 steps every other variant in this library
    typically needs. inference_steps therefore defaults to 1, matching
    the paper's own headline result; raising it trades away some of that
    speed advantage for potentially higher accuracy.

    W and Psi can use independent optimizers and learning rates --
    set_optimizer()/set_learning_rate() affect W, set_psi_optimizer()/
    set_feedback_rate() affect Psi.

    UNPROVEN IN PRODUCTION -- W's gradient has been independently
    verified via finite-difference check (see tests/tDirectKPVerify.cpp),
    but Psi's own alignment behavior (whether it actually correlates with
    W's transpose chain over training, per the paper's Appendix A.1) has
    NOT yet been verified with a real, sustained training run. Treat any
    accuracy conclusion from real training with appropriate skepticism
    until that verification is done.
    """

    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size)

    def add_layer(
        self,
        size: int,
        next_size: int,
        terminal_size: int,
        lr: float = 1e-6,
        ir: float = 0.1,
        fl: float = 1e-4,
        lmbda: float = 1e-2,
        act: str = "relu",
    ) -> None:
        """
        @param terminal_size The size of the network's FINAL output layer
            (e.g. 10 for MNIST) -- required on every add_layer() call, not
            inferred, since the true terminal layer isn't known until the
            whole network has been assembled.
        @param fl Learning rate for this layer's Psi (feedback) update,
            independent of lr (which only affects W).
        """
        super().add_layer(
            size, next_size, terminal_size,
            lr=lr, ir=ir, fl=fl, lmbda=lmbda,
            activation=act, activation_deriv="d" + act,
        )

    def set_optimizer(self, optimizer: str) -> None:
        """Sets W's optimizer: ADAM, ADAMW, or SGD."""
        super().set_optimizer(optimizer)

    def set_psi_optimizer(self, optimizer: str) -> None:
        """Sets Psi's optimizer: ADAM, ADAMW, or SGD. Independent of W's
        optimizer -- the paper treats these as separately-tuned in every
        experiment it reports."""
        super().set_psi_optimizer(optimizer)

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self) -> None:
        super().randomize_weights()

    def set_learning_rate(self, lr: float) -> None:
        """Sets W's learning rate, on every layer."""
        super().set_learning_rate(lr)

    def set_feedback_rate(self, fl: float) -> None:
        """Sets Psi's learning rate, on every layer. Independent of W's
        learning rate."""
        super().set_feedback_rate(fl)

    def train_step(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        inference_steps: int = 1,
    ) -> float:
        """Full DKP-PC train step: forward-projection init, terminal
        error, direct feedback alignment update, settling, weight update.
        All four phases run in C++; this is a single call, unlike
        SimplePCN/GaussSeidelPCN, which expose a separate
        train_step_with_projection() -- DKP-PC's phase 0 already includes
        forward-projection as a required step of the algorithm itself,
        not an optional add-on.

        @param inference_steps Defaults to 1, matching the paper's own
            headline result. See class docstring.
        """
        return super().train_step(X.flatten(), Y.flatten(), inference_steps)

    def predict(self, X: npt.NDArray[np.float32], inference_steps: int) -> npt.NDArray[np.float32]:
        return super().predict(X.flatten(), inference_steps)

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        epochs: int,
        inference_steps: int = 1,
        initial_lr: float = 0.01,
        decay_rate: float = 1.0,
        shuffle: bool = True,
    ) -> "DKPPCN":
        """
        Runs a full multi-epoch training loop with a live rich progress
        display. Delegates per-batch execution to train_step().

        @param inference_steps Defaults to 1, matching the paper's own
            headline result -- see class docstring. Every other fit() in
            this library calls this parameter "steps" with no default;
            it's named and defaulted differently here deliberately, since
            for DKP-PC specifically, 1 is not an arbitrary starting guess
            but the paper's own validated operating point.
        """
        _fit_with_progress(self, X, Y, epochs, inference_steps, initial_lr, decay_rate, shuffle)
        return self
