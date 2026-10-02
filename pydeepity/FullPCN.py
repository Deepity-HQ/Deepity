from typing import Optional

import numpy as np
import numpy.typing as npt

from ._backend import dy
from .layer import Activation, Layer, Linear
from .utils import _fit_with_progress


class FullPCN(dy.FullPCNetwork):
    """
    The "everything" Predictive Coding Network: every PC mechanism this
    library implements, bolted onto one class as independently toggleable
    extras, each OFF by default so a plain FullPCN reproduces DKPPCN
    (DirectKPPCNetwork) exactly.

    DKP direct feedback (Psi) is always active, same as DKPPCN, so
    inference_steps=1 is a reasonable default operating point even before
    any extra is enabled. The extras, each a configure() keyword:

        use_mu_pc_scaling:
            muPC's full Table 1 parameterization (per-layer forward scale
            `a` plus unit-variance weight init), replacing plain PC's
            fan-scaled init. Must be decided at configure() time: `a` is
            computed from the full, now-known architecture and baked in
            by compile(), so toggling it later has no effect. See
            Innocenti et al., https://arxiv.org/abs/2505.13124.

        use_residual_connections:
            Adds a skip connection (mu += z, after activation) on every
            middle hidden layer. Requires matching width between a
            layer's input and output; compile() raises ValueError
            otherwise. Also compile()-time-only, like use_mu_pc_scaling.

        use_ipc:
            Incremental PC: weights update every settling step instead
            of once at the end. Safe to toggle any time after configure().
            See Salvatori et al., https://arxiv.org/abs/2212.00720.

        use_epc:
            Error-based PC: each settling step becomes one full
            forward+backward sweep through every hidden layer (the exact
            PC gradient via reverse-mode AD), instead of the usual
            one-hop local message -- orders of magnitude fewer settling
            steps on deep networks, at a higher per-step cost. Safe to
            toggle any time after configure(). See Goemaere et al.,
            https://arxiv.org/abs/2505.20137.

        use_momentum / momentum_beta:
            EMA-smooths the settling update direction before applying it
            to z, instead of applying it directly. Safe to toggle any
            time after configure().

        use_cross_entropy:
            Softmax cross-entropy energy on the terminal layer only
            (plain Gaussian energy everywhere else). Safe to toggle any
            time after configure().

    use_mu_pc_scaling and use_residual_connections are compile()-time
    decisions: set them via configure(), not afterwards. The other four
    extras are read fresh every settling step and can be flipped at any
    point via their own set_use_*() method.
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
        batch_size: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        resolved_batch_size = (
            dy.auto_batch_size()
            if batch_size is None
            else batch_size
        )

        self.architecture = architecture
        self.device = device

        self._configured = False
        self._learning_rate: float = 1e-6
        self._inference_rate: float = 0.1
        self._feedback_rate: float = 1e-4
        self._lambda: float = 1e-2
        self._optimizer: str = "SGD"
        self._psi_optimizer: str = "SGD"

        self._validate_architecture()

        super().__init__(resolved_batch_size, device)

    def _validate_architecture(self) -> None:
        if not self.architecture:
            raise ValueError("FullPCN requires at least one layer.")

        if not isinstance(self.architecture[0], Linear):
            raise TypeError("Architecture must begin with a Linear layer.")

        if not isinstance(self.architecture[-1], Linear) and not isinstance(
            self.architecture[-1], Activation
        ):
            raise TypeError(
                "Network architecture must end with a Linear or Activation layer."
            )

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
        use_mu_pc_scaling: bool = False,
        use_residual_connections: bool = False,
        use_ipc: bool = False,
        use_epc: bool = False,
        use_momentum: bool = False,
        momentum_beta: float = 0.9,
        use_cross_entropy: bool = False,
        seed: Optional[int] = None,
    ) -> "FullPCN":
        """
        Configure and initialize the network.

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

        use_mu_pc_scaling, use_residual_connections, use_ipc, use_epc,
        use_momentum, momentum_beta, use_cross_entropy:
            See the class docstring; every extra is OFF by default.

        seed:
            Optional seed for weight randomization, for reproducible
            runs. None (the default) uses a nondeterministic seed.
        """
        self._learning_rate = learning_rate
        self._inference_rate = inference_rate
        self._feedback_rate = feedback_rate
        self._lambda = lmbda
        self._optimizer = optimizer
        self._psi_optimizer = psi_optimizer

        self._build_backend()

        # use_mu_pc_scaling/use_residual_connections must be set before
        # compile() (their effect is baked in there); the other toggles
        # loop over self.layers immediately, so they also need
        # _build_backend() to have already run. Setting all of them here,
        # right before compile(), satisfies both requirements at once.
        # Calling super() directly (not self.set_use_*()): those wrappers
        # guard on _require_configured(), which isn't true yet this early
        # in configure() itself.
        super().set_use_mu_pc_scaling(use_mu_pc_scaling)
        super().set_use_residual_connections(use_residual_connections)
        super().set_use_ipc(use_ipc)
        super().set_use_epc(use_epc)
        super().set_use_momentum(use_momentum, momentum_beta)
        super().set_use_cross_entropy(use_cross_entropy)

        super().set_optimizer(optimizer)
        super().set_psi_optimizer(psi_optimizer)

        self.compile()
        self.randomize_weights(seed=seed)

        self._configured = True

        return self

    def _require_configured(self) -> None:
        if not self._configured:
            raise RuntimeError(
                "FullPCN has not been configured. "
                "Call net.configure(...) before training or prediction."
            )

    def set_use_mu_pc_scaling(self, enabled: bool) -> None:
        """
        Enable/disable muPC's Table 1 parameterization. Only takes effect
        if set before compile() (i.e. via configure()'s own kwarg, or
        before a manual compile() call); calling this after the network
        is already configured has no effect.
        """
        super().set_use_mu_pc_scaling(enabled)

    def set_use_residual_connections(self, enabled: bool) -> None:
        """
        Enable/disable residual connections on middle hidden layers. Only
        takes effect if set before compile(), same caveat as
        set_use_mu_pc_scaling().
        """
        super().set_use_residual_connections(enabled)

    def set_use_ipc(self, enabled: bool) -> None:
        """Enable/disable incremental PC. Safe to call at any time."""
        self._require_configured()
        super().set_use_ipc(enabled)

    def set_use_epc(self, enabled: bool) -> None:
        """Enable/disable ePC settling. Safe to call at any time."""
        self._require_configured()
        super().set_use_epc(enabled)

    def set_use_momentum(self, enabled: bool, beta: float = 0.9) -> None:
        """Enable/disable momentum settling. Safe to call at any time."""
        self._require_configured()
        super().set_use_momentum(enabled, beta)

    def set_use_cross_entropy(self, enabled: bool) -> None:
        """
        Enable/disable softmax cross-entropy energy on the terminal
        layer. Safe to call at any time.
        """
        self._require_configured()
        super().set_use_cross_entropy(enabled)

    def set_optimizer(self, optimizer: str) -> None:
        """Set the optimizer used for the forward weights W."""
        super().set_optimizer(optimizer)
        self._optimizer = optimizer

    def set_psi_optimizer(self, optimizer: str) -> None:
        """Set the optimizer used for the feedback weights Psi."""
        super().set_psi_optimizer(optimizer)
        self._psi_optimizer = optimizer

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate for the forward weights W."""
        super().set_learning_rate(learning_rate)
        self._learning_rate = learning_rate

    def set_feedback_rate(self, feedback_rate: float) -> None:
        """Set the learning rate for the feedback weights Psi."""
        super().set_feedback_rate(feedback_rate)
        self._feedback_rate = feedback_rate

    def set_inference_rate(self, inference_rate: float) -> None:
        """Set the inference/settling rate on every layer."""
        self._require_configured()

        for layer in self.layers:
            layer.set_inference_rate(inference_rate)

        self._inference_rate = inference_rate

    def compile(self) -> None:
        super().compile()

    def randomize_weights(self, seed: Optional[int] = None) -> None:
        """Randomize all network weights. `seed` gives a reproducible run."""
        super().randomize_weights(seed)

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
    ) -> "FullPCN":
        """Train the network for multiple epochs."""
        self._require_configured()

        if initial_lr is None:
            initial_lr = self._learning_rate

        _fit_with_progress(
            self,
            X,
            Y,
            epochs,
            inference_steps,
            initial_lr,  # type: ignore
            decay_rate,
            shuffle,
        )

        return self
