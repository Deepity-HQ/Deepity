from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from ._backend import dy
from ._interop import InputArray, to_numpy
from ._serialization import build_metadata, read_training_state, shard_bytes, training_metadata
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

    Note:
        With ReLU hidden layers and inference_steps > 1, settling hands
        each hidden unit a common-mode pull (-W^T W z of the layer above)
        that scales with the output width times the init variance. The
        previous layer's output is non-negative, so it accumulates in each
        row's mean weight, pre-activations go negative and the units go
        silent. Pass configure(center_weights=True) to counter this, or
        use a smaller init. Sigmoid hidden layers do not need it.
    """

    def __init__(
        self,
        *architecture: Layer | Activation,
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
        self._flags: dict = {}
        self.default_inference_steps: int = 1

        self._validate_architecture()

        super().__init__(batch_size, device)

    def _validate_architecture(self) -> None:
        if not self.architecture:
            raise ValueError("DKPPCN requires at least one layer.")

        if not isinstance(self.architecture[0], Linear):
            raise TypeError("Architecture must begin with a Linear layer.")

        if not isinstance(self.architecture[-1], Linear) and not isinstance(self.architecture[-1], Activation):
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
        center_weights: bool = False,
        psi_decay: Optional[float] = None,
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

        center_weights:
            Re-center each hidden layer's W rows after every update. Recommended for ReLU hidden layers with
            inference_steps > 1.

        psi_decay:
            Weight decay for the feedback weights Psi, independent of `lmbda` (which keeps
            applying to W). Use it if Psi grows without bound during long runs, which can end in
            non-finite outputs. It follows the same convention as `lmbda`: with SGD, Psi is
            multiplied by (1 - psi_decay) every step, so use small values (for example 1e-4);
            with ADAMW, each step subtracts feedback_rate * psi_decay * Psi, so larger values
            (0.1 to 1) are needed. If None, Psi follows the optimizer's default: no decay with
            SGD, `lmbda` with ADAMW.
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
        self.set_center_weights(center_weights)
        self.set_psi_decay(-1.0 if psi_decay is None else psi_decay)
        self._flags = {"center_weights": center_weights, "psi_decay": psi_decay}

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

        self._optimizer = optimizer

    def set_psi_optimizer(self, optimizer: str) -> None:
        """Set the optimizer used for the feedback weights Psi."""
        super().set_psi_optimizer(optimizer)

        self._psi_optimizer = optimizer

    def set_center_weights(self, enabled: bool) -> None:
        """Re-center each hidden layer's W rows after every weight update."""
        super().set_center_weights(enabled)

        self._flags["center_weights"] = enabled

    def set_psi_decay(self, decay: float) -> None:
        """Set the weight decay for the feedback weights Psi (negative keeps the optimizer default)."""
        super().set_psi_decay(decay)

        self._flags["psi_decay"] = None if decay < 0 else decay

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
        """Allocate backend buffers for the network built by
        _build_backend(). Called automatically by configure()."""
        super().compile()

    def randomize_weights(self, seed: Optional[int] = None, dist: str = "normal") -> None:
        """Randomize all network weights. `seed` gives a reproducible run;
        `dist` e.g. "normal(0,1)" or "uniform(-0.3,0.3)" (the default
        "normal" has no "(a,b)" and so deliberately falls back to each
        layer's own existing init)."""
        super().randomize_weights(seed, dist)

    def _current_hyperparameters(self) -> dict:
        return {
            "learning_rate": self._learning_rate,
            "inference_rate": self._inference_rate,
            "feedback_rate": self._feedback_rate,
            "lmbda": self._lambda,
            "optimizer": self._optimizer,
            "psi_optimizer": self._psi_optimizer,
            **self._flags,
        }

    def save(self, filepath: str, max_shard_bytes: Optional[int] = None) -> bool:
        """Write a self-describing model file: weights, architecture, hyperparameters and the
        default settling steps. `deepity.load(filepath)` rebuilds the network from it.
        A path ending in `.safetensors.index.json` writes a sharded model (shards of at most
        `max_shard_bytes`, default 2 GiB) that `deepity.load` opens through the index.
        Raises on any failure."""
        self._require_configured()

        return super().save(filepath, build_metadata(self, "DKPPCN", self._current_hyperparameters(), None), shard_bytes(filepath, max_shard_bytes))

    def save_checkpoint(
        self, filepath: str, training_state: Optional[dict] = None, max_shard_bytes: Optional[int] = None
    ) -> bool:
        """Write a training checkpoint: everything `save` writes plus the optimizer state (Adam/AdamW
        moments and step counters), so `deepity.resume(filepath)` continues training exactly.
        `training_state` is any JSON-serializable dict you want back on resume (epoch, global step,
        learning-rate schedule position, RNG state). A `.safetensors.index.json` path shards it as `save`
        does. CPU networks only. Raises on any failure."""
        self._require_configured()

        metadata = {
            **build_metadata(self, "DKPPCN", self._current_hyperparameters(), None),
            **training_metadata(training_state),
        }

        return super().save_checkpoint(filepath, metadata, shard_bytes(filepath, max_shard_bytes))

    def load_checkpoint(self, filepath: str) -> dict:
        """Restore weights and optimizer state from a checkpoint into this already-configured
        network. Returns the `training_state` dict that was saved with it (empty if none)."""
        self._require_configured()

        super().load_checkpoint(filepath)

        return read_training_state(filepath)

    def train_step(
        self,
        X: InputArray,
        Y: InputArray,
        inference_steps: int = 1,
    ) -> float:
        """Run one training step on a single batch: a direct feedback
        alignment update, then settling for `inference_steps`
        iterations before updating weights.
        @return Total settling energy accumulated over `inference_steps`."""
        self._require_configured()

        return super().train_step(
            to_numpy(X, np.float32).flatten(),
            to_numpy(Y, np.float32).flatten(),
            inference_steps,
        )

    def predict(
        self,
        X: InputArray,
        inference_steps: Optional[int] = None,
    ) -> npt.NDArray[np.float32]:
        """Run inference on `X`, settling for `inference_steps` iterations (default: the model's
        `default_inference_steps`, which a loaded model file sets).
        @return The terminal layer's beliefs."""
        self._require_configured()

        return np.asarray(
            super().predict(
                to_numpy(X, np.float32).flatten(),
                self.default_inference_steps if inference_steps is None else inference_steps,
            )
        )

    def fit(
        self,
        X: InputArray,
        Y: InputArray,
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
            to_numpy(X, np.float32),
            to_numpy(Y, np.float32),
            epochs,
            inference_steps,
            initial_lr, # type: ignore
            decay_rate,
            shuffle,
        )

        return self
