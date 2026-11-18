"""Self-describing model files (see docs/model-format.md).

A saved model stores its architecture, hyperparameters and settling defaults as JSON in the file's
metadata, so `deepity.load(path)` can rebuild the network with no hand-built architecture.
Weights never pass through Python: C++ writes and reads them.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from ._backend import dy
from .layer import Activation, Convolution, GeLU, Layer, Linear, ReLU, Sigmoid, TanH

FORMAT_VERSION = 1
# "iterative" is the full settling model; "forward" is reserved for forward-only exports.
INFERENCE_MODES = ("iterative",)

_ACTIVATIONS: dict[str, type[Activation]] = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": TanH,
    "gelu": GeLU,
}


def _network_classes() -> dict[str, type]:
    # Imported here because the network modules import this one.
    from .ConvolutionalPCN import ConvolutionalPCN
    from .DKPPCN import DKPPCN
    from .FullPCN import FullPCN
    from .GaussSeidelPCN import GaussSeidelPCN
    from .SequentialPCN import SequentialPCN
    from .SimpleConvolutionalPCN import SimpleConvolutionalPCN
    from .SimplePCN import SimplePCN

    return {
        cls.__name__: cls
        for cls in (
            DKPPCN, FullPCN, SequentialPCN, SimplePCN, SimpleConvolutionalPCN, GaussSeidelPCN, ConvolutionalPCN
        )
    }


def architecture_to_json(architecture: tuple[Layer | Activation, ...]) -> list[dict[str, Any]]:
    """One entry per weighted layer, with the activation that follows it ("linear" if none)."""
    layers: list[dict[str, Any]] = []
    for item in architecture:
        if isinstance(item, Linear):
            layers.append({"kind": "linear", "in": item.in_n, "out": item.out_n, "activation": "linear"})
        elif isinstance(item, Convolution):
            layers.append(
                {
                    "kind": "convolution",
                    "out_channels": item.out_channels,
                    "kernel": [item.kernel_h, item.kernel_w],
                    "stride": [item.stride_h, item.stride_w],
                    "pad": [item.pad_h, item.pad_w],
                    "activation": "linear",
                }
            )
        elif isinstance(item, Activation):
            if not layers:
                raise ValueError("An activation cannot appear before the first weighted layer.")
            layers[-1]["activation"] = item.to_string()
        else:
            raise TypeError(f"Cannot describe architecture element {type(item).__name__}.")
    return layers


def architecture_from_json(layers: list[dict[str, Any]]) -> tuple[Layer | Activation, ...]:
    architecture: list[Layer | Activation] = []
    for entry in layers:
        kind = entry.get("kind")
        needed = {"linear": ("in", "out"), "convolution": ("out_channels", "kernel", "stride", "pad")}.get(kind)
        if needed is None:
            raise ValueError(f"Unknown layer kind {kind!r} in model file. This version knows: linear, convolution.")
        missing = [k for k in needed if k not in entry]
        if missing:
            raise ValueError(f"A {kind} layer in the model file lacks {missing}.")
        if kind == "linear":
            architecture.append(Linear(int(entry["in"]), int(entry["out"])))
        else:
            architecture.append(
                Convolution(
                    int(entry["out_channels"]),
                    *entry["kernel"],
                    *entry["stride"],
                    *entry["pad"],
                )
            )
        activation = entry.get("activation", "linear")
        if activation != "linear":
            if activation not in _ACTIVATIONS:
                raise ValueError(f"Unknown activation {activation!r} in model file.")
            architecture.append(_ACTIVATIONS[activation]())
    return tuple(architecture)


_MAX_CARD_BYTES = 64 << 10


def card_to_json(card: dict[str, Any]) -> str:
    """Validates a model card and returns its JSON. A card is a dict of plain values (license,
    description, authors, training_data, metrics, url, ...), at most 64 KiB once serialized."""
    if not isinstance(card, dict):
        raise TypeError("net.card must be a dict.")
    try:
        text = json.dumps(card, separators=(",", ":"))
    except TypeError as error:
        raise TypeError(f"net.card must be JSON serializable: {error}") from error
    if len(text.encode()) > _MAX_CARD_BYTES:
        raise ValueError(f"net.card is larger than {_MAX_CARD_BYTES} bytes.")
    return text


def read_card(path: str) -> dict[str, Any]:
    """The model card stored in a model file, or `{}` if it has none."""
    return json.loads(dy.read_model_metadata(path).get("deepity.card", "{}"))


def build_metadata(net: Any, network: str, hyperparameters: dict[str, Any], dtype: Optional[str]) -> dict[str, str]:
    """The `deepity.*` metadata entries for `net`, ready to pass to the C++ save."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        created_by = "pydeepity " + version("pydeepity")
    except PackageNotFoundError:
        created_by = "pydeepity (uninstalled)"

    model: dict[str, Any] = {
        "network": network,
        "batch_size": int(net.batch_size),
        "device": getattr(net, "device", "cpu"),
        "layers": architecture_to_json(net.architecture),
        "hyperparameters": hyperparameters,
    }
    if dtype is not None:
        model["dtype"] = dtype
    if getattr(net, "input_shape", None) is not None:
        model["input_shape"] = [int(d) for d in net.input_shape]

    card = getattr(net, "card", None)

    # Only networks that carry a default settling count record one; the others take `steps` per call.
    inference = {"mode": "iterative"}
    if hasattr(net, "default_inference_steps"):
        inference["inference_steps"] = int(net.default_inference_steps)
    return {
        "deepity.model": json.dumps(model, separators=(",", ":")),
        "deepity.inference": json.dumps(inference, separators=(",", ":")),
        "deepity.created_by": created_by,
        **({"deepity.card": card_to_json(card)} if card else {}),
    }


def for_inference_metadata(metadata: dict[str, str], for_inference: bool) -> dict[str, str]:
    """Adds the marker that makes a save leave out training-only tensors."""
    return {**metadata, "deepity.training_tensors": "omitted"} if for_inference else metadata


def _json_field(meta: dict[str, str], key: str, path: str, default: Optional[str] = None) -> dict[str, Any]:
    """Parses a JSON object stored under `key`, naming the file and key when it is malformed."""
    try:
        value = json.loads(meta[key] if default is None else meta.get(key, default))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {key} is not valid JSON ({error}).") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key} must be a JSON object.")
    return value


def _known_hyperparameters(net: Any, saved: dict[str, Any], path: str) -> dict[str, Any]:
    """Drops saved hyperparameters this build's configure() does not take (a newer file may add some),
    with a warning, so an older Deepity still loads a newer file's weights."""
    import inspect
    import warnings

    parameters = inspect.signature(net.configure).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return saved
    unknown = sorted(set(saved) - set(parameters))
    if unknown:
        warnings.warn(f"{path}: ignoring hyperparameters this version does not know: {unknown}.", stacklevel=3)
    return {k: v for k, v in saved.items() if k in parameters}


def _build(path: str, device: Optional[str], batch_size: Optional[int]) -> tuple[Any, dict[str, str]]:
    """Reads the file's metadata and returns a freshly configured network (weights not loaded yet)."""
    meta = dy.read_model_metadata(path)
    if "deepity.model" not in meta:
        raise ValueError(
            f"{path} has no architecture ('deepity.model'). Build the network yourself, "
            "then call net.load(path) to load its weights."
        )
    model = _json_field(meta, "deepity.model", path)
    if not isinstance(model.get("layers"), list):
        raise ValueError(f"{path}: deepity.model has no 'layers' list.")

    classes = _network_classes()
    name = model.get("network")
    if name not in classes:
        raise ValueError(f"{path} describes network {name!r}, which this build cannot load. Known: {sorted(classes)}.")

    import inspect

    accepted = inspect.signature(classes[name].__init__).parameters
    kwargs: dict[str, Any] = {"batch_size": batch_size if batch_size is not None else model.get("batch_size")}
    if "device" in accepted:
        kwargs["device"] = device if device is not None else model.get("device", "cpu")
    elif device not in (None, "cpu"):
        raise ValueError(f"{name} runs on the CPU only, so device={device!r} is not available.")
    if "dtype" in model:
        kwargs["dtype"] = model["dtype"]
    if "input_shape" in model:
        kwargs["input_shape"] = tuple(model["input_shape"])

    net = classes[name](*architecture_from_json(model["layers"]), **kwargs)
    net.configure(**_known_hyperparameters(net, model.get("hyperparameters", {}), path))

    if "deepity.card" in meta:
        net.card = json.loads(meta["deepity.card"])

    inference = _json_field(meta, "deepity.inference", path, default="{}")
    mode = inference.get("mode", "iterative")
    if mode not in INFERENCE_MODES:
        raise ValueError(f"{path} uses inference mode {mode!r}, which this build cannot run. Known: {list(INFERENCE_MODES)}.")
    if "inference_steps" in inference:
        net.default_inference_steps = int(inference["inference_steps"])
    return net, meta


class ModelFileMixin:
    """Self-describing save, checkpoint and import for the network classes whose C++ side exposes the
    `_save`/`_save_checkpoint`/`_load_checkpoint`/`_load_weights` methods (see `BindModelFiles`).

    Subclasses list this mixin before their native base, set `_center_weights` and the usual
    private hyperparameter attributes, and implement `_current_hyperparameters()`.
    """

    _center_weights: bool = False

    def set_center_weights(self, enabled: bool) -> None:
        super().set_center_weights(enabled)  # type: ignore[misc]
        self._center_weights = bool(enabled)

    def _current_hyperparameters(self) -> dict[str, Any]:
        raise NotImplementedError

    def _file_metadata(self) -> dict[str, str]:
        return build_metadata(self, type(self).__name__, self._current_hyperparameters(), None)

    def save(self, filepath: str, max_shard_bytes: Optional[int] = None, for_inference: bool = False) -> bool:
        """Write a self-describing model file: weights, architecture, hyperparameters. `deepity.load(filepath)`
        rebuilds the network from it. A path ending in `.safetensors.index.json` writes shards.
        If `for_inference`, the file leaves out what only training uses (the DKP feedback weights Psi), so it is
        smaller. It loads and predicts identically; a model loaded from it that trains again starts with a fresh Psi."""
        self._require_configured()  # type: ignore[attr-defined]
        return self._save(filepath, for_inference_metadata(self._file_metadata(), for_inference), shard_bytes(filepath, max_shard_bytes))  # type: ignore[attr-defined]

    def save_checkpoint(
        self, filepath: str, training_state: Optional[dict] = None, max_shard_bytes: Optional[int] = None
    ) -> bool:
        """Write a training checkpoint: everything `save` writes plus the optimizer state (Adam/AdamW
        moments and step counters), so `deepity.resume(filepath)` continues training exactly.
        `training_state` is any JSON-serializable dict you want back on resume. CPU networks only."""
        self._require_configured()  # type: ignore[attr-defined]
        metadata = {**self._file_metadata(), **training_metadata(training_state)}
        return self._save_checkpoint(filepath, metadata, shard_bytes(filepath, max_shard_bytes))  # type: ignore[attr-defined]

    def load_checkpoint(self, filepath: str) -> dict:
        """Restore weights and optimizer state from a checkpoint into this configured network.
        Returns the `training_state` dict that was saved with it (empty if none)."""
        self._require_configured()  # type: ignore[attr-defined]
        self._load_checkpoint(filepath)  # type: ignore[attr-defined]
        return read_training_state(filepath)

    def load_weights(self, filepath: str, strict: bool = True, rename=None):
        """Load weights from any safetensors file; see `deepity.import_weights` for PyTorch sources."""
        self._require_configured()  # type: ignore[attr-defined]
        return self._load_weights(filepath, strict, rename)  # type: ignore[attr-defined]


def load(path: str, device: Optional[str] = None, batch_size: Optional[int] = None) -> Any:
    """Rebuilds the network described by the model file at `path` and loads its weights.

    `device` and `batch_size` override what the file recorded (for example to load a model saved
    on a GPU machine onto a CPU). Works on a checkpoint too (weights only). Raises if the file is
    missing, corrupt, from a newer Deepity, or describes a network this build does not know.
    """
    net, _ = _build(path, device, batch_size)
    net.load(path)
    return net


def resume(path: str, device: Optional[str] = None, batch_size: Optional[int] = None) -> tuple[Any, dict[str, Any]]:
    """Rebuilds a network from a checkpoint written by `net.save_checkpoint(...)` and restores
    its weights and optimizer state. Returns `(net, training_state)`, where `training_state` is
    the dict passed to `save_checkpoint` (epoch, step, RNG state, ...), or `{}` if none was given."""
    net, meta = _build(path, device, batch_size)
    return net, net.load_checkpoint(path)


INDEX_SUFFIX = ".safetensors.index.json"
DEFAULT_SHARD_BYTES = 2 << 30


def shard_bytes(path: str, requested: Optional[int]) -> int:
    """0 for a single file; otherwise the shard size limit. An index path shards by default."""
    if requested is not None and requested <= 0:
        raise ValueError("max_shard_bytes must be a positive number of bytes.")
    if str(path).endswith(INDEX_SUFFIX):
        return requested if requested is not None else DEFAULT_SHARD_BYTES
    if requested is not None:
        raise ValueError(f"max_shard_bytes needs a path ending in {INDEX_SUFFIX}, got {path!r}.")
    return 0


def write_tensors(
    path: str,
    tensors: dict[str, Any],
    metadata: Optional[dict[str, str]] = None,
    max_shard_bytes: Optional[int] = None,
) -> None:
    """Writes a dict of numpy arrays to a safetensors file in one call, with a content hash.
    A path ending in `.safetensors.index.json` shards it (see `net.save`). `metadata` is a
    string-to-string dict; keys starting with `deepity.` are reserved. Readable by any safetensors
    reader and by `deepity.read_tensors`."""
    bad = [k for k in (metadata or {}) if k.startswith("deepity.")]
    if bad:
        raise ValueError(f"Metadata keys {bad} are reserved for Deepity.")
    import numpy as np

    arrays = {name: np.asarray(value, order="C") for name, value in tensors.items()}
    dy.write_tensors(path, arrays, metadata or {}, shard_bytes(path, max_shard_bytes))


def training_metadata(training_state: Optional[dict[str, Any]]) -> dict[str, str]:
    """Metadata entry carrying the caller's training state (must be JSON-serializable)."""
    if training_state is None:
        return {}
    return {"deepity.training": json.dumps(training_state, separators=(",", ":"))}


def read_training_state(path: str) -> dict[str, Any]:
    return json.loads(dy.read_model_metadata(path).get("deepity.training", "{}"))
