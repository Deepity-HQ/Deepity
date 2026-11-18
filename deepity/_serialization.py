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
    from .DKPPCN import DKPPCN
    from .FullPCN import FullPCN

    return {"DKPPCN": DKPPCN, "FullPCN": FullPCN}


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
        if kind == "linear":
            architecture.append(Linear(int(entry["in"]), int(entry["out"])))
        elif kind == "convolution":
            architecture.append(
                Convolution(
                    int(entry["out_channels"]),
                    *entry["kernel"],
                    *entry["stride"],
                    *entry["pad"],
                )
            )
        else:
            raise ValueError(f"Unknown layer kind {kind!r} in model file.")
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
        "device": net.device,
        "layers": architecture_to_json(net.architecture),
        "hyperparameters": hyperparameters,
    }
    if dtype is not None:
        model["dtype"] = dtype

    card = getattr(net, "card", None)

    inference = {"mode": "iterative", "inference_steps": int(net.default_inference_steps)}
    return {
        "deepity.model": json.dumps(model, separators=(",", ":")),
        "deepity.inference": json.dumps(inference, separators=(",", ":")),
        "deepity.created_by": created_by,
        **({"deepity.card": card_to_json(card)} if card else {}),
    }


def _build(path: str, device: Optional[str], batch_size: Optional[int]) -> tuple[Any, dict[str, str]]:
    """Reads the file's metadata and returns a freshly configured network (weights not loaded yet)."""
    meta = dy.read_model_metadata(path)
    if "deepity.model" not in meta:
        raise ValueError(
            f"{path} has no architecture ('deepity.model'). Build the network yourself, "
            "then call net.load(path) to load its weights."
        )
    model = json.loads(meta["deepity.model"])

    classes = _network_classes()
    name = model.get("network")
    if name not in classes:
        raise ValueError(f"{path} describes network {name!r}, which this build cannot load. Known: {sorted(classes)}.")

    kwargs: dict[str, Any] = {
        "batch_size": batch_size if batch_size is not None else model.get("batch_size"),
        "device": device if device is not None else model.get("device", "cpu"),
    }
    if "dtype" in model:
        kwargs["dtype"] = model["dtype"]

    net = classes[name](*architecture_from_json(model["layers"]), **kwargs)
    net.configure(**model.get("hyperparameters", {}))

    if "deepity.card" in meta:
        net.card = json.loads(meta["deepity.card"])

    inference = json.loads(meta.get("deepity.inference", "{}"))
    mode = inference.get("mode", "iterative")
    if mode not in INFERENCE_MODES:
        raise ValueError(f"{path} uses inference mode {mode!r}, which this build cannot run. Known: {list(INFERENCE_MODES)}.")
    if "inference_steps" in inference:
        net.default_inference_steps = int(inference["inference_steps"])
    return net, meta


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
