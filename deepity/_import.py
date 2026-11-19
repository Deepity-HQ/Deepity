"""Import weights trained elsewhere (PyTorch, safetensors) into a Deepity network.

The arrays are written to a temporary safetensors file and loaded by the C++ reader, so shape,
dtype and padding checks are the same as for a native model file.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from ._backend import dy
from ._serialization import write_tensors
from ._sources import resolve

Source = Union[str, os.PathLike, Mapping[str, Any]]
Rename = Union[None, str, Mapping[str, str], Callable[[str], str]]


def _to_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    # torch tensors (any device, any float type incl. bfloat16) go through .float() first.
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype.is_floating_point:
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind == "f":
        array = array.astype(dtype, copy=False)
    # A PyTorch conv weight [out, in, kh, kw] is the same row-major data as [out, in*kh*kw].
    return array.reshape(array.shape[0], -1) if array.ndim > 2 else array


def _load_source(source: Source, dtype: np.dtype) -> dict[str, np.ndarray]:
    if isinstance(source, Mapping):
        return {str(k): _to_numpy(v, dtype) for k, v in source.items()}
    path = resolve(source)
    if path.endswith(".gguf"):
        return {k: _to_numpy(v, dtype) for k, v in dy.read_gguf(path)[0].items()}
    if path.endswith((".safetensors", ".index.json")):
        return {k: _to_numpy(v, dtype) for k, v in dy.read_tensors(path).items()}
    try:
        import torch
    except ImportError as error:
        raise ImportError(f"Reading {path!r} needs PyTorch (pip install torch). Safetensors files do not.") from error
    # weights_only keeps an untrusted pickle from running code.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    return {str(k): _to_numpy(v, dtype) for k, v in state.items() if hasattr(v, "shape")}


def _sequential_names(names: list[str]) -> dict[str, str]:
    """Pairs each module's `weight` and `bias` in file order with layers 0, 1, 2, ... of the network.

    `0.weight`, `0.bias`, `2.weight`, `2.bias` (a Sequential with an activation between) and
    `fc1.weight`, `fc2.weight` both become `layers.0.weight`, `layers.1.weight`, and so on.
    """
    mapping: dict[str, str] = {}
    layer_of: dict[str, int] = {}
    for name in names:
        prefix, _, leaf = name.rpartition(".")
        if leaf not in ("weight", "bias"):
            continue
        if prefix not in layer_of:
            layer_of[prefix] = len(layer_of)
        mapping[name] = f"layers.{layer_of[prefix]}.{leaf}"
    return mapping


def read_gguf(path: Union[str, os.PathLike]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reads a GGUF file (the llama.cpp format, versions 2 and 3) and returns `(tensors, metadata)`.

    Float tensors, including every block quantization (Q4_0 to Q8_0, the K-quants and the IQ quants), come
    back as float32 arrays in row-major shape; integer tensors as int64. A type that cannot be decoded
    (Q8_1, the ternary formats) raises an error naming the tensor. `metadata` is the key-value header; arrays longer than 256 entries
    (token lists) appear as `{"array_length": n}`. Raises on a malformed, truncated or unsafe file.
    """
    import json

    tensors, metadata = dy.read_gguf(resolve(path))
    return tensors, json.loads(metadata)


def import_weights(
    net: Any,
    source: Source,
    rename: Rename = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Loads weights from a PyTorch `state_dict`, a `.pt`/`.pth` file, a GGUF file, a safetensors file or a dict
    of arrays into an already-built, configured network.

    `rename` says which Deepity tensor each source name fills (`layers.0.weight`, `layers.0.bias`,
    `layers.1.weight`, ...): `None` keeps the names, `"sequential"` pairs each module's weight and
    bias in order with layers 0, 1, 2 (what a `torch.nn.Sequential` or a stack of Linear layers
    needs), a dict maps names, and a callable maps each name.

    Returns `(missing, unexpected)`: network tensors the source did not fill, and source tensors
    nothing used. Feedback weights (`psi`) are never expected from a foreign file, so they are not
    reported as missing. The terminal layer's own tensors usually are, since a PyTorch model has no
    counterpart. With `strict=True` any remaining name in either list raises instead.
    Convolution weights are flattened to `[out, in*kh*kw]`. Floats are converted to the network's dtype.
    """
    dtype = np.dtype(getattr(net, "dtype", "float32"))
    tensors = _load_source(source, dtype)

    if rename == "sequential":
        table = _sequential_names(list(tensors))
        mapper: Optional[Callable[[str], str]] = lambda n: table.get(n, n)
    elif isinstance(rename, Mapping):
        mapper = lambda n: rename.get(n, n)
    else:
        mapper = rename

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "import.safetensors")
        write_tensors(path, tensors)
        missing, unexpected = net.load_weights(path, False, mapper)
    missing = [name for name in missing if not name.endswith(".psi")]
    if strict and (missing or unexpected):
        raise ValueError(f"Import names do not match. Missing from the source: {missing}. Not used by the network: {unexpected}.")
    return missing, unexpected
