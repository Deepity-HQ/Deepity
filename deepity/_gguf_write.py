"""Write GGUF files (the llama.cpp container) from numpy arrays or a Deepity network.

Float tensors can be stored as F32, F16, BF16 or a block quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to
Q6_K); integer tensors keep their width. GGUF is read
by llama.cpp tooling and by `deepity.read_gguf`. Network weights are only useful to other tools if they
know the architecture, so a Deepity export carries the model description under `deepity.model`.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import Any, Optional

import numpy as np

from ._backend import dy
from ._gguf_quant import QUANTIZERS

_ALIGNMENT = 32
_MAX_NAME_BYTES = 64
_MAX_DIMS = 4
_FLOAT_TYPES = {"f32": 0, "f16": 1, "bf16": 30, **{name: info[0] for name, info in QUANTIZERS.items()}}
_INT_TYPES = {np.dtype(np.int8): 24, np.dtype(np.int16): 25, np.dtype(np.int32): 26, np.dtype(np.int64): 27}
_F64 = 28
_RESERVED_KEYS = {"general.alignment"}


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _scalar(value: Any) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        if 0 <= value < 1 << 32:
            return 4, struct.pack("<I", value)
        if -(1 << 63) <= value < 1 << 63:
            return 11, struct.pack("<q", value)
        if 0 <= value < 1 << 64:
            return 10, struct.pack("<Q", value)
        raise ValueError(f"integer metadata value {value} does not fit in 64 bits")
    if isinstance(value, float):
        return 12, struct.pack("<d", value)
    if isinstance(value, str):
        return 8, _string(value)
    raise TypeError(f"unsupported GGUF metadata value of type {type(value).__name__}")


def _value(value: Any) -> tuple[int, bytes]:
    if isinstance(value, dict):
        return 8, _string(json.dumps(value, separators=(",", ":")))
    if isinstance(value, (list, tuple)):
        items = [_scalar(v) for v in value]
        kinds = {kind for kind, _ in items}
        if len(kinds) > 1:  # mixed ints and floats widen to float64, anything else is an error
            if kinds <= {4, 11, 10, 12}:
                items = [_scalar(float(v)) for v in value]
            else:
                raise TypeError("a GGUF array must hold values of one type")
        kind = items[0][0] if items else 4
        return 9, struct.pack("<IQ", kind, len(items)) + b"".join(raw for _, raw in items)
    return _scalar(value)


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    return np.where(nan, np.uint16(0x7FC0), rounded)


def _encode(name: str, array: np.ndarray, dtype: Optional[str]) -> tuple[int, bytes]:
    """The GGML type id and raw little-endian bytes for one tensor."""
    if array.dtype in _INT_TYPES:
        return _INT_TYPES[array.dtype], array.astype(array.dtype.newbyteorder("<")).tobytes()
    if array.dtype == np.float64 and dtype is None:
        return _F64, array.astype("<f8").tobytes()
    if array.dtype.kind != "f":
        raise TypeError(f"tensor '{name}' has dtype {array.dtype}, which GGUF cannot store here")
    target = dtype or ("f16" if array.dtype == np.float16 else "f32")
    if target in QUANTIZERS and (array.ndim < 2 or array.shape[-1] % QUANTIZERS[target][1] != 0):
        target = "f32"  # biases and widths that are not whole blocks stay unquantized, as llama.cpp does
    if target == "f32":
        return 0, array.astype("<f4").tobytes()
    if target == "f16":
        return 1, array.astype("<f2").tobytes()
    if target == "bf16":
        return 30, _bf16_bits(array).astype("<u2").tobytes()
    type_id, _, quantize = QUANTIZERS[target]
    return type_id, quantize(array)


def write_gguf(
    path: str,
    tensors: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
    dtype: Optional[str] = None,
    architecture: str = "deepity",
) -> None:
    """Writes `tensors` (name -> numpy array) and `metadata` (str -> str/int/float/bool/list/dict) to a GGUF file.

    `dtype` is `None` (keep each float tensor's own width: float32, float16; float64 is kept too),
    or `"f32"`, `"f16"`, `"bf16"` or a block quantization (`"q4_0"`, `"q4_1"`, `"q5_0"`, `"q5_1"`, `"q8_0"`,
    `"q2_k"`, `"q3_k"`, `"q4_k"`, `"q5_k"`, `"q6_k"`) for every float tensor. With a block quantization, tensors
    that are not at least 2-D with a last dimension divisible by the block size (32, or 256 for the K-quants)
    stay float32. The quantizers are this library's own, so the bytes are valid GGUF that decodes within
    quantization error but differ from llama.cpp's; IQ quants are read-only. Shapes are written in GGUF's
    fastest-axis-first order. Dicts become JSON strings. The write is atomic.
    """
    if dtype is not None and dtype not in _FLOAT_TYPES:
        raise ValueError(f"dtype must be one of {sorted(_FLOAT_TYPES)} or None, got {dtype!r}")
    meta = dict(metadata or {})
    bad = [k for k in meta if k in _RESERVED_KEYS or not isinstance(k, str) or not k]
    if bad:
        raise ValueError(f"invalid or reserved metadata keys: {bad}")
    meta.setdefault("general.architecture", architecture)

    kv = b"".join(_string(key) + struct.pack("<I", kind) + raw for key, (kind, raw) in ((k, _value(v)) for k, v in meta.items()))
    infos, blobs, offset = b"", [], 0
    for name, value in tensors.items():
        if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
            raise ValueError(f"tensor name '{name}' is longer than GGUF's {_MAX_NAME_BYTES}-byte limit")
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError(f"tensor '{name}' is empty; GGUF cannot store a tensor with no elements")
        if array.ndim > _MAX_DIMS:
            raise ValueError(f"tensor '{name}' has {array.ndim} dimensions; GGUF allows {_MAX_DIMS}")
        ggml_type, raw = _encode(name, array, dtype)
        dims = list(reversed(array.shape)) or [1]
        infos += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", ggml_type, offset)
        padding = (-len(raw)) % _ALIGNMENT
        blobs.append(raw + b"\0" * padding)
        offset += len(raw) + padding
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + kv + infos
    head += b"\0" * ((-len(head)) % _ALIGNMENT)

    directory = os.path.dirname(os.path.abspath(path))
    handle, tmp_path = tempfile.mkstemp(prefix=".gguf-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(head)
            for blob in blobs:
                out.write(blob)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def export_gguf(net: Any, path: str, dtype: Optional[str] = None, architecture: str = "deepity") -> None:
    """Writes `net`'s weights to a GGUF file, with its description (architecture and hyperparameters)
    under `deepity.model`. See `write_gguf` for `dtype`."""
    from ._serialization import build_metadata

    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "weights.safetensors")
        net.save(weights)
        tensors = {name: np.array(value) for name, value in dy.read_tensors(weights).items()}
    # The terminal layer has no weights, so it is saved as empty tensors that GGUF cannot hold.
    tensors = {name: value for name, value in tensors.items() if value.size > 0}
    file_meta = build_metadata(net, type(net).__name__, net._current_hyperparameters() if hasattr(net, "_current_hyperparameters") else {}, None)
    metadata: dict[str, Any] = {"general.name": type(net).__name__}
    for key in ("deepity.model", "deepity.inference", "deepity.created_by"):
        if key in file_meta:
            metadata[key] = file_meta[key]
    write_gguf(path, tensors, metadata, dtype, architecture)
