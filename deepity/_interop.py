"""Normalizes network input arrays so train_step()/predict()/fit() accept
not just numpy, but any object exporting the DLPack protocol (PyTorch CPU
tensors, JAX arrays, etc.). The compiled extension's bound functions only
ever ingest a C-contiguous numpy array; nanobind happens to also accept a
torch.Tensor directly (zero-copy, via its own ndarray exchange protocol),
but never a jax.Array. Funneling everything through this one helper means
every network class gets the same framework support without re-deriving
detection logic, and torch/numpy inputs stay exactly as zero-copy as they
are today (np.from_dlpack() shares the buffer; .astype(copy=False) only
copies on an actual dtype mismatch, same as it always has for numpy)."""

from __future__ import annotations

from typing import Protocol, Union, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class _SupportsDLPack(Protocol):
    """Structural type for anything to_numpy()/is_cuda_array() actually
    accept beyond a plain numpy array -- torch.Tensor and jax.Array both
    satisfy this without either needing to be imported here."""

    def __dlpack__(self, *, stream: int | None = ...) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...


# train_step()/predict()/fit()'s real input type, across every network
# class this module's helpers are used from: not just npt.NDArray, since
# that's what used to leave type checkers flagging a bare torch.Tensor
# argument as an error despite it being a genuinely supported input.
InputArray = Union[npt.NDArray[np.float32], _SupportsDLPack]


def to_numpy(x, dtype) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if hasattr(x, "__dlpack__"):
        arr = np.from_dlpack(x).astype(dtype, copy=False)
        if not arr.flags.writeable:
            # Some bound functions (e.g. clamp_input()) require a
            # writable buffer even though they only read from it.
            # Immutable sources (JAX arrays) export a read-only DLPack
            # view, so only here -- never for numpy/torch -- is a copy
            # unavoidable.
            arr = np.array(arr)
        return arr
    return np.asarray(x, dtype=dtype)


# dlpack.h's DLDeviceType.kDLCUDA -- the DLPack spec's own device-type
# code, not something Deepity defines.
_DLPACK_DEVICE_TYPE_CUDA = 2


def is_cuda_array(x) -> bool:
    """True if x reports a CUDA DLPack device. False for anything
    without __dlpack_device__ (plain numpy, Python lists, etc.)."""
    dlpack_device = getattr(x, "__dlpack_device__", None)
    if dlpack_device is None:
        return False
    device_type, _ = dlpack_device()
    return int(device_type) == _DLPACK_DEVICE_TYPE_CUDA


def to_cuda_capsule(x, stream_ptr: int):
    """Exports x as a DLPack capsule, explicitly synchronized against
    stream_ptr (an integer CUDA stream handle, e.g. a CUDA-device
    FullPCNetwork's own `stream_handle` property) before this returns.

    This is NOT the same as a bare x.__dlpack__() call: with no stream
    argument, DLPack's spec has the producer synchronize against the
    *legacy default* stream, not whatever stream the actual consumer
    (Deepity's own CUDA backend, which never uses the default stream)
    will read from. Passing stream_ptr makes the producer (e.g.
    PyTorch) insert a real cross-stream wait first, so the pointer
    inside the returned capsule is immediately safe to read from
    stream_ptr -- never pass a bare tensor into a *_device() binding
    without going through this first.
    """
    return x.__dlpack__(stream=stream_ptr)
