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

import numpy as np


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
