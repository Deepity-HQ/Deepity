"""Export a network's forward path to ONNX.

The exported graph is the prediction with zero settling steps: `act(x W0^T + b0) W1^T + b1 ...`,
which is what `net.predict(x, inference_steps=0)` computes. It does not contain the iterative
settling, so it matches `predict` with the default settling only to the extent that settling
changes the output. Needs the optional `onnx` package.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from ._backend import dy
from ._serialization import architecture_to_json

_OPSET = 20
_ACTIVATION_OPS = {"relu": "Relu", "sigmoid": "Sigmoid", "tanh": "Tanh", "gelu": "Gelu"}


def export_onnx(net: Any, path: str, dynamic_batch: bool = True) -> None:
    """Writes the forward path of `net` (dense layers only) to `path` as an ONNX model.

    Input `x` is `[batch, in]`, output `y` is `[batch, out]`; with `dynamic_batch` the batch size is
    left free. Raises on a network with convolution layers (not supported yet) or if `onnx` is missing.
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise ImportError("ONNX export needs the onnx package (pip install onnx).") from error

    layers = architecture_to_json(net.architecture)
    if any(layer["kind"] != "linear" for layer in layers):
        raise NotImplementedError("ONNX export supports dense (Linear) networks only for now.")

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = os.path.join(tmp, "weights.safetensors")
        net.save(weights_path)
        tensors = dy.read_tensors(weights_path)

    dtype = np.dtype(getattr(net, "dtype", "float32"))
    elem = TensorProto.DOUBLE if dtype == np.float64 else TensorProto.FLOAT
    batch = "batch" if dynamic_batch else int(net.batch_size)

    nodes, initializers = [], []
    previous = "x"
    for i, layer in enumerate(layers):
        weight = np.array(tensors[f"layers.{i}.weight"], dtype=dtype)
        bias = np.array(tensors[f"layers.{i}.bias"], dtype=dtype)
        initializers += [numpy_helper.from_array(weight, f"W{i}"), numpy_helper.from_array(bias, f"b{i}")]
        activation = layer["activation"]
        if activation != "linear" and activation not in _ACTIVATION_OPS:
            raise NotImplementedError(f"No ONNX operator for activation {activation!r}.")
        last = i == len(layers) - 1
        gemm_out = "y" if last and activation == "linear" else f"h{i}"
        nodes.append(helper.make_node("Gemm", [previous, f"W{i}", f"b{i}"], [gemm_out], transB=1))
        previous = gemm_out
        if activation != "linear":
            previous = "y" if last else f"a{i}"
            # Deepity's GELU is the tanh approximation, not the exact erf form.
            extra = {"approximate": "tanh"} if activation == "gelu" else {}
            nodes.append(helper.make_node(_ACTIVATION_OPS[activation], [gemm_out], [previous], **extra))

    graph = helper.make_graph(
        nodes,
        "deepity_forward",
        [helper.make_tensor_value_info("x", elem, [batch, layers[0]["in"]])],
        [helper.make_tensor_value_info("y", elem, [batch, layers[-1]["out"]])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="deepity")
    onnx.checker.check_model(model)
    onnx.save(model, path)
