"""Export a network's forward path to ONNX.

The exported graph is the prediction with zero settling steps: `act(x W0^T + b0) W1^T + b1 ...`,
which is what `net.predict(x, 0)` computes for dense networks and `net.predict_with_projection(x, 0)`
for convolutional ones (their plain `predict(x, 0)` leaves the output at zero). It does not contain the iterative
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
    """Writes the forward path of `net` to `path` as an ONNX model.

    Dense networks take `x` as `[batch, in]`; convolutional ones take it flattened as
    `[batch, channels * height * width]`, like `predict`, and the graph reshapes it. The output `y`
    is `[batch, out]`. With `dynamic_batch` the batch size is left free. Raises if `onnx` is missing.
    """
    try:
        import onnx  # type: ignore[import-not-found]
        from onnx import TensorProto, helper, numpy_helper  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("ONNX export needs the onnx package (pip install onnx).") from error

    layers = architecture_to_json(net.architecture)
    if type(net).__name__ == "GaussSeidelPCN":
        # It has no separate terminal layer: its output is the last layer's belief, which is the previous
        # layer's prediction, so the last Linear never takes part in predict().
        if len(layers) < 2:
            raise ValueError("A GaussSeidelPCN needs at least two Linear layers to export.")
        layers = layers[:-1]
    convolutional = any(layer["kind"] == "convolution" for layer in layers)
    if convolutional and any(layer["kind"] != "convolution" for layer in layers):
        raise NotImplementedError("ONNX export does not support mixing dense and convolution layers.")

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = os.path.join(tmp, "weights.safetensors")
        net.save(weights_path)
        tensors = dy.read_tensors(weights_path)

    dtype = np.dtype(getattr(net, "dtype", "float32"))
    elem = TensorProto.DOUBLE if dtype == np.float64 else TensorProto.FLOAT
    batch = "batch" if dynamic_batch else int(net.batch_size)

    nodes, initializers = [], []
    previous = "x"
    if convolutional:
        channels, height, width = (int(d) for d in net.input_shape)
        in_width = channels * height * width
        initializers.append(numpy_helper.from_array(np.array([-1, channels, height, width], dtype=np.int64), "x_shape"))
        nodes.append(helper.make_node("Reshape", ["x", "x_shape"], ["x4"]))
        previous = "x4"
    else:
        in_width = layers[0]["in"]

    # A convolution with 0 output channels is the terminal marker and has no weights.
    weighted = [(i, layer) for i, layer in enumerate(layers) if layer.get("out_channels", 1) > 0]
    for position, (i, layer) in enumerate(weighted):
        weight = np.array(tensors[f"layers.{i}.weight"], dtype=dtype)
        bias = np.array(tensors[f"layers.{i}.bias"], dtype=dtype)
        activation = layer["activation"]
        if activation != "linear" and activation not in _ACTIVATION_OPS:
            raise NotImplementedError(f"No ONNX operator for activation {activation!r}.")
        last = position == len(weighted) - 1
        linear_out = "y" if last and activation == "linear" and not convolutional else f"h{i}"
        if convolutional:
            kh, kw = layer["kernel"]
            weight = weight.reshape(layer["out_channels"], -1, kh, kw)
            nodes.append(
                helper.make_node(
                    "Conv", [previous, f"W{i}", f"b{i}"], [linear_out],
                    kernel_shape=[kh, kw], strides=layer["stride"],
                    pads=[layer["pad"][0], layer["pad"][1], layer["pad"][0], layer["pad"][1]],
                )
            )
        else:
            nodes.append(helper.make_node("Gemm", [previous, f"W{i}", f"b{i}"], [linear_out], transB=1))
        initializers += [numpy_helper.from_array(weight, f"W{i}"), numpy_helper.from_array(bias, f"b{i}")]
        previous = linear_out
        if activation != "linear":
            previous = f"a{i}" if convolutional or not last else "y"
            # Deepity's GELU is the tanh approximation, not the exact erf form.
            extra = {"approximate": "tanh"} if activation == "gelu" else {}
            nodes.append(helper.make_node(_ACTIVATION_OPS[activation], [linear_out], [previous], **extra))

    if convolutional:
        nodes.append(helper.make_node("Flatten", [previous], ["y"], axis=1))
        out_width = None
    else:
        out_width = layers[-1]["out"]

    graph = helper.make_graph(
        nodes,
        "deepity_forward",
        [helper.make_tensor_value_info("x", elem, [batch, in_width])],
        [helper.make_tensor_value_info("y", elem, [batch, out_width])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="deepity")
    onnx.checker.check_model(model)
    onnx.save(model, path)
