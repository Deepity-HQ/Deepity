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


def export_onnx(net: Any, path: str, dynamic_batch: bool = True, settling_steps: int = 0) -> None:
    """Writes the forward path of `net` to `path` as an ONNX model.

    Dense networks take `x` as `[batch, in]`; convolutional ones take it flattened as
    `[batch, channels * height * width]`, like `predict`, and the graph reshapes it. The output `y`
    is `[batch, out]`. With `dynamic_batch` the batch size is left free. Raises if `onnx` is missing.

    `settling_steps > 0` makes the graph match `net.predict(x, settling_steps)`. `SimplePCN` and
    `SequentialPCN`, whose predictions start from a cleared state and depend on the step count, get that
    many settling steps unrolled (the graph grows with the count). `FullPCN`, `DKPPCN` and `GaussSeidelPCN` predict from the forward pass, where
    settling changes nothing, so the plain forward graph already matches for any step count. Other
    networks raise `NotImplementedError`.
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise ImportError("ONNX export needs the onnx package (pip install onnx).") from error

    if settling_steps < 0:
        raise ValueError("settling_steps must not be negative.")
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

    if settling_steps > 0:
        _check_settling_supported(net, convolutional, settling_steps)
    if settling_steps > 0 and type(net).__name__ in _UNROLLED:
        previous, nodes_extra, init_extra = _unroll_settling(helper, numpy_helper, elem, dtype, tensors, layers, net, settling_steps)
        nodes += nodes_extra
        initializers += init_extra
        graph = helper.make_graph(
            nodes,
            "deepity_settled",
            [helper.make_tensor_value_info("x", elem, [batch, in_width])],
            [helper.make_tensor_value_info("y", elem, [batch, layers[-1]["out"]])],
            initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="deepity")
        onnx.checker.check_model(model)
        onnx.save(model, path)
        return

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


_MAX_UNROLLED_STEPS = 2000
# FullPCN, DKPPCN and GaussSeidelPCN predict from the forward projection, where every error is already zero, so their
# settling changes nothing and the plain forward graph is exact for any step count.
_NO_OP_SETTLING = ("FullPCN", "DKPPCN", "GaussSeidelPCN")
_UNROLLED = ("SimplePCN", "SequentialPCN")


def _check_settling_supported(net: Any, convolutional: bool, steps: int) -> None:
    kind = type(net).__name__
    if kind not in _NO_OP_SETTLING + _UNROLLED:
        raise NotImplementedError(f"Unrolled settling is supported for SimplePCN and SequentialPCN only (not {kind}).")
    if convolutional:
        raise NotImplementedError("Unrolled settling is not supported for convolutional networks.")
    if steps > _MAX_UNROLLED_STEPS:
        raise ValueError(f"settling_steps above {_MAX_UNROLLED_STEPS} would make an unreasonably large graph.")


def _unroll_settling(helper, numpy_helper, elem, dtype, tensors, layers, net, steps):
    """Settling from a cleared state, step by step, as ONNX nodes, for SimplePCN and SequentialPCN.

    Beliefs start at zero except z_0 = x (clamped). One step reads every error before changing any
    belief: e_i = z_i - mu_{i-1}.
      SimplePCN: mu_i = f_i(z_i) W_i^T + b_i; hidden z_i += ir * (f_i'(z_i) * (e_{i+1} W_i) - e_i),
        terminal z_n -= ir * e_n.
      SequentialPCN: mu_i = f_i(z_i W_i^T + b_i) with preactivation a_i, and learned per-feature
        precisions p_i (stored in the file as `layers.i.p`); hidden
        z_i += ir * (((e_{i+1} * p_{i+1}) * f_i'(a_i)) W_i - p_i * e_i), terminal z_n -= ir * p_n * e_n.
    """
    precision = type(net).__name__ == "SequentialPCN"
    nodes: list = []
    inits: list = []
    counter = [0]

    def name(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def const(value: float) -> str:
        key = name("c")
        inits.append(numpy_helper.from_array(np.array(value, dtype=dtype), key))
        return key

    def op(kind: str, inputs: list[str], **attrs: Any) -> str:
        out = name("t")
        nodes.append(helper.make_node(kind, inputs, [out], **attrs))
        return out

    one = const(1.0)
    n = len(layers)
    sizes = [layers[0]["in"]] + [layer["out"] for layer in layers]
    for i, layer in enumerate(layers):
        weight = np.array(tensors[f"layers.{i}.weight"], dtype=dtype)
        bias = np.array(tensors[f"layers.{i}.bias"], dtype=dtype)
        inits += [numpy_helper.from_array(weight, f"W{i}"), numpy_helper.from_array(bias, f"b{i}")]
        if layer["activation"] != "linear" and layer["activation"] not in _ACTIVATION_OPS:
            raise NotImplementedError(f"No ONNX operator for activation {layer['activation']!r}.")
    ir = const(float(net._inference_rate))
    prec = [None] + [const_vec(inits, numpy_helper, f"p{i}", np.array(tensors[f"layers.{i}.p"], dtype=dtype)) for i in range(1, n + 1)] if precision else None

    def activate(z: str, i: int) -> str:
        act = layers[i]["activation"]
        if act == "linear":
            return z
        extra = {"approximate": "tanh"} if act == "gelu" else {}
        return op(_ACTIVATION_OPS[act], [z], **extra)

    def derivative(z: str, act: str) -> Optional[str]:
        if act == "linear":
            return None
        if act == "relu":
            return op("Cast", [op("Greater", [z, const(0.0)])], to=elem)
        if act == "sigmoid":
            s = op("Sigmoid", [z])
            return op("Mul", [s, op("Sub", [one, s])])
        if act == "tanh":
            t = op("Tanh", [z])
            return op("Sub", [one, op("Mul", [t, t])])
        # gelu, tanh form: 0.5(1+t) + 0.5 x (1-t^2) k (1 + 3 c x^2), t = tanh(k (x + c x^3))
        k, c = float(np.sqrt(2 / np.pi)), 0.044715
        x2 = op("Mul", [z, z])
        inner = op("Mul", [const(k), op("Mul", [z, op("Add", [one, op("Mul", [const(c), x2])])])])
        t = op("Tanh", [inner])
        left = op("Mul", [const(0.5), op("Add", [one, t])])
        sech2 = op("Sub", [one, op("Mul", [t, t])])
        poly = op("Mul", [const(k), op("Add", [one, op("Mul", [const(3 * c), x2])])])
        right = op("Mul", [op("Mul", [const(0.5), z]), op("Mul", [sech2, poly])])
        return op("Add", [left, right])

    # The batch size is dynamic, so the zero beliefs are built from x's shape: sum x over its columns,
    # scale by 0, then expand to each layer's width.
    zero_col = op("Mul", [op("ReduceSum", ["x", const_int(inits, numpy_helper, name("ax"), [1])], keepdims=1), const(0.0)])
    z = ["x"] + [op("Expand", [zero_col, const_int(inits, numpy_helper, name("sh"), [1, sizes[i]])]) for i in range(1, n + 1)]

    for _ in range(steps):
        pre = [None] * n
        if precision:
            mu = []
            for i in range(n):
                pre[i] = op("Gemm", [z[i], f"W{i}", f"b{i}"], transB=1)
                mu.append(activate(pre[i], i))
        else:
            mu = [op("Gemm", [activate(z[i], i), f"W{i}", f"b{i}"], transB=1) for i in range(n)]
        err = [None] + [op("Sub", [z[i], mu[i - 1]]) for i in range(1, n + 1)]
        new = ["x"]
        for i in range(1, n + 1):
            pull = err[i] if not precision else op("Mul", [err[i], prec[i]])
            if i == n:
                delta = op("Mul", [ir, op("Sub", [const(0.0), pull])])
            elif precision:
                gated = op("Mul", [err[i + 1], prec[i + 1]])
                d = derivative(pre[i], layers[i]["activation"])
                gated = gated if d is None else op("Mul", [gated, d])
                delta = op("Mul", [ir, op("Sub", [op("MatMul", [gated, f"W{i}"]), pull])])
            else:
                back = op("MatMul", [err[i + 1], f"W{i}"])
                d = derivative(z[i], layers[i]["activation"])
                driven = back if d is None else op("Mul", [d, back])
                delta = op("Mul", [ir, op("Sub", [driven, pull])])
            new.append(op("Add", [z[i], delta]))
        z = new
    nodes.append(helper.make_node("Identity", [z[n]], ["y"]))
    return "y", nodes, inits


def const_int(inits: list, numpy_helper, key: str, values: list[int]) -> str:
    inits.append(numpy_helper.from_array(np.array(values, dtype=np.int64), key))
    return key


def const_vec(inits: list, numpy_helper, key: str, value: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(value, key))
    return key
