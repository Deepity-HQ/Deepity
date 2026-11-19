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
from typing import Any, Optional

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

    `settling_steps > 0` makes the graph match `net.predict(x, settling_steps)`. `SimplePCN`,
    `SequentialPCN`, `SimpleConvolutionalPCN` and `ConvolutionalPCN`, whose predictions start from a cleared state
    and depend on the step count, get that many settling steps unrolled (the graph grows with the count). For a
    convolutional network this is the plain `predict`, which starts from zero beliefs, not the forward-projected
    `predict_with_projection` that `settling_steps=0` matches. `FullPCN`, `DKPPCN` and `GaussSeidelPCN` predict from the forward pass, where
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
    if settling_steps > 0 and type(net).__name__ in _UNROLLED + _UNROLLED_CONV:
        g = _Graph(helper, numpy_helper, elem, dtype)
        (_unroll_conv_settling if convolutional else _unroll_settling)(g, tensors, layers, net, settling_steps)
        out_width = int(np.prod(_conv_shapes(net, layers)[-1])) if convolutional else layers[-1]["out"]
        graph = helper.make_graph(
            nodes + g.nodes,
            "deepity_settled",
            [helper.make_tensor_value_info("x", elem, [batch, in_width])],
            [helper.make_tensor_value_info("y", elem, [batch, out_width])],
            initializers + g.inits,
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
_UNROLLED_CONV = ("SimpleConvolutionalPCN", "ConvolutionalPCN")


def _check_settling_supported(net: Any, convolutional: bool, steps: int) -> None:
    kind = type(net).__name__
    if convolutional and kind not in _UNROLLED_CONV:
        raise NotImplementedError(f"Unrolled settling for convolutional networks is supported for SimpleConvolutionalPCN and ConvolutionalPCN only (not {kind}).")
    if not convolutional and kind not in _NO_OP_SETTLING + _UNROLLED:
        raise NotImplementedError(f"Unrolled settling is supported for SimplePCN and SequentialPCN only (not {kind}).")
    if steps > _MAX_UNROLLED_STEPS:
        raise ValueError(f"settling_steps above {_MAX_UNROLLED_STEPS} would make an unreasonably large graph.")


class _Graph:
    """Nodes and constants for an unrolled settling graph, with the small operator helpers both networks use."""

    def __init__(self, helper: Any, numpy_helper: Any, elem: int, dtype: np.dtype) -> None:
        self.helper, self.numpy_helper, self.elem, self.dtype = helper, numpy_helper, elem, dtype
        self.nodes: list = []
        self.inits: list = []
        self._count = 0
        self._one: Optional[str] = None

    @property
    def one(self) -> str:
        if self._one is None:
            self._one = self.const(1.0)
        return self._one

    def name(self, prefix: str) -> str:
        self._count += 1
        return f"{prefix}{self._count}"

    def const(self, value: float) -> str:
        key = self.name("c")
        self.inits.append(self.numpy_helper.from_array(np.array(value, dtype=self.dtype), key))
        return key

    def const_int(self, values: list[int]) -> str:
        key = self.name("i")
        self.inits.append(self.numpy_helper.from_array(np.array(values, dtype=np.int64), key))
        return key

    def array(self, key: str, value: np.ndarray) -> str:
        self.inits.append(self.numpy_helper.from_array(value, key))
        return key

    def op(self, kind: str, inputs: list[str], **attrs: Any) -> str:
        out = self.name("t")
        self.nodes.append(self.helper.make_node(kind, inputs, [out], **attrs))
        return out

    def activate(self, z: str, act: str) -> str:
        if act == "linear":
            return z
        extra = {"approximate": "tanh"} if act == "gelu" else {}
        return self.op(_ACTIVATION_OPS[act], [z], **extra)

    def derivative(self, z: str, act: str) -> Optional[str]:
        """f'(z) for the pre-activation `z`, or None for a linear activation."""
        if act == "linear":
            return None
        op, const = self.op, self.const
        if act == "relu":
            return op("Cast", [op("Greater", [z, const(0.0)])], to=self.elem)
        one = self.one
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

    def zero_beliefs(self, shape: list[int]) -> str:
        """A zero tensor of `[batch] + shape`, taking the (dynamic) batch size from `x`."""
        column = self.op("Mul", [self.op("ReduceSum", ["x", self.const_int([1])], keepdims=1), self.const(0.0)])
        column = self.op("Reshape", [column, self.const_int([-1] + [1] * len(shape))])
        return self.op("Expand", [column, self.const_int([1] + shape)])


def _unroll_settling(g: _Graph, tensors, layers, net, steps):
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
    dtype, op, const = g.dtype, g.op, g.const
    n = len(layers)
    sizes = [layers[0]["in"]] + [layer["out"] for layer in layers]
    for i, layer in enumerate(layers):
        g.array(f"W{i}", np.array(tensors[f"layers.{i}.weight"], dtype=dtype))
        g.array(f"b{i}", np.array(tensors[f"layers.{i}.bias"], dtype=dtype))
        if layer["activation"] != "linear" and layer["activation"] not in _ACTIVATION_OPS:
            raise NotImplementedError(f"No ONNX operator for activation {layer['activation']!r}.")
    ir = const(float(net._inference_rate))
    prec = [None] + [g.array(f"p{i}", np.array(tensors[f"layers.{i}.p"], dtype=dtype)) for i in range(1, n + 1)] if precision else None

    z = ["x"] + [g.zero_beliefs([sizes[i]]) for i in range(1, n + 1)]
    for _ in range(steps):
        pre = [None] * n
        if precision:
            mu = []
            for i in range(n):
                pre[i] = op("Gemm", [z[i], f"W{i}", f"b{i}"], transB=1)
                mu.append(g.activate(pre[i], layers[i]["activation"]))
        else:
            mu = [op("Gemm", [g.activate(z[i], layers[i]["activation"]), f"W{i}", f"b{i}"], transB=1) for i in range(n)]
        err = [None] + [op("Sub", [z[i], mu[i - 1]]) for i in range(1, n + 1)]
        new = ["x"]
        for i in range(1, n + 1):
            pull = err[i] if not precision else op("Mul", [err[i], prec[i]])
            if i == n:
                delta = op("Mul", [ir, op("Sub", [const(0.0), pull])])
            elif precision:
                gated = op("Mul", [err[i + 1], prec[i + 1]])
                d = g.derivative(pre[i], layers[i]["activation"])
                gated = gated if d is None else op("Mul", [gated, d])
                delta = op("Mul", [ir, op("Sub", [op("MatMul", [gated, f"W{i}"]), pull])])
            else:
                back = op("MatMul", [err[i + 1], f"W{i}"])
                d = g.derivative(z[i], layers[i]["activation"])
                driven = back if d is None else op("Mul", [d, back])
                delta = op("Mul", [ir, op("Sub", [driven, pull])])
            new.append(op("Add", [z[i], delta]))
        z = new
    g.nodes.append(g.helper.make_node("Identity", [z[n]], ["y"]))


def _conv_shapes(net: Any, layers: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Each layer's own (channels, height, width): the shape of its belief, which is its input."""
    shapes = [tuple(int(d) for d in net.input_shape)]
    for layer in layers[:-1]:
        c, h, w = shapes[-1]
        (kh, kw), (sh, sw), (ph, pw) = layer["kernel"], layer["stride"], layer["pad"]
        shapes.append((layer["out_channels"], (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1))
    return shapes


def _unroll_conv_settling(g: _Graph, tensors, layers, net, steps):
    """Settling for SimpleConvolutionalPCN and ConvolutionalPCN, from a cleared state, as ONNX nodes.

    Layer i owns the belief z_i of its own input, shape [batch, C, H, W]; z_0 = x is clamped and the last layer is
    the terminal one (its z is the output). Errors are read for every layer before any belief moves:
    e_i = z_i - mu_{i-1}, with mu_i = f_i(conv(z_i, W_i) + b_i) and preactivation a_i. A step is
      z_i += ir * (convT(e_{i+1} * p_{i+1} * f_i'(a_i), W_i) - p_i * e_i)
    for hidden layers and z_n -= ir * p_n * e_n for the terminal one. `SimpleConvolutionalPCN` has no
    precisions (p = 1); `ConvolutionalPCN` stores them as `layers.i.p`.
    """
    precision = type(net).__name__ == "ConvolutionalPCN"
    dtype, op, const = g.dtype, g.op, g.const
    n = len(layers)
    if any(layer["out_channels"] <= 0 for layer in layers[:-1]) or layers[-1]["out_channels"] != 0:
        raise NotImplementedError("Unrolled convolutional settling needs a terminal Convolution with 0 output channels, last.")
    shapes = _conv_shapes(net, layers)
    for i, layer in enumerate(layers[:-1]):
        c, h, w = shapes[i]
        kh, kw = layer["kernel"]
        g.array(f"W{i}", np.array(tensors[f"layers.{i}.weight"], dtype=dtype).reshape(layer["out_channels"], c, kh, kw))
        g.array(f"b{i}", np.array(tensors[f"layers.{i}.bias"], dtype=dtype))
        if layer["activation"] != "linear" and layer["activation"] not in _ACTIVATION_OPS:
            raise NotImplementedError(f"No ONNX operator for activation {layer['activation']!r}.")
    ir = const(float(net._inference_rate))
    prec = [None] + [g.array(f"p{i}", np.array(tensors[f"layers.{i}.p"], dtype=dtype).reshape(shapes[i])) for i in range(1, n)] if precision else None

    def attrs(i: int) -> dict[str, Any]:
        (kh, kw), (sh, sw), (ph, pw) = layers[i]["kernel"], layers[i]["stride"], layers[i]["pad"]
        return {"kernel_shape": [kh, kw], "strides": [sh, sw], "pads": [ph, pw, ph, pw]}

    def conv_transpose(gated: str, i: int) -> str:
        # The transposed convolution must come back to layer i's own height and width, which a strided
        # forward convolution can leave a few pixels short of: output_padding makes up the difference.
        _, h, w = shapes[i]
        _, oh, ow = shapes[i + 1]
        (kh, kw), (sh, sw), (ph, pw) = layers[i]["kernel"], layers[i]["stride"], layers[i]["pad"]
        pad_out = [h - ((oh - 1) * sh - 2 * ph + kh), w - ((ow - 1) * sw - 2 * pw + kw)]
        return op("ConvTranspose", [gated, f"W{i}"], output_padding=pad_out, **attrs(i))

    z = ["x4"] + [g.zero_beliefs(list(shapes[i])) for i in range(1, n)]
    for _ in range(steps):
        pre = [op("Conv", [z[i], f"W{i}", f"b{i}"], **attrs(i)) for i in range(n - 1)]
        mu = [g.activate(pre[i], layers[i]["activation"]) for i in range(n - 1)]
        err = [None] + [op("Sub", [z[i], mu[i - 1]]) for i in range(1, n)]
        new = ["x4"]
        for i in range(1, n):
            pull = err[i] if not precision else op("Mul", [err[i], prec[i]])
            if i == n - 1:
                delta = op("Mul", [ir, op("Sub", [const(0.0), pull])])
            else:
                gated = err[i + 1] if not precision else op("Mul", [err[i + 1], prec[i + 1]])
                d = g.derivative(pre[i], layers[i]["activation"])
                gated = gated if d is None else op("Mul", [gated, d])
                delta = op("Mul", [ir, op("Sub", [conv_transpose(gated, i), pull])])
            new.append(op("Add", [z[i], delta]))
        z = new
    g.nodes.append(g.helper.make_node("Flatten", [z[n - 1]], ["y"], axis=1))
