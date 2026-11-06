@mainpage pydeepity

Python bindings for [Deepity](../index.html), a Predictive Coding (PC)
neural network library. This page documents the `deepity` package (the
pure-Python layer users actually import — distributed on PyPI as
`pydeepity`), which wraps the compiled `deepity` extension module built from
Deepity's C++ engine.

For the underlying C++ API, see the [main Deepity documentation](../index.html).
For install/build instructions, see the
[GitHub README](https://github.com/ra4ster/deepity#readme).

@section py_quickstart Quick Start

```python
import numpy as np
import deepity as dy

net = dy.FullPCN(
    dy.Linear(2, 32), dy.ReLU(),
    dy.Linear(32, 1),
).configure(learning_rate=1e-3)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

for _ in range(1500):
    net.train_step(X, Y, inference_steps=150)

print(net.predict(X, inference_steps=150))
```

More complete, runnable examples live in `examples/` (e.g. `examples/xor.py`,
`examples/train_mnist_deep.py`).

@section py_networks Networks

Each class wraps a templated C++ network and exposes a common
`configure()` / `train_step()` / `predict()` / `fit()` surface:

- @ref deepity.SequentialPCN.SequentialPCN "SequentialPCN" — plain, no precision weighting
- @ref deepity.SimplePCN.SimplePCN "SimplePCN" — precision-weighted
- @ref deepity.ConvolutionalPCN.ConvolutionalPCN "ConvolutionalPCN" / @ref deepity.SimpleConvolutionalPCN.SimpleConvolutionalPCN "SimpleConvolutionalPCN" — convolutional
- @ref deepity.GaussSeidelPCN.GaussSeidelPCN "GaussSeidelPCN" — sequential layer sweeps
- @ref deepity.DKPPCN.DKPPCN "DKPPCN" — Direct Kolen-Pollack feedback alignment
- @ref deepity.FullPCN.FullPCN "FullPCN" — DKPPCN plus opt-in muPC scaling, residual
  connections, iPC, ePC, momentum, and cross-entropy, every extra OFF
  by default

@section py_layers Architecture Building Blocks

Passed positionally to a network's constructor to describe its
architecture:

- @ref deepity.layer.Linear "Linear" — a weighted layer
- @ref deepity.layer.Activation "Activation" and its concrete subclasses
  (@ref deepity.layer.Sigmoid "Sigmoid", @ref deepity.layer.TanH "TanH",
  @ref deepity.layer.ReLU "ReLU", @ref deepity.layer.GeLU "GeLU")
- @ref deepity.layer.Convolution "Convolution"

@section py_backend Backend Module

`deepity._backend.dy` is the raw compiled nanobind extension; everything
in @ref py_networks and @ref py_layers above wraps it. A handful of
standalone utilities are re-exported directly, with no wrapper class:
`StreamAlignedBatcher`, `get_l2_cache_bytes`, `auto_batch_size`,
`dynamic_thread`, the `omp_*` thread-count queries, `randomize` (fills a
numpy array in place with Deepity's own CPU/GPU RNG, so a project can seed
weights/data without a separate randomization dependency), and the
in-place SIMD activation functions
(`relu`/`drelu`/`tanh`/`dtanh`/`sigmoid`/`dsigmoid`).

@section py_elsewhere Elsewhere

- [Deepity C++ API reference](../index.html)
- [GitHub README](https://github.com/ra4ster/deepity#readme)
