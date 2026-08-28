[![CI](https://github.com/ra4ster/deepity/actions/workflows/ci.yml/badge.svg)](https://github.com/ra4ster/deepity/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ra4ster/deepity)](https://github.com/ra4ster/deepity/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/ra4ster/deepity)](https://github.com/ra4ster/deepity/releases)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/ra4ster/deepity/blob/main/pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](Chttps://github.com/ra4ster/deepity/blob/main/CMakeLists.txt)
[![Stars](https://img.shields.io/github/stars/ra4ster/deepity?style=social)](https://github.com/ra4ster/deepity/stargazers)
[![Issues](https://img.shields.io/github/issues/ra4ster/deepity)](https://github.com/ra4ster/deepity/issues)
[![Last Commit](https://img.shields.io/github/last-commit/ra4ster/deepity)](https://github.com/ra4ster/deepity/commits/main)

![](resources/Deepity.png)

## What is this?

[`ngc-learn`](https://github.com/NACLab/ngc-learn) is the reference implementation for predictive coding networks (PCNs) — the biologically-motivated alternative to backprop from Whittington & Bogacz's 2017 paper. It's built on JAX, JIT-compiled, and GPU-capable.

Deepity is a from-scratch C++ reimplementation of the same algorithm. Not a port — a **reconstruction**, built by reading `ngc-learn`'s actual source and documentation line by line, gradient-checking every formula independently, and testing every hyperparameter claim against real training runs instead of trusting the docs at face value.

**The result: `GaussSeidelPCN`, one of Deepity's two PCN variants, beats `ngc-learn`'s own published accuracy on MNIST — on CPU, in plain C++, no JIT compiler involved.**

<div align="center">
<img src="resources/accuracy_comparison.png" alt="Accuracy comparison: ngc-learn vs SimplePCN vs GaussSeidelPCN" width="700" />
</div>

*All three curves are real, measured per-epoch accuracy — same architecture (784→512→512→10), same MNIST data, same 20-step settling budget. Nothing here is illustrative.*

---

## Two variants, one honest trade-off

Predictive coding networks "settle" toward a solution over several iterative steps before each weight update. There are two structurally different ways to run that settling loop, and we built both:

| | **SimplePCN** | **GaussSeidelPCN** |
|---|---|---|
| Settling dynamics | Synchronous (Jacobi) | Sequential-sweep (Gauss-Seidel) |
| Test accuracy | **93.04%** | **96.86%** |
| Speed vs. `ngc-learn` | ~1.3x slower | ~2x slower |
| Best for | Speed-sensitive use | Accuracy-sensitive use |

`ngc-learn` itself scored **95.09%** test accuracy in its own real run. Neither number above is cherry-picked or run once — both were validated across multiple random seeds.

Why does Gauss-Seidel win on accuracy but lose on speed? Sequential-sweep settling lets later layers see earlier layers' *already-updated* values within the same timestep — information propagates through the whole network depth in a single step, instead of needing one step per layer of depth like synchronous settling does. That's a real advantage for convergence quality. It's also why the settling loop can't be as aggressively parallelized, which is where the speed cost comes from.

We think that trade-off is worth exposing directly, not averaging away.

---

## What we actually found

Three specific, verified discoveries — each one contradicted an earlier, reasonable-looking assumption:

**1. `ngc-learn`'s activation is applied *before* the linear transform, not after.**
The obvious-looking formulation is `μ = φ(Wz + b)`. The real one, confirmed against both the documented ODEs and the actual source, is `μ = W·φ(z) + b`. Small-looking difference, real consequence for both the forward pass and the weight gradient.

**2. The feedback pathway is *not* `W` transposed — it's a separate, randomly-initialized, never-updated matrix.**
We initially assumed `ngc-learn`'s backward error signal used the standard backprop-style transposed weight matrix, matching the clean textbook notation `(W)ᵀ·e`. The real wiring uses an independent `StaticSynapse` — this is [feedback alignment](https://arxiv.org/abs/1411.0247) (Lillicrap et al.), not backprop. Implementing this specific piece is what actually closed the accuracy gap.

**3. Weight-init magnitude and optimizer choice aren't independent knobs.**
A uniform `±0.3` weight init — much larger than the size-scaled Gaussian init most libraries default to — performed *worse* than our own default when paired with plain SGD (73% vs 86%). Paired with Adam instead, it jumped to 94%. Adam's per-parameter adaptive step size can absorb a larger initial weight scale in a way SGD's fixed step size can't.

None of these were obvious from the paper alone. All three came from reading real, running source code and testing the result directly — not from theorizing about what "should" work.

---

## Performance engineering

Deepity is CPU-first by design, and the numbers above already account for real infrastructure work, not just algorithm changes:

- **Fused error/energy computation** — collapses a copy + subtract + separate-re-read sequence into a single memory pass over `z`/`μ`.
- **Mu-caching** — a clamped layer's outgoing prediction is provably constant for the entire settling loop (weights don't change mid-settle); skipping its recomputation is an *exact* optimization, not an approximation, and one that a JIT-compiled static graph structurally can't replicate as cheaply.
- **Optional Intel MKL backend** — build with `-DDEEPITY_USE_MKL=ON` (falls back to OpenBLAS automatically if MKL isn't found) for a substantial speedup on Intel hardware.
- **Contiguous memory arena** — every layer's buffers packed into one flat allocation, maximizing cache locality with zero pointer-chasing across the layer hierarchy.

---

## Building from source

```bash
git clone https://github.com/ra4ster/deepity
cd deepity
python build.py
```

```bash
python build.py [Release/Debug] [OpenBLAS/MKL] [--native/--fast/--distributed] [--clean] [--jobs=N] [--no-cuda]
```

`pip install rich` first for a live build dashboard — falls back to plain stdout either way.

**Requirements:** CMake, a C++20 compiler with AVX2/AVX-512 support, Python 3.9+ (for `pydeepity` bindings and `build.py`), [Ninja](https://ninja-build.org/) (optional, auto-detected).

### Python

```python
import numpy as np
from pydeepity import GaussSeidelPCN

net = GaussSeidelPCN(batch_size=256)
net.add_layer(784, 512, lr=0.001, ir=0.04, act="tanh")
net.add_layer(512, 512, lr=0.001, ir=0.04, act="tanh")
net.add_layer(512, 10, lr=0.001, ir=0.04, act="tanh")
net.add_layer(10, 0, lr=0.001, ir=0.04, act="linear")
net.set_optimizer("ADAMW")
net.compile()
net.randomize_weights()

energy = net.train_step_with_projection(X_batch, Y_batch, steps=20)
predictions = net.predict(X_batch, steps=20)
```

Working examples live in [`examples/`](examples/), including MNIST end-to-end training for both variants.

---

## The debugging story (the honest version)

Real dead ends we hit and had to walk back from, in case they're useful to someone else working in this space:

- Assumed `ngc-learn`'s update rule matched a clean textbook gradient derivation. It doesn't — the real algorithm deliberately drops a term, per Whittington & Bogacz's own point that PC *approximates* backprop rather than reproducing it exactly.
- Spent real time chasing a `2x` speed gap against `ngc-learn`'s JIT-compiled JAX backend with increasingly heavy-machinery proposals (custom compiler backends, alternative BLAS libraries) before realizing the actual, already-proven lever (mu-caching) was sitting unused.
- A one-line accuracy regression turned out to be an evaluation script silently calling the wrong training method (`train_step` instead of `train_step_with_projection`) — a reminder that a training/eval mismatch can look exactly like an algorithm bug.

If any of this overlaps with something you're working on, we'd genuinely like to hear from you — see below.

---

## Contributing

This is early-stage, actively-changing research code, and we're specifically looking for people interested in predictive coding, computational neuroscience, or CPU numerical performance work. Contributions, issues, and "I tried to reproduce this and got a different number" reports are all welcome — please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) first.

## Project structure

```plaintext
include/deepity/           # Public headers (Layer hierarchy, Activations, MemoryArena, ...)
include/deepity/layers/    # SimplePCLayer, GaussSeidelPCLayer, ConvPCLayer, ...
include/deepity/networks/  # SimplePCNetwork, GaussSeidelPCNetwork, ...
src/                        # C++ source implementations
bindings/                   # Python bindings (pybind11)
pydeepity/                  # Compiled Python extension module (generated)
examples/                   # Runnable Python examples (XOR, MNIST, benchmarking)
tests/                      # C++ gradient-check and verification suites
resources/                  # Images and benchmark assets
CMakeLists.txt               # Build configuration (OpenBLAS/MKL, CUDA, arch profiles)
build.py                    # Cross-platform CMake build & test runner
```

## License

Deepity is distributed under the terms in [`LICENSE`](LICENSE).

<small><i>Ra4ster (Jack R) @ 2026 ❤️</i></small>
