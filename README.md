<p align="center">
  <img src="resources/deepity-mark.png" alt="Deepity logo" width="250"/>
</p>

# Deepity

A high-performance C++ / CUDA engine for Predictive Coding Networks (PCNs) and Direct Kolen-Pollack algorithms.

Deepity bypasses standard backpropagation's memory overhead using local, layer-by-layer learning rules instead of one global backward pass -- see [Why Predictive Coding?](#why-predictive-coding) below for what that actually means.

**Current Benchmark:** 98.22% MNIST test accuracy in 952.6s on CPU — matching ngc-learn's own PC reference implementation's accuracy in ~1.79x less wall-clock time, on an identical architecture/settling-step/batch-size/epoch-count configuration. See [Performance & Benchmarks](#performance--benchmarks) below for the full picture, including where Deepity does *not* win (e.g. wall-clock vs. single-pass backprop).

[Website & Documentation](https://ra4ster.github.io/Deepity) • [Current Benchmarks](#performance--benchmarks) • [Contributing](#contributing)

---

## Why Predictive Coding?

Standard backpropagation (as in PyTorch/TensorFlow) computes a loss at the output, then propagates error gradients backward through every layer in one global pass -- which means holding the entire forward computation graph in memory until that backward pass runs.

Deepity trains networks differently. Each layer holds a belief about its own activity and compares it against a top-down prediction from the layer above; the mismatch -- the local prediction error -- is what drives learning, resolved one layer at a time instead of via a single global backward pass. Deepity implements two ways of turning that local error into a weight update:

- **Classic settling** (`SimplePCN`, `FullPCN`, and most of the network variants below): for each batch, every layer's belief is iteratively refined over several "settling steps" (`T` in the benchmarks below) until the prediction errors across the network stabilize, and only then do weights update. More settling steps means a better-converged update at proportionally more compute per batch -- most of the speed/accuracy trade-offs in the benchmarks below come down to T.
- **Direct Kolen-Pollack** (`DKPPCN`): trains a separate feedback weight matrix alongside the forward weights, rather than reusing the forward weights symmetrically for top-down predictions the way classic settling does, and in practice needs far fewer iterations per batch as a result -- `DKPPCN` below uses 2 inference steps where `SimplePCN` uses 20, which is most of why it's dramatically faster, at a real accuracy cost (see the MNIST comparison below).

**Key Advantages:**

- **O(1) Memory Footprint:** no backward computation graph is ever stored -- each layer's update only depends on its own local prediction error, not on gradients computed elsewhere in the network.
- **No weight-transport requirement:** standard backprop needs each layer's backward pass to use the exact transpose of its forward weights; Deepity's local updates don't inherit that constraint, and DKP relaxes it further by learning the feedback pathway instead of assuming it. This locality is also the standard motivation for predictive coding's interest in neuromorphic and edge hardware in the wider research literature -- Deepity doesn't yet have benchmarks on either, so take that as context rather than a claim about this specific engine.

## Performance & Benchmarks

Development and testing for the CUDA backend happens on the Ohio Supercomputer Center (Project PAS0350), on NVIDIA A100 GPUs. The numbers below are all CPU-only, run on one machine (12th Gen Intel Core i7-12700H, 20 logical threads, native + MKL build), so they measure the engine's single-machine CPU path specifically, not the CUDA backend. Every command below is reproducible from the repo root, and raw output for each run is committed under `logs/`.

### MNIST, matched configuration

784&rarr;512&rarr;512&rarr;10 (sigmoid), T=20 settling steps/batch, batch size 250, 100 epochs, AdamW lr=0.00373 with 0.94/epoch decay -- identical across the first two columns so the comparison is apples-to-apples.

| | Deepity `SimplePCN` | ngc-learn `pc_discrim` | PyTorch backprop (control) |
|---|---|---|---|
| Test / dev accuracy | **98.22%** | 98% (best dev) | 97.51% |
| Wall-clock, 100 epochs | **952.6s** | 1707.7s | 627.8s |
| Command | `python examples/mnist.py 7 100 20` | [ngc-museum](https://github.com/NACLab/ngc-museum)'s `pc_discrim` exhibit, unmodified | `python examples/test_ffnn_deep.py 7 100` |
| Raw log | `logs/SimplePCN_vs_ngclearn_mnist.txt` | `logs/ngclearn_pc_discrim_mnist.txt` | `logs/bp_control_mnist.txt` |

- **Deepity vs. ngc-learn** is the matched comparison -- identical architecture, settling-step count, batch size, epoch count, forward-projection init, and optimizer schedule, run as two independent predictive-coding implementations on the same machine. Deepity reaches equal-or-better accuracy in ~1.79x less time than ngc-learn's own PC reference implementation.
- **The PyTorch column is a reference point, not a PC run.** It's single-pass backprop (one forward + one backward per batch) against the same targets via MSE loss. Backprop is faster in wall-clock here, since it does one pass per batch where PC does T=20 settling passes -- the 1.79x figure is Deepity vs. another PC implementation, not Deepity vs. backprop.

A separate, cheaper variant, `DKPPCN` (784&rarr;512&rarr;10, batch size 1000, only 2 inference steps/batch instead of 20), reaches 95.97% in 49.6s on the same CPU (`python examples/mnist_dkppcn.py 7 50 2 cpu`, `logs/DKPPCN_native_mkl.txt`) -- a different point on the speed/accuracy trade-off curve.

### Raw kernel throughput (GFLOPS)

`tests/tGflopsBenchmark.cpp`, built as the `GflopsBenchmark` target via Google Benchmark, measures sustained GFLOPS for a full PC train step (`SimplePCN`, 784-512-256-64-10, batch 256, T=20):

```bash
python build.py Release OpenBLAS --fast
./build/Release/bin/GflopsBenchmark
```

FLOP accounting, from the file's own header comment: for a layer transitioning $\text{size} \to \text{nextSize}$ at a given $\text{batchSize}$, each settling step runs two GEMMs of matched cost -- the forward prediction GEMM and the feedback GEMM -- each costing $2 \cdot \text{batchSize} \cdot \text{size} \cdot \text{nextSize}$ FLOPs (one multiply + one add per multiply-accumulate, the standard convention). `UpdateWeights()` runs a third, equally-sized GEMM once per *train step* (not once per settling step) for the weight gradient. Summed over every weighted layer transition:

$$
\text{FLOPs}_{\text{train-step}} = \sum_{\text{layers}} \text{batchSize} \cdot \text{size} \cdot \text{nextSize} \cdot (4 \cdot \text{numSteps} + 2)
$$

Plugged in for the benchmarked network ($\text{batchSize}=256$, $\text{numSteps}=20$, constant factor $256 \cdot (4 \cdot 20 + 2) = 256 \cdot 82 = 20{,}992$):

| Layer transition | size × nextSize |
|---|---|
| 784 → 512 | 401,408 |
| 512 → 256 | 131,072 |
| 256 → 64 | 16,384 |
| 64 → 10 | 640 |
| **Σ (size × nextSize)** | **549,504** |

$$
\text{FLOPs}_{\text{train-step}} = 549{,}504 \times 20{,}992 \approx 11.54\ \text{GFLOP}
$$

That 11.54 GFLOP figure is fixed by the architecture, batch size, and step count. What's measured is wall time per train step; Google Benchmark's `FLOPS` counter divides the fixed number above by that measured time. E.g. the first of the runs logged below took 48.6ms/iteration:

$$
\text{GFLOPS/s} = \frac{11.54 \times 10^{9}\ \text{FLOPs}}{0.0486\ \text{s}} \approx 237\ \text{GFLOPS/s}
$$

-- matching that run's own `FLOPS=237.29G/s` counter output. Eight consecutive runs land between **232 and 259 GFLOPS/s** (this is a laptop with CPU frequency scaling enabled, not a pinned benchmark server, hence the spread -- see `logs/gflops_benchmark.txt` for all raw runs).

### Activation-kernel speedup

`tests/tActivations.cpp`, built as the `ActivationBenchmark` target, compares Deepity's SIMD kernels (hand-written AVX2/AVX-512/SSE intrinsics calling SLEEF for the transcendental ones -- see `include/deepity/utils/activations/`) against naive `std::tanh`/`std::exp`-based scalar loops at matched input sizes:

```bash
./build/Release/bin/ActivationBenchmark
```

Throughput in millions of elements/second; speedup = Deepity ÷ naive; raw output in `logs/activation_benchmark.txt`:

**Tanh**

| Size | 256 | 512 | 784 | 16,384 | 131,072 |
|---|---|---|---|---|---|
| Deepity (SIMD) | 303.6 | 394.8 | 460.7 | 559.8 | 3044.0 |
| `std::tanh` (naive) | 223.7 | 243.2 | 250.8 | 258.6 | 261.2 |
| Speedup | **1.36x** | **1.62x** | **1.84x** | **2.16x** | **11.66x** |

**Sigmoid**

| Size | 256 | 512 | 784 | 16,384 | 131,072 |
|---|---|---|---|---|---|
| Deepity (SIMD) | 892.8 | 1279.6 | 1421.4 | 1989.7 | 2091.5 |
| naive (`1/(1+exp(-x))`) | 358.1 | 416.9 | 457.3 | 505.2 | 506.2 |
| Speedup | **2.49x** | **3.07x** | **3.11x** | **3.94x** | **4.13x** |

**ReLU**

| Size | 256 | 512 | 784 | 16,384 | 131,072 |
|---|---|---|---|---|---|
| Deepity (SIMD) | 1323.5 | 2643.1 | 4031.7 | 18520.3 | 16724.8 |
| naive (`max(0,x)`) | 1504.4 | 2917.8 | 4102.4 | 19042.4 | 16681.0 |
| Speedup | 0.88x | 0.91x | 0.98x | 0.97x | 1.00x |

**GELU**

| Size | 256 | 512 | 784 | 16,384 | 131,072 |
|---|---|---|---|---|---|
| Deepity (SIMD) | 357.2 | 416.6 | 431.5 | 462.3 | 456.4 |
| naive (erf-based) | 153.4 | 162.6 | 169.6 | 179.7 | 166.1 |
| Speedup | **2.33x** | **2.56x** | **2.54x** | **2.57x** | **2.75x** |

Tanh, Sigmoid, and GELU win at every tested size, with the margin generally growing with size. Tanh's jump at the largest size (131,072) is `Tanh.h`'s settling loop also crossing an OpenMP threshold (`n > 16384`) at that size, adding multi-core parallelism on top of SIMD width. ReLU sits at 0.88x-1.00x, essentially tied with the naive loop -- expected, since `relu(x) = max(0, x)` is one compare with no multiply, so it's memory-bandwidth-bound rather than compute-bound, and the compiler's own auto-vectorization of a one-line loop already reaches the same ceiling.

_(Note: the CUDA backend is a separate code path with its own characteristics -- TF32-enabled im2col+cuBLAS convolution, CUDA graph capture for the settling loop -- and isn't covered by the CPU numbers above. ImageNet-scale training on it is in active development; see `research/imagenet_pc_vgg7_ce.py`, which now supports checkpoint/resume and a CLI-overridable settling-step count for sweeping T.)_

## Quick Start

### Prerequisites

- CMake 3.21+ and Ninja
- A C++20 compiler (Clang recommended, GCC and MSVC also supported)
- OpenBLAS or Intel MKL
- Python 3.9+ with development headers, plus `nanobind` (for the `pydeepity` bindings)
- CUDA Toolkit (optional -- the CUDA backend is detected automatically if present; the project builds and runs CPU-only otherwise)

See `CONTRIBUTING.md` for the full prerequisite list and known-working manual CMake invocations (e.g. Windows + Clang).

### Building from Source

The build is driven by `build.py`, which wraps CMake/Ninja configuration, compilation, and running the test suite in one command:

```bash
git clone https://github.com/Ra4ster/Deepity.git
cd Deepity
python build.py Release OpenBLAS --fast
```

`Release`/`Debug` and `OpenBLAS`/`MKL` select the build type and BLAS vendor; `--fast` targets a portable AVX2/FMA baseline (see `python build.py --list-profiles` for the other options, and `python build.py --help` for CUDA/test/job-count flags). This also builds the `pydeepity` Python extension module unless `--no-python-bindings` is passed.

### Basic Initialization (Python)

Deepity is a C++ engine, but most users drive it via the `pydeepity` bindings (powered by `nanobind`). Networks are declared with `Linear`/`Activation` objects, then built with `configure()`:

```python
import numpy as np
from pydeepity import SimplePCN, Linear, TanH

net = SimplePCN(
    Linear(2, 8), TanH(),
    Linear(8, 1),
    batch_size=1,
)
net.configure(learning_rate=0.05, inference_rate=0.3, lmbda=0.0001, optimizer="SGD")

x = np.array([1.0, -1.0], dtype=np.float32)
y = np.array([1.0], dtype=np.float32)
energy = net.train_step(x, y, steps=50)  # settles + updates weights locally
```

See `research/imagenet.py` / `research/imagenet_pc_vgg7_ce.py`, and `examples/mnist.py`, for larger, end-to-end training scripts.

_(Note: If you prefer to use the engine natively in C++, see `include/deepity/networks/DiscriminativePCNetwork.h` -- most `pydeepity` wrapper classes correspond directly to a C++ network class under `include/deepity/networks/`.)_

## Architecture & Roadmap

Deepity is currently pivoting from an academic research project into a scalable infrastructure tool.

- [x] Core CUDA kernel fused optimization
- [x] DKP-PC acceleration
- [x] ImageNet data loading pipeline (see `research/imagenet.py`)
- [ ] Stable, large-scale ImageNet training (actively being tuned)
- [x] Cross-platform CI/CD (Ubuntu + Windows wheel builds, pyright, on every push -- see `.github/workflows`)
- [ ] cuDNN-backed convolution (Winograd / implicit-GEMM), replacing the current im2col + cuBLAS-GEMM path in `CUDABackendConv.cu` -- TF32 is already on by default, but im2col+GEMM still trails cuDNN's own specialized small-kernel (e.g. 3x3) conv algorithms
- [ ] Settling-step count (T) reduction for genuine multi-step PC -- T is currently copied straight from [PCX](https://github.com/liukidar/pcax)'s (a JAX-based predictive-coding research library, used here as an external reference point) own per-architecture recipe (e.g. T=12 for VGG-7); now a CLI-overridable argument on `research/imagenet_pc_vgg7_ce.py` for sweeping, but whether a smaller T reaches comparable accuracy for proportionally less compute per batch is still untested
- [x] Checkpoint/resume for long training runs -- `ModelIO` (safetensors-backed) checkpoints after every epoch; rerunning the same command resumes from `<prefix>.safetensors` + `<prefix>_meta.json`, carrying over `global_step`, `best_acc`, and the data-shuffling RNG state so the LR schedule and data order pick up exactly where they left off
- [ ] mimalloc-backed allocation path, replacing the current `MemoryArena` allocator

## Contributing

We are actively looking for contributors to help build out the auxiliary systems. If you have experience with **C++/CUDA performance work** or **Python (Build Systems/DevOps)**, check out the open issues marked `good first issue` or reach out directly.

## License

MIT License @ 2026
