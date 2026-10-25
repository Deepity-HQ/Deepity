<p align="center">
  <img src="resources/deepity-mark.png" alt="Deepity logo" width="250"/>
</p>

# Deepity

**A fast C++ / CUDA engine for training neural networks with predictive coding — a brain-inspired alternative to backpropagation — with a simple Python interface.**

[Website & Documentation](https://ra4ster.github.io/Deepity) • [Quick Start](#quick-start) • [Benchmarks](#performance--benchmarks) • [Contributing](#contributing)

> **At a glance:** Deepity's `SimplePCN` reaches **98.21%** MNIST test accuracy in **487.5s** on a laptop CPU — faster than the two established predictive-coding libraries, jpc (JAX) and ngc-learn, at matching or better accuracy on an identical setup. Details, and where Deepity does *not* win, are in [Performance & Benchmarks](#performance--benchmarks).

---

## Contents

- [What is Deepity?](#what-is-deepity)
- [New to predictive coding? Start here](#new-to-predictive-coding-start-here)
- [Why use predictive coding?](#why-use-predictive-coding)
- [Quick Start](#quick-start)
- [Which network should I use?](#which-network-should-i-use)
- [Performance & Benchmarks](#performance--benchmarks)
- [Building from source](#building-from-source)
- [Architecture & Roadmap](#architecture--roadmap)
- [Contributing](#contributing) • [License](#license)

---

## What is Deepity?

Most neural networks today (anything you'd build in PyTorch or TensorFlow) learn with **backpropagation**: the network makes a prediction, measures how wrong it is at the output, and sends that error backward through every layer in one big pass.

**Predictive Coding Networks (PCNs)** learn differently. Each layer learns from a *local* signal — how far its own activity is from what the layer next to it predicted — instead of waiting for one global error to flow back from the output. The idea comes from neuroscience, where it's a leading theory of how the brain learns.

Deepity is an engine for building and training these networks quickly:

- **A C++ core** with hand-optimized CPU kernels (SIMD, Intel MKL or OpenBLAS) and a **CUDA backend** for NVIDIA GPUs.
- **A Python package, `pydeepity`,** so you can define and train networks in a few lines, without writing any C++.
- **Several PCN variants,** from a plain beginner-friendly network to one that bundles recent research techniques you can switch on one at a time.

You don't need to know anything about predictive coding to try it. The next section explains the core idea in plain terms.

---

## New to predictive coding? Start here

Think of each layer of the network as holding a **guess** about what it's seeing. Every layer also makes a **prediction** about what the layer next to it should be seeing. Wherever a prediction doesn't match the guess, that mismatch is a **prediction error**.

Training a PCN on one batch of data works in two phases:

1. **Settling (inference).** The input is fixed at the bottom of the network and the correct answer at the top. Then every layer repeatedly nudges its guess to reduce the prediction errors around it. Each round of nudging is one **settling step**, and the number of rounds is called **T**. After enough steps, the guesses stop changing much: the network has "settled" on a consistent explanation of the input and the answer.
2. **Learning.** Each layer then adjusts its weights to make its predictions closer to the settled guesses. This update uses only that layer's own prediction error. No layer needs to know what's happening elsewhere in the network.

The total amount of prediction error across the network is called its **energy**. Settling lowers the energy for the current batch, and learning lowers it for future batches. When you call `train_step` in Deepity, the number it returns is this energy.

**The main trade-off to know about:** more settling steps (a larger T) give a better-converged update, but cost proportionally more compute per batch. Most of the speed and accuracy differences in the benchmarks below come down to T.

Deepity implements two ways of turning prediction errors into weight updates:

- **Classic settling** (`SimplePCN`, `FullPCN`, and most other variants): every layer's guess is refined over several settling steps, as described above, before the weights update.
- **Direct Kolen-Pollack, or DKP** (`DKPPCN`): trains a separate set of *feedback* weights alongside the normal forward weights, instead of reusing the forward weights to send predictions back down. In practice it needs far fewer settling steps: `DKPPCN` below uses 2 where `SimplePCN` uses 20, which is most of why it's dramatically faster, at a real accuracy cost (see the [MNIST comparison](#mnist-matched-configuration)).

---

## Why use predictive coding?

- **No stored backward graph.** Backpropagation has to keep the entire forward computation in memory until the backward pass runs. In a PCN, each layer's update depends only on its own local prediction error, so Deepity never stores a backward computation graph. *(O(1) memory footprint for the learning signal.)*
- **No weight-transport requirement.** In backpropagation, each layer's backward pass must use the exact transpose of its forward weights, which is one of the main reasons it's considered biologically implausible. Deepity's local updates don't inherit that constraint, and DKP relaxes it further by *learning* the feedback pathway instead of assuming it.
- **A good fit for future hardware.** That same locality is why the research literature is interested in predictive coding for neuromorphic and edge hardware. Deepity doesn't have benchmarks on either yet, so treat this as context rather than a claim about this engine.

---

## Quick Start

### 1. Install

Deepity currently builds from source. You'll need CMake 3.21+, Ninja, a C++20 compiler, OpenBLAS or Intel MKL, and Python 3.9+ with `nanobind` (the full list is in [Building from source](#building-from-source)). Then:

```bash
git clone https://github.com/Ra4ster/Deepity.git
cd Deepity
python build.py Release OpenBLAS --fast
```

This builds the C++ engine, runs the test suite, and builds the `pydeepity` Python package.

### 2. Train your first network

A tiny network with 2 inputs, 8 hidden units and 1 output, trained on a single example:

```python
import numpy as np
from pydeepity import SimplePCN, Linear, TanH

# Declare the layers, the same way you would in PyTorch.
net = SimplePCN(
    Linear(2, 8), TanH(),
    Linear(8, 1),
    batch_size=1,
)

# Set how fast the weights learn (learning_rate) and how fast
# each layer's guess moves during settling (inference_rate).
net.configure(learning_rate=0.05, inference_rate=0.3, lmbda=0.0001, optimizer="SGD")

x = np.array([1.0, -1.0], dtype=np.float32)   # input
y = np.array([1.0], dtype=np.float32)          # target

# Settle for 50 steps, then update the weights locally.
# The return value is the network's energy (total prediction error).
energy = net.train_step(x, y, steps=50)
```

Run `train_step` in a loop over your data and the energy should fall as the network learns.

### 3. Next steps

- **A full training script:** `examples/mnist.py` trains on MNIST end to end. It's the script behind the headline benchmark.
- **Larger experiments:** `research/imagenet.py` and `research/imagenet_pc_vgg7_ce.py` train convolutional PCNs on ImageNet-scale data.
- **Using C++ directly:** start with `include/deepity/networks/DiscriminativePCNetwork.h`. Most `pydeepity` classes correspond directly to a C++ network class under `include/deepity/networks/`.

---

## Which network should I use?

| If you want... | Use | Notes |
|---|---|---|
| The simplest, most standard predictive coding | `SimplePCN` | Classic settling. The best place to start, and the network behind the headline MNIST result. |
| Much faster training, accepting some accuracy loss | `DKPPCN` | Direct Kolen-Pollack feedback, so very few settling steps are needed (2 instead of 20 in the benchmarks). |
| Recent research techniques, switchable one at a time | `FullPCN` | Includes DKP and adds optional extras such as µPC scaling, residual connections, incremental PC (iPC), error-based PC (ePC), momentum settling and a cross-entropy output. Everything is off by default. |
| Image data | The convolutional variants | Convolutional PCNs, used in the ImageNet research scripts. |

---

## Performance & Benchmarks

**How to read this section:** the headline comparison pits Deepity against other predictive-coding libraries running exactly the same network, data and training settings, so the only difference is the implementation. A standard PyTorch backpropagation run is included as a reference point.

Development and testing for the CUDA backend happens on the Ohio Supercomputer Center (Project PAS0350), on NVIDIA A100 GPUs. The numbers below are all CPU-only, run on one machine (12th Gen Intel Core i7-12700H, 20 logical threads, native + MKL build), so they measure the engine's single-machine CPU path specifically, not the CUDA backend. Every command below is reproducible from the repo root, and raw output for each run is committed under `logs/`.

### MNIST, matched configuration

MNIST is a standard dataset of handwritten digits (0–9); the task is to classify each image.

The network is 784&rarr;512&rarr;512&rarr;10 (sigmoid), with T=20 settling steps/batch, batch size 250, 100 epochs, and AdamW lr=0.00373 with 0.94/epoch decay — identical across the first two columns so the comparison is apples-to-apples.

| | Deepity `SimplePCN` | jpc (JAX) | ngc-learn `pc_discrim` | PyTorch backprop (control) |
|---|---|---|---|---|
| Test / dev accuracy | **98.21%** | 97.55% | 98% (best dev) | 97.51% |
| Wall-clock, 100 epochs | **487.5s** | 531.4s | 1707.7s | 627.8s |
| Command | `python examples/mnist.py 7 100 20` | `python benchmark_jpc_mnist.py 7 100` | [ngc-museum](https://github.com/NACLab/ngc-museum)'s `pc_discrim` exhibit, unmodified | `python examples/test_ffnn_deep.py 7 100` |
| Raw log | `logs/mnist_simd_fusion_confirm.txt` | `logs/jpc_mnist_run.txt` | `logs/ngclearn_pc_discrim_mnist.txt` | `logs/bp_control_mnist.txt` |

- **Deepity vs. jpc** is the closest comparison: both are predictive-coding implementations on the same architecture, settling-step count, batch size, epoch count and optimizer schedule. jpc is built on JAX/XLA; Deepity is hand-tuned C++. Deepity trains faster at matching accuracy.
- **Deepity vs. ngc-learn:** identical architecture, settling-step count, batch size, epoch count, forward-projection init and optimizer schedule, run as two independent predictive-coding implementations on the same machine. Deepity reaches equal-or-better accuracy in less time.
- **The PyTorch column is a reference point, not a PC run.** It's single-pass backprop (one forward and one backward pass per batch) against the same targets via MSE loss. It does one pass per batch where PC does T=20 settling passes, so it isn't directly comparable to the PC columns.

**The faster option:** `DKPPCN` (784&rarr;512&rarr;10, batch size 1000, only 2 inference steps/batch instead of 20) reaches 95.97% in 49.6s on the same CPU (`python examples/mnist_dkppcn.py 7 50 2 cpu`, `logs/DKPPCN_native_mkl.txt`). That's a different point on the speed/accuracy trade-off curve.

### Raw kernel throughput (GFLOPS)

**What this measures:** GFLOPS is billions of floating-point operations per second, a standard measure of how efficiently code uses the processor. Higher means more of the hardware's raw compute is doing useful work.

`tests/tGflopsBenchmark.cpp`, built as the `GflopsBenchmark` target via Google Benchmark, measures sustained GFLOPS for a full PC train step (`SimplePCN`, 784-512-256-64-10, batch 256, T=20):

```bash
python build.py Release OpenBLAS --fast
./build/Release/bin/GflopsBenchmark
```

**How the FLOPs are counted** (from the file's own header comment): for a layer transitioning $\text{size} \to \text{nextSize}$ at a given $\text{batchSize}$, each settling step runs two matrix multiplications (GEMMs) of matched cost — the forward prediction GEMM and the feedback GEMM — each costing $2 \cdot \text{batchSize} \cdot \text{size} \cdot \text{nextSize}$ FLOPs (one multiply plus one add per multiply-accumulate, the standard convention). `UpdateWeights()` runs a third, equally sized GEMM once per *train step* (not once per settling step) for the weight gradient. Summed over every weighted layer transition:

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

That 11.54 GFLOP figure is fixed by the architecture, batch size and step count. What's measured is the wall time per train step; Google Benchmark's `FLOPS` counter divides the fixed number above by that measured time:

$$
\text{GFLOPS/s} = \frac{11.54 \times 10^{9}\ \text{FLOPs}}{\text{measured wall time}} \approx \textbf{690 GFLOPS/s}
$$

See `logs/gflops_benchmark.txt` for the raw runs.

The same FLOP-accounting formula applied to the full [MNIST, matched configuration](#mnist-matched-configuration) run — 329 TFLOP total across 100 epochs (24,000 train steps), at 487.5s wall-clock — gives **675 GFLOPS/s** sustained, end to end.

### Activation-kernel speedup

**What this measures:** activation functions (tanh, sigmoid, ReLU, GELU) are applied to every value in every layer at every settling step, so their speed matters. Deepity uses hand-written SIMD kernels, which process many values per CPU instruction, instead of plain one-value-at-a-time loops.

`tests/tActivations.cpp`, built as the `ActivationBenchmark` target, compares Deepity's SIMD kernels (hand-written AVX2/AVX-512/SSE intrinsics calling SLEEF for the transcendental ones; see `include/deepity/utils/activations/`) against naive `std::tanh`/`std::exp`-based scalar loops at matched input sizes:

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

Tanh, Sigmoid and GELU win at every tested size, with the margin generally growing with size. Tanh's jump at the largest size (131,072) is because `Tanh.h`'s settling loop also crosses an OpenMP threshold (`n > 16384`) at that size, adding multi-core parallelism on top of SIMD width. ReLU sits at 0.88x–1.00x, essentially tied with the naive loop. That's expected: `relu(x) = max(0, x)` is one comparison with no multiply, so it's limited by memory bandwidth rather than compute, and the compiler's own auto-vectorization of a one-line loop already reaches the same ceiling.

_Note: the CUDA (GPU) backend is a separate code path with its own characteristics — TF32-enabled im2col+cuBLAS convolution, and CUDA graph capture for the settling loop — and isn't covered by the CPU numbers above. ImageNet-scale training on it is in active development; see `research/imagenet_pc_vgg7_ce.py`, which supports checkpoint/resume and a command-line-overridable settling-step count for sweeping T._

---

## Building from source

### Prerequisites

- CMake 3.21+ and Ninja
- A C++20 compiler (Clang recommended; GCC and MSVC are also supported)
- OpenBLAS or Intel MKL
- Python 3.9+ with development headers, plus `nanobind` (for the `pydeepity` bindings)
- CUDA Toolkit (optional: the CUDA backend is detected automatically if present; otherwise the project builds and runs CPU-only)

See `CONTRIBUTING.md` for the full prerequisite list and known-working manual CMake invocations (for example, Windows + Clang).

### Build

`build.py` handles CMake/Ninja configuration, compilation and running the test suite in one command:

```bash
git clone https://github.com/Ra4ster/Deepity.git
cd Deepity
python build.py Release OpenBLAS --fast
```

- `Release` / `Debug` selects the build type.
- `OpenBLAS` / `MKL` selects the linear-algebra library.
- `--fast` targets a portable AVX2/FMA baseline. Run `python build.py --list-profiles` for the other options, and `python build.py --help` for CUDA, test and job-count flags.

The `pydeepity` Python package is built too, unless you pass `--no-python-bindings`.

---

## Architecture & Roadmap

Deepity started as an academic research project and is now becoming a scalable infrastructure tool.

**Done**

- [x] Fused, optimized CUDA kernels
- [x] DKP-PC acceleration
- [x] ImageNet data loading pipeline (see `research/imagenet.py`)
- [x] Cross-platform CI/CD (Ubuntu + Windows wheel builds and pyright type checks on every push; see `.github/workflows`)
- [x] Checkpoint/resume for long training runs: `ModelIO` (safetensors-backed) saves a checkpoint after every epoch, and rerunning the same command resumes from `<prefix>.safetensors` + `<prefix>_meta.json`, carrying over `global_step`, `best_acc` and the data-shuffling RNG state, so the learning-rate schedule and data order pick up exactly where they left off

**In progress / planned**

- [ ] Stable, large-scale ImageNet training (actively being tuned)
- [ ] cuDNN-backed convolution (Winograd / implicit-GEMM), replacing the current im2col + cuBLAS-GEMM path in `CUDABackendConv.cu`. TF32 is already on by default, but im2col+GEMM still trails cuDNN's specialized small-kernel (e.g. 3x3) convolution algorithms.
- [ ] Fewer settling steps (smaller T) for genuine multi-step PC. T is currently taken directly from [PCX](https://github.com/liukidar/pcax) (a JAX-based predictive-coding research library, used here as an external reference point) and its per-architecture recipes, e.g. T=12 for VGG-7. It's now a command-line argument on `research/imagenet_pc_vgg7_ce.py` for sweeping, but whether a smaller T reaches comparable accuracy for proportionally less compute per batch is still untested.
- [ ] mimalloc-backed allocation path, replacing the current `MemoryArena` allocator

---

## Contributing

We're actively looking for contributors to help build out the auxiliary systems. If you have experience with **C++/CUDA performance work** or **Python (build systems/DevOps)**, check out the open issues marked `good first issue` or reach out directly.

## License

MIT License @ 2026
