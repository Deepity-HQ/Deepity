<p align="center">
  <img src="resources/deepity-mark.png" alt="Deepity logo" width="250"/>
</p>

# Deepity ⚡️

A high-performance C++ / CUDA engine for Predictive Coding Networks (PCNs) and Direct Kolen-Pollack algorithms.

Deepity bypasses standard backpropagation memory bottlenecks using localized learning rules, allowing for highly efficient neuromorphic and edge-compute scaling.

**Current Benchmark:** see [Performance & Benchmarks](#performance--benchmarks) below for what's actually measured today.

[Website & Documentation](https://ra4ster.github.io/Deepity) • [Current Benchmarks](#performance--benchmarks) • [Contributing](#contributing)

---

## Why Predictive Coding?

Standard backpropagation (like PyTorch/TensorFlow) requires massive global memory overhead for backward passes. Deepity utilizes **local Hebbian learning rules**, allowing weight updates to occur simultaneously with forward passes.

**Key Advantages:**

- **O(1) Memory Footprint:** No massive gradient graphs stored in VRAM.
- **Streaming Token Processing:** Continuous learning without catastrophic forgetting.
- **Hardware Symbiosis:** Maps perfectly to neuromorphic architectures and extreme edge devices.

## Performance & Benchmarks

Development and testing happens on the Ohio Supercomputer Center (Project PAS0350), on NVIDIA A100 GPUs.

The actual benchmark suite (`tests/tReadme.cpp`, built as the `ReadmeBenchmark` target via Google Benchmark) measures the CPU/CUDA settling-loop's own raw throughput and GFLOPS on synthetic workloads -- it does not train on MNIST or any other real dataset, and there is no PyTorch/standard-backprop comparison anywhere in this repository. Build it yourself and run it to get current numbers for your own hardware:

```bash
python build.py Release OpenBLAS --fast
./build/Release/bin/ReadmeBenchmark
```

_(Note: ImageNet-scale training is in active development -- see `imagenet.py` at the repo root -- and the CUDA backend generally is still being hardened.)_

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

See `imagenet.py` / `mnist.py` at the repo root for larger, end-to-end training scripts.

_(Note: If you prefer to use the engine natively in C++, see the `DiscriminativePCNetwork` example above -- most `pydeepity` wrapper classes correspond directly to a C++ network class under `include/deepity/networks/`.)_

## Architecture & Roadmap

Deepity is currently pivoting from an academic research project into a scalable infrastructure tool.

- [x] Core CUDA kernel fused optimization
- [x] DKP-PC acceleration
- [x] ImageNet data loading pipeline (see `imagenet.py`)
- [ ] Stable, large-scale ImageNet training (actively being tuned)
- [x] Cross-platform CI/CD (Ubuntu + Windows wheel builds, pyright, on every push -- see `.github/workflows`)

## Contributing

We are actively looking for contributors to help build out the auxiliary systems. If you have experience with **C++/CUDA performance work** or **Python (Build Systems/DevOps)**, check out the open issues marked `good first issue` or reach out directly.

## License

MIT License @ 2026
