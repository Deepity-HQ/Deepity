@mainpage Deepity

Deepity is a Predictive Coding (PC) library engineered from the ground up for
zero-overhead, ultra-low-variance inference and learning. It is aggressively
CPU-optimized to extract maximum throughput from modern hardware, with a CUDA
backend currently in development.

This page is the entry point into the **API reference**. For build
instructions, benchmarks, and contribution guidelines, see the
[GitHub README](https://github.com/ra4ster/deepity#readme).

@section overview Overview

Deepity implements Predictive Coding Networks (PCNs) as an alternative to
standard backpropagation-based training. The library is written in C++20,
uses custom AVX2/AVX-512 SIMD kernels for activation functions, and packs
layer buffers into a single contiguous memory arena for cache locality. A
`nanobind`-based Python extension (`pydeepity`) exposes the same native
engine to Python with negligible overhead.

@section quickstart Quick Start

@subsection quickstart_cpp C++

The @ref Deep::DiscriminativePCNetwork "DiscriminativePCNetwork" class
manages layer hierarchies, bidirectionality, and dynamic thread scaling
based on batch size automatically:

@code{.cpp}
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/utils/Activations.h>
#include <vector>
#include <random>
#include <iostream>

int main() {
// Initialize a network with a batch size of 4 (e.g., for XOR)
Deep::DiscriminativePCNetwork net(4);

    // ... configure layers, load data (see examples/xor.py for a full
    // worked example, and DiscriminativePCNetwork's own docs below for
    // the layer-configuration API) ...

    // Train using the clean TrainStep API (150 relaxation steps per epoch)
    for (int epoch = 0; epoch < 1500; ++epoch) {
        float energy = net.TrainStep(X, Y, 150);
    }

    // Run inference using the Predict API
    std::vector<float> predictions = net.Predict(X, 150);

    for (float pred : predictions) {
        std::cout << "Prediction: " << pred << "\n";
    }

    return 0;

}
@endcode

@subsection quickstart_py Python

The same engine is available from Python via the compiled `pydeepity`
extension module. Runnable examples live in `examples/`, including
`examples/xor.py` (minimal training loop) and
`examples/train_mnist_deep.py` (end-to-end MNIST training).

@section architecture Core Architecture

@subsection architecture_networks Networks

Top-level network abstractions, each managing a layer stack and the
predictive-coding relaxation loop:

- @ref Deep::SimplePCNetwork — no precision weighting
- @ref Deep::DiscriminativePCNetwork — the original, precision-weighted PCN
- @ref Deep::ConvPCNetwork / @ref Deep::SimpleConvPCNetwork — convolutional,
  im2col-based
- @ref Deep::GaussSeidelPCNetwork — sequential (Gauss-Seidel) layer sweeps
  instead of simultaneous updates
- @ref Deep::DirectKPPCNetwork — Direct Kolen-Pollack feedback alignment
  (DFA) in place of symmetric weight transport
- @ref Deep::FullPCNetwork — DirectKPPCNetwork's DFA structure plus
  opt-in muPC scaling and residual connections, every extra OFF by
  default so it reproduces DirectKPPCNetwork exactly

@subsection architecture_layers Layers

All layers derive from a common @ref Deep::Layer "Layer" base:

- @ref Deep::Layer (base class)
- @ref Deep::SimplePCLayer
- @ref Deep::SimpleConvPCLayer / @ref Deep::ConvPCLayer
- @ref Deep::DiscriminativePCLayer
- @ref Deep::GaussSeidelPCLayer
- @ref Deep::DirectKPPCLayer
- @ref Deep::FullPCLayer

@subsection architecture_support Supporting Components

- @ref Deep::MemoryArena / @ref Deep::DeviceMemoryArena — contiguous buffer
  allocation for layer state
- @ref Deep::AdamUpdate / @ref Deep::AdamWUpdate — weight-update optimizer
- @ref Deep::StreamAlignedBatcher — batch alignment for SIMD kernels
- @ref Deep::Im2Col — convolution lowering
- @ref Deep::ModelIO — model save/load
- @ref Deep::ProfileAccumulator / @ref Timer — internal profiling utilities
- Activation functions (Elliot Sigmoid approximation, vectorized ReLU) —
  see `Activations.h`

@section performance Performance

On a 12th Gen Intel Core i7-12700H, Deepity sustains roughly 123 GFLOPS
during predictive-coding inference and learning when compiled with Clang
(LLVM), via custom SIMD micro-kernels, 64-byte-aligned buffers, and a
contiguous memory arena. Full benchmark methodology and GCC-vs-Clang
comparisons are in the
[README](https://github.com/ra4ster/deepity#-performance-at-a-glance).

@section roadmap Roadmap

- [x] SIMD micro-kernels (AVX2/AVX-512 Padé approximations)
- [x] Contiguous flat-memory buffers
- [x] PCNetwork abstraction (Layer hierarchy & bidirectional inference)
- [x] Python bindings (nanobind + NumPy support)
- [x] API reference documentation (Doxygen)
- [x] Multithreading and Precision Metrics
- [x] Memory Arena Contiguity
- [x] File IO Support (@ref Deep::ModelIO)
- [ ] CUDA accelerated engine (GPU GEMM operations for massive scales) —
  in active development on the `dev/CUDA` branch
- [ ] Java port

@section elsewhere Elsewhere

- [GitHub README](https://github.com/ra4ster/deepity#readme) — install,
  build (`build.py`), and benchmark details
- [CONTRIBUTING.md](https://github.com/ra4ster/deepity/blob/main/CONTRIBUTING.md)
