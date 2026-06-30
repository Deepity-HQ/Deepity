# ![Deepity](resources/Deepity.png)

Deepity is a Predictive Coding (PC) model library designed for ultra-low variance, high-speed inference and updating. The library is heavily CPU-optimized, with CUDA support on the horizon.

<p align="center">
<img src="resources/LayerCPUMetrics.png" alt="PCN Layer CPU Metrics" width="300">
</p>

---

<p align="center">
  <img src="resources/pcn_profile.svg"
       alt="PCN Profiler"
       class="svg-icon"
       width="800">
</p>

## 🚀 Performance at a Glance

When running on optimized hardware (*a Dell Inc. Inspiron 16 Plus 7620 with a 12th Gen Intel® Core™ i7-12700H × 20*), Deepity hits a sustained execution speed that rivals embedded hardware:

$$
\frac{180\text{ GigaFLOPs}}{1.18956\text{ s}} \approx 151.32\text{ GFLOPS}
$$

The Python bindings maintain this performance with negligible overhead — benchmarked at **3.5× faster** than an equivalent NumPy implementation:

<p align="center">
<img src="resources/PyTest.png" alt="Python Benchmark Results" width="400">
</p>

| Implementation | Avg (ms) | Min (ms) | Max (ms) |
| --- | --- | --- | --- |
| **Deepity (Python)** | **1201.6** | **1200.3** | **1205.3** |
| NumPy (naive) | 4201.6 | 4147.5 | 4281.3 |

*For reference, performing similar operations using standard academic libraries like `ngc-learn` typically yields under 15 GFLOPS due to high-level language memory overhead.*

---

## 🛠️ Example Usage

### C++

Getting a predictive coding layer up and running in Deepity is straightforward:

```cpp
#include "PCNLayer.h"
#include "Activations.h"
#include <vector>

int main(void) {

    // Initialize a layer: 1000 inputs, 100 outputs
    // Defaults: stepSize=30, activation=Deep::relu,
    // learning rate = inference rate = 1e-6
    Deep::PCLayer pc(1000, 100);

    // Your input data (must match pc.GetInputSize())
    std::vector<float> input_sample(1000, 0.5f);

    // Run inference and update beliefs/weights
    pc.RunPrediction(input_sample.data());

    // Flush any remaining partial batch
    pc.Flush();

    #ifdef _DEBUG
    pc.DebugStats(); // Inspect layer health, NaNs, and statistics
    #endif

    return 0;
}
```

### Python

Deepity ships with Python bindings via pybind11:

```python
import deepity as deep
import numpy as np

net = deep.PCNetwork()
net.add_layer(784, 256, lr=1e-4, ir=1e-4, step_size=30, act="relu")
net.add_layer(256, 64)
net.add_layer(64, 10)
net.compile()

# Feed an input bottom-up
x = np.random.uniform(0.0, 1.0, 784).astype(np.float32)
net.inference_step(x)
net.flush_inference()

print("Energy:", net.total_energy())
net.save("model.bin")
```

---

## ⚡ Core Architecture Features

**Custom SIMD Activation Functions**
Deepity bypasses standard C++ library bottlenecks by implementing highly optimized activation functions using AVX2 and AVX-512 intrinsics.

**High-Performance Tanh**
Bypasses expensive `expf` evaluations using a highly tuned Padé rational polynomial approximation, yielding up to a ~40% speedup over `std::tanh`.

<p align="center">
<img src="resources/ActivationCPUMetrics.png" alt="Activation CPU Metrics" width="200">
</p>

**Vectorized ReLU**
Processes up to 16 floats per clock cycle, completely saturating standard single-core RAM bandwidth limits (~15.8 GB/s).

**Strict 64-byte Alignment**
To prevent hardware exceptions and segmentation faults when loading wide 256-bit or 512-bit registers, Deepity enforces strict 64-byte memory boundaries (via `std::align_val_t{64}`) for all internal sequential sub-buffers ($W$, $z$, $p$, $err$).

**Contiguous Arena Allocator**
All layer buffers in a network are packed into a single contiguous memory block, maximising cache locality and eliminating pointer-chasing overhead across layers.

---

## 📊 Single-Layer Benchmarks & Design Decisions

### 1. The Impact of Batch Size

Batching provides massive performance scaling. A batch size of **256** proved to be the sweet spot for maximum CPU throughput.

| Batch Size | Time (ms) |
| --- | --- |
| 1 (None) | 4484 |
| 16 | 3149 |
| 32 | 2634 |
| 64 | 2338 |
| 128 | 2263 |
| **256** | **2233** |
| 512 | 2265 |

### 2. Random Number Generation

We compared `OpenRAND` against the standard C++ `std::mt19937` generator. Results were within a 5% margin of error, so OpenRAND was chosen for its parallelism-friendly design.

| Generator | Time (ms) |
| --- | --- |
| OpenRAND | 4712 |
| mt19937 | 4482 |

### 3. Memory Layout: Contiguous vs. Separate

Packing all layer attributes into a single flat array showed zero performance penalty over separate heap allocations, while providing simpler alignment guarantees and better cache behaviour.

| Layout | Time (ms) |
| --- | --- |
| Separate vectors | 4481 |
| Contiguous block | 4484 |

---

## 📅 Roadmap

- [x] SIMD micro-kernels — AVX2/AVX-512 Padé approximations for activations
- [x] Contiguous flat-memory buffers — cache-friendly, pointer-chasing-free
- [x] PCNetwork abstraction — layer hierarchy with bidirectional inference and generation
- [x] Python bindings — full pybind11 port with NumPy array support
- [ ] CUDA accelerated engine — moving GEMM operations to GPU for massive model scales
- [ ] Java port
- [ ] API reference documentation — Doxygen HTML guides

---

## 🏗️ Project Structure

```
includes/       # Public headers (PCNLayer.h, PCNNetwork.h, Activations.h, Optimize.h)
src/            # C++ source (PCNLayer.cpp, PCNNetwork.cpp)
pybind/         # Python bindings (binding.cpp)
tests/          # C++ and Python test suites
bin/            # Build outputs (library, executables, Python .so)
resources/      # Images and benchmark assets
```

---

Ra4ster (Jack R) @ 2026 ❤️