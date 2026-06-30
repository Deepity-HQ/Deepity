# ![Deepity](resources/Deepity.png)

Deepity is a Predictive Coding (PC) library engineered from the ground up for zero-overhead, ultra-low variance inference and learning. It is aggressively CPU-optimized to extract maximum throughput from modern hardware, with a CUDA backend currently in development.

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

Deepity is built for speed. On a **Dell Inspiron 16 Plus 7620** (12th Gen Intel Core i7-12700H, 20 logical processors), the engine sustains approximately **116 GFLOPS** during predictive-coding inference and learning.

**Benchmark Configuration:**
* **Architecture:** 784 → 512 → 256 → 64 → 10
* **Batch size:** 256
* **Iterations:** 157
* **Average runtime:** 1.244 s

The dominant computation consists of batched single-precision matrix multiplications (`SGEMM`), corresponding to roughly 144.4 GFLOPs of floating-point work:

$$\frac{144.4\text{ GFLOPs}}{1.24379\text{ s}}\approx 116.1\text{ GFLOPS}$$

By utilizing native C++ extensions via pybind11, Deepity maintains this performance footprint in Python with negligible overhead—running **~3.5× faster** than an equivalent, highly vectorized NumPy implementation.

<p align="center">
<img src="resources/PyTest.png" alt="Python Benchmark Results" width="400">
</p>

| Implementation | Avg (ms) | Min (ms) | Max (ms) |
| :--- | ---: | ---: | ---: |
| **Deepity (Python)** | **1201.6** | **1200.3** | **1205.3** |
| NumPy (Naive) | 4201.6 | 4147.5 | 4281.3 |

*Note: High-level research frameworks routinely incur heavy penalties from Python execution and tensor abstractions. Deepity bypasses this by keeping the entire inference loop in native C++ memory.*

---

## ⚡ Core Architecture & Optimizations

Deepity achieves its low-variance execution times through strict memory management and custom hardware intrinsics.

**Custom SIMD Micro-Kernels**
We bypass standard C++ library bottlenecks by implementing highly optimized activation functions using raw AVX2 and AVX-512 intrinsics. 

**Rational Polynomial Tanh**
Deepity avoids expensive `expf` instruction calls by utilizing a highly tuned Padé rational polynomial approximation. This yields up to a 40% speedup over `std::tanh` without sacrificing necessary precision.

<p align="center">
<img src="resources/ActivationCPUMetrics.png" alt="Activation CPU Metrics" width="200">
</p>

**Saturated Vectorized ReLU**
The ReLU implementation processes up to 16 floats per clock cycle, completely saturating standard single-core RAM bandwidth limits (~15.8 GB/s).

**Strict 64-Byte Alignment**
To prevent hardware exceptions and segmentation faults when loading wide 256-bit or 512-bit registers, Deepity enforces strict 64-byte memory boundaries (`std::align_val_t{64}`) for all internal sequential sub-buffers.

**Contiguous Arena Allocator**
All layer buffers in a network are packed into a single contiguous memory block. This maximizes L1/L2 cache locality and eliminates pointer-chasing overhead across the layer hierarchy.

---

## 📊 Design Decisions & Benchmarks

During development, we benchmarked several architectural approaches to find the absolute ceiling for CPU throughput.

### The Impact of Batching
Batching provides massive scaling. A batch size of **256** proved to be the sweet spot for maximizing CPU utilization before cache eviction penalties take over.

| Batch Size | Time (ms) |
| --- | --- |
| 1 (None) | 4484 |
| 16 | 3149 |
| 64 | 2338 |
| **256** | **2233** |
| 512 | 2265 |

### Memory Layout: Contiguous vs. Separate
Packing all layer attributes into a single flat array showed zero performance penalty over separate heap allocations, while providing vastly simpler alignment guarantees and predictable cache behavior.

| Layout | Time (ms) |
| --- | --- |
| Separate vectors | 4481 |
| Contiguous block | 4484 |

### Random Number Generation
We tested `OpenRAND` against the standard `std::mt19937` generator. Because the results were within a 5% margin of error, we opted for the standard library MT implementation to minimize external dependencies.

---

## 🛠️ Example Usage

### C++

Running a Predictive Coding layer in Deepity is built to be straightforward and explicit.

```cpp
#include "PCLayer.h"
#include "Activations.h"
#include <vector>

int main() {
    Deep::PCLayer pc(1000, 100);
    std::vector<float> input_sample(1000, 0.5f);
    
    pc.ClampState(input_sample);

    for (int i = 0; i < 157; ++i) {
        pc.CalculateState();
        pc.UpdateState();
    }

    pc.UpdateWeights();
    pc.UnclampState();

    #ifdef _DEBUG
    pc.DebugStats();
    #endif

    return 0;
}
```

### Python

Deepity ships with lightweight Python bindings that accept standard NumPy arrays.

```python
import deepity as deep
import numpy as np

# Build network
net = deep.PCNetwork()
net.add_layer(784, 256, lr=1e-4, act="relu", dact="drelu")
net.add_layer(256, 64,  lr=1e-4, act="relu", dact="drelu")
net.add_layer(64, 10,   lr=1e-4, act="relu", dact="drelu")

net.randomize_weights()

# Clamp input
x = np.random.uniform(0.0, 1.0, 784).astype(np.float32)
net.clamp(x)

# Inference loop
for _ in range(50):
    energy = net.calculate_state()
    net.update_state()

# Learning step
net.update_weights()
print(f"Final energy: {energy:.4f}")
```

---

## 📅 Roadmap

- [x] SIMD micro-kernels (AVX2/AVX-512 Padé approximations)
- [x] Contiguous flat-memory buffers
- [x] PCNetwork abstraction (Layer hierarchy & bidirectional inference)
- [x] Python bindings (pybind11 + NumPy support)
- [ ] CUDA accelerated engine (GPU GEMM operations for massive scales)
- [ ] Java port
- [ ] API reference documentation (Doxygen)

---

## 🏗️ Project Structure

```text
includes/       # Public headers (PCLayer.h, PCNetwork.h, Activations.h)
src/            # C++ source (PCLayer.cpp, PCNetwork.cpp)
pybind/         # Python bindings (binding.cpp)
tests/          # C++ and Python test suites
bin/            # Build outputs (library, executables, Python .so)
resources/      # Images and benchmark assets
```

---
*Ra4ster (Jack R) @ 2026 ❤️*