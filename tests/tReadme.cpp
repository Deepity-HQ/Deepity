#include <benchmark/benchmark.h>
#include <vector>
#include <cmath>
#include <random>
#include "DiscriminativePCNetwork.h"
#include "DiscriminativePCLayer.h"
#include "Activations.h"

using namespace Deep;

// ============================================================================
// 1. GFLOPS & END-TO-END BENCHMARK (784 -> 512 -> 256 -> 64 -> 10)
// ============================================================================
static void BM_Readme_GFLOPS_Network(benchmark::State& state) {
    const int batchSize = 256;
    DiscriminativePCNetwork net(batchSize);

    // Architecture: 784 -> 512 -> 256 -> 64 -> 10
    net.AddLayer(784, 512, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(512, 256, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(256, 64,  0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(64,  10,  0.05f, 0.3f, 0.0f, 0.0001f, Deep::linear, Deep::dLinear);

    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::vector<float> dummyX(batchSize * 784, 0.5f);
    std::vector<float> dummyY(batchSize * 10, 0.1f);

    for (auto _ : state) {

        net.TrainStep(dummyX, dummyY, 157);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Readme_GFLOPS_Network)->Unit(benchmark::kMillisecond);

// ============================================================================
// 2. PYTHON/NUMPY COMPARISON: 10,000 INPUTS THROUGHPUT
// ============================================================================
static void BM_Readme_10k_Throughput(benchmark::State& state) {
    const int batchSize = 256;
    const int totalSamples = 10000;
    const int batches = (totalSamples + batchSize - 1) / batchSize;

    DiscriminativePCNetwork net(batchSize);
    net.AddLayer(784, 512, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(512, 10,  0.05f, 0.3f, 0.0f, 0.0001f, Deep::linear, Deep::dLinear);
    net.Compile(); // Allocate contiguous block
    
    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::vector<float> dummyX(batchSize * 784, 0.5f);
    std::vector<float> dummyY(batchSize * 10, 0.1f);

    for (auto _ : state) {
        for (int b = 0; b < batches; b++) {
            net.TrainStep(dummyX, dummyY, 50);
        }
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Readme_10k_Throughput)->Unit(benchmark::kMillisecond);

// ============================================================================
// 3. SIZE 128 WORKLOADS (GCC vs Clang Metrics)
// ============================================================================
static DiscriminativePCNetwork Create128Network() {
    DiscriminativePCNetwork net(1);
    net.AddLayer(128, 128, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(128, 128, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::linear, Deep::dLinear);
    net.Compile(); // Required for v2 MemoryArena architecture
    return net;
}

static void BM_Network_Inference_128(benchmark::State& state) {
    auto net = Create128Network();
    std::vector<float> dummyX(128, 0.5f);
    for (auto _ : state) {
        net.Predict(dummyX, 1);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Network_Inference_128);

static void BM_Network_TrainSample_128(benchmark::State& state) {
    auto net = Create128Network();
    std::vector<float> dummyX(128, 0.5f);
    std::vector<float> dummyY(128, 0.1f);
    
    for (auto _ : state) {
        net.Clamp(dummyX);
        net.GetTerminalLayer()->ClampState(dummyY);
        net.CalculateState();
        net.UpdateState();
        net.UpdateWeights();
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Network_TrainSample_128);

static void BM_Layer_UpdateWeights_128(benchmark::State& state) {
    // Isolated layer for raw weight update speed
    DiscriminativePCLayer layer(128, 128, 1, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    std::mt19937 rng(42);
    layer.RandomizeWeights(rng);
    
    for (auto _ : state) {
        layer.UpdateWeights();
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Layer_UpdateWeights_128);

// ============================================================================
// 4. ACTIVATION FUNCTIONS (Naive vs Deepity SIMD + Sigmoids)
// ============================================================================
void naive_tanh(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = std::tanh(x[i]);
}

void naive_sigmoid(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

static void BM_Activation_StdTanh(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<float> data(size, 0.5f);
    for (auto _ : state) {
        naive_tanh(data.data(), size);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Activation_StdTanh)->Arg(10048)->Arg(1000064);

static void BM_Activation_DeepityTanh(benchmark::State& state) {
    size_t size = state.range(0);
    size_t bytes = (size * sizeof(float) + 63) & ~63; // 64-byte alignment padding
    float* data = static_cast<float*>(std::aligned_alloc(64, bytes));
    std::fill_n(data, size, 0.5f);

    for (auto _ : state) {
        Deep::tanh(data, size);
        benchmark::ClobberMemory();
    }
    std::free(data);
}
BENCHMARK(BM_Activation_DeepityTanh)->Arg(10048)->Arg(1000064);

static void BM_Activation_DeepityDTanh(benchmark::State& state) {
    size_t size = state.range(0);
    size_t bytes = (size * sizeof(float) + 63) & ~63;
    float* data = static_cast<float*>(std::aligned_alloc(64, bytes));
    std::fill_n(data, size, 0.5f);

    for (auto _ : state) {
        Deep::dTanh(data, size, true);
        benchmark::ClobberMemory();
    }
    std::free(data);
}
BENCHMARK(BM_Activation_DeepityDTanh)->Arg(10048)->Arg(1000064);

static void BM_Activation_NaiveSigmoid(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<float> data(size, 0.5f);
    for (auto _ : state) {
        naive_sigmoid(data.data(), size);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Activation_NaiveSigmoid)->Arg(10048)->Arg(1000064);

static void BM_Activation_DeepitySigmoid(benchmark::State& state) {
    size_t size = state.range(0);
    size_t bytes = (size * sizeof(float) + 63) & ~63;
    float* data = static_cast<float*>(std::aligned_alloc(64, bytes));
    std::fill_n(data, size, 0.5f);

    for (auto _ : state) {
        Deep::sigmoid(data, size);
        benchmark::ClobberMemory();
    }
    std::free(data);
}
BENCHMARK(BM_Activation_DeepitySigmoid)->Arg(10048)->Arg(1000064);

// ============================================================================
// 5. IMPACT OF BATCHING (1, 16, 64, 256, 512)
// ============================================================================
static void BM_ImpactOfBatching(benchmark::State& state) {
    int batchSize = state.range(0);
    DiscriminativePCNetwork net(batchSize);
    net.AddLayer(512, 512, 0.05f, 0.3f, 0.0f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.Compile(); // Crucial: memory is now bound
    
    std::vector<float> dummyX(batchSize * 512, 0.5f);
    std::vector<float> dummyY(batchSize * 512, 0.1f);

    for (auto _ : state) {
        net.TrainStep(dummyX, dummyY, 10);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ImpactOfBatching)
    ->Arg(1)->Arg(16)->Arg(64)->Arg(256)->Arg(512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();