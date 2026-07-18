#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <vector>

#include "Activations.h"
#include "DiscriminativePCNetwork.h"

using namespace Deep;

namespace
{

//------------------------------------------------------------
// Common Test Data
//------------------------------------------------------------

std::mt19937 g_rng(42);

constexpr int kInferenceSteps = 50;

const std::vector<float> kInput = {
    -1.0f,
     1.0f
};

const std::vector<float> kTarget = {
     1.0f
};

//------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------

DiscriminativePCNetwork CreateNetwork(
    int inputSize,
    int hiddenSize,
    int outputSize)
{
    DiscriminativePCNetwork net(1);

    net.AddLayer(
        inputSize,
        hiddenSize,
        0.05f,
        0.30f,
        0.0001f,
        Deep::tanh,
        Deep::dTanh);

    net.AddLayer(
        hiddenSize,
        outputSize,
        0.05f,
        0.30f,
        0.0001f,
        Deep::tanh,
        Deep::dTanh);

    net.AddLayer(
        outputSize,
        0,
        0.05f,
        0.30f,
        0.0001f,
        Deep::linear,
        Deep::dLinear);

    net.RandomizeWeights(g_rng);

    return net;
}

DiscriminativePCLayer CreateLayer(
    int size,
    int nextSize)
{
    DiscriminativePCLayer layer(
        size,
        nextSize,
        1,
        0.05f,
        0.30f,
        0.0001f,
        Deep::tanh,
        Deep::dTanh);

    layer.RandomizeWeights(g_rng);

    return layer;
}

//------------------------------------------------------------
// Activation Benchmarks
//------------------------------------------------------------

static void BM_Tanh(benchmark::State& state)
{
    std::vector<float> values(state.range(0), 0.5f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::tanh(values.data(), values.size());

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_Tanh)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_dTanh(benchmark::State& state)
{
    std::vector<float> values(state.range(0), 0.5f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::dTanh(values.data(), values.size(), false);

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_dTanh)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_ReLU(benchmark::State& state)
{
    std::vector<float> values(state.range(0), -0.5f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::relu(values.data(), values.size());

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_ReLU)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_dReLU(benchmark::State& state)
{
    std::vector<float> values(state.range(0), -0.5f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::dRelu(values.data(), values.size(), false);

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_dReLU)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_Sigmoid(benchmark::State& state)
{
    std::vector<float> values(state.range(0), 0.25f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::sigmoid(values.data(), values.size());

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_Sigmoid)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_dSigmoid(benchmark::State& state)
{
    std::vector<float> values(state.range(0), 0.25f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::dSigmoid(values.data(), values.size());

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_dSigmoid)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_Linear(benchmark::State& state)
{
    std::vector<float> values(state.range(0), 1.0f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::linear(values.data(), values.size());

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_Linear)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

static void BM_dLinear(benchmark::State& state)
{
    std::vector<float> values(state.range(0), 1.0f);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(values.data());

        Deep::dLinear(values.data(), values.size());

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * values.size());
}

BENCHMARK(BM_dLinear)
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

//------------------------------------------------------------
// Layer Benchmarks
//------------------------------------------------------------

static void BM_Layer_ResetState(benchmark::State& state)
{
    auto layer = CreateLayer(
        state.range(0),
        state.range(0));

    for (auto _ : state)
    {
        layer.ResetState();

        benchmark::DoNotOptimize(layer.GetBeliefs());
    }
}

BENCHMARK(BM_Layer_ResetState)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

static void BM_Layer_CalculateState(benchmark::State& state)
{
    auto layer = CreateLayer(
        state.range(0),
        state.range(0));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            layer.CalculateState());
    }
}

BENCHMARK(BM_Layer_CalculateState)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

static void BM_Layer_UpdateState(benchmark::State& state)
{
    auto layer = CreateLayer(
        state.range(0),
        state.range(0));

    layer.CalculateState();

    for (auto _ : state)
    {
        layer.UpdateState();

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Layer_UpdateState)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

static void BM_Layer_UpdateWeights(benchmark::State& state)
{
    auto layer = CreateLayer(
        state.range(0),
        state.range(0));

    layer.CalculateState();

    for (auto _ : state)
    {
        layer.UpdateWeights();

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Layer_UpdateWeights)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

//------------------------------------------------------------
// Network Benchmarks
//------------------------------------------------------------

static void BM_Network_ResetState(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    for (auto _ : state)
    {
        net.ResetState();

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Network_ResetState)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

static void BM_Network_Clamp(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    for (auto _ : state)
    {
        net.Clamp(kInput);

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Network_Clamp)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

static void BM_Network_CalculateState(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    net.Clamp(kInput);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(
            net.CalculateState());
    }
}

BENCHMARK(BM_Network_CalculateState)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

static void BM_Network_UpdateState(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    net.Clamp(kInput);
    net.CalculateState();

    for (auto _ : state)
    {
        net.UpdateState();

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Network_UpdateState)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

static void BM_Network_UpdateWeights(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    net.Clamp(kInput);
    net.GetTerminalLayer()->ClampState(kTarget);

    for (auto _ : state)
    {
        net.UpdateWeights();

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_Network_UpdateWeights)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

//------------------------------------------------------------
// End-to-End Inference Benchmarks
//------------------------------------------------------------

static void BM_Network_Inference(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    for (auto _ : state)
    {
        net.ResetState();
        net.Clamp(kInput);

        for (int i = 0; i < kInferenceSteps; ++i)
        {
            benchmark::DoNotOptimize(net.CalculateState());
            net.UpdateState();
        }

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * kInferenceSteps);
}

BENCHMARK(BM_Network_Inference)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

//------------------------------------------------------------
// Training Benchmarks
//------------------------------------------------------------

static void BM_Network_TrainSample(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    for (auto _ : state)
    {
        net.ResetState();

        net.Clamp(kInput);
        net.GetTerminalLayer()->ClampState(kTarget);

        for (int i = 0; i < kInferenceSteps; ++i)
        {
            net.CalculateState();
            net.UpdateState();
        }

        net.UpdateWeights();
        net.GetTerminalLayer()->UnclampState();

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_Network_TrainSample)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

static void BM_Network_TrainEpoch(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    static const std::array<std::vector<float>, 4> inputs =
    {{
        {-1.f,-1.f},
        {-1.f, 1.f},
        { 1.f,-1.f},
        { 1.f, 1.f}
    }};

    static const std::array<std::vector<float>, 4> targets =
    {{
        {-1.f},
        { 1.f},
        { 1.f},
        {-1.f}
    }};

    for (auto _ : state)
    {
        for (size_t sample = 0; sample < inputs.size(); ++sample)
        {
            net.ResetState();

            net.Clamp(inputs[sample]);
            net.GetTerminalLayer()->ClampState(targets[sample]);

            for (int i = 0; i < kInferenceSteps; ++i)
            {
                net.CalculateState();
                net.UpdateState();
            }

            net.UpdateWeights();
            net.GetTerminalLayer()->UnclampState();
        }

        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(
        state.iterations() * inputs.size());
}

BENCHMARK(BM_Network_TrainEpoch)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

//------------------------------------------------------------
// Scaling Benchmarks
//------------------------------------------------------------

static void BM_Scaling_Inference(benchmark::State& state)
{
    const int hidden = static_cast<int>(state.range(0));

    auto net = CreateNetwork(
        hidden,
        hidden,
        hidden);

    std::vector<float> input(hidden, 0.5f);

    for (auto _ : state)
    {
        net.ResetState();
        net.Clamp(input);

        for (int i = 0; i < 20; ++i)
        {
            net.CalculateState();
            net.UpdateState();
        }

        benchmark::ClobberMemory();
    }

    state.SetComplexityN(hidden);
}

BENCHMARK(BM_Scaling_Inference)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Complexity();

static void BM_Scaling_Training(benchmark::State& state)
{
    const int hidden = static_cast<int>(state.range(0));

    auto net = CreateNetwork(
        hidden,
        hidden,
        hidden);

    std::vector<float> input(hidden, 0.25f);
    std::vector<float> target(hidden, 0.75f);

    for (auto _ : state)
    {
        net.ResetState();

        net.Clamp(input);
        net.GetTerminalLayer()->ClampState(target);

        for (int i = 0; i < 20; ++i)
        {
            net.CalculateState();
            net.UpdateState();
        }

        net.UpdateWeights();
        net.GetTerminalLayer()->UnclampState();

        benchmark::ClobberMemory();
    }

    state.SetComplexityN(hidden);
}

BENCHMARK(BM_Scaling_Training)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Complexity();

//------------------------------------------------------------
// Weight Initialization
//------------------------------------------------------------

static void BM_RandomizeWeights(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    for (auto _ : state)
    {
        net.RandomizeWeights(g_rng);

        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_RandomizeWeights)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

//------------------------------------------------------------
// Full Prediction Pipeline
//------------------------------------------------------------

static void BM_Predict(benchmark::State& state)
{
    auto net = CreateNetwork(
        2,
        state.range(0),
        1);

    for (auto _ : state)
    {
        net.ResetState();
        net.Clamp(kInput);

        for (int i = 0; i < kInferenceSteps; ++i)
        {
            net.CalculateState();
            net.UpdateState();
        }

        benchmark::DoNotOptimize(
            net.GetTerminalLayer()->GetBeliefs()[0]);
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_Predict)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

//------------------------------------------------------------

} // namespace

BENCHMARK_MAIN();