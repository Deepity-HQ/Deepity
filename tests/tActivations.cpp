/**
 * @file tActivations.cpp
 * @brief Compares Deep::'s SIMD/SLEEF-backed activation functions
 * against naive std-library loops, across array sizes matching real
 * layer widths used elsewhere in this project (256 as a small hidden
 * layer, 512/784 matching the MNIST architecture, 16384/131072 as
 * larger, batch-scale sizes).
 *
 * Data is reset from a pristine copy before each timed call (a cheap
 * memcpy, excluded from the timed region via PauseTiming/ResumeTiming)
 * rather than reusing the same buffer repeatedly -- these functions
 * mutate their input in place, and tanh/sigmoid are contractions toward
 * a fixed point, so repeated application without a reset would
 * gradually change the value distribution across iterations and skew
 * later timings.
 *
 * Run with --benchmark_format=json to produce output matching this
 * project's existing benchmark-result convention (see logs/results.json).
 */
#include <benchmark/benchmark.h>
#include <deepity/Activations.h>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

namespace
{
    std::vector<float> MakeRandomInput(size_t n, uint32_t seed = 42)
    {
        std::vector<float> data(n);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        for (auto &v : data)
            v = dist(rng);
        return data;
    }
}

// ─── tanh ────────────────────────────────────────────────────────────

static void BM_Deep_Tanh(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        Deep::tanh(working.data(), working.size());
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Deep_Tanh)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

static void BM_Std_Tanh(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        for (auto &v : working)
            v = std::tanh(v);
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Std_Tanh)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

// ─── sigmoid ─────────────────────────────────────────────────────────

static void BM_Deep_Sigmoid(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        Deep::sigmoid(working.data(), working.size());
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Deep_Sigmoid)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

static void BM_Std_Sigmoid(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        for (auto &v : working)
            v = 1.0f / (1.0f + std::exp(-v));
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Std_Sigmoid)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

// ─── relu ────────────────────────────────────────────────────────────

static void BM_Deep_Relu(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        Deep::relu(working.data(), working.size());
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Deep_Relu)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

static void BM_Std_Relu(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        for (auto &v : working)
            v = std::max(0.0f, v);
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Std_Relu)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

BENCHMARK_MAIN();