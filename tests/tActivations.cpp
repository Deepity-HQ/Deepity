#include <benchmark/benchmark.h>
#include <deepity/utils/Activations.h>
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

// ─── gelu ────────────────────────────────────────────────────────────

static void BM_Deep_Gelu(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        Deep::gelu(working.data(), working.size());
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Deep_Gelu)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

static void BM_Std_Gelu(benchmark::State &state)
{
    size_t n = state.range(0);
    std::vector<float> original = MakeRandomInput(n);
    std::vector<float> working(n);

    constexpr float MAGIC_GELU_1 = 0.7978845608f;
    constexpr float MAGIC_GELU_2 = 0.044715f;
    for (auto _ : state)
    {
        state.PauseTiming();
        std::memcpy(working.data(), original.data(), n * sizeof(float));
        state.ResumeTiming();

        for (auto &v : working)
        {
            float inner = MAGIC_GELU_1 * v + (MAGIC_GELU_2 * MAGIC_GELU_1) * v * v * v;
            v *= 0.5f * (1.0f + std::tanh(inner));
        }
        benchmark::DoNotOptimize(working.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Std_Gelu)->Arg(256)->Arg(512)->Arg(784)->Arg(16384)->Arg(131072);

BENCHMARK_MAIN();