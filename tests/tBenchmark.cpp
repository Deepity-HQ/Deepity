#include <benchmark/benchmark.h>
#include "PCNetwork.h"
#include "Activations.h"
#include <vector>
#include <random>

static void BM_Network_Inference(benchmark::State &state)
{

    Deep::PCNetwork net;
    net.AddLayer(784, 512, 1e-6f, Deep::tanh, Deep::dTanh);
    net.AddLayer(512, 256, 1e-6f, Deep::tanh, Deep::dTanh);
    net.AddLayer(256, 64,  1e-6f, Deep::tanh, Deep::dTanh);
    net.AddLayer(64,  10,  1e-6f, Deep::tanh, Deep::dTanh);
    net.AddLayer(10,  0,   1e-6f, Deep::tanh, Deep::dTanh);

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::vector<float> input(784);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float &x : input) x = dist(rng);

    net.Clamp(input);

    for (auto _ : state)
    {
        net.CalculateState();
        net.UpdateState();
        net.UpdateWeights();
    }

    state.SetItemsProcessed(state.iterations() * net.GetBatchSize() );
}

BENCHMARK(BM_Network_Inference)
    ->Arg(1)
    ->Arg(16)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();