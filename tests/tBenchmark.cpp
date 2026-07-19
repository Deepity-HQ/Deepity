#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "DiscriminativePCNetwork.h"
#include "Activations.h"

using namespace Deep;

namespace {

    DiscriminativePCNetwork CreateBatchedNetwork(int batchSize, int inputSize, int hiddenSize, int outputSize) {
        DiscriminativePCNetwork net(batchSize);

        net.AddLayer(inputSize, hiddenSize, 0.05f, 0.30f, 0.000f, 0.0001f, Deep::tanh, Deep::dTanh);
        net.AddLayer(hiddenSize, outputSize, 0.05f, 0.30f, 0.000f, 0.0001f, Deep::tanh, Deep::dTanh);
        net.AddLayer(outputSize, 0, 0.05f, 0.30f, 0.000f, 0.0001f, Deep::linear, Deep::dLinear);

        std::mt19937 rng(42); 
        net.RandomizeWeights(rng);

        return net;
    }

    static void BM_Find_OMP_Threshold(benchmark::State& state) {
        const int batchSize = static_cast<int>(state.range(0));
        
        // Fix the hidden size to a realistic, moderately large dimension
        const int hiddenSize = 256; 

        auto net = CreateBatchedNetwork(batchSize, hiddenSize, hiddenSize, hiddenSize);

        std::vector<float> input(batchSize * hiddenSize, 0.5f);
        std::vector<float> target(batchSize * hiddenSize, 0.75f);

        for (auto _ : state) {
            // Drop inference steps to 10 so the massive batch sizes don't take forever to benchmark
            net.TrainStep(input, target, 10); 
            benchmark::ClobberMemory();
        }

        // Track throughput (items per second) to easily spot the scaling crossover
        state.SetItemsProcessed(state.iterations() * batchSize);
    }
    
    // Scale the batch size by multiples of 4: 16 -> 64 -> 256 -> 1024 -> 4096 -> 16384
    BENCHMARK(BM_Find_OMP_Threshold)
        ->RangeMultiplier(4)
        ->Range(16, 16384)
        ->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();