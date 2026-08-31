/**
 * @file tGflopsBenchmark.cpp
 * @brief Measures real, sustained GFLOPS for a full predictive-coding
 * train step, replacing the project's older, now-removed
 * LayerCPUMetrics.png benchmark (a stale 784-512-256-64-10 test whose
 * exact FLOP-counting methodology was never documented).
 *
 * FLOP accounting, made explicit here rather than assumed: for a layer
 * transitioning size -> nextSize with a given batchSize, each settling
 * step runs two GEMMs of matched cost (the forward prediction GEMM in
 * ComputeMuOnly(), and the feedback GEMM in UpdateState()), each costing
 * 2 * batchSize * size * nextSize FLOPs (the standard convention: one
 * multiply and one add per multiply-accumulate). UpdateWeights() runs a
 * third, equally-sized GEMM once per train step (not once per settling
 * step) for the weight gradient.
 *
 * Total FLOPs per train step, summed over all non-terminal layers:
 *   sum_over_layers[ batchSize * size * nextSize * (4 * numSteps + 2) ]
 */
#include <benchmark/benchmark.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <random>
#include <vector>

static void BM_TrainStepGFLOPS(benchmark::State &state)
{
    const int batchSize = 256;
    const int numSteps = 20;
    // Matches the project's prior GFLOPS benchmark architecture, for a
    // directly comparable before/after number.
    const std::vector<int> layerSizes = {784, 512, 256, 64, 10};

    Deep::SimplePCNetwork net(batchSize);
    for (size_t i = 0; i + 1 < layerSizes.size(); ++i)
    {
        net.AddLayer(layerSizes[i], layerSizes[i + 1], 0.001f, 0.08f, 0.0f,
                     Deep::ActivationType::SIGMOID, Deep::ActivationType::dSIGMOID);
    }
    net.AddLayer(layerSizes.back(), 0, 0.001f, 0.08f, 0.0f,
                 Deep::ActivationType::LINEAR, Deep::ActivationType::dLINEAR);
    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::vector<float> X(batchSize * layerSizes.front());
    std::vector<float> Y(batchSize * layerSizes.back(), 0.001f);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : X)
        v = dist(rng);
    for (int b = 0; b < batchSize; ++b)
        Y[b * layerSizes.back() + (b % layerSizes.back())] = 0.999f;

    // Total FLOPs for one train step, summed over every non-terminal
    // layer transition, per the accounting in the file header.
    double flopsPerStep = 0.0;
    for (size_t i = 0; i + 1 < layerSizes.size(); ++i)
    {
        double size = layerSizes[i];
        double nextSize = layerSizes[i + 1];
        flopsPerStep += (double)batchSize * size * nextSize * (4.0 * numSteps + 2.0);
    }

    for (auto _ : state)
    {
        net.TrainStepWithProjection(X, Y, numSteps);
    }

    state.counters["FLOPS"] = benchmark::Counter(
        flopsPerStep, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}
BENCHMARK(BM_TrainStepGFLOPS)->Unit(benchmark::kMillisecond)->Iterations(157);

BENCHMARK_MAIN();