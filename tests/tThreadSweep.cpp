#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <cblas.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
using namespace Deep;

double RunConfig(int batchSize, int threads, int steps, int nBatches)
{
    omp_set_num_threads(threads);
    openblas_set_num_threads(threads);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    DiscriminativePCNetwork net(batchSize);
    net.AddLayer(784, 512, 0.002f, 0.05f, 0.0f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(512, 10, 0.002f, 0.05f, 0.0f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(10, 0, 0.002f, 0.05f, 0.0f, 0.0001f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();

    // Force thread count again AFTER Compile() -- DynamicThread() runs
    // inside AddLayer()'s layer constructors and would otherwise override
    // this with its own (currently hardcoded) choice.
    omp_set_num_threads(threads);
    openblas_set_num_threads(threads);

    net.RandomizeWeights(rng);

    std::vector<float> X((size_t)batchSize * 784);
    std::vector<float> Y((size_t)batchSize * 10);
    for (auto &v : X)
        v = dist(rng);
    for (auto &v : Y)
        v = dist(rng);

    auto start = std::chrono::steady_clock::now();
    for (int b = 0; b < nBatches; ++b)
        net.TrainStep(X, Y, steps);
    auto end = std::chrono::steady_clock::now();

    double totalSeconds = std::chrono::duration<double>(end - start).count();
    return (totalSeconds / (nBatches * steps)) * 1000.0; // ms/step
}

int main()
{
    const int STEPS = 60;    // matches real training's confirmed value, not the profiler's 150
    const int N_BATCHES = 3; // short -- just enough for a stable reading per cell

    std::vector<int> batchSizes = {256, 512, 1024};
    std::vector<int> threadCounts = {4, 8, 12, 16, 24, 32, 48};

    std::cout << "Sweeping batch_size x thread_count (" << STEPS << " steps, "
              << N_BATCHES << " batches per cell)...\n\n";

    std::cout << "batch\\threads";
    for (int t : threadCounts)
        std::cout << "\t" << t;
    std::cout << "\n";

    for (int bsz : batchSizes)
    {
        std::cout << bsz;
        for (int t : threadCounts)
        {
            double msPerStep = RunConfig(bsz, t, STEPS, N_BATCHES);
            std::cout << "\t" << msPerStep;
            std::cout.flush();
        }
        std::cout << "\n";
    }

    std::cout << "\nLook for the LOWEST ms/step value in the whole grid -- that's the\n";
    std::cout << "real Pareto-optimal (batch_size, thread_count) combination for this\n";
    std::cout << "machine, not necessarily matching today's batch=256/threads=8 guess.\n";

    return 0;
}