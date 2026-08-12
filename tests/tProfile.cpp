// tProfile.cpp
//
// Small, synthetic-data, MNIST-shaped profiling harness. No Python, no real
// MNIST loading -- deliberately fast to iterate on (seconds, not the 50
// minutes we're trying to explain), so the OMP/OpenBLAS thread-count sweep
// and the log(p)->log_p A/B test are actually practical to run repeatedly.
//
// Build with -DPCN_PROFILE (see CMakeLists.txt changes needed below) --
// WITHOUT that flag, this measures nothing (PCN_TIME is a no-op) and the
// numbers would be meaningless.

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "DiscriminativePCNetwork.h"
#include "Profile.h"

using namespace Deep;

int main()
{
#ifndef PCN_PROFILE
    std::cout << "WARNING: built without -DPCN_PROFILE -- all timers are "
                 "no-ops. This run measures nothing.\n";
#endif

    const int BATCH_SIZE = 256;      // real MNIST batch size, not a toy value
    const int INFERENCE_STEPS = 150; // matches your real hyperparameters
    const int N_BATCHES = 200;       // representative, but seconds not hours

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    DiscriminativePCNetwork net(BATCH_SIZE);
    net.AddLayer(784, 512, 0.002f, 0.05f, 0.0f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(512, 10, 0.002f, 0.05f, 0.0f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(10, 0, 0.002f, 0.05f, 0.0f, 0.0001f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();
    net.RandomizeWeights(rng);

    std::vector<float> X((size_t)BATCH_SIZE * 784);
    std::vector<float> Y((size_t)BATCH_SIZE * 10);
    for (auto &v : X) v = dist(rng);
    for (auto &v : Y) v = dist(rng);

    std::cout << "Profiling: " << N_BATCHES << " batches, batch_size=" << BATCH_SIZE
               << ", inference_steps=" << INFERENCE_STEPS << "\n";

    // --- Wall-clock timer wraps the ENTIRE run, measured independently of
    // the phase accumulators, per review point 2. ---
    auto wallStart = std::chrono::steady_clock::now();

    for (int b = 0; b < N_BATCHES; ++b)
    {
        net.TrainStep(X, Y, INFERENCE_STEPS);
    }

    auto wallEnd = std::chrono::steady_clock::now();
    double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();

    // --- Report ---
    PrintAllProfiles(wallSeconds);

    std::cout << "\nRun with:\n";
    std::cout << "  OMP_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<m> ./tProfile\n";
    std::cout << "and re-run to sweep the thread-count grid from the review.\n";

    return 0;
}