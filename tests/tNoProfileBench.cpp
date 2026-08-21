// Identical settling-loop logic to tProfile.cpp, but linked against
// PLAIN Deepity -- NOT DeepityProfiled, NOT PCN_PROFILE. GEMM (2.1ms)
// and tanh() (0.05ms) are both confirmed fast in isolation; if THIS is
// also fast (~2-5ms/step), that conclusively proves PCN_PROFILE/
// DeepityProfiled itself is the source of the ~106ms/step gap seen in
// DeepityProfile -- not the actual computation.
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <deepity/networks/DiscriminativePCNetwork.h>
using namespace Deep;

int main(void)
{
    const int BATCH_SIZE = 256;
    const int INFERENCE_STEPS = 150;
    const int N_BATCHES = 5;

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
    for (auto &v : X)
        v = dist(rng);
    for (auto &v : Y)
        v = dist(rng);

    std::cout << "Plain Deepity (NO PCN_PROFILE): " << N_BATCHES << " batches, "
              << INFERENCE_STEPS << " steps each\n\n";

    auto wallStart = std::chrono::steady_clock::now();

    for (int b = 0; b < N_BATCHES; ++b)
    {
        net.TrainStep(X, Y, INFERENCE_STEPS);
        std::cout << "  batch " << (b + 1) << "/" << N_BATCHES << " done\n";
    }

    auto wallEnd = std::chrono::steady_clock::now();
    double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();

    std::cout << "\n=== Result ===\n";
    std::cout << "Wall clock total: " << wallSeconds << " s\n";
    std::cout << "Per-step average: " << (wallSeconds / (N_BATCHES * INFERENCE_STEPS)) * 1000.0 << " ms\n\n";

    std::cout << "GEMM alone: ~2.1ms/step. tanh() alone: ~0.05ms/step.\n";
    std::cout << "If this is ~2-10ms/step, PCN_PROFILE/DeepityProfiled itself\n";
    std::cout << "was the source of the ~106ms/step gap -- not the real computation.\n";

    return 0;
}