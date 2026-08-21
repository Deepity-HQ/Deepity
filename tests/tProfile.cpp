// tProfile.cpp -- SIMPLIFIED for this investigation.
//
// The original per-phase instrumentation (PCN_TIME() calls inside
// DiscriminativePCLayer.cpp) was never carried forward through today's
// restructuring -- the registry in Profile.h is empty, confirmed by the
// missing per-layer sections in the last run's output. Rather than
// re-instrument the whole library again, this measures the two coarse
// phases that actually matter for "where is the Windows slowdown coming
// from" directly at the call site: the settling loop (CalculateState +
// UpdateState, repeated INFERENCE_STEPS times) vs UpdateWeights(). No
// changes to any library source file needed -- just std::chrono around
// the existing public methods.
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <deepity/networks/DiscriminativePCNetwork.h>

using namespace Deep;

int main(void)
{
    std::cout << "OpenBLAS selected CPU kernel: " << openblas_get_corename() << "\n\n";
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
    for (auto &v : X) v = dist(rng);
    for (auto &v : Y) v = dist(rng);

    std::cout << "Profiling (coarse, call-site timing): " << N_BATCHES << " batches, batch_size="
              << BATCH_SIZE << ", inference_steps=" << INFERENCE_STEPS << "\n\n";

    double totalSettle = 0.0;
    double totalCalcState = 0.0;
    double totalUpdateState = 0.0;
    double totalUpdateWeights = 0.0;
    double totalResetClamp = 0.0;

    auto wallStart = std::chrono::steady_clock::now();

    for (int b = 0; b < N_BATCHES; ++b)
    {
        auto t0 = std::chrono::steady_clock::now();
        net.ResetState();
        net.Clamp(X);
        net.GetTerminalLayer()->ClampState(Y);
        auto t1 = std::chrono::steady_clock::now();
        totalResetClamp += std::chrono::duration<double>(t1 - t0).count();

        for (int s = 0; s < INFERENCE_STEPS; ++s)
        {
            auto cs0 = std::chrono::steady_clock::now();
            net.CalculateState();
            auto cs1 = std::chrono::steady_clock::now();
            totalCalcState += std::chrono::duration<double>(cs1 - cs0).count();

            net.UpdateState();
            auto us1 = std::chrono::steady_clock::now();
            totalUpdateState += std::chrono::duration<double>(us1 - cs1).count();
        }

        auto t2 = std::chrono::steady_clock::now();
        net.UpdateWeights();
        auto t3 = std::chrono::steady_clock::now();
        totalUpdateWeights += std::chrono::duration<double>(t3 - t2).count();

        net.GetTerminalLayer()->UnclampState();

        std::cout << "  batch " << (b + 1) << "/" << N_BATCHES << " done\n";
    }

    auto wallEnd = std::chrono::steady_clock::now();
    double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
    totalSettle = totalCalcState + totalUpdateState;

    std::cout << "\n=== Coarse phase breakdown (" << N_BATCHES << " batches, "
              << INFERENCE_STEPS << " steps each) ===\n";
    std::cout << "  Wall clock total:        " << wallSeconds << " s\n";
    std::cout << "  ResetState+Clamp:        " << totalResetClamp << " s  ("
              << (100.0 * totalResetClamp / wallSeconds) << "%)\n";
    std::cout << "  CalculateState (sum):    " << totalCalcState << " s  ("
              << (100.0 * totalCalcState / wallSeconds) << "%)\n";
    std::cout << "  UpdateState (sum):       " << totalUpdateState << " s  ("
              << (100.0 * totalUpdateState / wallSeconds) << "%)\n";
    std::cout << "  UpdateWeights (sum):     " << totalUpdateWeights << " s  ("
              << (100.0 * totalUpdateWeights / wallSeconds) << "%)\n";
    std::cout << "  --------\n";
    std::cout << "  Settle loop total:       " << totalSettle << " s  ("
              << (100.0 * totalSettle / wallSeconds) << "%)\n";
    std::cout << "  Per-step avg (Calc+Upd): " << (totalSettle / (N_BATCHES * INFERENCE_STEPS)) * 1000.0 << " ms\n";
    std::cout << "  Per-step CalculateState: " << (totalCalcState / (N_BATCHES * INFERENCE_STEPS)) * 1000.0 << " ms\n";
    std::cout << "  Per-step UpdateState:    " << (totalUpdateState / (N_BATCHES * INFERENCE_STEPS)) * 1000.0 << " ms\n";

    return 0;
}
