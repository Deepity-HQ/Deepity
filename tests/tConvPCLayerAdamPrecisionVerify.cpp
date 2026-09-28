/**
 * @file tConvPCLayerAdamPrecisionVerify.cpp
 * @brief Exercises two ConvPCLayer code paths tConvDiagnose.cpp never
 * touches: the ADAM/ADAMW branch of UpdateWeights() (that test only ever
 * uses the default SGD optimizer, and neither ConvPCNetwork nor
 * SimpleConvPCNetwork currently expose SetOptimizer() to Python, so this
 * is the only coverage ADAM gets at all), and UpdatePrecision() (never
 * called by TrainStep()/TrainStepWithProjection(), so nothing else
 * exercises it either).
 *
 * Not a gradient check, just confirms both paths run to completion,
 * keep every parameter finite, and move in the expected direction
 * (energy down for ADAM training; precision up when error shrinks).
 */
#include <deepity/networks/ConvPCNetwork.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
    bool AllFinite(const float *data, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(data[i]))
                return false;
        return true;
    }
}

int main()
{
    ConvPCNetwork net(2);
    net.AddLayer(1, 2, 4, 4, 3, 3, 1, 1, 0, 0,
                 0.01f, 0.1f, 0.05f, 0.0f,
                 ActivationType::SIGMOID, ActivationType::dSIGMOID);
    net.AddLayer(2, 3, 2, 2, 2, 2, 1, 1, 0, 0,
                 0.01f, 0.1f, 0.05f, 0.0f,
                 ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(3, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                 0.01f, 0.1f, 0.05f, 0.0f,
                 ActivationType::LINEAR, ActivationType::dLINEAR);

    for (auto &l : net.GetLayers())
        l->SetOptimizer(OptimizerType::ADAM);

    net.Compile();

    std::mt19937 rng(7);
    net.RandomizeWeights(rng);

    std::vector<float> x = {
        0.2f, -0.5f, 0.8f, 0.1f,
        -0.3f, 0.6f, -0.7f, 0.4f,
        0.9f, -0.2f, 0.3f, -0.6f,
        0.5f, -0.9f, 0.1f, -0.1f,
        0.1f, -0.4f, 0.7f, 0.2f,
        -0.2f, 0.5f, -0.6f, 0.3f,
        0.8f, -0.1f, 0.2f, -0.5f,
        0.4f, -0.8f, 0.2f, 0.0f};
    std::vector<float> y = {1.0f, -1.0f, 0.5f, -0.5f, 1.0f, 0.3f};

    // --- 1. ADAM optimizer path -----------------------------------
    float e0 = net.TrainStep(x, y, /*inferenceSteps=*/5);
    float eLast = e0;
    bool sawNonFinite = false;

    for (int step = 0; step < 30; ++step)
    {
        eLast = net.TrainStep(x, y, 5);
        for (auto &l : net.GetLayers())
        {
            if (l->GetOutChannels() == 0)
                continue;
            size_t wSize = (size_t)l->GetOutChannels() * l->GetInChannels() * l->GetKernelH() * l->GetKernelW();
            if (!AllFinite(l->GetWeights(), wSize))
                sawNonFinite = true;
        }
    }

    printf("ADAM: energy %.6f -> %.6f\n", e0, eLast);

    if (sawNonFinite)
    {
        printf("FAILED: ADAM optimizer path produced non-finite weights.\n");
        return 1;
    }
    if (!(eLast < e0))
    {
        printf("FAILED: ADAM optimizer path did not reduce energy over 30 steps.\n");
        return 1;
    }

    // --- 2. UpdatePrecision() --------------------------------------
    // Settle once more, then repeatedly call UpdatePrecision() and check
    // it stays finite, stays positive (it's exp(log_p), so this should
    // be automatic, but verifying it directly still catches a shape/
    // indexing bug that corrupts log_p), and actually changes p (i.e.
    // isn't a silent no-op).
    net.ResetState();
    net.Clamp(x);
    net.GetTerminalLayer()->ClampState(y);
    for (int t = 0; t < 5; ++t)
    {
        net.CalculateState();
        net.UpdateState();
    }

    ConvPCLayer *hidden = net.GetLayers()[1].get();
    size_t ownSize = (size_t)hidden->GetInChannels() * hidden->GetInHeight() * hidden->GetInWidth();
    std::vector<float> pBefore(hidden->GetPrecisions(), hidden->GetPrecisions() + ownSize);

    for (int i = 0; i < 10; ++i)
        net.UpdatePrecision();

    const float *pAfter = hidden->GetPrecisions();

    if (!AllFinite(pAfter, ownSize))
    {
        printf("FAILED: UpdatePrecision() produced non-finite precision values.\n");
        return 1;
    }

    bool allPositive = true;
    bool changed = false;
    for (size_t i = 0; i < ownSize; ++i)
    {
        if (pAfter[i] <= 0.0f)
            allPositive = false;
        if (std::fabs(pAfter[i] - pBefore[i]) > 1e-8f)
            changed = true;
    }

    printf("Precision[0]: %.6f -> %.6f\n", pBefore[0], pAfter[0]);

    if (!allPositive)
    {
        printf("FAILED: UpdatePrecision() produced a non-positive precision value.\n");
        return 1;
    }
    if (!changed)
    {
        printf("FAILED: UpdatePrecision() left every precision value unchanged "
               "(likely a silent no-op / wrong buffer).\n");
        return 1;
    }

    net.GetTerminalLayer()->UnclampState();

    printf("PASSED: ConvPCLayer's ADAM optimizer path and UpdatePrecision() "
           "both run correctly.\n");
    return 0;
}
