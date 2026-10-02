/**
 * @file tSimpleConvPCLayerStateVerify.cpp
 * @brief Finite-difference gradient check for SimpleConvPCLayer's dE/dz,
 * specifically exercising the Col2Im(W^T @ (e_above * f'(mu))) feedback
 * term in UpdateState() for a genuinely UNCLAMPED middle layer -- mirrors
 * tConvPCLayerStateVerify.cpp's approach exactly (same settle/perturb/
 * compare structure), adapted for SimpleConvPCLayer's simpler one-hop
 * (no precision weighting) update rule.
 *
 * SIGN CONVENTION, matching tConvPCLayerStateVerify.cpp exactly: with
 * ir=1, dz_dt = z_after - z_before = -dE/dz, i.e. dE/dz = -(z_after -
 * z_before).
 */
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <cstdio>
#include <cmath>
#include <random>

using namespace Deep;

namespace
{
    void SettleNetwork(SimpleConvPCNetwork &net, const std::vector<float> &x,
                       const std::vector<float> &y, int steps)
    {
        net.ResetState();
        net.Clamp(x);
        net.GetTerminalLayer()->ClampState(y);

        for (int t = 0; t < steps; ++t)
        {
            net.CalculateState();
            net.UpdateState();
        }
    }

    float TotalEnergy(SimpleConvPCNetwork &net)
    {
        return net.CalculateState();
    }
}

int main()
{
    const size_t batchSize = 1;
    const float eps = 3e-3f;
    const float tolerance = 1e-2f;

    SimpleConvPCNetwork net(batchSize);

    // Layer 0 (input, clamped): 1x4x4 -> conv 3x3 stride1 -> 2 channels, 2x2.
    net.AddLayer(1, 2, 4, 4, 3, 3, 1, 1, 0, 0,
                 0.0f, 1.0f, 0.0f,
                 ActivationType::SIGMOID, ActivationType::dSIGMOID);
    // Layer 1 (middle, UNCLAMPED, the layer under test): 2x2x2 -> conv
    // 2x2 stride1 -> 3 channels, 1x1. ir=1.0 strips the update-rate
    // scaling.
    net.AddLayer(2, 3, 2, 2, 2, 2, 1, 1, 0, 0,
                 0.0f, 1.0f, 0.0f,
                 ActivationType::TANH, ActivationType::dTANH);
    // Layer 2 (terminal, clamped): 3x1x1 -> flattened size 3.
    net.AddLayer(3, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                 0.0f, 1.0f, 0.0f,
                 ActivationType::LINEAR, ActivationType::dLINEAR);

    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::vector<float> x = {
        0.2f, -0.5f, 0.8f, 0.1f,
        -0.3f, 0.6f, -0.7f, 0.4f,
        0.9f, -0.2f, 0.3f, -0.6f,
        0.5f, -0.9f, 0.1f, -0.1f};
    std::vector<float> y = {1.0f, -1.0f, 0.5f};

    SimpleConvPCLayer *testLayer = net.GetLayers()[1].get();
    const size_t testIdx = 0;

    SettleNetwork(net, x, y, /*steps=*/5);

    float *z = testLayer->GetBeliefs();
    float zOriginal = z[testIdx];

    z[testIdx] = zOriginal + eps;
    float energyPlus = TotalEnergy(net);

    z[testIdx] = zOriginal - eps;
    float energyMinus = TotalEnergy(net);

    z[testIdx] = zOriginal;

    printf("Baseline energy: %.6f, energyPlus: %.6f, energyMinus: %.6f\n",
           TotalEnergy(net), energyPlus, energyMinus);

    float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

    SettleNetwork(net, x, y, /*steps=*/5);
    net.CalculateState();

    float zBefore = z[testIdx];
    testLayer->UpdateState();
    float zAfter = z[testIdx];

    float impliedGrad = -(zAfter - zBefore);

    printf("Numerical dE/dz[%zu]:  % .6f\n", testIdx, numericalGrad);
    printf("Analytical dE/dz[%zu]: % .6f\n", testIdx, impliedGrad);

    float diff = std::fabs(numericalGrad - impliedGrad);
    float relDenom = std::fabs(numericalGrad) + 1e-6f;
    float relError = diff / relDenom;

    printf("Absolute difference: %.6f, relative error: %.6f\n", diff, relError);

    if (relError > tolerance)
    {
        printf("FAILED: gradient mismatch exceeds tolerance (%.4f), the "
               "Col2Im-based feedback term in UpdateState() does not match "
               "the true energy gradient.\n", tolerance);
        return 1;
    }

    printf("PASSED: SimpleConvPCLayer's Col2Im feedback term in UpdateState() "
           "matches the true dE/dz for a genuinely unclamped middle "
           "layer.\n");
    return 0;
}
