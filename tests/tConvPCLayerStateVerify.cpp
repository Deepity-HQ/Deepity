/**
 * @file tConvPCLayerStateVerify.cpp
 * @brief Finite-difference gradient check for ConvPCLayer's dE/dz, the
 * check ConvPCLayer.h's own file-level warning says has never been done.
 *
 * The previously-existing gradient check (per that warning) verified only
 * dE/dW with BOTH layers clamped, which never actually calls the
 * Col2Im-based feedback term in UpdateState() at all: a clamped layer's
 * UpdateState() returns before touching dz_dt. This test instead checks
 * dE/dz for a genuinely UNCLAMPED middle layer, which is the only way to
 * exercise that code path, Col2Im(W^T @ (e_above * p_above * f'(mu))),
 * the exact feedback term the warning flags as unverified.
 *
 * SIGN CONVENTION, matching tDirectKPVerify.cpp's convention exactly:
 * UpdateState()'s dz_dt is the layer's own term (-p*e) plus the feedback
 * term, and z += ir * dz_dt applies it. Both terms are, by construction,
 * the NEGATIVE gradient of total network energy w.r.t. this layer's z
 * (own term: d/dz of 0.5*p*e^2 is p*e, so -p*e is -dE/dz; feedback term:
 * Col2Im is Im2Col's adjoint, so Col2Im(W^T @ (p_above*e_above*f'(mu)))
 * is exactly -d(E_above)/dz via the chain rule through mu). So with
 * ir=1, dz_dt = z_after - z_before = -dE/dz, i.e. dE/dz = -(z_after -
 * z_before), the same "-(after - before) with the rate stripped to 1"
 * trick tDirectKPVerify.cpp uses for dE/dW.
 */
#include <deepity/networks/ConvPCNetwork.h>
#include <cstdio>
#include <cmath>
#include <random>

using namespace Deep;

namespace
{
    // Settles to a non-trivial (z, e, mu) state before testing the
    // gradient AT that state, matching tDirectKPVerify.cpp's approach.
    void SettleNetwork(ConvPCNetwork &net, const std::vector<float> &x,
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

    // Total energy across every layer at the network's CURRENT z, without
    // calling UpdateState(), CalculateState() alone recomputes each
    // layer's e/mu from whatever z currently holds, so perturbing z and
    // re-calling this is safe and doesn't drift anything else.
    float TotalEnergy(ConvPCNetwork &net)
    {
        return net.CalculateState();
    }
}

int main()
{
    const size_t batchSize = 1;
    const float eps = 3e-3f;
    const float tolerance = 1e-2f; // matches tDirectKPVerify.cpp's tolerance

    ConvPCNetwork net(batchSize);

    // Layer 0 (input, clamped): 1x4x4 -> conv 3x3 stride1 -> 2 channels, 2x2.
    net.AddLayer(1, 2, 4, 4, 3, 3, 1, 1, 0, 0,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 ActivationType::SIGMOID, ActivationType::dSIGMOID);
    // Layer 1 (middle, UNCLAMPED, the layer under test): 2x2x2 -> conv
    // 2x2 stride1 -> 3 channels, 1x1. ir=1.0 strips the update-rate
    // scaling, same trick tDirectKPVerify.cpp uses via lr=1.0.
    net.AddLayer(2, 3, 2, 2, 2, 2, 1, 1, 0, 0,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 ActivationType::TANH, ActivationType::dTANH);
    // Layer 2 (terminal, clamped): 3x1x1 -> flattened size 3.
    net.AddLayer(3, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                 0.0f, 1.0f, 0.0f, 0.0f,
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

    ConvPCLayer *testLayer = net.GetLayers()[1].get();
    const size_t testIdx = 0; // check z[0] of the unclamped middle layer

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

    // --- Analytical gradient (via the real UpdateState() code) ----
    SettleNetwork(net, x, y, /*steps=*/5);
    net.CalculateState(); // refresh e/mu for every layer immediately
                          // before testLayer->UpdateState() reads them,
                          // matching tDirectKPVerify.cpp's placement note.

    float zBefore = z[testIdx];
    testLayer->UpdateState(); // isolates the analytical dz_dt to just
                              // this one layer, like tDirectKPVerify.cpp
                              // calling testLayer->UpdateWeights() alone.
    float zAfter = z[testIdx];

    // ir=1 above strips scaling, so this recovers dE/dz directly,
    // see the sign-convention note in the file header.
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

    printf("PASSED: ConvPCLayer's Col2Im feedback term in UpdateState() "
           "matches the true dE/dz for a genuinely unclamped middle "
           "layer.\n");
    return 0;
}
