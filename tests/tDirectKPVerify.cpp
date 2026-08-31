/**
 * @file tDirectKPVerify.cpp
 * @brief Finite-difference gradient check for DirectKPPCLayer's core PC
 * math (dE/dW), matching this project's existing tGaussSeidelVerify.cpp
 * convention.
 *
 * Deliberately checks ONLY dE/dW here, not Psi. Psi is not a gradient of
 * this (or any) energy function -- it's a feedback-alignment matrix that
 * is only supposed to gradually correlate with W's transpose chain over
 * many training steps (see DirectKPPCNetwork.h's class docs and the
 * paper's Appendix A.1). A finite-difference check against it wouldn't
 * be testing the right thing; that needs a separate, longer-horizon
 * alignment test instead (see the discussion this test came out of).
 *
 * SIGN CONVENTION, worked out explicitly since it's easy to get backwards:
 * For layer L (connecting to layerAbove via W_L), the true gradient is
 *   dE/dW_L = -(e_{L+1} @ zF_L^T)
 * (the minus sign falls out of e_{L+1} = z_{L+1} - (W_L @ zF_L + b_L)).
 * The layer's real UpdateWeights() applies
 *   W_L += (lr/batchSize) * e_{L+1}^T @ zF_L
 * which is W -= lr * dE/dW -- ordinary gradient descent, just written
 * with the sign already folded into the formula rather than an explicit
 * subtraction. This test uses lr=1, batchSize=1 to strip away scaling,
 * so the "implied gradient" is simply -(W_after - W_before), and that
 * should match the finite-difference gradient directly (not its
 * negative) if UpdateWeights() is correct.
 */
#include <deepity/networks/DirectKPPCNetwork.h>
#include <cstdio>
#include <cmath>
#include <random>

using namespace Deep;

namespace
{
    // Runs enough settling steps to reach a non-trivial, realistic
    // (z, e) state before testing the gradient AT that state -- matching
    // how UpdateWeights() is actually used in practice (after settling
    // completes), not testing an artificial, freshly-initialized state.
    void SettleNetwork(DirectKPPCNetwork &net, const std::vector<float> &x,
                       const std::vector<float> &y, int steps)
    {
        net.ResetState();
        net.Clamp(x);
        net.ProjectForward();
        net.GetTerminalLayer()->ClampState(y);

        for (int t = 0; t < steps; ++t)
            net.Step();
    }

    // Total energy across every layer at the network's CURRENT state,
    // without touching z or clamps -- just CalculateState()'s error/
    // energy computation, summed. Does not call UpdateState(), so
    // repeated calls with a perturbed W are safe and don't drift z.
    float TotalEnergy(DirectKPPCNetwork &net)
    {
        float e = 0.0f;
        for (auto &layer : net.GetLayers())
            e += layer->CalculateState();
        return e;
    }
}

int main()
{
    const int batchSize = 1;
    const size_t terminalSize = 2;
    const float eps = 3e-3;
    const float tolerance = 1e-2f; // finite-difference error is O(eps^2);
                                   // loose enough to tolerate float32
                                   // precision, tight enough to catch a
                                   // real sign/shape bug

    DirectKPPCNetwork net(batchSize);
    net.AddLayer(3, 3, terminalSize, 0.0f, 0.08f, 0.0f, 0.0f,
                 ActivationType::SIGMOID, ActivationType::dSIGMOID);
    net.AddLayer(3, (size_t)terminalSize, terminalSize, 1.0f, 0.08f, 0.0f, 0.0f,
                 ActivationType::LINEAR, ActivationType::dLINEAR);
    net.AddLayer(terminalSize, 0, terminalSize, 0.0f, 0.08f, 0.0f, 0.0f,
                 ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    net.SetOptimizer(OptimizerType::SGD);
    net.SetLearningRate(1.0f); // strips scaling -- see file header

    std::vector<float> x = {0.2f, -0.5f, 0.8f, 0.1f};
    std::vector<float> y = {1.0f, -1.0f};

    // Layer index 2 (0-indexed): the 3->terminalSize layer, whose W we're
    // checking. Its layerAbove is the terminal layer directly, keeping
    // the gradient formula simple while Psi (from layer 1) still exists
    // and is exercised elsewhere in the network.
    DirectKPPCLayer *testLayer = net.GetLayers()[1].get();
    float *W = const_cast<float *>(testLayer->GetWeights());
    const size_t testIdx = 0; // check W[0] specifically

    SettleNetwork(net, x, y, /*steps=*/5);

    float wOriginal = W[testIdx];

    // Perturb W only -- z stays exactly where settling left it.
    W[testIdx] = wOriginal + eps;
    float energyPlus = TotalEnergy(net);

    W[testIdx] = wOriginal - eps;
    float energyMinus = TotalEnergy(net);

    W[testIdx] = wOriginal;

    printf("Baseline energy: %.6f, energyPlus: %.6f, energyMinus: %.6f\n",
           TotalEnergy(net), energyPlus, energyMinus);

    float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

    // --- Analytical gradient (via the real UpdateWeights() code) ----
    SettleNetwork(net, x, y, /*steps=*/5);
    TotalEnergy(net); // <--- IT NEEDS TO GO EXACTLY HERE

    float wBefore = W[testIdx];
    testLayer->UpdateWeights();
    float wAfter = W[testIdx];

    // lr=1, batchSize=1 above strips scaling, so this recovers the raw
    // gradient directly -- see the sign-convention note in the header.
    float impliedGrad = -(wAfter - wBefore);

    printf("Numerical dE/dW[%zu]:  % .6f\n", testIdx, numericalGrad);
    printf("Analytical dE/dW[%zu]: % .6f\n", testIdx, impliedGrad);

    float diff = std::fabs(numericalGrad - impliedGrad);
    float relDenom = std::fabs(numericalGrad) + 1e-6f;
    float relError = diff / relDenom;

    printf("Absolute difference: %.6f, relative error: %.6f\n", diff, relError);

    if (relError > tolerance)
    {
        printf("FAILED: gradient mismatch exceeds tolerance (%.4f)\n", tolerance);
        return 1;
    }

    printf("PASSED\n");
    return 0;
}