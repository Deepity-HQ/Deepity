/**
 * @file tGaussSeidelPCLayerVerify.cpp
 * @brief Finite-difference gradient check for GaussSeidelPCLayer's weight
 * update (dE/dW), matching tDirectKPVerify.cpp's exact methodology,
 * referenced by that file as "tGaussSeidelVerify.cpp" but never actually
 * written until now.
 *
 * Deliberately checks ONLY dE/dW here, not E. E is not a gradient of this
 * (or any) energy function, it's an independently-initialized,
 * never-updated feedback-alignment matrix (Lillicrap et al.) that
 * UpdateState() uses in place of W for the feedback term. A
 * finite-difference check against it wouldn't be testing the right thing,
 * exactly as DirectKPPCLayer's own Psi is deliberately excluded from its
 * gradient check.
 *
 * GaussSeidelPCLayer's own CalculateState() is NOT read-only (it just
 * calls UpdateState(), which mutates z), unlike every other layer's
 * CalculateState(), so it can't be used the way tDirectKPVerify.cpp uses
 * CalculateState() for its TotalEnergy() helper. Instead, TotalEnergy()
 * here directly re-runs sweeps 2 and 3 (ComputePrediction() then
 * ComputeError() on every layer), the read-only half of a Gauss-Seidel
 * step, so it can be called repeatedly with a perturbed W without
 * disturbing z.
 *
 * SIGN CONVENTION (identical derivation to tDirectKPVerify.cpp): for
 * layer L (connecting to layerAbove via W_L),
 *   dE/dW_L = -(e_{L+1} @ zF_L^T)
 * and UpdateWeights() applies W_L += (lr/batchSize) * e_{L+1}^T @ zF_L,
 * i.e. W -= lr * dE/dW. With lr=1, batchSize=1, the implied gradient is
 * simply -(W_after - W_before).
 */
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <cstdio>
#include <cmath>
#include <random>

using namespace Deep;

namespace
{
    // Runs a full settling step (Step()'s three sweeps) `steps` times,
    // starting from a clean, clamped state.
    void SettleNetwork(GaussSeidelPCNetwork &net, const std::vector<float> &x,
                       const std::vector<float> &y, int steps)
    {
        net.ResetState();
        net.Clamp(x);
        net.ProjectForward();
        net.GetTerminalLayer()->ClampState(y);

        for (int t = 0; t < steps; ++t)
            net.Step();
    }

    // Total energy at the network's CURRENT state, without touching z:
    // re-runs sweeps 2 (ComputePrediction) and 3 (ComputeError) only, so
    // repeated calls with a perturbed W are safe and don't drift z or
    // require sweep 1 (UpdateState(), which DOES mutate z).
    float TotalEnergy(GaussSeidelPCNetwork &net)
    {
        for (auto &layer : net.GetLayers())
            layer->ComputePrediction();

        float total = 0.0f;
        for (auto &layer : net.GetLayers())
            total += layer->ComputeError();
        return total;
    }
}

int main()
{
    const int batchSize = 1;
    const float eps = 3e-3f;
    const float tolerance = 1e-2f; // finite-difference error is O(eps^2);
                                   // loose enough to tolerate float32
                                   // precision, tight enough to catch a
                                   // real sign/shape bug

    GaussSeidelPCNetwork net(batchSize);
    net.AddLayer(3, 3, 0.0f, 0.08f, 0.0f, ActivationType::SIGMOID, ActivationType::dSIGMOID);
    net.AddLayer(3, 2, 0.0f, 0.08f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.AddLayer(2, 0, 0.0f, 0.08f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    net.SetOptimizer(OptimizerType::SGD);
    net.SetLearningRate(1.0f); // strips scaling, see file header

    std::vector<float> x = {0.2f, -0.5f, 0.8f};
    std::vector<float> y = {1.0f, -1.0f};

    // Layer index 1 (0-indexed): the 3->2 layer, whose W we're checking.
    GaussSeidelPCLayer *testLayer = net.GetLayers()[1].get();
    float *W = const_cast<float *>(testLayer->GetWeights());
    const size_t testIdx = 0; // check W[0] specifically

    SettleNetwork(net, x, y, /*steps=*/5);

    float wOriginal = W[testIdx];

    // Perturb W only, z stays exactly where settling left it.
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
    TotalEnergy(net); // refresh mu/e to match the just-settled z/W exactly

    float wBefore = W[testIdx];
    testLayer->UpdateWeights();
    float wAfter = W[testIdx];

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

    printf("PASSED: GaussSeidelPCLayer's UpdateWeights() matches the true dE/dW "
           "(E's feedback-alignment role in UpdateState() correctly stays out of "
           "the weight-gradient path).\n");
    return 0;
}
