/**
 * @file tSimpleConvPCVerify.cpp
 * @brief Finite-difference gradient check for SimpleConvPCLayer's
 * UpdateWeights() (SGD path) -- prerequisite #1 named explicitly in
 * SimpleConvPCLayer.h's own file-level warning, before trusting this
 * class for real training.
 *
 * Both layers CLAMPED, matching ConvPCLayer.h's own noted methodology
 * for isolating the weight-gradient path specifically (the feedback
 * term, which needs a genuinely UNCLAMPED middle layer, was already
 * separately verified for ConvPCLayer via tConvFeedbackVerify.cpp --
 * that methodology needs re-running against THIS class too, but as a
 * separate, second test).
 *
 * Same finite-difference methodology as tDirectKPVerify.cpp/
 * tGaussSeidelVerify.cpp: for a handful of random W elements, compare
 * UpdateWeights()'s actual, analytical delta against a numerically
 * estimated dE/dW (central difference), checking both sign (matches
 * descent) and magnitude (relative error).
 */
#include <deepity/layers/SimpleConvPCLayer.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
    // Runs one full "clamped weight-gradient" energy evaluation: forward
    // pass through layer0 (clamped input -> mu), then layer1's error
    // against that mu. Returns the resulting energy. Matches exactly
    // what UpdateWeights() itself depends on (colBuffer from
    // ComputeMuOnly(), local_grad from layer1's error).
    float EvaluateEnergy(SimpleConvPCLayer &layer0, SimpleConvPCLayer &layer1)
    {
        layer0.ComputeMuOnly();
        return layer1.CalculateState();
    }
}

int main()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int inChannels = 1, outChannels = 2;
    const int inH = 4, inW = 4, kH = 3, kW = 3;
    const int batchSize = 1;

    SimpleConvPCLayer layer0(inChannels, outChannels, inH, inW, kH, kW,
                              /*strideH=*/1, /*strideW=*/1, /*padH=*/1, /*padW=*/1,
                              batchSize, /*lr=*/0.01f, /*ir=*/0.1f, /*lmbda=*/0.0f,
                              ActivationType::TANH, ActivationType::dTANH);

    // Terminal layer: inChannels/inHeight/inWidth MUST match layer0's
    // outChannels/outHeight/outWidth exactly (2, 4, 4) -- outChannels=0
    // marks it terminal, matching every other Layer convention tonight.
    SimpleConvPCLayer layer1(outChannels, 0, layer0.GetOutHeight(), layer0.GetOutWidth(),
                              1, 1, 1, 1, 0, 0,
                              batchSize, 0.0f, 0.1f, 0.0f,
                              ActivationType::LINEAR, ActivationType::dLINEAR);

    layer0.SetLayerAbove(&layer1);
    layer1.SetLayerBelow(&layer0);

    layer0.RandomizeWeights(rng);

    std::vector<float> input(inChannels * inH * inW);
    for (auto &v : input)
        v = dist(rng);
    layer0.ClampState(input);

    std::vector<float> target(outChannels * layer0.GetOutHeight() * layer0.GetOutWidth());
    for (auto &v : target)
        v = dist(rng);
    layer1.ClampState(target);

    // Real forward pass + error, exactly as UpdateWeights() itself
    // depends on -- this also fills colBuffer (via ComputeMuOnly) which
    // UpdateWeights() reads.
    EvaluateEnergy(layer0, layer1);

    float *W = layer0.GetWeights();
    size_t colRows = (size_t)inChannels * kH * kW;
    size_t Wsize = (size_t)outChannels * colRows;

    std::vector<float> W_before(W, W + Wsize);

    layer0.UpdateWeights();

    std::vector<float> W_after(W, W + Wsize);

    const float eps = 1e-3f;
    const float lr_batch = layer0.GetLearningRate() / batchSize;
    float worstRelErr = 0.0f;
    bool allPassed = true;

    std::uniform_int_distribution<size_t> idxDist(0, Wsize - 1);
    for (int trial = 0; trial < 8; ++trial)
    {
        size_t idx = idxDist(rng);
        float delta = W_after[idx] - W_before[idx];

        // Perturb +eps, restore everything else, re-evaluate energy
        for (size_t i = 0; i < Wsize; ++i)
            W[i] = W_before[i];

        layer0.ClampState(input);
        W[idx] = W_before[idx] + eps;
        float Eplus = EvaluateEnergy(layer0, layer1);

        for (size_t i = 0; i < Wsize; ++i)
            W[i] = W_before[i];
        layer0.ClampState(input);
        W[idx] = W_before[idx] - eps;
        float Eminus = EvaluateEnergy(layer0, layer1);

        float numericalGrad = (Eplus - Eminus) / (2.0f * eps);
        float expectedDelta = -lr_batch * numericalGrad;

        float relDenom = std::fabs(expectedDelta) + 1e-6f;
        float relErr = std::fabs(delta - expectedDelta) / relDenom;
        worstRelErr = std::max(worstRelErr, relErr);

        bool matchesDescent = (delta * (-numericalGrad)) >= 0.0f;
        if (!matchesDescent || relErr > 0.05f)
            allPassed = false;

        printf("  W[%zu]: delta=%g  numeric_dE/dW=%g  expected_delta=%g  %s  rel_err=%g\n",
               idx, delta, numericalGrad, expectedDelta,
               matchesDescent ? "MATCHES DESCENT" : "SIGN MISMATCH", relErr);
    }

    // Restore W to its true post-update state before exiting, for cleanliness.
    for (size_t i = 0; i < Wsize; ++i)
        W[i] = W_after[i];

    printf("\nWorst relative error: %g\n", worstRelErr);
    printf(allPassed ? "PASS\n" : "FAIL\n");

    return allPassed ? 0 : 1;
}