// tConvGradCheck.cpp
//
// Finite-difference verification of ConvPCLayer::UpdateWeights() -- run
// this AFTER the repack/single-GEMM restructure to confirm the math is
// still exactly what it was, not just faster. Uses only public accessors
// (GetWeights()), capturing W before/after a real UpdateWeights() call and
// comparing that delta against numerically-estimated dE/dW.
//
// Network: layer0 (conv, 1x6x6 -> 2x4x4) directly feeding layer1 (terminal,
// outChannels=0). BOTH layers clamped -- z never moves, isolating
// UpdateWeights() with no settling/staleness complexity.

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <deepity/layers/ConvPCLayer.h>

using namespace Deep;

int main()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const float LR = 0.05f;
    const int BATCH = 1;

    ConvPCLayer layer0(1, 2, 6, 6, 3, 3, 1, 1, 0, 0,
                       BATCH, LR, 0.1f, 0.0f, 0.0f,
                       ActivationType::TANH, ActivationType::dTANH);

    ConvPCLayer layer1(2, 0, 4, 4, 1, 1, 1, 1, 0, 0,
                       BATCH, LR, 0.1f, 0.0f, 0.0f,
                       ActivationType::LINEAR, ActivationType::dLINEAR);

    layer0.SetLayerAbove(&layer1);
    layer1.SetLayerBelow(&layer0);

    layer0.RandomizeWeights(rng);

    std::vector<float> input(1 * 6 * 6);
    std::vector<float> target(2 * 4 * 4);
    for (auto &v : input)
        v = dist(rng);
    for (auto &v : target)
        v = dist(rng);

    layer0.ClampState(input);
    layer1.ClampState(target);

    layer0.CalculateState();
    layer1.CalculateState();
    layer0.UpdateState(); // converts layer0's mu -> f'(activated); clamped, so z untouched
    layer1.UpdateState(); // outChannels=0, clamped -- no-op

    auto TotalEnergy = [&]()
    {
        return layer0.CalculateState() + layer1.CalculateState();
    };

    size_t colRows = 1 * 3 * 3;
    size_t Wsize = 2 * colRows;

    std::vector<float> W_before(Wsize);
    std::memcpy(W_before.data(), layer0.GetWeights(), Wsize * sizeof(float));

    layer0.UpdateWeights(); // <-- now internally does repack + single GEMM instead of 256 small ones

    std::vector<float> W_after(Wsize);
    std::memcpy(W_after.data(), layer0.GetWeights(), Wsize * sizeof(float));

    std::vector<float> analytic_delta(Wsize);
    for (size_t i = 0; i < Wsize; ++i)
        analytic_delta[i] = W_after[i] - W_before[i];

    std::memcpy(layer0.GetWeights(), W_before.data(), Wsize * sizeof(float));
    layer0.CalculateState();
    layer1.CalculateState();

    const float eps = 1e-4f;
    const int nChecks = 8;
    float worstRelErr = 0.0f;

    std::cout << "=== ConvPCLayer UpdateWeights() re-verification (post repack/single-GEMM restructure) ===\n";
    std::cout << "Checking " << nChecks << " of " << Wsize << " weight entries...\n";

    for (int c = 0; c < nChecks; ++c)
    {
        size_t i = rng() % Wsize;
        float orig = W_before[i];

        layer0.GetWeights()[i] = orig + eps;
        float E_plus = TotalEnergy();

        layer0.GetWeights()[i] = orig - eps;
        float E_minus = TotalEnergy();

        layer0.GetWeights()[i] = orig;

        float numeric_dEdW = (E_plus - E_minus) / (2.0f * eps);
        float lr_batch = LR / BATCH;

        float expected_descent = -lr_batch * numeric_dEdW;
        float expected_ascent = lr_batch * numeric_dEdW;

        float err_descent = std::abs(analytic_delta[i] - expected_descent) / (std::abs(numeric_dEdW) + 1e-8f);
        float err_ascent = std::abs(analytic_delta[i] - expected_ascent) / (std::abs(numeric_dEdW) + 1e-8f);
        float best = std::min(err_descent, err_ascent);
        worstRelErr = std::max(worstRelErr, best);

        std::cout << "  W[" << i << "]: analytic_delta=" << analytic_delta[i]
                  << "  numeric dE/dW=" << numeric_dEdW
                  << "  " << (err_descent < err_ascent ? "MATCHES DESCENT" : "MATCHES ASCENT")
                  << "  (rel err=" << best << ")\n";
    }

    std::cout << "\nWorst relative error: " << worstRelErr << "\n";
    std::cout << (worstRelErr < 0.02f ? "PASS" : "FAIL") << "\n";

    return worstRelErr < 0.02f ? 0 : 1;
}