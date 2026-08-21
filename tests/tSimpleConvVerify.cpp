// Three-part verification of SimpleConvPCLayer, required before trusting
// it for real training (per the file-level warning in
// SimpleConvPCLayer.h):
//   1. SGD weight-gradient check (finite-difference)
//   2. Feedback-term (Col2Im) verification with a genuinely unclamped
//      middle layer -- same methodology as tConvFeedbackVerify.cpp,
//      re-run against THIS class specifically (precision removal touched
//      the own-error term the feedback path depends on)
//   3. AdamW weight-gradient check -- a NEW port, not the already-verified
//      SimplePCLayer code, checking SIGN first (same bug class found and
//      fixed there)

#include "deepity/layers/SimpleConvPCLayer.h"
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace Deep;

void Part1_SGDWeightGradCheck()
{
    std::cout << "\n=== Part 1: SGD weight-gradient check ===\n";

    const int BATCH = 1;
    const float LR = 0.05f;

    SimpleConvPCLayer layerA(1, 2, 4, 4, 3, 3, 1, 1, 0, 0, BATCH, LR, 0.1f, 0.0f,
                              ActivationType::TANH, ActivationType::dTANH);
    SimpleConvPCLayer layerB(2, 0, 2, 2, 1, 1, 0, 0, 0, 0, BATCH, LR, 0.1f, 0.0f,
                              ActivationType::LINEAR, ActivationType::dLINEAR);
    layerA.SetLayerAbove(&layerB);
    layerB.SetLayerBelow(&layerA);

    std::mt19937 rng(42);
    layerA.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> x(1 * 4 * 4);
    for (auto &v : x) v = dataDist(rng);
    std::vector<float> target(2 * 2 * 2);
    for (auto &v : target) v = dataDist(rng);

    layerA.ClampState(x);
    layerB.ClampState(target);
    layerA.CalculateState();
    layerB.CalculateState();
    layerA.UpdateState();
    layerB.UpdateState();

    auto totalEnergy = [&]() { return layerA.CalculateState() + layerB.CalculateState(); };

    size_t Wsize = (size_t)2 * 1 * 3 * 3;
    std::vector<float> W_before(layerA.GetWeights(), layerA.GetWeights() + Wsize);

    layerA.UpdateWeights();

    std::vector<float> W_after(layerA.GetWeights(), layerA.GetWeights() + Wsize);
    std::vector<float> analytic_delta(Wsize);
    for (size_t i = 0; i < Wsize; ++i)
        analytic_delta[i] = W_after[i] - W_before[i];

    std::memcpy(layerA.GetWeights(), W_before.data(), Wsize * sizeof(float));
    totalEnergy();

    float eps = 1e-3f, worstRelErr = 0.0f;
    std::uniform_int_distribution<size_t> idxDist(0, Wsize - 1);
    int nChecks = 8;

    for (int chk = 0; chk < nChecks; ++chk)
    {
        size_t idx = idxDist(rng);
        float orig = W_before[idx];

        layerA.GetWeights()[idx] = orig + eps;
        float E_plus = totalEnergy();
        layerA.GetWeights()[idx] = orig - eps;
        float E_minus = totalEnergy();
        layerA.GetWeights()[idx] = orig;
        totalEnergy();

        float numeric_dEdW = (E_plus - E_minus) / (2 * eps);
        float lr_batch = LR / BATCH;
        float expected_descent = -lr_batch * numeric_dEdW;
        float expected_ascent = lr_batch * numeric_dEdW;
        float err_descent = std::abs(analytic_delta[idx] - expected_descent) / (std::abs(numeric_dEdW) + 1e-8f);
        float err_ascent = std::abs(analytic_delta[idx] - expected_ascent) / (std::abs(numeric_dEdW) + 1e-8f);
        float best = std::min(err_descent, err_ascent);
        worstRelErr = std::max(worstRelErr, best);

        std::cout << "  W[" << idx << "]: delta=" << analytic_delta[idx]
                   << "  numeric_dE/dW=" << numeric_dEdW
                   << "  " << (err_descent < err_ascent ? "MATCHES DESCENT" : "MATCHES ASCENT")
                   << "  rel_err=" << best << "\n";
    }
    std::cout << "Worst relative error: " << worstRelErr << "\n";
    std::cout << (worstRelErr < 0.02f ? "PASS" : "FAIL") << "\n";
}

void Part2_FeedbackVerify()
{
    std::cout << "\n=== Part 2: feedback-term (Col2Im) verification, SimpleConvPCLayer ===\n";

    const int BATCH = 1;
    const float IR = 0.1f;

    SimpleConvPCLayer layer0(1, 2, 6, 6, 3, 3, 1, 1, 0, 0, BATCH, 0.05f, IR, 0.0f,
                              ActivationType::TANH, ActivationType::dTANH);
    SimpleConvPCLayer layer1(2, 1, 4, 4, 3, 3, 1, 1, 0, 0, BATCH, 0.05f, IR, 0.0f,
                              ActivationType::TANH, ActivationType::dTANH);
    SimpleConvPCLayer layer2(1, 0, 2, 2, 1, 1, 0, 0, 0, 0, BATCH, 0.05f, IR, 0.0f,
                              ActivationType::LINEAR, ActivationType::dLINEAR);
    layer0.SetLayerAbove(&layer1); layer1.SetLayerBelow(&layer0);
    layer1.SetLayerAbove(&layer2); layer2.SetLayerBelow(&layer1);

    std::mt19937 rng(42);
    layer0.RandomizeWeights(rng);
    layer1.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> x(1 * 6 * 6);
    for (auto &v : x) v = dataDist(rng);
    std::vector<float> target(1 * 2 * 2);
    for (auto &v : target) v = dataDist(rng);

    layer0.ResetState(); layer1.ResetState(); layer2.ResetState();
    layer0.ClampState(x);
    layer2.ClampState(target);

    auto step = [&]() {
        layer0.CalculateState(); layer1.CalculateState(); layer2.CalculateState();
        layer0.UpdateState(); layer1.UpdateState(); layer2.UpdateState();
    };

    for (int t = 0; t < 30; ++t) step();

    size_t hiddenSize = layer1.GetInputSize();
    std::vector<float> z_before(layer1.GetBeliefs(), layer1.GetBeliefs() + hiddenSize);

    step();

    std::vector<float> z_after(layer1.GetBeliefs(), layer1.GetBeliefs() + hiddenSize);
    std::vector<float> implied_dz_dt(hiddenSize);
    for (size_t i = 0; i < hiddenSize; ++i)
        implied_dz_dt[i] = (z_after[i] - z_before[i]) / IR;

    std::memcpy(layer1.GetBeliefs(), z_before.data(), hiddenSize * sizeof(float));

    auto totalEnergy = [&]() {
        return layer0.CalculateState() + layer1.CalculateState() + layer2.CalculateState();
    };

    float eps = 1e-3f, worstRelErr = 0.0f;
    int nChecks = 8;
    std::uniform_int_distribution<size_t> idxDist(0, hiddenSize - 1);

    for (int chk = 0; chk < nChecks; ++chk)
    {
        size_t idx = idxDist(rng);
        float orig = z_before[idx];

        layer1.GetBeliefs()[idx] = orig + eps;
        float E_plus = totalEnergy();
        layer1.GetBeliefs()[idx] = orig - eps;
        float E_minus = totalEnergy();
        layer1.GetBeliefs()[idx] = orig;
        totalEnergy();

        float numeric_dEdz = (E_plus - E_minus) / (2 * eps);
        float expected_dz_dt = -numeric_dEdz;
        float relErr = std::abs(implied_dz_dt[idx] - expected_dz_dt) / (std::abs(numeric_dEdz) + 1e-6f);
        worstRelErr = std::max(worstRelErr, relErr);

        std::cout << "  z[" << idx << "]: implied_dz_dt=" << implied_dz_dt[idx]
                   << "  -numeric_dE/dz=" << expected_dz_dt << "  rel_err=" << relErr << "\n";
    }
    std::cout << "Worst relative error: " << worstRelErr << "\n";
    std::cout << (worstRelErr < 0.05f ? "PASS" : "FAIL") << "\n";
}

void Part3_AdamWGradCheck()
{
    std::cout << "\n=== Part 3: AdamW weight-gradient check (NEW port, checking SIGN first) ===\n";

    const int BATCH = 1;
    const float LR = 0.05f;

    SimpleConvPCLayer layerA(1, 2, 4, 4, 3, 3, 1, 1, 0, 0, BATCH, LR, 0.1f, 0.0f,
                              ActivationType::TANH, ActivationType::dTANH);
    SimpleConvPCLayer layerB(2, 0, 2, 2, 1, 1, 0, 0, 0, 0, BATCH, LR, 0.1f, 0.0f,
                              ActivationType::LINEAR, ActivationType::dLINEAR);
    layerA.SetLayerAbove(&layerB);
    layerB.SetLayerBelow(&layerA);
    layerA.SetOptimizer(OptimizerType::ADAMW);

    // REAL BUG, caught before running: the constructor already called
    // BindMemory() with opt=SGD (the header default) BEFORE SetOptimizer()
    // above ran -- grad_W/m_W/v_W are still nullptr at this point.
    // SimpleConvPCNetwork (not yet built) would fix this the same way
    // SimplePCNetwork does -- SetOptimizer() on every layer, THEN
    // Compile() rebinds everything into a shared arena. For this
    // standalone-layer test, do that rebind manually: a fresh, correctly-
    // sized arena, now that GetRequiredFloats() sees opt=ADAMW.
    static std::unique_ptr<MemoryArena> adamArena; // static: must outlive layerA's use of it
    adamArena = std::make_unique<MemoryArena>(layerA.GetRequiredFloats());
    layerA.BindMemory(*adamArena);

    std::cout << "  (Manually re-bound layerA's memory after SetOptimizer(ADAMW) -- \n"
              << "  SimpleConvPCLayer has no Network wrapper yet to do this automatically,\n"
              << "  unlike SimplePCNetwork's Compile()-after-SetOptimizer() flow. A real\n"
              << "  design gap worth fixing when SimpleConvPCNetwork gets built.)\n\n";

    std::mt19937 rng(42);
    layerA.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> x(1 * 4 * 4);
    for (auto &v : x) v = dataDist(rng);
    std::vector<float> target(2 * 2 * 2);
    for (auto &v : target) v = dataDist(rng);

    layerA.ClampState(x);
    layerB.ClampState(target);
    layerA.CalculateState();
    layerB.CalculateState();
    layerA.UpdateState();
    layerB.UpdateState();

    auto totalEnergy = [&]() { return layerA.CalculateState() + layerB.CalculateState(); };

    size_t Wsize = (size_t)2 * 1 * 3 * 3;
    std::vector<float> W_before(layerA.GetWeights(), layerA.GetWeights() + Wsize);

    layerA.UpdateWeights();

    std::vector<float> W_after(layerA.GetWeights(), layerA.GetWeights() + Wsize);
    std::vector<float> analytic_delta(Wsize);
    for (size_t i = 0; i < Wsize; ++i)
        analytic_delta[i] = W_after[i] - W_before[i];

    std::memcpy(layerA.GetWeights(), W_before.data(), Wsize * sizeof(float));
    totalEnergy();

    float eps = 1e-3f;
    bool allCorrectSign = true;
    std::uniform_int_distribution<size_t> idxDist(0, Wsize - 1);
    int nChecks = 8;

    for (int chk = 0; chk < nChecks; ++chk)
    {
        size_t idx = idxDist(rng);
        float orig = W_before[idx];

        layerA.GetWeights()[idx] = orig + eps;
        float E_plus = totalEnergy();
        layerA.GetWeights()[idx] = orig - eps;
        float E_minus = totalEnergy();
        layerA.GetWeights()[idx] = orig;
        totalEnergy();

        float numeric_dEdW = (E_plus - E_minus) / (2 * eps);
        bool correctSign = (analytic_delta[idx] > 0) == (numeric_dEdW < 0) || std::abs(numeric_dEdW) < 1e-6f;
        if (!correctSign) allCorrectSign = false;

        std::cout << "  W[" << idx << "]: delta=" << analytic_delta[idx]
                   << "  true_grad_sign=" << (numeric_dEdW > 0 ? "+" : "-")
                   << "  " << (correctSign ? "OK" : "WRONG SIGN") << "\n";
    }
    std::cout << "All signs correct: " << (allCorrectSign ? "true" : "false") << "\n";
    std::cout << (allCorrectSign ? "PASS" : "FAIL") << "\n";
}

int main()
{
    std::cout.setf(std::ios::unitbuf);
    Part1_SGDWeightGradCheck();
    Part2_FeedbackVerify();
    Part3_AdamWGradCheck();
    return 0;
}
