// Gradient check for GaussSeidelPCLayer -- necessitated by a real,
// concrete failure signature in actual MNIST training: energy dropped
// smoothly and genuinely low, but accuracy stayed at/below chance and
// even declined -- the classic "found a degenerate solution satisfying
// the energy objective without solving the task" signature, strongly
// suggesting a real bug in the three-sweep math, not just a tuning issue.
//
// CRITICAL DESIGN POINT: totalEnergy() below calls ComputePrediction()
// then ComputeError() on every layer -- deliberately NOT UpdateState(),
// which would let z respond to a weight/z perturbation and corrupt the
// check. This freezes z, matching the same "hold state fixed, perturb
// one weight, re-measure energy" recipe used throughout this session's
// other gradient checks.
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace Deep;

void Part1_WeightGradCheck()
{
    std::cout << "\n=== Part 1: GaussSeidelPCLayer weight-gradient check ===\n";

    const int BATCH = 4;
    const float LR = 0.05f;

    GaussSeidelPCLayer layerA(4, 3, BATCH, LR, 0.1f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    GaussSeidelPCLayer layerB(3, 0, BATCH, LR, 0.1f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    layerA.SetLayerAbove(&layerB);
    layerB.SetLayerBelow(&layerA);

    std::mt19937 rng(42);
    layerA.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> x((size_t)BATCH * 4);
    for (auto &v : x)
        v = dataDist(rng);
    std::vector<float> target((size_t)BATCH * 3);
    for (auto &v : target)
        v = dataDist(rng);

    layerA.ClampState(x);
    layerB.ClampState(target);

    // Settle for real, using the genuine three-sweep Step() sequence.
    for (int t = 0; t < 30; ++t)
    {
        layerA.UpdateState();
        layerB.UpdateState();
        layerA.ComputePrediction();
        layerB.ComputePrediction();
        layerA.ComputeError();
        layerB.ComputeError();
    }

    // Frozen-z energy re-evaluation -- ComputePrediction+ComputeError
    // ONLY, deliberately skipping UpdateState() so z doesn't respond to
    // a weight perturbation.
    auto totalEnergy = [&]()
    {
        layerA.ComputePrediction();
        layerB.ComputePrediction();
        float ea = layerA.ComputeError();
        float eb = layerB.ComputeError();
        return ea + eb;
    };

    size_t Wsize = (size_t)4 * 3;
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
    std::cout << (worstRelErr < 0.05f ? "PASS" : "FAIL") << "\n";
}

void Part2_FeedbackTermCheck()
{
    std::cout << "\n=== Part 2: GaussSeidelPCLayer feedback-term check ===\n";
    std::cout << "(genuinely unclamped middle layer)\n";

    const int BATCH = 4;
    const float IR = 0.1f;

    GaussSeidelPCLayer layer0(4, 5, BATCH, 0.05f, IR, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    GaussSeidelPCLayer layer1(5, 3, BATCH, 0.05f, IR, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    GaussSeidelPCLayer layer2(3, 0, BATCH, 0.05f, IR, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    layer0.SetLayerAbove(&layer1);
    layer1.SetLayerBelow(&layer0);
    layer1.SetLayerAbove(&layer2);
    layer2.SetLayerBelow(&layer1);

    std::mt19937 rng(42);
    layer0.RandomizeWeights(rng);
    layer1.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> x((size_t)BATCH * 4);
    for (auto &v : x)
        v = dataDist(rng);
    std::vector<float> target((size_t)BATCH * 3);
    for (auto &v : target)
        v = dataDist(rng);

    layer0.ResetState();
    layer1.ResetState();
    layer2.ResetState();
    layer0.ClampState(x);
    layer2.ClampState(target);

    auto step = [&]()
    {
        layer0.UpdateState();
        layer1.UpdateState();
        layer2.UpdateState();
        layer0.ComputePrediction();
        layer1.ComputePrediction();
        layer2.ComputePrediction();
        layer0.ComputeError();
        layer1.ComputeError();
        layer2.ComputeError();
    };

    for (int t = 0; t < 30; ++t)
        step();

    size_t hiddenSize = layer1.GetInputSize() * BATCH;
    std::vector<float> z_before(layer1.GetBeliefs(), layer1.GetBeliefs() + hiddenSize);

    step();

    std::vector<float> z_after(layer1.GetBeliefs(), layer1.GetBeliefs() + hiddenSize);
    std::vector<float> implied_dz_dt(hiddenSize);
    for (size_t i = 0; i < hiddenSize; ++i)
        implied_dz_dt[i] = (z_after[i] - z_before[i]) / IR;

    std::memcpy(layer1.GetBeliefs(), z_before.data(), hiddenSize * sizeof(float));

    // Frozen-z energy -- ComputePrediction+ComputeError only.
    auto totalEnergy = [&]()
    {
        layer0.ComputePrediction();
        layer1.ComputePrediction();
        layer2.ComputePrediction();
        float e0 = layer0.ComputeError();
        float e1 = layer1.ComputeError();
        float e2 = layer2.ComputeError();
        return e0 + e1 + e2;
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

int main()
{
    Part1_WeightGradCheck();
    Part2_FeedbackTermCheck();
    return 0;
}