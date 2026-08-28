// Targeted check: a MIDDLE layer's own weight gradient, with BOTH a real
// layerBelow AND layerAbove -- the exact configuration the earlier Part 1
// check never tested (it only used layerBelow=nullptr, matching an
// input-layer situation, not a middle layer's). This is where the real
// bug was found empirically: 17.67% sign agreement vs the working Jacobi
// implementation on real MNIST data.
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace Deep;

int main()
{
    std::cout << "=== Middle-layer weight-gradient check (layerBelow AND layerAbove both real) ===\n";

    const int BATCH = 4;
    const float LR = 0.05f;

    GaussSeidelPCLayer layer0(4, 5, BATCH, LR, 0.1f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    GaussSeidelPCLayer layer1(5, 3, BATCH, LR, 0.1f, 0.0f, ActivationType::TANH, ActivationType::dTANH); // THE layer under test
    GaussSeidelPCLayer layer2(3, 0, BATCH, LR, 0.1f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    layer0.SetLayerAbove(&layer1); layer1.SetLayerBelow(&layer0);
    layer1.SetLayerAbove(&layer2); layer2.SetLayerBelow(&layer1);

    std::mt19937 rng(42);
    layer0.RandomizeWeights(rng);
    layer1.RandomizeWeights(rng);
    layer1.SetOptimizer(OptimizerType::ADAMW); // matches EXACTLY what the failing
                                                 // real-data comparison used

    // Manual rebind -- GetRequiredFloats()/BindMemory() already ran during
    // construction using the default SGD, before SetOptimizer() above.
    // Without this, grad_W/m_W/v_W are still nullptr. Same fix needed for
    // SimpleConvPCLayer earlier this session.
    static std::unique_ptr<MemoryArena> adamArena;
    adamArena = std::make_unique<MemoryArena>(layer1.GetRequiredFloats());
    layer1.BindMemory(*adamArena);
    layer1.RandomizeWeights(rng); // re-randomize -- BindMemory() zeroed weights again

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> x((size_t)BATCH * 4);
    for (auto &v : x) v = dataDist(rng);
    std::vector<float> target((size_t)BATCH * 3);
    for (auto &v : target) v = dataDist(rng);

    layer0.ResetState(); layer1.ResetState(); layer2.ResetState();
    layer0.ClampState(x);
    layer2.ClampState(target);
    layer2.ComputeError(); // matches the corrected TrainStepWithProjection fix

    auto step = [&]() {
        layer0.UpdateState(); layer1.UpdateState(); layer2.UpdateState();
        layer0.ComputePrediction(); layer1.ComputePrediction(); layer2.ComputePrediction();
        layer0.ComputeError(); layer1.ComputeError(); layer2.ComputeError();
    };

    for (int t = 0; t < 20; ++t) step();

    // Frozen-z energy -- ComputePrediction+ComputeError only.
    auto totalEnergy = [&]() {
        layer0.ComputePrediction(); layer1.ComputePrediction(); layer2.ComputePrediction();
        float e0 = layer0.ComputeError();
        float e1 = layer1.ComputeError();
        float e2 = layer2.ComputeError();
        return e0 + e1 + e2;
    };

    size_t Wsize = (size_t)5 * 3; // layer1's OWN weights
    std::vector<float> W_before(layer1.GetWeights(), layer1.GetWeights() + Wsize);

    layer1.UpdateWeights(); // THE call under test

    std::vector<float> W_after(layer1.GetWeights(), layer1.GetWeights() + Wsize);
    std::vector<float> analytic_delta(Wsize);
    for (size_t i = 0; i < Wsize; ++i)
        analytic_delta[i] = W_after[i] - W_before[i];

    std::memcpy(layer1.GetWeights(), W_before.data(), Wsize * sizeof(float));
    totalEnergy();

    float eps = 1e-3f, worstRelErr = 0.0f;
    int nCorrectSign = 0;
    std::uniform_int_distribution<size_t> idxDist(0, Wsize - 1);
    int nChecks = 12;

    for (int chk = 0; chk < nChecks; ++chk)
    {
        size_t idx = idxDist(rng);
        float orig = W_before[idx];

        layer1.GetWeights()[idx] = orig + eps;
        float E_plus = totalEnergy();
        layer1.GetWeights()[idx] = orig - eps;
        float E_minus = totalEnergy();
        layer1.GetWeights()[idx] = orig;
        totalEnergy();

        float numeric_dEdW = (E_plus - E_minus) / (2 * eps);
        float lr_batch = LR / BATCH;
        float expected_descent = -lr_batch * numeric_dEdW;
        float expected_ascent = lr_batch * numeric_dEdW;
        float err_descent = std::abs(analytic_delta[idx] - expected_descent) / (std::abs(numeric_dEdW) + 1e-8f);
        float err_ascent = std::abs(analytic_delta[idx] - expected_ascent) / (std::abs(numeric_dEdW) + 1e-8f);
        float best = std::min(err_descent, err_ascent);
        worstRelErr = std::max(worstRelErr, best);
        bool correctSign = err_descent < err_ascent;
        if (correctSign) nCorrectSign++;

        std::cout << "  W[" << idx << "]: delta=" << analytic_delta[idx]
                   << "  numeric_dE/dW=" << numeric_dEdW
                   << "  " << (correctSign ? "MATCHES DESCENT" : "WRONG SIGN (matches ascent)")
                   << "  rel_err=" << best << "\n";
    }
    std::cout << "\nCorrect-sign fraction: " << (100.0f * nCorrectSign / nChecks) << "%\n";
    std::cout << "Worst relative error: " << worstRelErr << "\n";
    std::cout << (worstRelErr < 0.05f ? "PASS" : "FAIL") << "\n";

    return 0;
}
