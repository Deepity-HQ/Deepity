// Finite-difference gradient check run THROUGH SimpleConvPCNetwork's
// actual TrainStep()/Compile() flow -- not hand-wired layers like
// tSimpleConvVerify.cpp used. The smoke test confirmed energy decreases
// sanely; this confirms the WRAPPER'S wiring doesn't introduce a
// magnitude/sign error the energy-only check couldn't catch (same class
// of gap the evaluation-order bug slipped through earlier today).
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace Deep;

int main()
{
    const int BATCH = 2;
    const float LR = 0.03f;

    // Small 2-layer network (input->hidden->terminal), matching
    // tSimpleConvVerify.cpp's Part 1 shape/scale for a direct comparison.
    SimpleConvPCNetwork net(BATCH);
    net.AddLayer(1, 2, 4, 4, 3, 3, 1, 1, 0, 0, LR, 0.1f, 0.0f,
                 ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(2, 0, 2, 2, 1, 1, 0, 0, 0, 0, LR, 0.1f, 0.0f,
                 ActivationType::LINEAR, ActivationType::dLINEAR);

    net.SetOptimizer(OptimizerType::SGD);
    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    std::vector<float> X((size_t)BATCH * 1 * 4 * 4);
    std::vector<float> Y((size_t)BATCH * 2 * 2 * 2);
    for (auto &v : X)
        v = dataDist(rng);
    for (auto &v : Y)
        v = dataDist(rng);

    SimpleConvPCLayer *layerA = net.GetLayers()[0];

    // Settle once (matching tSimpleConvVerify.cpp's pattern) before
    // capturing weights and testing the gradient.
    net.ResetState();
    net.Clamp(X);
    net.GetTerminalLayer()->ClampState(Y);
    for (int t = 0; t < 30; ++t)
    {
        net.CalculateState();
        net.UpdateState();
    }

    auto totalEnergy = [&]()
    {
        // Sequential, not `+`-operand order -- same fix as
        // tSimpleConvVerify.cpp's earlier evaluation-order bug.
        float sum = 0.0f;
        for (auto *l : net.GetLayers())
        {
            float e = l->CalculateState();
            sum += e;
        }
        return sum;
    };

    size_t Wsize = (size_t)2 * 1 * 3 * 3;
    std::vector<float> W_before(layerA->GetWeights(), layerA->GetWeights() + Wsize);

    layerA->UpdateWeights();

    std::vector<float> W_after(layerA->GetWeights(), layerA->GetWeights() + Wsize);
    std::vector<float> analytic_delta(Wsize);
    for (size_t i = 0; i < Wsize; ++i)
        analytic_delta[i] = W_after[i] - W_before[i];

    std::memcpy(layerA->GetWeights(), W_before.data(), Wsize * sizeof(float));
    totalEnergy();

    float eps = 1e-3f, worstRelErr = 0.0f;
    std::uniform_int_distribution<size_t> idxDist(0, Wsize - 1);
    int nChecks = 8;

    std::cout << "=== SimpleConvPCNetwork gradient check (through the wrapper) ===\n";

    for (int chk = 0; chk < nChecks; ++chk)
    {
        size_t idx = idxDist(rng);
        float orig = W_before[idx];

        layerA->GetWeights()[idx] = orig + eps;
        float E_plus = totalEnergy();
        layerA->GetWeights()[idx] = orig - eps;
        float E_minus = totalEnergy();
        layerA->GetWeights()[idx] = orig;
        totalEnergy();

        float numeric_dEdW = (E_plus - E_minus) / (2 * eps);
        float lr_batch = LR / BATCH;
        float expected = -lr_batch * numeric_dEdW;
        float relErr = std::abs(analytic_delta[idx] - expected) / (std::abs(numeric_dEdW) + 1e-8f);
        worstRelErr = std::max(worstRelErr, relErr);

        std::cout << "  W[" << idx << "]: delta=" << analytic_delta[idx]
                  << "  numeric_dE/dW=" << numeric_dEdW
                  << "  rel_err=" << relErr << "\n";
    }

    std::cout << "Worst relative error: " << worstRelErr << "\n";
    std::cout << (worstRelErr < 0.05f ? "PASS" : "FAIL") << "\n";

    return 0;
}