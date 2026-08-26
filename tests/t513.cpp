// Isolates ONE 512->512 SimplePCLayer, standalone, matching Layer 1's
// real situation from the granular profile (fed by a 512-wide "layer
// below", feeding into a 10-wide "layer above") but stripped of
// everything else in the network -- to find exactly what's
// pathologically slow about this specific shape. 30.82ms calc / 27.06ms
// update measured in the real network is ~5x more than its FLOP count
// vs Layer 0 would predict.
#include <deepity/layers/SimplePCLayer.h>
#include <random>
#include <vector>
#include <iostream>
#include <chrono>

using namespace Deep;

int main()
{
    const int BATCH = 256;
    const int STEPS = 20;
    const int N_REPS = 40;

    // Matches Layer 1 exactly: 512 in, 512 out, same activation, wired
    // to a layer below (512-wide, standing in for Layer 0) and a layer
    // above (10-wide, standing in for Layer 2).
    SimplePCLayer below(512, 512, BATCH, 0.001f, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    SimplePCLayer layer1(512, 512, BATCH, 0.001f, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    SimplePCLayer above(512, 10, BATCH, 0.001f, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);

    below.SetLayerAbove(&layer1);
    layer1.SetLayerBelow(&below);
    layer1.SetLayerAbove(&above);
    above.SetLayerBelow(&layer1);

    std::mt19937 rng(7);
    below.RandomizeWeights(rng);
    layer1.RandomizeWeights(rng);
    above.RandomizeWeights(rng);

    std::uniform_real_distribution<float> initDist(-0.3f, 0.3f);
    auto reinit = [&](SimplePCLayer &l, size_t inSz, size_t outSz)
    {
        if (outSz == 0)
            return;
        float *W = l.GetWeights();
        for (size_t i = 0; i < inSz * outSz; ++i)
            W[i] = initDist(rng);
    };
    reinit(below, 512, 512);
    reinit(layer1, 512, 512);
    reinit(above, 512, 10);

    std::vector<float> belowInput((size_t)BATCH * 512);
    std::mt19937 dataRng(123);
    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    for (auto &v : belowInput)
        v = dataDist(dataRng);

    below.ClampState(belowInput); // "below" acts as a clamped input stand-in

    double t_calc = 0, t_update = 0;
    auto totalStart = std::chrono::steady_clock::now();

    for (int rep = 0; rep < N_REPS; ++rep)
    {
        below.CalculateState(); // seed below's own mu once, matching project_forward's role

        for (int step = 0; step < STEPS; ++step)
        {
            auto t0 = std::chrono::steady_clock::now();
            layer1.CalculateState();
            auto t1 = std::chrono::steady_clock::now();
            t_calc += std::chrono::duration<double>(t1 - t0).count();

            t0 = t1;
            layer1.UpdateState();
            t1 = std::chrono::steady_clock::now();
            t_update += std::chrono::duration<double>(t1 - t0).count();
        }
    }

    auto totalEnd = std::chrono::steady_clock::now();
    double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();

    std::cout << "=== Isolated 512->512 layer, " << N_REPS << " reps x " << STEPS << " steps ===\n\n";
    std::cout << "Total: " << totalTime << "s\n";
    std::cout << "calculate_state: " << 1000 * t_calc / (N_REPS * STEPS) << " ms/call  (avg per single call)\n";
    std::cout << "update_state:    " << 1000 * t_update / (N_REPS * STEPS) << " ms/call  (avg per single call)\n\n";
    std::cout << "Compare against the real network's measured Layer 1 costs:\n";
    std::cout << "  calc: 30.82ms/batch / 20 steps = 1.541 ms/call\n";
    std::cout << "  update: 27.06ms/batch / 20 steps = 1.353 ms/call\n";
    std::cout << "\nIf THIS isolated test also shows ~1.5ms/call, the slowness is\n";
    std::cout << "intrinsic to a 512x512 SimplePCLayer regardless of context. If\n";
    std::cout << "it's much faster here, something about the FULL 4-layer network's\n";
    std::cout << "context (thread contention between layers, memory layout, etc)\n";
    std::cout << "is the real cause -- not the layer itself in isolation.\n";

    return 0;
}