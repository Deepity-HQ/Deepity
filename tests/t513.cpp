// Confirms the cache-aliasing theory directly: SAME isolated-layer test
// as tLayer1Isolate.cpp, but at size=513 instead of 512 -- a single
// dimension off from a power of two. If the ~1.5ms/call slowness was
// genuinely caused by 512's power-of-two stride aliasing in the CPU
// cache, 513 should show DRAMATICALLY better throughput despite being
// virtually the same amount of actual work (~0.4% more FLOPs).
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
    const int SIZE = 513; // ONLY difference from tLayer1Isolate.cpp: 512 -> 513

    SimplePCLayer below(SIZE, SIZE, BATCH, 0.001f, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    SimplePCLayer layer1(SIZE, SIZE, BATCH, 0.001f, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    SimplePCLayer above(SIZE, 10, BATCH, 0.001f, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);

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
        if (outSz == 0) return;
        float *W = l.GetWeights();
        for (size_t i = 0; i < inSz * outSz; ++i) W[i] = initDist(rng);
    };
    reinit(below, SIZE, SIZE);
    reinit(layer1, SIZE, SIZE);
    reinit(above, SIZE, 10);

    std::vector<float> belowInput((size_t)BATCH * SIZE);
    std::mt19937 dataRng(123);
    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    for (auto &v : belowInput) v = dataDist(dataRng);

    below.ClampState(belowInput);

    double t_calc = 0, t_update = 0;
    auto totalStart = std::chrono::steady_clock::now();

    for (int rep = 0; rep < N_REPS; ++rep)
    {
        below.CalculateState();

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

    std::cout << "=== Isolated " << SIZE << "->" << SIZE << " layer (non-power-of-2), "
              << N_REPS << " reps x " << STEPS << " steps ===\n\n";
    std::cout << "Total: " << totalTime << "s\n";
    std::cout << "calculate_state: " << 1000*t_calc/(N_REPS*STEPS) << " ms/call\n";
    std::cout << "update_state:    " << 1000*t_update/(N_REPS*STEPS) << " ms/call\n\n";
    std::cout << "Compare against size=512's measured: calc=1.670ms, update=1.487ms\n";
    std::cout << "If 513 is DRAMATICALLY faster despite ~0.4% MORE actual work,\n";
    std::cout << "that confirms cache-associativity aliasing at the 512 power-of-2\n";
    std::cout << "stride as the real cause -- a well-known, well-understood HPC\n";
    std::cout << "pathology with a simple, standard fix (pad the leading dimension).\n";

    return 0;
}
