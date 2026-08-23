// Granular profiling of the EXACT configuration currently under test:
// 784->512->512->10, ADAMW, mu_cache_threshold=0, forward-projection
// init, batch=256, steps=20. Isolates where the remaining ~12ms/batch
// gap vs ngc-learn (82ms measured vs their 70.31ms) actually lives --
// project_forward specifically, settling loop (per-layer), weight
// updates, or fixed per-region overhead (thread spin-up, virtual
// dispatch) that doesn't show up as "real compute" anywhere.
#include <deepity/networks/SimplePCNetwork.h>
#include <random>
#include <vector>
#include <iostream>
#include <chrono>

using namespace Deep;

int main()
{
    const int BATCH = 256;
    const float LR = 0.001f;
    const int STEPS = 20;
    const int N_BATCHES = 40; // matches the earlier timing test's sample size

    SimplePCNetwork net(BATCH);
    net.AddLayer(784, 512, LR, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(512, 512, LR, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(512, 10, LR, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(10, 0, LR, 0.08f, 0.0001f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.SetOptimizer(OptimizerType::ADAMW);
    net.Compile();

    std::mt19937 rng(7);
    net.RandomizeWeights(rng);

    // Match the real +-0.3 uniform init directly in C++ this time
    std::uniform_real_distribution<float> initDist(-0.3f, 0.3f);
    for (auto *layer : net.GetLayers())
    {
        if (layer->GetOutputSize() == 0) continue; // terminal has no weights
        size_t wsz = layer->GetInputSize() * layer->GetOutputSize();
        float *W = layer->GetWeights();
        for (size_t i = 0; i < wsz; ++i)
            W[i] = initDist(rng);
    }

    net.SetMuCacheThreshold(0.0f);

    std::mt19937 dataRng(123);
    std::uniform_real_distribution<float> dataDist(0.0f, 1.0f);
    std::vector<float> X((size_t)BATCH * 784);
    std::vector<float> Y((size_t)BATCH * 10, 0.001f);
    for (auto &v : X) v = dataDist(dataRng);
    for (int b = 0; b < BATCH; ++b)
        Y[b * 10 + (b % 10)] = 0.999f; // fake one-hot-ish labels, real values don't matter for timing

    // Granular phase timers
    double t_reset = 0, t_clamp_input = 0, t_project = 0, t_clamp_target = 0;
    double t_calc[4] = {0, 0, 0, 0};
    double t_update[4] = {0, 0, 0, 0};
    double t_weights = 0, t_unclamp = 0;

    auto &layers = net.GetLayers();

    auto totalStart = std::chrono::steady_clock::now();

    for (int rep = 0; rep < N_BATCHES; ++rep)
    {
        auto t0 = std::chrono::steady_clock::now();
        net.ResetState();
        auto t1 = std::chrono::steady_clock::now();
        t_reset += std::chrono::duration<double>(t1 - t0).count();

        t0 = t1;
        net.Clamp(X);
        t1 = std::chrono::steady_clock::now();
        t_clamp_input += std::chrono::duration<double>(t1 - t0).count();

        t0 = t1;
        net.ProjectForward();
        t1 = std::chrono::steady_clock::now();
        t_project += std::chrono::duration<double>(t1 - t0).count();

        t0 = t1;
        layers.back()->ClampState(Y);
        t1 = std::chrono::steady_clock::now();
        t_clamp_target += std::chrono::duration<double>(t1 - t0).count();

        for (int step = 0; step < STEPS; ++step)
        {
            for (size_t i = 0; i < layers.size(); ++i)
            {
                t0 = std::chrono::steady_clock::now();
                layers[i]->CalculateState();
                t1 = std::chrono::steady_clock::now();
                t_calc[i] += std::chrono::duration<double>(t1 - t0).count();
            }
            for (size_t i = 0; i < layers.size(); ++i)
            {
                t0 = std::chrono::steady_clock::now();
                layers[i]->UpdateState();
                t1 = std::chrono::steady_clock::now();
                t_update[i] += std::chrono::duration<double>(t1 - t0).count();
            }
        }

        t0 = std::chrono::steady_clock::now();
        net.UpdateWeights();
        t1 = std::chrono::steady_clock::now();
        t_weights += std::chrono::duration<double>(t1 - t0).count();

        t0 = t1;
        layers.back()->UnclampState();
        t1 = std::chrono::steady_clock::now();
        t_unclamp += std::chrono::duration<double>(t1 - t0).count();
    }

    auto totalEnd = std::chrono::steady_clock::now();
    double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();
    double msPerBatch = 1000.0 * totalTime / N_BATCHES;

    double calcSum = 0, updateSum = 0;
    for (int i = 0; i < 4; ++i) { calcSum += t_calc[i]; updateSum += t_update[i]; }

    std::cout << "=== Granular breakdown, " << N_BATCHES << " batches ===\n\n";
    std::cout << "Total: " << totalTime << "s  (" << msPerBatch << " ms/batch)\n";
    std::cout << "ngc-learn reference: 70.31 ms/batch\n\n";

    auto pct = [&](double t) { return 100.0 * t / totalTime; };

    std::cout << "reset_state:      " << 1000*t_reset/N_BATCHES << " ms/batch  (" << pct(t_reset) << "%)\n";
    std::cout << "clamp_input:      " << 1000*t_clamp_input/N_BATCHES << " ms/batch  (" << pct(t_clamp_input) << "%)\n";
    std::cout << "project_forward:  " << 1000*t_project/N_BATCHES << " ms/batch  (" << pct(t_project) << "%)\n";
    std::cout << "clamp_target:     " << 1000*t_clamp_target/N_BATCHES << " ms/batch  (" << pct(t_clamp_target) << "%)\n";
    std::cout << "settling (calc):  " << 1000*calcSum/N_BATCHES << " ms/batch  (" << pct(calcSum) << "%)\n";
    for (int i = 0; i < 4; ++i)
        std::cout << "  layer " << i << " calc: " << 1000*t_calc[i]/N_BATCHES << " ms/batch\n";
    std::cout << "settling (update):" << 1000*updateSum/N_BATCHES << " ms/batch  (" << pct(updateSum) << "%)\n";
    for (int i = 0; i < 4; ++i)
        std::cout << "  layer " << i << " update: " << 1000*t_update[i]/N_BATCHES << " ms/batch\n";
    std::cout << "update_weights:   " << 1000*t_weights/N_BATCHES << " ms/batch  (" << pct(t_weights) << "%)\n";
    std::cout << "unclamp:          " << 1000*t_unclamp/N_BATCHES << " ms/batch  (" << pct(t_unclamp) << "%)\n";

    double accountedFor = t_reset + t_clamp_input + t_project + t_clamp_target + calcSum + updateSum + t_weights + t_unclamp;
    double unaccounted = totalTime - accountedFor;
    std::cout << "\nUnaccounted (measurement overhead, fixed costs not captured above): "
              << 1000*unaccounted/N_BATCHES << " ms/batch  (" << pct(unaccounted) << "%)\n";

    return 0;
}
