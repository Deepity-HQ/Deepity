// Mu-cache threshold sweep. threshold=0 is a sanity re-check that the
// generalization from the boolean flag didn't break the already-validated
// exact L0 behavior. threshold>0 extends caching to UNCLAMPED layers
// (L1, L2) as a genuine approximation -- reports BOTH trajectory
// deviation from the exact baseline AND speedup for each threshold, so
// there's real data to pick a value from rather than a single guess.
#include <deepity/layers/SimplePCLayer.h>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace Deep;

struct Network
{
    std::vector<SimplePCLayer *> layers;

    ~Network()
    {
        for (auto *l : layers)
            delete l;
    }

    void Wire()
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
        {
            layers[i]->SetLayerAbove(layers[i + 1]);
            layers[i + 1]->SetLayerBelow(layers[i]);
        }
    }

    float CalculateState()
    {
        float e = 0.0f;
        for (auto *l : layers)
            e += l->CalculateState();
        return e;
    }

    void UpdateState()
    {
        for (auto *l : layers)
            l->UpdateState();
    }

    void SetMuCacheThreshold(float t)
    {
        for (auto *l : layers)
            l->SetMuCacheThreshold(t);
    }
};

Network BuildNetwork(int batchSize, float lr, std::mt19937 &rng)
{
    Network net;
    net.layers.push_back(new SimplePCLayer(784, 512, batchSize, lr, 0.08f, 0.0001f,
                                           ActivationType::TANH, ActivationType::dTANH));
    net.layers.push_back(new SimplePCLayer(512, 256, batchSize, lr, 0.08f, 0.0001f,
                                           ActivationType::TANH, ActivationType::dTANH));
    net.layers.push_back(new SimplePCLayer(256, 10, batchSize, lr, 0.08f, 0.0001f,
                                           ActivationType::TANH, ActivationType::dTANH));
    net.layers.push_back(new SimplePCLayer(10, 0, batchSize, lr, 0.08f, 0.0001f,
                                           ActivationType::LINEAR, ActivationType::dLINEAR));
    net.Wire();

    for (auto *l : net.layers)
        l->RandomizeWeights(rng);

    return net;
}

// Runs one full settle, returns (final energy, final terminal beliefs,
// worst z-diff seen at every step against a supplied exact-baseline
// trajectory -- pass nullptr to just record the trajectory instead of
// comparing against one).
struct SettleResult
{
    float finalEnergy;
    std::vector<std::vector<float>> zTrajectory; // [step][layer] -> flattened z, only every 10th step + last
};

SettleResult RunSettle(Network &net, const std::vector<float> &X, const std::vector<float> &Y, int steps)
{
    net.layers[0]->ClampState(X);
    net.layers.back()->ClampState(Y);

    SettleResult result;
    float energy = 0.0f;

    for (int step = 0; step < steps; ++step)
    {
        energy = net.CalculateState();
        net.UpdateState();

        if (step % 10 == 0 || step == steps - 1)
        {
            std::vector<float> snapshot;
            for (auto *l : net.layers)
            {
                size_t n = l->GetBatchSize() * l->GetInputSize();
                const float *z = l->GetBeliefs();
                snapshot.insert(snapshot.end(), z, z + n);
            }
            result.zTrajectory.push_back(snapshot);
        }
    }

    result.finalEnergy = energy;
    return result;
}

int main()
{
    const int BATCH = 250;
    const float LR = 0.06f / 15.0f;
    const int STEPS = 60;
    const int N_TIMING_REPS = 15;

    std::mt19937 dataRng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X((size_t)BATCH * 784);
    std::vector<float> Y((size_t)BATCH * 10);
    for (auto &v : X)
        v = dist(dataRng);
    for (auto &v : Y)
        v = dist(dataRng);

    // Exact baseline (threshold=-1, caching fully disabled) -- ground truth
    // to compare every threshold against.
    std::mt19937 seedRng(42);
    Network baseline = BuildNetwork(BATCH, LR, seedRng);
    baseline.SetMuCacheThreshold(-1.0f);
    SettleResult baselineResult = RunSettle(baseline, X, Y, STEPS);

    std::cout << "Baseline (no caching): final energy = " << baselineResult.finalEnergy << "\n\n";

    std::vector<float> thresholds = {-1.0f, 0.0f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f};
    double baselineMsPerStep = 0.0;

    std::cout << std::left;
    std::cout.width(12);
    std::cout << "threshold";
    std::cout.width(16);
    std::cout << "final_energy";
    std::cout.width(16);
    std::cout << "worst_z_diff";
    std::cout.width(14);
    std::cout << "ms/step";
    std::cout << "speedup%\n";

    for (size_t idx = 0; idx < thresholds.size(); ++idx)
    {
        float threshold = thresholds[idx];
        std::mt19937 seedRng2(42); // SAME seed -- identical initial weights every time
        Network net = BuildNetwork(BATCH, LR, seedRng2);
        net.SetMuCacheThreshold(threshold);

        SettleResult result = RunSettle(net, X, Y, STEPS);

        float worstDiff = 0.0f;
        for (size_t s = 0; s < result.zTrajectory.size(); ++s)
        {
            for (size_t i = 0; i < result.zTrajectory[s].size(); ++i)
            {
                float d = std::abs(result.zTrajectory[s][i] - baselineResult.zTrajectory[s][i]);
                worstDiff = std::max(worstDiff, d);
            }
        }

        // Timing (separate network instance, same threshold, many reps)
        std::mt19937 seedRng3(42);
        Network timingNet = BuildNetwork(BATCH, LR, seedRng3);
        timingNet.SetMuCacheThreshold(threshold);

        auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < N_TIMING_REPS; ++rep)
        {
            for (auto *l : timingNet.layers)
                l->ResetState();
            RunSettle(timingNet, X, Y, STEPS);
        }
        auto end = std::chrono::steady_clock::now();
        double totalTime = std::chrono::duration<double>(end - start).count();
        double msPerStep = 1000.0 * totalTime / (N_TIMING_REPS * STEPS);

        if (idx == 0)
            baselineMsPerStep = msPerStep; // threshold=-1 row IS the baseline

        double speedup = 100.0 * (baselineMsPerStep - msPerStep) / baselineMsPerStep;

        std::string thresholdLabel = (threshold < 0.0f) ? "disabled" : std::to_string(threshold);
        std::cout.width(12);
        std::cout << thresholdLabel;
        std::cout.width(16);
        std::cout << result.finalEnergy;
        std::cout.width(16);
        std::cout << worstDiff;
        std::cout.width(14);
        std::cout << msPerStep;
        std::cout << speedup << "\n";
    }

    std::cout << "\nworst_z_diff should be ~0 for threshold=0 (sanity re-check of the\n";
    std::cout << "L0-only refactor). For threshold>0, watch how worst_z_diff and\n";
    std::cout << "speedup% trade off -- that curve is the real decision to make, not\n";
    std::cout << "a single pass/fail.\n";

    return 0;
}