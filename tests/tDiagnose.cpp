// tDiagnose.cpp
//
// Diagnostic harness for DiscriminativePCNetwork, aimed at answering:
// "why is accuracy capped, and is it the algorithm or the task?"
//
// Strategy: train on a SYNTHETIC classification task with a known, easy
// answer (well-separated Gaussian blobs) instead of MNIST. If this network
// can't get near 100% on an easy task, the shortfall is architectural/
// algorithmic, not MNIST-specific difficulty. We also directly inspect
// per-unit precision and activation variance, instead of reading one
// aggregate energy number.
//
// Runs two conditions back-to-back with identical seed/data: precision OFF
// (pr=0) and precision ON (pr>0), so we can see directly whether precision
// helps, hurts, or is neutral for actual classification accuracy.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "DiscriminativePCNetwork.h"

using namespace Deep;

// ---------------------------------------------------------------------------
// Synthetic dataset: N_CLASSES well-separated Gaussian blobs in INPUT_DIM
// dimensions. A linear classifier can solve this near-perfectly, so this is
// a floor test: if the PCN can't get close to 100%, something is wrong
// independent of MNIST's real difficulty.
//
// NOTE: dimensions padded to multiples of 16 (N_CLASSES=16, BATCH_SIZE=48)
// to avoid a MemoryArena capacity-vs-alignment-padding mismatch -- see
// CONTRIBUTING.md, rule 5 in "Numerical / algorithmic gotchas", and the
// Compile() ordering note.
// ---------------------------------------------------------------------------

constexpr int N_TRAIN = 500;
constexpr int N_TEST = 200;
constexpr int INPUT_DIM = 32;  // multiple of 16
constexpr int N_CLASSES = 16;  // padded from 5 -- extra classes unused
constexpr int HIDDEN = 64;     // multiple of 16
constexpr int BATCH_SIZE = 48; // multiple of 16

struct Dataset
{
    std::vector<float> X; // N * INPUT_DIM, row-major, tanh-range
    std::vector<float> Y; // N * N_CLASSES, soft one-hot
    std::vector<int> labels;
    int n;
};

Dataset MakeBlobs(int n, std::mt19937 &rng)
{
    Dataset d;
    d.n = n;
    d.X.resize((size_t)n * INPUT_DIM);
    d.Y.resize((size_t)n * N_CLASSES);
    d.labels.resize(n);

    // Fixed, well-separated class centers (unit-ish vectors, spread out)
    static std::vector<std::vector<float>> centers;
    if (centers.empty())
    {
        std::mt19937 centerRng(7);
        std::uniform_real_distribution<float> u(-0.8f, 0.8f);
        for (int c = 0; c < N_CLASSES; c++)
        {
            std::vector<float> center(INPUT_DIM);
            for (float &v : center)
                v = u(centerRng);
            centers.push_back(center);
        }
    }

    std::normal_distribution<float> noise(0.0f, 0.15f); // small noise, easy separation
    std::uniform_int_distribution<int> classPick(0, N_CLASSES - 1);

    for (int i = 0; i < n; i++)
    {
        int cls = classPick(rng);
        d.labels[i] = cls;
        for (int j = 0; j < INPUT_DIM; j++)
        {
            float v = centers[cls][j] + noise(rng);
            d.X[(size_t)i * INPUT_DIM + j] = std::clamp(v, -1.0f, 1.0f);
        }
        for (int c = 0; c < N_CLASSES; c++)
            d.Y[(size_t)i * N_CLASSES + c] = (c == cls) ? 0.9f : -0.9f;
    }
    return d;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<float> Slice(const std::vector<float> &flat, int stride, int start, int count)
{
    return std::vector<float>(flat.begin() + (size_t)start * stride,
                               flat.begin() + (size_t)(start + count) * stride);
}

// Precision distribution stats for one layer: mean/stddev/min/max of log(p).
// p = exp(log_p) exactly, so log(p[i]) reconstructs log_p without needing a
// raw accessor.
void ReportPrecisionStats(const std::string &name, DiscriminativePCLayer *layer, int size)
{
    const float *p = layer->GetPrecisions();
    std::vector<float> logp(size);
    for (int i = 0; i < size; i++)
        logp[i] = std::log(MAX(p[i], 1e-8f));

    float mean = std::accumulate(logp.begin(), logp.end(), 0.0f) / size;
    float sqsum = 0.0f;
    for (float v : logp)
        sqsum += (v - mean) * (v - mean);
    float stddev = std::sqrt(sqsum / size);
    auto [lo, hi] = std::minmax_element(logp.begin(), logp.end());

    std::cout << "  [" << name << "] log(p): mean=" << mean << " stddev=" << stddev
               << " min=" << *lo << " max=" << *hi << "\n";
}

// Dead-unit check: for the hidden layer, how many units have near-zero
// variance in their belief (z) across the batch -- i.e. respond almost
// identically regardless of which input was presented. A healthy hidden
// layer should have most units varying meaningfully across a diverse batch.
void ReportDeadUnits(const std::string &name, DiscriminativePCLayer *layer, int size, int batchSize)
{
    const float *z = layer->GetBeliefs(); // batchSize x size, row-major
    int dead = 0;
    for (int unit = 0; unit < size; unit++)
    {
        float mean = 0.0f;
        for (int b = 0; b < batchSize; b++)
            mean += z[(size_t)b * size + unit];
        mean /= batchSize;

        float var = 0.0f;
        for (int b = 0; b < batchSize; b++)
        {
            float d = z[(size_t)b * size + unit] - mean;
            var += d * d;
        }
        var /= batchSize;

        if (var < 1e-6f)
            dead++;
    }
    std::cout << "  [" << name << "] dead/near-constant units: " << dead << " / " << size << "\n";
}

// Trains + evaluates one configuration, returns test accuracy.
float RunCondition(const std::string &label, float pr, int epochs,
                    const Dataset &train, const Dataset &test)
{
    std::cout << "\n=== " << label << " (pr=" << pr << ") ===\n";

    std::mt19937 mt(42); // same seed across conditions -- isolates pr's effect
    DiscriminativePCNetwork net(BATCH_SIZE);
    net.AddLayer(INPUT_DIM, HIDDEN, 0.02f, 0.2f, pr, 0.0001f, tanh, dTanh);
    net.AddLayer(HIDDEN, N_CLASSES, 0.02f, 0.2f, pr, 0.0001f, tanh, dTanh);
    net.AddLayer(N_CLASSES, 0, 0.02f, 0.2f, pr, 0.0001f, tanh, dTanh);
    net.Compile();
    net.RandomizeWeights(mt);

    auto &layers = net.GetLayers();
    auto *hidden = static_cast<DiscriminativePCLayer *>(layers[1]);
    int hiddenSize = (int)hidden->GetInputSize();

    int inferenceSteps = 30;
    int nBatches = train.n / BATCH_SIZE;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int b = 0; b < nBatches; b++)
        {
            auto xb = Slice(train.X, INPUT_DIM, b * BATCH_SIZE, BATCH_SIZE);
            auto yb = Slice(train.Y, N_CLASSES, b * BATCH_SIZE, BATCH_SIZE);

            net.ResetState();
            net.Clamp(xb);
            net.GetTerminalLayer()->ClampState(yb);

            for (int t = 0; t < inferenceSteps; t++)
            {
                net.CalculateState();
                net.UpdateState();
            }

            net.UpdateWeights();
            if (pr > 0.0f)
                net.UpdatePrecision();

            net.GetTerminalLayer()->UnclampState();
        }
    }

    // Diagnostics on final state (post-training, using last training batch's
    // settled activity for the dead-unit check).
    ReportPrecisionStats("hidden", hidden, hiddenSize);
    ReportDeadUnits("hidden", hidden, hiddenSize, BATCH_SIZE);

    // Evaluation on held-out synthetic test set
    int correct = 0;
    int nTestBatches = test.n / BATCH_SIZE;
    for (int b = 0; b < nTestBatches; b++)
    {
        auto xb = Slice(test.X, INPUT_DIM, b * BATCH_SIZE, BATCH_SIZE);

        net.ResetState();
        net.Clamp(xb);
        for (int t = 0; t < inferenceSteps; t++)
        {
            net.CalculateState();
            net.UpdateState();
        }

        const float *out = net.GetTerminalLayer()->GetBeliefs(); // BATCH_SIZE x N_CLASSES
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            int predicted = 0;
            float best = -1e9f;
            for (int c = 0; c < N_CLASSES; c++)
            {
                float v = out[(size_t)i * N_CLASSES + c];
                if (v > best) { best = v; predicted = c; }
            }
            int trueLabel = test.labels[b * BATCH_SIZE + i];
            if (predicted == trueLabel)
                correct++;
        }
    }

    float acc = 100.0f * correct / (nTestBatches * BATCH_SIZE);
    std::cout << "  Held-out test accuracy: " << correct << "/" << (nTestBatches * BATCH_SIZE)
               << " (" << std::fixed << std::setprecision(2) << acc << "%)\n";
    std::cout << std::defaultfloat << std::setprecision(6); // reset
    return acc;
}

int main(void)
{
    std::mt19937 dataRng(123);
    Dataset train = MakeBlobs(N_TRAIN, dataRng);
    Dataset test = MakeBlobs(N_TEST, dataRng);

    std::cout << "Synthetic task: " << N_CLASSES << " well-separated Gaussian blobs, "
               << INPUT_DIM << "-dim, low noise.\n";
    std::cout << "A working classifier should reach ~95-100% here. If this network\n";
    std::cout << "can't, the shortfall is architectural, not MNIST-specific.\n";

    int epochs = 40;

    float accNoPrecision = RunCondition("Precision OFF", 0.0f, epochs, train, test);
    float accWithPrecision = RunCondition("Precision ON", 0.001f, epochs, train, test);

    std::cout << "\n========================================\n";
    std::cout << "Precision OFF: " << accNoPrecision << "%\n";
    std::cout << "Precision ON:  " << accWithPrecision << "%\n";
    std::cout << "========================================\n";

    if (accNoPrecision < 85.0f && accWithPrecision < 85.0f)
        std::cout << "\nBoth conditions underperform on an EASY synthetic task.\n"
                   << "-> Shortfall is likely architectural/algorithmic, not data-specific.\n"
                   << "   Check dead-unit counts above; if high, capacity or inference\n"
                   << "   steps/rates are the next thing to investigate.\n";
    else if (std::abs(accNoPrecision - accWithPrecision) > 5.0f)
        std::cout << "\nPrecision measurably changes accuracy on this easy task.\n"
                   << "-> Precision weighting has a real effect on learning dynamics here,\n"
                   << "   worth tuning 'pr' deliberately rather than treating it as inert.\n";
    else
        std::cout << "\nBoth conditions solve the easy task well and similarly.\n"
                   << "-> The algorithm itself is sound at this scale; MNIST's accuracy\n"
                   << "   ceiling is more likely about capacity/training budget/steps at\n"
                   << "   MNIST's larger scale, not a lurking correctness bug.\n";

    return 0;
}