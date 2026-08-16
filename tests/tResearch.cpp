// Decisive, unambiguous demonstration: can this library reach ~100%
// accuracy on a clearly-separable task? Fixes two real issues from the
// last investigation:
//   1. Dataset now uses a FIXED seed (was random_device-seeded, meaning
//      every run got a genuinely different, luck-dependent task).
//   2. Cluster separation increased, noise decreased -- unambiguously
//      easy, not "should be easy but wasn't checked."
//   3. Evaluated on the FULL held-out test set (all batches), not one
//      noisy shuffled 100-sample batch -- removes evaluation noise from
//      the accuracy readout entirely.

#include "SimplePCNetwork.h"
#include "StreamAlignedBatcher.h"
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

constexpr int N_CLASSES = 4;
constexpr int INPUT_DIM = 16;
constexpr int HIDDEN_DIM = 32;
constexpr int SAMPLES_PER_CLASS = 1000; // more samples -- more stable statistics
constexpr int PER_CLASS_BATCH = 25;
constexpr int STEPS = 60;
constexpr int EPOCHS = 30;
constexpr float LR = 0.02f; // showed the cleanest monotonic energy descent last run
constexpr float DECAY_RATE = 0.98f;
constexpr unsigned DATA_SEED = 1234; // FIXED -- was random_device before, meaning every
                                      // run got a genuinely different, luck-dependent task

struct SyntheticDataset
{
    std::vector<float> X;
    std::vector<float> Y;
    std::vector<int> labels;
    size_t numSamples;
};

SyntheticDataset GenerateSamplesFromCenters(std::mt19937 &rng, const std::vector<std::vector<float>> &centers)
{
    SyntheticDataset ds;
    ds.numSamples = (size_t)N_CLASSES * SAMPLES_PER_CLASS;
    ds.X.resize(ds.numSamples * INPUT_DIM);
    ds.Y.resize(ds.numSamples * N_CLASSES, -0.9f);
    ds.labels.resize(ds.numSamples);

    std::normal_distribution<float> noise(0.0f, 0.15f);
    size_t row = 0;
    for (int c = 0; c < N_CLASSES; ++c)
        for (int s = 0; s < SAMPLES_PER_CLASS; ++s)
        {
            for (int d = 0; d < INPUT_DIM; ++d)
                ds.X[row * INPUT_DIM + d] = centers[c][d] + noise(rng);
            ds.Y[row * N_CLASSES + c] = 0.9f;
            ds.labels[row] = c;
            ++row;
        }

    for (int d = 0; d < INPUT_DIM; ++d)
    {
        float minV = ds.X[d], maxV = ds.X[d];
        for (size_t i = 0; i < ds.numSamples; ++i)
        {
            float v = ds.X[i * INPUT_DIM + d];
            minV = std::min(minV, v);
            maxV = std::max(maxV, v);
        }
        float range = maxV - minV;
        if (range < 1e-6f) range = 1.0f;
        for (size_t i = 0; i < ds.numSamples; ++i)
            ds.X[i * INPUT_DIM + d] = 2.0f * (ds.X[i * INPUT_DIM + d] - minV) / range - 1.0f;
    }
    return ds;
}

int main()
{
    std::mt19937 dataRng(DATA_SEED); // FIXED seed -- reproducible across runs

    // Generate cluster centers ONCE, shared between train and test --
    // only the noise differs between the two draws now, not the
    // underlying class locations (the actual bug: calling the old
    // GenerateSyntheticData twice gave train and test COMPLETELY
    // UNRELATED random centers per class, guaranteeing generalization
    // failure regardless of whether the library itself worked at all).
    std::vector<std::vector<float>> centers(N_CLASSES, std::vector<float>(INPUT_DIM));
    std::uniform_real_distribution<float> centerDist(-5.0f, 5.0f);
    for (int c = 0; c < N_CLASSES; ++c)
        for (int d = 0; d < INPUT_DIM; ++d)
            centers[c][d] = centerDist(dataRng);

    SyntheticDataset train = GenerateSamplesFromCenters(dataRng, centers);
    SyntheticDataset test = GenerateSamplesFromCenters(dataRng, centers);

    const int batchSize = N_CLASSES * PER_CLASS_BATCH;

    std::mt19937 netRng(42);
    Deep::SimplePCNetwork net(batchSize);
    net.AddLayer(INPUT_DIM, HIDDEN_DIM, LR, 0.08f, 0.0001f, Deep::ActivationType::TANH, Deep::ActivationType::dTANH);
    net.AddLayer(HIDDEN_DIM, N_CLASSES, LR, 0.08f, 0.0001f, Deep::ActivationType::TANH, Deep::ActivationType::dTANH);
    net.AddLayer(N_CLASSES, 0, LR, 0.08f, 0.0001f, Deep::ActivationType::LINEAR, Deep::ActivationType::dLINEAR);

    for (auto *layer : net.GetLayers())
        layer->SetOptimizer(Deep::OptimizerType::SGD);

    net.Compile();
    net.RandomizeWeights(netRng);

    Deep::StreamAlignedBatcher batcher(
        train.X.data(), train.Y.data(), train.labels.data(),
        train.numSamples, INPUT_DIM, N_CLASSES, N_CLASSES, PER_CLASS_BATCH, 42);

    size_t nBatches = batcher.NumBatchesPerEpoch();
    std::vector<float> X_batch(batchSize * INPUT_DIM), Y_batch(batchSize * N_CLASSES);
    std::vector<int> labels_batch(batchSize);

    std::cout << "Clean demo: " << EPOCHS << " epochs, " << nBatches << " batches/epoch, "
              << "lr=" << LR << ", steps=" << STEPS << ", fixed data seed=" << DATA_SEED << "\n\n";

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        float currentLr = LR * std::pow(DECAY_RATE, (float)epoch);
        for (auto *layer : net.GetLayers())
            layer->SetLearningRate(currentLr);

        float lastEnergy = 0.0f;
        for (size_t b = 0; b < nBatches; ++b)
        {
            batcher.GetBatch(X_batch.data(), Y_batch.data(), labels_batch.data());
            lastEnergy = net.TrainStep(X_batch, Y_batch, STEPS);
        }

        // FULL held-out test set evaluation -- every sample, not one
        // noisy shuffled batch.
        int correct = 0, total = 0;
        for (size_t i = 0; i + batchSize <= test.numSamples; i += batchSize)
        {
            std::vector<float> X_eval(test.X.begin() + i * INPUT_DIM,
                                       test.X.begin() + (i + batchSize) * INPUT_DIM);
            net.ResetState();
            net.Clamp(X_eval);
            for (int t = 0; t < 200; ++t)
            {
                net.CalculateState();
                net.UpdateState();
            }
            const float *pred = net.GetTerminalLayer()->GetBeliefs();

            // DEBUG: print raw prediction values for the first 5 examples
            // of the first batch, final epoch only -- directly reveals
            // whether the terminal's free-running eval settling produces
            // NaN/extreme values (instability) or plausible-but-wrong
            // numbers (a different kind of bug).
            if (epoch == EPOCHS - 1 && i == 0)
            {
                std::cout << "\n  DEBUG: raw predictions, first 5 eval examples:\n";
                for (int r = 0; r < 5; ++r)
                {
                    std::cout << "    sample " << r << " (true label=" << test.labels[i + r] << "): [";
                    for (int c = 0; c < N_CLASSES; ++c)
                        std::cout << pred[r * N_CLASSES + c] << (c < N_CLASSES - 1 ? ", " : "");
                    std::cout << "]\n";
                }
                std::cout << "\n";
            }

            for (int r = 0; r < batchSize; ++r)
            {
                int argmax = 0;
                float best = pred[r * N_CLASSES];
                for (int c = 1; c < N_CLASSES; ++c)
                    if (pred[r * N_CLASSES + c] > best) { best = pred[r * N_CLASSES + c]; argmax = c; }
                if (argmax == test.labels[i + r])
                    ++correct;
                ++total;
            }
        }
        float acc = 100.0f * correct / total;

        std::cout << "  epoch " << epoch << "  lr=" << currentLr
                  << "  train_energy=" << lastEnergy
                  << "  FULL_test_acc=" << acc << "%  (" << correct << "/" << total << ")\n";
    }

    return 0;
}
