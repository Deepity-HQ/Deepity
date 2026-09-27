// tConvDiagnose.cpp
//
// Synthetic floor test for ConvPCNetwork, mirroring tDiagnose.cpp's
// approach for the discriminative network: a task with a KNOWN, easy
// answer, so if the network can't solve it, the shortfall is architectural,
// not "the real dataset is just hard."
//
// Unlike tDiagnose.cpp's flat Gaussian blobs, this task has genuine SPATIAL
// structure (a class-specific patch placed at a distinct location in a
// small image) -- something a flat/dense network could still solve by
// memorizing pixel positions, but that specifically exercises convolution's
// actual mechanism (Im2Col/Col2Im, spatial weight sharing) rather than just
// being a relabeled flat-vector problem.
//
// Architecture collapses spatial dims to 1x1 by the second conv layer, so
// the terminal layer's flattened size is exactly N_CLASSES -- same
// one-hot/argmax readout pattern used in every earlier gate/blob test.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <deepity/networks/ConvPCNetwork.h>

using namespace Deep;

constexpr int IMG = 12; // input image is IMG x IMG, 1 channel
constexpr int N_CLASSES = 5;
constexpr int N_TRAIN = 400;
constexpr int N_TEST = 100;
constexpr int BATCH_SIZE = 25;

struct Dataset
{
    std::vector<float> X; // N * 1 * IMG * IMG, row-major, tanh-range
    std::vector<float> Y; // N * N_CLASSES, soft one-hot
    std::vector<int> labels;
    int n;
};

Dataset MakeSpatialBlobs(int n, std::mt19937 &rng)
{
    Dataset d;
    d.n = n;
    d.X.assign((size_t)n * IMG * IMG, -0.8f); // background
    d.Y.resize((size_t)n * N_CLASSES);
    d.labels.resize(n);

    // Fixed, well-separated patch locations per class -- spread across the
    // image so classes are trivially distinguishable BY POSITION, which is
    // exactly what convolution's spatial structure should exploit.
    static std::vector<std::pair<int, int>> patchOrigin;
    if (patchOrigin.empty())
    {
        // 5 classes -> 5 non-overlapping 3x3 patch locations in a 12x12 image.
        patchOrigin = {{1, 1}, {1, 8}, {8, 1}, {8, 8}, {4, 4}};
    }

    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_int_distribution<int> classPick(0, N_CLASSES - 1);

    for (int i = 0; i < n; i++)
    {
        int cls = classPick(rng);
        d.labels[i] = cls;

        float *img = d.X.data() + (size_t)i * IMG * IMG;
        for (int p = 0; p < IMG * IMG; p++)
            img[p] = std::clamp(-0.8f + noise(rng), -1.0f, 1.0f);

        auto [r0, c0] = patchOrigin[cls];
        for (int dr = 0; dr < 3; dr++)
            for (int dc = 0; dc < 3; dc++)
                img[(r0 + dr) * IMG + (c0 + dc)] = std::clamp(0.9f + noise(rng), -1.0f, 1.0f);

        for (int c = 0; c < N_CLASSES; c++)
            d.Y[(size_t)i * N_CLASSES + c] = (c == cls) ? 0.9f : -0.9f;
    }
    return d;
}

std::vector<float> SliceImages(const std::vector<float> &flat, int start, int count)
{
    return std::vector<float>(flat.begin() + (size_t)start * IMG * IMG,
                              flat.begin() + (size_t)(start + count) * IMG * IMG);
}

std::vector<float> SliceLabels(const std::vector<float> &flat, int start, int count)
{
    return std::vector<float>(flat.begin() + (size_t)start * N_CLASSES,
                              flat.begin() + (size_t)(start + count) * N_CLASSES);
}

int main()
{
    std::mt19937 dataRng(123);
    Dataset train = MakeSpatialBlobs(N_TRAIN, dataRng);
    Dataset test = MakeSpatialBlobs(N_TEST, dataRng);

    std::cout << "Synthetic task: " << N_CLASSES << " classes, each a distinct 3x3 patch\n";
    std::cout << "location in a " << IMG << "x" << IMG << " image (spatially separable).\n";
    std::cout << "A working ConvPCNetwork should reach ~90-100% here.\n\n";

    std::mt19937 netRng(42);
    ConvPCNetwork net(BATCH_SIZE);

    // Layer 0: 1x12x12 -> conv 5x5 stride2 -> 8 channels, 4x4
    net.AddLayer(1, 8, IMG, IMG, 5, 5, 2, 2, 0, 0,
                 0.02f, 0.2f, 0.0f, 0.0001f,
                 ActivationType::TANH, ActivationType::dTANH);

    // Layer 1: 8x4x4 -> conv 4x4 stride1 -> N_CLASSES channels, 1x1
    // (collapses all remaining spatial extent -- effectively a
    // fully-connected layer expressed as a convolution)
    net.AddLayer(8, N_CLASSES, 4, 4, 4, 4, 1, 1, 0, 0,
                 0.02f, 0.2f, 0.0f, 0.0001f,
                 ActivationType::TANH, ActivationType::dTANH);

    // Layer 2 (terminal): N_CLASSES x 1 x 1 -> flattened size = N_CLASSES exactly
    net.AddLayer(N_CLASSES, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                 0.02f, 0.2f, 0.0f, 0.0001f,
                 ActivationType::LINEAR, ActivationType::dLINEAR);

    net.Compile();
    net.RandomizeWeights(netRng);

    const int INFERENCE_STEPS = 30;
    const int EPOCHS = 40;
    const int nBatches = N_TRAIN / BATCH_SIZE;

    std::cout << "Training...\n";
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float epochEnergy = 0.0f;
        for (int b = 0; b < nBatches; b++)
        {
            auto xb = SliceImages(train.X, b * BATCH_SIZE, BATCH_SIZE);
            auto yb = SliceLabels(train.Y, b * BATCH_SIZE, BATCH_SIZE);
            epochEnergy += net.TrainStep(xb, yb, INFERENCE_STEPS);
        }
        if (epoch % 5 == 0 || epoch == EPOCHS - 1)
            std::cout << "  Epoch " << std::setw(3) << epoch
                      << "  avg energy=" << std::fixed << std::setprecision(4)
                      << (epochEnergy / nBatches) << "\n";
    }

    std::cout << "\nEvaluating on held-out test set...\n";
    int correct = 0;
    int nTestBatches = N_TEST / BATCH_SIZE;
    for (int b = 0; b < nTestBatches; b++)
    {
        auto xb = SliceImages(test.X, b * BATCH_SIZE, BATCH_SIZE);
        auto preds = net.Predict(xb, INFERENCE_STEPS); // BATCH_SIZE x N_CLASSES flattened

        for (int i = 0; i < BATCH_SIZE; i++)
        {
            int predicted = 0;
            float best = -1e9f;
            for (int c = 0; c < N_CLASSES; c++)
            {
                float v = preds[(size_t)i * N_CLASSES + c];
                if (v > best)
                {
                    best = v;
                    predicted = c;
                }
            }
            int trueLabel = test.labels[b * BATCH_SIZE + i];
            if (predicted == trueLabel)
                correct++;
        }
    }

    float acc = 100.0f * correct / (nTestBatches * BATCH_SIZE);
    std::cout << "Held-out test accuracy: " << correct << "/" << (nTestBatches * BATCH_SIZE)
              << " (" << std::fixed << std::setprecision(2) << acc << "%)\n\n";

    if (acc >= 90.0f)
        std::cout << "PASS -- ConvPCNetwork solves an easy, spatially-separable task.\n"
                  << "The architecture and Im2Col/Col2Im wiring are sound end-to-end.\n";
    else if (acc >= 50.0f)
        std::cout << "PARTIAL -- better than chance (" << (100.0f / N_CLASSES)
                  << "% baseline) but not solving cleanly.\n"
                  << "Worth checking epochs/inference steps/learning rate before\n"
                  << "suspecting a correctness bug -- gradient checks already passed.\n";
    else
        std::cout << "FAIL -- at or near chance level. Something is wrong beyond\n"
                  << "hyperparameters; worth re-checking the network wiring\n"
                  << "(layer shapes, AddLayer order) before MNIST.\n";

    return acc >= 90.0f ? 0 : 1;
}
