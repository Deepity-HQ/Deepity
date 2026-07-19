#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include "DiscriminativePCNetwork.h"
#include "Activations.h"
#include "Timer.h"

float WeightNorm(Deep::DiscriminativePCLayer *layer, size_t count)
{
    const float *w = layer->GetWeights();
    float sumSq = 0.0f;
    for (size_t i = 0; i < count; ++i)
        sumSq += w[i] * w[i];
    return std::sqrt(sumSq);
}

int main(void)
{
    Deep::DiscriminativePCNetwork net(4); // batchSize=4, all 4 XOR examples trained together
    Timer timer;

    net.AddLayer(2, 8, 0.05f, 0.3f, 0.000f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(8, 1, 0.05f, 0.3f, 0.000f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(1, 0, 0.05f, 0.3f, 0.000f, 0.0001f, Deep::linear, Deep::dLinear);

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    auto &layers = net.GetLayers();
    auto *layer0 = static_cast<Deep::DiscriminativePCLayer *>(layers[0]);
    auto *layer1 = static_cast<Deep::DiscriminativePCLayer *>(layers[1]);
    float initialW0Norm = WeightNorm(layer0, 2 * 8);

    // Flattened, row-major: 4 examples x 2 inputs / x 1 target
    std::vector<float> flatX = {
        -1.0f, -1.0f,
        -1.0f, +1.0f,
        +1.0f, -1.0f,
        +1.0f, +1.0f};
    std::vector<float> flatY = {
        -1.0f,
        +1.0f,
        +1.0f,
        -1.0f};

    int epochs = 5000;
    int inferenceSteps = 50;
    int reportEvery = 100; // dense enough to catch a cliff, not just its aftermath

    std::cout << "Starting Discriminative PC XOR Test (batched, instrumented)...\n";

    double start = timer.elapsed();
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        net.ResetState();
        net.Clamp(flatX);
        net.GetTerminalLayer()->ClampState(flatY);

        float energy = 0.0f;
        for (int t = 0; t < inferenceSteps; t++)
        {
            energy = net.CalculateState();
            net.UpdateState();
        }

        net.UpdateWeights();
        net.GetTerminalLayer()->UnclampState();

        if (epoch % reportEvery == 0)
        {
            float w0norm = WeightNorm(layer0, 2 * 8);
            float w1norm = WeightNorm(layer1, 8 * 1);

            std::cout << "Epoch " << std::setw(5) << epoch
                       << " | Energy: " << std::fixed << std::setprecision(4) << energy / 4.0f
                       << " | W0 norm: " << w0norm
                       << " | W1 norm: " << w1norm
                       << " | Elapsed: " << timer.elapsed() - start << "s\n";
            start = timer.elapsed();

            if (w0norm > 10.0f * initialW0Norm && w0norm > 20.0f)
            {
                std::cout << "  *** WARNING: W0 norm jumped far beyond init scale ("
                           << initialW0Norm << " -> " << w0norm << "). "
                           << "Possible instability/cliff at this epoch. ***\n";
            }
        }
    }

    // Testing -- one batched forward pass, read out all 4 rows
    std::cout << "\n=== Predictions ===\n";

    net.ResetState();
    net.Clamp(flatX);

    for (int t = 0; t < inferenceSteps; t++)
    {
        net.CalculateState();
        net.UpdateState();
    }

    float *beliefs = net.GetTerminalLayer()->GetBeliefs();
    int correct = 0;

    for (int i = 0; i < 4; i++)
    {
        float pred = beliefs[i];
        float target = flatY[i];
        bool signCorrect = (pred > 0 && target > 0) || (pred < 0 && target < 0);
        if (signCorrect) correct++;

        std::cout << "Input: [" << std::setw(2) << flatX[i * 2] << ", " << std::setw(2) << flatX[i * 2 + 1] << "]"
                   << " | Target: " << std::showpos << target
                   << " | Pred: " << std::noshowpos << std::setprecision(4) << pred
                   << (signCorrect ? " OK" : " WRONG") << "\n";
    }

    std::cout << "\nAccuracy: " << correct << "/4\n";
    std::cout << (correct == 4 ? "XOR LEARNED SUCCESSFULLY!" : "XOR NOT FULLY LEARNED.") << "\n";

    return 0;
}