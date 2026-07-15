#include <iostream>
#include <vector>
#include "PCNetwork.h"

using namespace Deep;

// Trains on a single (input, target) pair for a fixed number of iterations,
// interleaving CalculateState -> UpdateState -> UpdateWeights each step,
// matching the pattern used in tNetwork.cpp's benchmark loop.
//
// Returns the final energy and prints the learned output vs target so we
// can see convergence directly, without any Python/pybind11 involved.
float TrainOnExample(PCNetwork &net,
                      const std::vector<float> &input,
                      const std::vector<float> &target,
                      int iterations)
{
    net.Clamp(input);
    auto &layers = net.GetLayers();
    PCLayer *outputLayer = static_cast<PCLayer *>(layers.back());
    outputLayer->ClampState(target);

    float energy = 0.0f;
    for (int i = 0; i < iterations; ++i)
    {
        energy = net.CalculateState();
        net.UpdateState();          // settle only — no weight update yet
    }

    net.UpdateWeights();            // single update, after settling
    energy = net.CalculateState();  // re-measure post-update

    outputLayer->UnclampState();
    return energy;
}

// Reads out a prediction by clamping only the input and letting the
// terminal layer settle freely (not clamped to any target).
std::vector<float> Predict(PCNetwork &net,
                           const std::vector<float> &input,
                           int iterations)
{
    net.Clamp(input);

    auto &layers = net.GetLayers();
    PCLayer *outputLayer = static_cast<PCLayer *>(layers.back());

    for (int i = 0; i < iterations; ++i)
    {
        net.CalculateState();
        net.UpdateState();
    }

    float *beliefs = outputLayer->GetBeliefs();
    return std::vector<float>(beliefs, beliefs + outputLayer->GetInputSize());
}

int main()
{
    constexpr float INFERENCE_RATE = 0.01f;
    constexpr float LEARNING_RATE = 1e-5;
    constexpr int BATCH_SIZE = 1;
    constexpr int INFERENCE_STEPS = 50;
    constexpr int EPOCHS = 20000;

    std::mt19937 mt(42);

    PCNetwork net(BATCH_SIZE);
    net.AddLayer(2, 4, LEARNING_RATE, INFERENCE_RATE, tanh, dTanh);
    net.AddLayer(4, 1, LEARNING_RATE, INFERENCE_RATE, tanh, dTanh);
    net.AddLayer(1, 0, LEARNING_RATE, INFERENCE_RATE, tanh, dTanh); // terminal layer, matches tNetwork.cpp pattern
    net.RandomizeWeights(mt);

    // AND gate, scaled into tanh's [-1, 1] range.
    std::vector<std::vector<float>> inputs = {
        {-1.0f, -1.0f},
        {-1.0f, 1.0f},
        {1.0f, -1.0f},
        {1.0f, 1.0f},
    };
    std::vector<std::vector<float>> targets = {
        {-1.0f},
        {-1.0f},
        {-1.0f},
        {1.0f},
    };

    std::cout << "Training on AND gate (2 -> 4 -> 1 -> [1,0])..." << std::endl;

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        float lastEnergy = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i)
            lastEnergy = TrainOnExample(net, inputs[i], targets[i], INFERENCE_STEPS);

        if (epoch % 200 == 0)
        {
            auto &layers = net.GetLayers();
            const float *w = static_cast<PCLayer *>(layers[0])->GetWeights();
            float w0norm = 0.0f;
            for (size_t k = 0; k < 2 * 4; ++k)
                w0norm += w[k] * w[k];
            std::cout << "Epoch " << epoch << "  last energy=" << lastEnergy
                      << "  layer0 W norm=" << std::sqrt(w0norm) << std::endl;
        }
    }

    std::cout << "\nFinal predictions:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto pred = Predict(net, inputs[i], INFERENCE_STEPS);
        std::cout << "  input (" << inputs[i][0] << ", " << inputs[i][1]
                  << ") -> predicted " << pred[0]
                  << "   (target " << targets[i][0] << ")" << std::endl;
    }

    return 0;
}