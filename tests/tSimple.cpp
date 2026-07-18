#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include "DiscriminativePCNetwork.h"
#include "Activations.h"

// ---------------------------------------------------------
// Activation Functions
// ---------------------------------------------------------
void tanh_act(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        x[i] = std::tanh(x[i]);
}

void dtanh_act(float *x, size_t n)
{
    // Reconstructs tanh from the RAW pre-activation stored in sigma_prime
    for (size_t i = 0; i < n; i++)
    {
        float t = std::tanh(x[i]);
        x[i] = 1.0f - t * t;
    }
}

void linear_act(float *x, size_t /*n*/)
{
    // Identity -- no-op
}

void dlinear_act(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++)
        x[i] = 1.0f;
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main()
{
    // 1. Setup the Network
    // Pass batchSize=1 to the Network constructor
    Deep::DiscriminativePCNetwork net(1);

    // AddLayer(size, nextSize, lr, ir, lmbda, act, dAct)
    // Note: lr=0.05 (weights), ir=0.3 (fast inference)
    net.AddLayer(2, 8, 0.05f, 0.3f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(8, 1, 0.05f, 0.3f, 0.0001f, Deep::tanh, Deep::dTanh);
    net.AddLayer(1, 0, 0.05f, 0.3f, 0.0001f, Deep::linear, Deep::dLinear);

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    // 2. Data
    std::vector<std::vector<float>> X = {
        {-1.0f, -1.0f},
        {-1.0f, +1.0f},
        {+1.0f, -1.0f},
        {+1.0f, +1.0f}};

    // Soft targets: Prevents the hidden tanh derivatives from vanishing
    std::vector<std::vector<float>> Y = {
        {-1.0f},
        {+1.0f},
        {+1.0f},
        {-1.0f}};

    // 3. Training Loop
    int epochs = 5000;
    int inferenceSteps = 50;
    int reportEvery = 500;

    std::cout << "Starting Discriminative PC XOR Test...\n";

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float totalEnergy = 0.0f;

        // Shuffle training order each epoch to prevent oscillation
        std::vector<int> order = {0, 1, 2, 3};
        std::shuffle(order.begin(), order.end(), rng);

        for (int idx : order)
        {
            // Zero out hidden z from previous sample so inference starts neutral
            net.ResetState();

            net.Clamp(X[idx]);
            net.GetTerminalLayer()->ClampState(Y[idx]);

            // Inference
            for (int t = 0; t < inferenceSteps; t++)
            {
                totalEnergy += net.CalculateState();
                net.UpdateState();
            }

            // Weight update
            net.UpdateWeights();
            net.GetTerminalLayer()->UnclampState();
        }

        if (epoch % reportEvery == 0)
        {
            float avgEnergy = totalEnergy / (4 * inferenceSteps);
            std::cout << "Epoch " << std::setw(5) << epoch 
                      << " | Energy: " << std::fixed << std::setprecision(4) << avgEnergy << "\n";
        }
    }

    // 4. Testing
    std::cout << "\n=== Predictions ===\n";
    int correct = 0;

    for (size_t i = 0; i < X.size(); i++)
    {
        net.ResetState();
        net.Clamp(X[i]);

        // Free inference (no target clamped)
        for (int t = 0; t < inferenceSteps; t++)
        {
            net.CalculateState();
            net.UpdateState();
        }

        float pred = net.GetTerminalLayer()->GetBeliefs()[0];
        float target = Y[i][0];
        bool signCorrect = (pred > 0 && target > 0) || (pred < 0 && target < 0);
        if (signCorrect) correct++;

        std::cout << "Input: [" << std::setw(2) << X[i][0] << ", " << std::setw(2) << X[i][1] << "]"
                  << " | Target: " << std::showpos << target
                  << " | Pred: " << std::setprecision(4) << pred
                  << (signCorrect ? " OK" : " WRONG") << "\n";
    }

    std::cout << "\nAccuracy: " << correct << "/4\n";
    std::cout << (correct == 4 ? "XOR LEARNED SUCCESSFULLY!" : "XOR NOT FULLY LEARNED.") << "\n";

    return 0;
}