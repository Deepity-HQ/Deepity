#include <iostream>
#include <chrono>
#include "PCNetwork.h"

int main(void)
{
    std::cout << "Starting Predictive Coding Test." << std::endl;
    using namespace Deep;

    constexpr int NUM_RUNS = 10;
    constexpr float LEARNING_RATE = 0.01f;
    constexpr int BATCH_SIZE = 64;

    double totalTimeMs = 0.0;
    long long totalIterations = 0;

    std::mt19937 mt(11);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int run = 0; run < NUM_RUNS; ++run)
    {
        PCNetwork net(BATCH_SIZE);
        net.AddLayer(784, 512, LEARNING_RATE, tanh, dTanh);
        net.AddLayer(512, 256, LEARNING_RATE, tanh, dTanh);
        net.AddLayer(256, 64,  LEARNING_RATE, tanh, dTanh);
        net.AddLayer(64,  10,  LEARNING_RATE, tanh, dTanh);
        net.AddLayer(10,  0,   LEARNING_RATE, tanh, dTanh);

        net.RandomizeWeights(mt);

        std::vector<float> inputObservation(784);
        for (float &x : inputObservation)
            x = dist(mt);

        net.Clamp(inputObservation);

        float prevEnergy = 1e9f;
        float currentEnergy = 0.0f;
        int iteration = 0;

        auto start = std::chrono::high_resolution_clock::now();

        while (iteration < 157)
        {
            currentEnergy = net.CalculateState();

            if (std::abs(prevEnergy - currentEnergy) < 1e-3f)
                break;

            net.UpdateState();
            net.UpdateWeights();

            prevEnergy = currentEnergy;
            ++iteration;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        totalTimeMs += elapsed.count();
        totalIterations += iteration;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "Network: 784 -> 512 -> 256 -> 64 -> 10"  << std::endl;
    std::cout << "Runs: "             << NUM_RUNS           << std::endl;
    std::cout << "Average inference time: "
              << totalTimeMs / NUM_RUNS << " ms"            << std::endl;
    std::cout << "Average iterations: "
              << static_cast<double>(totalIterations) / NUM_RUNS << std::endl;
    std::cout << "Batch size: "       << BATCH_SIZE         << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "> Note: This network will not converge, since it is random." << std::endl;

    return 0;
}