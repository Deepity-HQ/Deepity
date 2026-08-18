#include <iostream>
#include <vector>
#include <random>
#include <deepity/networks/DiscriminativePCNetwork.h>

int main(void)
{
    std::cout << "[ASan] Initializing PCNetwork (Batch Size: 32)..." << std::endl;
    Deep::DiscriminativePCNetwork net(32);

    std::cout << "[ASan] Adding Layers..." << std::endl;
    // 64 -> 32
    net.AddLayer(64, 32, 0.06f, 0.08f, 0.0f, 0.0001f, Deep::ActivationType::TANH, Deep::ActivationType::dTANH);
    // 32 -> 10
    net.AddLayer(32, 10, 0.06f, 0.08f, 0.0f, 0.0001f, Deep::ActivationType::TANH, Deep::ActivationType::dTANH);
    // 10 -> 0 (Terminal)
    net.AddLayer(10, 0, 0.06f, 0.08f, 0.0f, 0.0001f, Deep::ActivationType::LINEAR, Deep::ActivationType::dLINEAR);

    std::cout << "[ASan] Compiling Memory Arena..." << std::endl;
    net.Compile();

    std::cout << "[ASan] Randomizing Weights..." << std::endl;
    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::cout << "[ASan] Generating Dummy Data..." << std::endl;
    std::vector<float> X(32 * 64, 0.5f);
    std::vector<float> Y(32 * 10, 0.9f);

    std::cout << "[ASan] Firing TrainStep (Euler Integration)..." << std::endl;
    try
    {
        // Run 10 inference steps to trigger all forward/backward SIMD loops
        float energy = net.TrainStep(X, Y, 10);
        std::cout << "[ASan] Success! Final Energy: " << energy << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ASan] Exception caught: " << e.what() << std::endl;
    }

    return 0;
}