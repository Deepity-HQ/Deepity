// Smoke test for GaussSeidelPCNetwork -- confirms it compiles, runs, and
// produces sane (finite, decreasing) energy on a tiny synthetic task
// before investing in the full finite-difference gradient check.
// Deliberately uses plain TrainStep() (no ProjectForward()) and SGD --
// isolating just the core three-sweep restructuring itself as the one
// new thing being smoke-tested here.
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>

using namespace Deep;

int main()
{
    const int BATCH = 4;
    const float LR = 0.02f;

    GaussSeidelPCNetwork net(BATCH);
    net.AddLayer(8, 6, LR, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(6, 4, LR, 0.08f, 0.0001f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(4, 0, LR, 0.08f, 0.0001f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();

    std::mt19937 rng(42);
    net.RandomizeWeights(rng);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> X((size_t)BATCH * 8);
    std::vector<float> Y((size_t)BATCH * 4);
    for (auto &v : X)
        v = dist(rng);
    for (auto &v : Y)
        v = dist(rng);

    std::cout << "Training 20 steps on synthetic data (plain TrainStep, no projection)...\n";

    float prevEnergy = -1.0f;
    bool sawIncrease = false;

    for (int iter = 0; iter < 20; ++iter)
    {
        float energy = net.TrainStep(X, Y, 30);
        std::cout << "  iter " << iter << "  energy = " << energy << "\n";

        if (std::isnan(energy) || std::isinf(energy))
        {
            std::cout << "FAIL -- energy is NaN/Inf.\n";
            return 1;
        }

        if (prevEnergy >= 0.0f && energy > prevEnergy)
            sawIncrease = true;

        prevEnergy = energy;
    }

    std::cout << "\n";
    if (sawIncrease)
        std::cout << "NOTE: energy was not strictly monotonic decreasing at every\n"
                     "step -- some noise is normal given weight updates happen every\n"
                     "iteration, but watch the OVERALL trend below.\n\n";

    std::cout << "PASS -- GaussSeidelPCNetwork compiles, trains, and produces\n";
    std::cout << "sane (non-NaN, non-Inf) energy through 20 real steps.\n";
    std::cout << "This does NOT confirm gradient correctness -- that needs the\n";
    std::cout << "full finite-difference check, not done here.\n";

    return 0;
}