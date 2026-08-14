#include "ConvPCLayer.h"
#include "MemoryArena.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

namespace Deep
{
    class PCNDiagnostics
    {
    public:
        static void RunHiddenLayerGradientCheck()
        {
            std::cout << "--- Starting ConvPCLayer Gradient Check ---" << std::endl;

            int batchSize = 1;
            float lr = 1e-6f;
            float ir = 0.1f;
            float pr = 0.01f;
            float lmbda = 1e-2f;

            ConvPCLayer l0(1, 2, 4, 4, 2, 2, 1, 1, 0, 0, batchSize, lr, ir, pr, lmbda, ActivationType::TANH, ActivationType::dTANH);
            ConvPCLayer l1(2, 2, 3, 3, 2, 2, 1, 1, 0, 0, batchSize, lr, ir, pr, lmbda, ActivationType::TANH, ActivationType::dTANH);
            ConvPCLayer l2(2, 0, 2, 2, 1, 1, 1, 1, 0, 0, batchSize, lr, ir, pr, lmbda, ActivationType::TANH, ActivationType::dTANH);

            // Wire the network pipeline
            l0.SetLayerAbove(&l1);
            l1.SetLayerBelow(&l0);
            l1.SetLayerAbove(&l2);
            l2.SetLayerBelow(&l1);

            // Bind memory arena
            size_t totalFloats = l0.GetRequiredFloats() + l1.GetRequiredFloats() + l2.GetRequiredFloats();
            MemoryArena arena(totalFloats);
            l0.BindMemory(arena);
            l1.BindMemory(arena);
            l2.BindMemory(arena);

            std::mt19937 rng(42);
            l0.RandomizeWeights(rng);
            l1.RandomizeWeights(rng);
            l2.RandomizeWeights(rng);

            // Inputs and targets
            std::vector<float> X(16, 0.5f); // 1 channel, 4x4
            std::vector<float> Y(8, 0.1f);  // 2 channels, 2x2

            l0.ResetState();
            l1.ResetState();
            l2.ResetState();

            l0.ClampState(X);
            l2.ClampState(Y);

            auto calcTotalEnergy = [&]()
            {
                return l0.CalculateState() + l1.CalculateState() + l2.CalculateState();
            };

            // Evaluate initial state before taking derivatives
            calcTotalEnergy();

            // 1. BACKUP Z BEFORE IT GETS CORRUPTED BY UPDATESTATE
            size_t hidden_size = l1.GetInputSize();
            std::vector<float> original_z_vec(hidden_size);
            for(size_t i = 0; i < hidden_size; ++i) {
                original_z_vec[i] = l1.z[i];
            }

            // 2. Compute Analytical Gradient
            l0.UpdateState();
            l1.UpdateState();
            l2.UpdateState();

            // Save the analytical gradients directly
            std::vector<float> analytical_grads(hidden_size);
            for(size_t i = 0; i < hidden_size; ++i) {
                analytical_grads[i] = -l1.dz_dt[i]; // dE/dz = -dz_dt
            }

            // 3. RESTORE Z SO NUMERICAL GRADIENT EVALUATES AT THE EXACT SAME POINT
            for(size_t i = 0; i < hidden_size; ++i) {
                l1.z[i] = original_z_vec[i];
            }

            float epsilon = 1e-4f;
            float max_diff = 0.0f;

            std::cout << "Checking " << hidden_size << " hidden beliefs..." << std::endl;

            for (size_t i = 0; i < hidden_size; ++i)
            {
                float original_z = l1.z[i];
                float analytical_grad = analytical_grads[i];

                // Compute Numerical Gradient safely
                l1.z[i] = original_z + epsilon;
                float e_plus = calcTotalEnergy();

                l1.z[i] = original_z - epsilon;
                float e_minus = calcTotalEnergy();

                l1.z[i] = original_z; // Restore

                float numerical_grad = (e_plus - e_minus) / (2.0f * epsilon);
                float diff = std::abs(analytical_grad - numerical_grad);
                if (diff > max_diff)
                    max_diff = diff;

                if (diff > 1e-3f)
                {
                    std::cout << "MISMATCH at index " << i << ": "
                              << "Analytical = " << std::fixed << std::setprecision(6) << analytical_grad
                              << ", Numerical = " << numerical_grad
                              << ", Diff = " << diff << std::endl;
                }
            }

            std::cout << "Max gradient difference: " << max_diff << std::endl;
            if (max_diff < 1e-3f)
            {
                std::cout << "SUCCESS: Feedback routing is mathematically sound." << std::endl;
            }
            else
            {
                std::cout << "FAILED: Col2Im or GEMM transpose routing contains an error." << std::endl;
            }
        }
    };
}

    int main(void) {
        Deep::PCNDiagnostics::RunHiddenLayerGradientCheck();
        return 0;
    }