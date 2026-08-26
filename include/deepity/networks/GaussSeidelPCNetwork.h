#pragma once
#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <deepity/MemoryArena.h>

/**
 * @file GaussSeidelPCNetwork.h
 * @brief Orchestrates GaussSeidelPCLayer's three-phase settling step in
 * the correct order -- the layer class alone cannot produce genuine
 * Gauss-Seidel dynamics; the SWEEP ORDER across layers is what makes it
 * Gauss-Seidel rather than Jacobi, and that only exists at this level.
 *
 * Per settling step, in order:
 *   1. UpdateState() on EVERY layer       (uses PREVIOUS timestep's mu/e)
 *   2. ComputePrediction() on EVERY layer (uses THIS timestep's fresh z)
 *   3. ComputeError() on EVERY layer      (uses THIS timestep's fresh
 *                                          z and layerBelow's fresh mu)
 *
 * Order among layers WITHIN a single sweep does not matter (verified via
 * the dependency analysis in GaussSeidelPCLayer.h's class docs -- each
 * layer's step-1/2/3 only depends on ITS OWN state plus a neighbor's
 * value from a sweep that has ALREADY fully completed). Only the ORDER
 * OF THE THREE SWEEPS THEMSELVES is load-bearing.
 *
 * @warning Not yet gradient-checked at the network level. The individual
 * layer's math should be independently verified before trusting any
 * accuracy conclusion drawn from real training with this class.
 */

namespace Deep
{
    class GaussSeidelPCNetwork
    {
    public:
        explicit GaussSeidelPCNetwork(int batchSize) noexcept;
        ~GaussSeidelPCNetwork() = default;

        GaussSeidelPCNetwork(const GaussSeidelPCNetwork &) = delete;
        GaussSeidelPCNetwork &operator=(const GaussSeidelPCNetwork &) = delete;

        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool));
        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      ActivationType aType, ActivationType dType);

        void RandomizeWeights(std::mt19937 &rng);
        void ResetState() noexcept;
        void Clamp(const std::vector<float> &input);

        /// @brief Runs ONE full Gauss-Seidel settling step (all three
        /// sweeps, in order) across every layer.
        /// @return Total energy, summed from every layer's
        /// ComputeError() -- meaningful only after all three sweeps have
        /// run for this step.
        float Step() noexcept;

        /// @brief Updates every non-terminal layer's weights. Called
        /// ONCE after the full settling loop completes, matching
        /// ngc-learn's evolve_process.
        void UpdateWeights() noexcept;

        /// @brief Seeds every hidden layer's z from a genuine forward
        /// pass through current weights (ComputePrediction() only, no
        /// error/energy side effects) -- an orthogonal optimization to
        /// the Gauss-Seidel restructuring itself. Assumes layers[0] is
        /// already clamped.
        void ProjectForward() noexcept;

        void SetOptimizer(OptimizerType o) noexcept
        {
            for (auto &layer : layers)
                layer->SetOptimizer(o);
        }

        void SetLearningRate(float lr) noexcept
        {
            for (auto &layer : layers)
                layer->SetLearningRate(lr);
        }

        GaussSeidelPCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }
        std::vector<std::unique_ptr<GaussSeidelPCLayer>> &GetLayers() noexcept { return layers; }
        const std::vector<std::unique_ptr<GaussSeidelPCLayer>> &GetLayers() const noexcept { return layers; }
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Full train step: reset, clamp input+target, settle for
        /// inferenceSteps, update weights once, unclamp.
        /// @return The final step's total energy, before weight updates.
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Same as TrainStep(), but with ProjectForward() called
        /// once after clamping the input, before the settling loop.
        float TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Forward prediction: clamp input, settle (no target
        /// clamped), read the terminal's beliefs.
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        void Compile();

    private:
        std::vector<std::unique_ptr<GaussSeidelPCLayer>> layers;
        std::unique_ptr<MemoryArena> arena;
        int batchSize;
    };
}