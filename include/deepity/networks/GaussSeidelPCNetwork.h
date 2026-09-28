#pragma once
#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Activations.h>
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/DeviceType.h>

/**
 * @file GaussSeidelPCNetwork.h
 * @brief Orchestrates GaussSeidelPCLayer's three-phase settling step in
 * the correct order. The layer class alone cannot produce genuine
 * Gauss-Seidel dynamics; the sweep order across layers is what makes it
 * Gauss-Seidel rather than Jacobi, and that only exists at this level.
 * Device-aware (DeviceType constructor parameter, backend member,
 * cpuArena/gpuArena split in Compile()).
 *
 * Per settling step, in order:
 *   1. UpdateState() on EVERY layer       (uses PREVIOUS timestep's mu/e)
 *   2. ComputePrediction() on EVERY layer (uses THIS timestep's fresh z)
 *   3. ComputeError() on EVERY layer      (uses THIS timestep's fresh
 *                                          z and layerBelow's fresh mu)
 *
 * Order among layers within a single sweep does not matter: each
 * layer's step-1/2/3 only depends on its own state plus a neighbor's
 * value from a sweep that has already fully completed. Only the order
 * of the three sweeps themselves is load-bearing.
 */

namespace Deep
{
    /// @brief Predictive Coding network built from GaussSeidelPCLayer,
    /// sweeping layers sequentially rather than updating them
    /// simultaneously, now device-aware.
    class GaussSeidelPCNetwork
    {
    public:
        /// @brief Constructs an empty network with a predetermined batch size.
        /// @param batchSize Batch size
        /// @param device Which device this network's layers run on.
        explicit GaussSeidelPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;
        /// @brief Default destructor.
        ~GaussSeidelPCNetwork() = default;

        /// @brief Copy construction is disabled; a network owns its
        /// layers' memory arena and is not meant to be duplicated.
        GaussSeidelPCNetwork(const GaussSeidelPCNetwork &) = delete;
        /// @brief Copy assignment is disabled; see the copy constructor.
        GaussSeidelPCNetwork &operator=(const GaussSeidelPCNetwork &) = delete;

        /// @brief Adds a layer to the network, using raw activation
        /// function pointers, converted to their equivalent
        /// ActivationType via To_AType() and forwarded to the
        /// ActivationType overload below (GaussSeidelPCLayer itself only
        /// accepts ActivationType).
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for weight updates
        /// @param ir inference rate (Euler integration step size)
        /// @param lmbda weight decay (L2 regularization) coefficient
        /// @param act activation function
        /// @param dAct derivative of previous activation function
        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool));
        /// @brief Adds a layer to the network, using a named
        /// ActivationType instead of raw function pointers.
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for weight updates
        /// @param ir inference rate (Euler integration step size)
        /// @param lmbda weight decay (L2 regularization) coefficient
        /// @param aType activation type
        /// @param dType activation derivative type
        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      ActivationType aType, ActivationType dType);

        /// @brief Sets the optimizer for EVERY layer added so far. Safe
        /// to call any time before Compile().
        void SetOptimizer(OptimizerType opt) noexcept
        {
            for (auto &layer : layers)
                layer->SetOptimizer(opt);
        }

        /// @brief Randomizes the weights of each layer.
        /// @param rng The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &rng);
        /// @brief Resets each layer's state without touching learned weights.
        void ResetState() noexcept;
        /// @brief Clamps the input to the first layer.
        /// @param input Reference to the input vector.
        void Clamp(const std::vector<float> &input);

        /// @brief Runs ONE full Gauss-Seidel settling step (all three
        /// sweeps, in order) across every layer.
        /// @return Total energy, summed from every layer's
        /// ComputeError(). Meaningful only after all three sweeps have
        /// run for this step.
        float Step() noexcept;

        /// @brief Updates every non-terminal layer's weights. Called
        /// once after the full settling loop completes.
        void UpdateWeights() noexcept;

        /// @brief Seeds every hidden layer's z from a genuine forward
        /// pass through current weights (ComputePrediction() only, no
        /// error/energy side effects). Assumes layers[0] is already
        /// clamped.
        void ProjectForward() noexcept;

        /// @brief Sets the learning rate used for weight updates, on
        /// every layer.
        /// @param lr The new learning rate.
        void SetLearningRate(float lr) noexcept
        {
            for (auto &layer : layers)
                layer->SetLearningRate(lr);
        }

        /// @brief Returns the network's terminal (final) layer.
        /// @return Pointer to the last layer added via AddLayer().
        GaussSeidelPCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }
        /// @brief Returns every layer in the network, in the order they were added.
        /// @return A reference to the internal layer list.
        std::vector<std::unique_ptr<GaussSeidelPCLayer>> &GetLayers() noexcept { return layers; }
        /// @brief Returns every layer in the network, in the order they were added.
        /// @return A const reference to the internal layer list.
        const std::vector<std::unique_ptr<GaussSeidelPCLayer>> &GetLayers() const noexcept { return layers; }
        /// @brief Returns the batch size for the network's layers.
        /// @return int batchSize
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

        /// @brief Sums every layer's required float count into a single
        /// contiguous arena and binds each layer into it.
        void Compile();

    private:
        /// @brief Every layer in the network, in the order they were added.
        std::vector<std::unique_ptr<GaussSeidelPCLayer>> layers;
        std::unique_ptr<IComputeBackend> backend;
        DeviceType device;
        /// @brief The contiguous memory block backing every layer's buffers.
        std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
        std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif
        /// @brief The batch size shared by every layer in the network.
        int batchSize;
    };
}
