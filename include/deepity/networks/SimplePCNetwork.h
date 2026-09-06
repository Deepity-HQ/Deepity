#pragma once
#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/SimplePCLayer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/backend/Backend.h>

/**
 * @file SimplePCNetwork.h
 * @brief Network wrapper for SimplePCLayer, mirroring DiscriminativePCNetwork
 * exactly (minus the UpdatePrecision() call, which no longer exists).
 *
 * This header includes implementations of PC layer-to-layer interaction.
 *
 * Usage:
 *  #include <deepity/SimplePCNetwork.h>
 *
 * Example:
 *  Deep::SimplePCNetwork network(1);
 *  network.addLayer({...});
 *  network.Clamp(input);
 *  network.CalculateState();
 *
 * @note All layers are stored in a vector.
 *
 * @note As of this revision, the network owns a single IComputeBackend
 * (CPU by default, GPU if requested at construction) and passes it down
 * into every layer it creates. `device` defaults to DEVICE_CPU, so every
 * existing caller (nanobind bindings, hand-written C++) keeps compiling
 * and behaving exactly as before -- passing DEVICE_GPU explicitly is
 * only needed for new, GPU-aware call sites.
 * @version 1.1
 * @date 2026-09-05
 * @author Jack Rose
 */

namespace Deep
{
    /// @brief An abstracted class for an array of `SimplePCLayer`
    class SimplePCNetwork
    {
    public:
        /// @brief Constructs an empty network with a predetermined batch size.
        /// @param batchSize Batch size
        /// @param device Which device this network's layers should run
        ///        on. Defaults to DEVICE_CPU, preserving existing
        ///        behavior exactly for anyone not explicitly requesting
        ///        DEVICE_GPU.
        explicit SimplePCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;

        SimplePCNetwork(const SimplePCNetwork &) = delete;
        SimplePCNetwork &operator=(const SimplePCNetwork &) = delete;

        ~SimplePCNetwork() = default;

        /// @brief Adds a layer to the network.
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for beliefs
        /// @param ir learning rate for weights
        /// @param lmbda weight decay (L2 regularization) coefficient
        /// @param act activation function
        /// @param dAct derivative of previous activation function
        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool));

        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      ActivationType aType, ActivationType dType);

        /// @brief Randomizes the weights of each layer
        /// @param rng The classic Mersenne Twister
        /// @param distribution A string representation of (normal, uniform) distribution. For example: "normal(0, 1)" for a standard normal.
        void RandomizeWeights(std::mt19937 &rng, const char *distribution);

        /// @brief Randomizes the weights of each layer
        /// @param rng The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &rng);

        /// @brief Resets each layer's state without touching learned weights.
        void ResetState() noexcept;

        /// @brief Clamps the input to the first layer, necessary for prediction
        /// @param input reference to input vector
        void Clamp(const std::vector<float> &input);

        /// @brief Calculates the state of each layer
        /// @param needEnergy ask for energy after
        /// @return Returns total energy if asked for
        float CalculateState(bool needEnergy = true);

        /// @brief Updates each layer's state
        void UpdateState();

        /// @brief Updates each layer's weights
        void UpdateWeights();

        // NOTE: no UpdatePrecision() -- precision doesn't exist in this class.

        /// @brief Returns the network's terminal (final) layer.
        /// @return Pointer to the last layer added via AddLayer().
        SimplePCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }

        /// @brief Returns every layer in the network, in the order they were added.
        /// @return A reference to the internal layer list.
        std::vector<std::unique_ptr<SimplePCLayer>> &GetLayers() noexcept { return layers; }

        /// @brief Returns every layer in the network, in the order they were added.
        /// @return A const reference to the internal layer list.
        const std::vector<std::unique_ptr<SimplePCLayer>> &GetLayers() const noexcept { return layers; }

        /// @brief Returns the batch size for the network's layers
        /// @return size_t batchSize
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Returns which device this network's layers run on.
        DeviceType GetDevice() const noexcept { return device; }

        /// @brief Sets the optimizer used for weight updates on every layer.
        /// @param o The optimizer type to apply.
        void SetOptimizer(OptimizerType o) noexcept
        {
            for (auto &layer : layers)
                layer->SetOptimizer(o);
        }

        /// @brief Runs a complete training step (clamp, settle, update, unclamp)
        /// @param x The batched input data
        /// @param y The batched target data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return The final energy state of the network before weight updates
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Runs a forward prediction pass (clamp, settle, read)
        /// @param x The batched input data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return A vector containing the batched predictions
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        /// @brief Runs a single, non-iterative forward pass through current
        /// weights, seeding each hidden layer's z from the PREVIOUS layer's
        /// mu -- giving the settling loop a genuine, current-weights-based
        /// starting point instead of zero-init.
        ///
        /// Reuses each layer's EXISTING CalculateState() (already-verified
        /// forward computation, mu = f(W@z+b)) as a side effect -- no new
        /// math, just a new sequence of existing calls. Assumes the input
        /// layer (layers[0]) is ALREADY clamped before this is called.
        ///
        /// @warning Layer 0's error (e) and this call's energy return value
        /// are meaningless here -- CalculateState() computes both mu (what
        /// we want) and e/energy (a side effect we're discarding, since the
        /// NEXT layer's z hasn't been set to a meaningful value yet at the
        /// point each layer's CalculateState() runs). Only mu is used.
        void ProjectForward() noexcept;

        /// @brief Full train step WITH forward-projection initialization,
        /// all in ONE call -- reset, clamp, project, settle, update weights,
        /// unclamp. Matches TrainStep()'s signature/return convention exactly,
        /// just with ProjectForward() inserted between clamping the input and
        /// clamping the target.
        ///
        /// Exists specifically to eliminate the Python/pybind boundary-
        /// crossing overhead of doing this same sequence via many separate
        /// calls from Python (reset_state, clamp_input, project_forward,
        /// clamp_state, then STEPS*2 individual calculate_state/update_state
        /// calls, update_weights, unclamp_state -- over 40 individual
        /// crossings per batch at STEPS=20). This does the whole sequence in
        /// ONE crossing instead.
        float TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        std::vector<float> PredictWithProjection(const std::vector<float> &x, int inferenceSteps);
        /// @brief Sets mu-cache threshold on every layer -- see
        /// SimplePCLayer::SetMuCacheThreshold() for semantics. Safe to call any
        /// time after Compile().
        void SetMuCacheThreshold(float threshold) noexcept;

        /// @brief Loads all layers into one contiguous block of memory
        /// (MemoryArena for DEVICE_CPU, DeviceMemoryArena for DEVICE_GPU).
        void Compile();

    private:
        std::vector<std::unique_ptr<SimplePCLayer>> layers;

        /// @brief The network's own compute backend, created once at
        /// construction and shared by every layer added afterward.
        std::unique_ptr<IComputeBackend> backend;
        /// @brief Which device `backend` actually is -- kept alongside
        /// it since IComputeBackend itself doesn't expose its own type,
        /// and Compile() needs to know which arena type to construct.
        DeviceType device;

        bool graphCaptured = false;
        int capturedInferenceSteps = -1;

        /// @brief Used when device == DEVICE_CPU. Only one of
        /// cpuArena/gpuArena is ever actually constructed for a given
        /// network -- they're kept as two separate members (rather than
        /// one unified type) because MemoryArena and DeviceMemoryArena
        /// share no common base, matching the same reasoning already
        /// applied when DeviceMemoryArena was designed.
        std::unique_ptr<MemoryArena> cpuArena;
        /// @brief Used when device == DEVICE_GPU. Only compiled at all
        /// when DEEPITY_USE_CUDA is defined, matching
        /// DeviceMemoryArena.h's own guard.
#if defined(DEEPITY_USE_CUDA)
        std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif

        int batchSize;
    };
}