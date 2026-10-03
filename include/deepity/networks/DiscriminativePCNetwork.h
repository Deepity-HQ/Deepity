#pragma once

#include <vector>
#include <memory>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/DeviceType.h>

/**
 * @file DiscriminativePCNetwork.h
 * @brief Network-level wrapper for DiscriminativePCLayer, the original
 * precision-weighted PC model. Device-aware (DeviceType constructor
 * parameter, backend member, cpuArena/gpuArena split), with auto-batch-
 * size detection (see autoSize below).
 *
 * @code{.cpp}
 * #include <deepity/networks/DiscriminativePCNetwork.h>
 *
 * Deep::DiscriminativePCNetwork network(1);
 * network.AddLayer(...);
 * network.Clamp(input);
 * network.CalculateState();
 * @endcode
 *
 * @version 2.0
 * @date 2026-09-27
 * @author Jack Rose
 */

namespace Deep
{
    class PCNDiagnostics;

    /// @brief An abstracted class for an array of `DiscriminativePCLayer`,
    /// device-aware.
    ///
    /// @see https://arxiv.org/pdf/2506.06332
    class DiscriminativePCNetwork
    {
        std::vector<std::unique_ptr<DiscriminativePCLayer>> layers;
        int batchSize;
        bool autoSize = true;

    public:
        /// @brief Default constructor; initializes the network with
        /// auto-batch-size detection (see AddLayer()'s fn-pointer
        /// overload's implementation for how the first layer's size
        /// triggers this).
        /// @param device Which device this network's layers run on.
        explicit DiscriminativePCNetwork(DeviceType device = DeviceType::DEVICE_CPU);
        /// @brief Batched constructor; initializes the network with a
        /// predetermined batch size.
        /// @param batchSize Batch size
        /// @param device Which device this network's layers run on.
        explicit DiscriminativePCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU);

        ~DiscriminativePCNetwork() = default;

        DiscriminativePCNetwork(const DiscriminativePCNetwork &) = delete;
        DiscriminativePCNetwork &operator=(const DiscriminativePCNetwork &) = delete;
        /// @brief Move constructor.
        DiscriminativePCNetwork(DiscriminativePCNetwork &&other) noexcept = default;
        /// @brief Move assignment operator.
        DiscriminativePCNetwork &operator=(DiscriminativePCNetwork &&other) noexcept = default;

        /// @brief Adds a layer to the network, using raw activation
        /// function pointers, converted to their equivalent
        /// ActivationType via To_AType() and forwarded to the
        /// ActivationType overload below (DiscriminativePCLayer itself
        /// only accepts ActivationType).
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for beliefs
        /// @param ir learning rate for weights
        /// @param pr learning rate for precision
        /// @param lmbda weight decay (L2 regularization) coefficient
        /// @param act activation function
        /// @param dAct derivative of previous activation function
        void AddLayer(int size, int nextSize, float lr, float ir, float pr, float lmbda,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool));

        /// @brief Adds a layer to the network, using a named
        /// ActivationType instead of raw function pointers.
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for beliefs
        /// @param ir learning rate for weights
        /// @param pr learning rate for precision
        /// @param lmbda weight decay (L2 regularization) coefficient
        /// @param aType activation type
        /// @param dType activation derivative type
        void AddLayer(int size, int nextSize, float lr, float ir, float pr, float lmbda,
                      Deep::ActivationType aType, Deep::ActivationType dType);

        /// @brief Sets the optimizer for EVERY layer added so far. Safe to
        /// call any time before Compile(), memory allocation is
        /// deferred to Compile(), not AddLayer().
        void SetOptimizer(OptimizerType opt) noexcept
        {
            for (auto &layer : layers)
                layer->SetOptimizer(opt);
        }

        /// @brief Randomizes the weights of each layer.
        /// @param rng The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &rng);

        /// @brief Clamps the input to the first layer, necessary for prediction.
        /// @param input reference to input vector
        void Clamp(const std::vector<float> &input);

        /// @brief Calculates the state of each layer.
        /// @param needEnergy Whether to compute and return the energy. On
        /// CUDABackend, false also means no blocking host sync anywhere
        /// in the call, required when running this inside a captured
        /// CUDA graph region.
        /// @return Returns total energy, or 0.0f if needEnergy is false.
        float CalculateState(bool needEnergy = true);

        /// @brief Updates each layer's state.
        void UpdateState();

        /// @brief Updates each layer's weights.
        void UpdateWeights();

        /// @brief Updates each layer's precision weighting.
        void UpdatePrecision();

        /// @brief Seeds every hidden layer's belief from a genuine
        /// forward pass through current weights, rather than zero-init.
        /// Assumes the input layer is already clamped.
        void ProjectForward() noexcept;
        /// @brief Same as TrainStep(), but with ProjectForward() called
        /// once after clamping the input, before the settling loop.
        /// @param x The batched input data
        /// @param y The batched target data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return The final energy state of the network before weight updates
        float TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);
        /// @brief Same as Predict(), but with ProjectForward() called
        /// once after clamping the input, before the settling loop.
        /// @param x The batched input data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return A vector containing the batched predictions
        std::vector<float> PredictWithProjection(const std::vector<float> &x, int inferenceSteps);

        /// @brief Resets each layer's state without touching learned weights.
        void ResetState() noexcept;

        /// @brief Returns every layer in the network, in the order they
        /// were added.
        /// @return A const reference to the internal layer list.
        const auto &GetLayers() const noexcept { return layers; }

        /// @brief Returns the batch size for the network's layers.
        /// @return size_t batchSize
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Returns the network's terminal (final) layer.
        /// @return Pointer to the last layer added via AddLayer(), or
        /// nullptr if no layers have been added.
        DiscriminativePCLayer *GetTerminalLayer() const
        {
            if (layers.empty())
                return nullptr;
            return layers.back().get();
        }

        /// @brief Sets the learning rate used for weight updates, on
        /// every layer.
        /// @param lr The new learning rate.
        void SetLearningRate(float lr)
        {
            for (auto &layer : layers)
                layer->SetLearningRate(lr);
        }
        /// @brief Sets the inference rate used for state updates, on
        /// every layer.
        /// @param ir The new inference rate.
        void SetInferenceRate(float ir)
        {
            for (auto &layer : layers)
                layer->SetInferenceRate(ir);
        }
        /// @brief Sets the learning rate used for precision updates, on
        /// every layer.
        /// @param pr The new precision rate.
        void SetPrecisionRate(float pr)
        {
            for (auto &layer : layers)
                layer->SetPrecisionRate(pr);
        }
        /// @brief Sets the weight-decay (L2 regularization) coefficient,
        /// on every layer.
        /// @param l The new lambda value.
        void SetLambda(float l)
        {
            for (auto &layer : layers)
                layer->SetLambda(l);
        }

        /// @brief Runs a complete training step (clamp, settle, update, unclamp).
        /// @param x The batched input data
        /// @param y The batched target data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return The final energy state of the network before weight updates
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Runs a forward prediction pass (clamp, settle, read).
        /// @param x The batched input data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return A vector containing the batched predictions
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        /// @brief Saves the network's learned parameters to disk.
        /// @param filename The destination file path.
        /// @return True on success, false otherwise.
        bool Save(const std::string &filename) const noexcept;
        /// @brief Loads the network's learned parameters from disk.
        /// @param filename The source file path.
        /// @return True on success, false otherwise.
        bool Load(const std::string &filename) noexcept;

        /// @brief Sums every layer's required float count into a single
        /// contiguous arena and binds each layer into it. Call after all
        /// AddLayer()/SetOptimizer() calls, before RandomizeWeights().
        void Compile();

    private:
        std::unique_ptr<IComputeBackend> backend;
        DeviceType device;
        std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
        std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif
        friend class PCNDiagnostics;

        // CUDA graph capture state, GPU-only: TrainStep() and
        // TrainStepWithProjection() capture different op sequences, so
        // each needs its own capture-validity tracking.
        bool graphCaptured = false;
        int capturedInferenceSteps = -1;
        bool graphCapturedWithProjection = false;
        int capturedInferenceStepsWithProjection = -1;
    };
}
