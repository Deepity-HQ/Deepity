#pragma once

#include <vector>
#include <memory>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/MemoryArena.h>

/**
 * @file DiscriminativePCLayer.h
 * @brief Defines the network-level implementation of a PC model.
 *
 * This header includes implementations of PC layer-to-layer interaction.
 *
 * Usage:
 *  #include <deepity/networks/DiscriminativePCNetwork.h>
 *
 * Example:
 *  Deep::DiscriminativePCNetwork network(1);
 *  network.addLayer({...});
 *  network.Clamp(input);
 *  network.CalculateState();
 *
 * @note All layers are stored in a vector.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    class PCNDiagnostics;
    /// @brief An abstracted class for an array of `DiscriminativePCLayer`
    ///
    /// @see https://arxiv.org/pdf/2506.06332
    class DiscriminativePCNetwork
    {
        std::vector<std::unique_ptr<DiscriminativePCLayer>> layers;
        int batchSize;
        bool autoSize = true;

    public:
        /// @brief Default constructor
        ///
        /// Initializes the network with auto-batch size detection.
        DiscriminativePCNetwork() : batchSize(0), autoSize(true) {}
        /// @brief Batched constructor
        /// @param batchSize Batch size
        ///
        /// Initializes the network with a predetermined batch size.
        DiscriminativePCNetwork(int batchSize) : batchSize(batchSize), autoSize(false) {}

        /// @brief Default constructor; deletes each layer.
        ~DiscriminativePCNetwork() = default;

        /// @brief Move constructor.
        DiscriminativePCNetwork(DiscriminativePCNetwork &&other) noexcept = default;
        /// @brief Move assignment operator.
        DiscriminativePCNetwork &operator=(DiscriminativePCNetwork &&other) noexcept = default;

        /// @brief Adds a layer to the network.
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

        /// @brief Randomizes the weights of each layer
        /// @param rng The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &rng);

        /// @brief Clamps the input to the first layer, necessary for prediction
        /// @param input reference to input vector
        void Clamp(const std::vector<float> &input);

        /// @brief Calculates the state of each layer
        /// @return Returns total energy
        float CalculateState();

        /// @brief Updates each layer's state
        void UpdateState();

        /// @brief Updates each layer's weights
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

        /// @brief Returns the batch size for the network's layers
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
                layer->SetLearningRate(ir);
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

        /// @brief Sets the optimizer used for weight updates, on every layer.
        /// @param opt The optimizer type to apply.
        void SetOptimizer(OptimizerType opt) noexcept
        {
            for (auto &layer : layers)
                layer->SetOptimizer(opt);
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

        /// @brief Saves the network's learned parameters to disk.
        /// @param filename The destination file path.
        /// @return True on success, false otherwise.
        bool Save(const std::string &filename) const noexcept;
        /// @brief Loads the network's learned parameters from disk.
        /// @param filename The source file path.
        /// @return True on success, false otherwise.
        bool Load(const std::string &filename) noexcept;

        /// @brief Loads all layers into one contiguous block of memory.
        void Compile();

    private:
        std::unique_ptr<MemoryArena> arena;
        friend class PCNDiagnostics;
    };
}