#pragma once

#include <vector>
#include "DiscriminativePCLayer.h"
#include "Optimize.h"

/**
 * @file DiscriminativePCLayer.h
 * @brief Defines the network-level implementation of a PC model.
 *
 * This header includes implementations of PC layer-to-layer interaction.
 *
 * Usage:
 *  #include <DiscriminativePCNetwork.h>
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
    /// @brief An abstracted class for an array of `DiscriminativePCLayer`
    ///
    /// @see https://arxiv.org/pdf/2506.06332
    class DiscriminativePCNetwork
    {
        std::vector<DiscriminativePCLayer *> layers;
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
        ~DiscriminativePCNetwork();

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

        void UpdatePrecision();

        void ResetState() noexcept;

        const std::vector<DiscriminativePCLayer *> &GetLayers() const noexcept { return layers; }

        /// @brief Returns the batch size for the network's layers
        /// @return size_t batchSize
        int GetBatchSize() const noexcept { return batchSize; }

        DiscriminativePCLayer *GetTerminalLayer() const
        {
            if (layers.empty())
                return nullptr;
            return layers.back();
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

        bool Save(const std::string &filename) const noexcept;
        bool Load(const std::string &filename) noexcept;
    };
}