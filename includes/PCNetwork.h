#pragma once
#include <vector>
#include "PCLayer.h"
#include "Optimize.h"

/**
 * @file PCLayer.h
 * @brief Defines the network-level implementation of a PC model.
 *
 * This header includes implementations of PC layer-to-layer interaction.
 *
 * Usage:
 *  #include <PCNetwork.h>
 *
 * Example:
 *  Deep::PCNetwork network(1);
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
    /// @brief An abstracted class for an array of `PCLayer`
    ///
    /// @see https://arxiv.org/pdf/2506.06332
    class PCNetwork
    {
        std::vector<PCLayer *> layers;
        int batchSize;
        bool autoSize = true;

    public:
        /// @brief Default constructor
        ///
        /// Initializes the network with auto-batch size detection.
        PCNetwork() : batchSize(0), autoSize(true) {}
        /// @brief Batched constructor
        /// @param batchSize Batch size
        /// 
        /// Initializes the network with a predetermined batch size.
        PCNetwork(int batchSize) : batchSize(batchSize), autoSize(false) {}

        /// @brief Default constructor; deletes each layer.
        ~PCNetwork();

        /// @brief Adds a layer to the network.
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for beliefs
        /// @param act activation function
        /// @param dAct derivative of previous activation function
        void AddLayer(int size, int nextSize, float lr,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t));
        
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

        /// @brief Returns the batch size for the network's layers
        /// @return size_t batchSize
        int GetBatchSize() const noexcept { return batchSize; }
    };
}