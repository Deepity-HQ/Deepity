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
    class PCNetwork
    {
        std::vector<PCLayer *> layers;
        int batchSize;
        bool autoSize = true;

    public:
        PCNetwork() : batchSize(0), autoSize(true) {}
        PCNetwork(int batchSize) : batchSize(batchSize), autoSize(false) {}

        ~PCNetwork();

        void AddLayer(int size, int nextSize, float lr,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t));

        void RandomizeWeights(std::mt19937 &rng);

        void Clamp(const std::vector<float> &input);

        float CalculateState();

        void UpdateState();

        void UpdateWeights();

        int GetBatchSize() const noexcept { return batchSize; }
    };
}