#pragma once
#include "deepity/Activations.h"
#include <memory>

/// @important This file is currently WIP! It is not suitable for use (yet).
/// @brief This is a Predictive Coding Network implemented with Direct Kolen-Pollock Feedback Alignment: @cite https://arxiv.org/html/2602.15571v2

namespace Deep
{
    class DKP_PCLayer
    {
    public:
        DKP_PCLayer(DKP_PCLayer *prevLayer, DKP_PCLayer *nextLayer, size_t inSize, size_t outSize, float lr = 1e-4);

    private:
        float lr;
        size_t inSize;
        size_t outSize;

        std::unique_ptr<float> feedback;

        DKP_PCLayer *prevLayer;
        DKP_PCLayer *nextLayer;

        ActivationFn act;
        DerivativeFn dAct;
    };
}