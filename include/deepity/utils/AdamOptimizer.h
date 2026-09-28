#pragma once

/**
 * @file AdamOptimizer.h
 * @brief The OptimizerType enum shared by every PC layer's UpdateWeights()
 * (actual Adam/AdamW math lives in IComputeBackend::AdamStep/AdamWStep).
 */

namespace Deep
{
    /// @brief Selects which weight-update rule a layer applies during UpdateWeights().
    enum class OptimizerType
    {
        SGD,   ///< Plain stochastic gradient descent.
        ADAM,  ///< Adam, no weight decay.
        ADAMW  ///< Adam with decoupled weight decay.
    };
}
