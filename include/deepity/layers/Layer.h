#pragma once
#include <cstddef>
#include <sleef.h>
#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

/**
 * @file Layer.h
 * @brief Virtual Layer.
 *
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

// Adam:
#define ALPHA 0.001
#define BETA1 0.9
#define BETA2 0.999
#define EPS 1e-8

namespace Deep
{
    /// @brief A deepity layer virtual class.
    class Layer
    {
    public:
        virtual ~Layer() = default;

        /// @brief Should return belief state `z`
        /// @return ? = 0
        virtual float *GetBeliefs() noexcept = 0;
        /// @brief Should return errors `e`
        /// @return ? = 0
        virtual const float *GetErrors() const noexcept = 0;
        /// @brief Should return input size
        /// @return ? = 0
        virtual size_t GetInputSize() const noexcept = 0;
        /// @brief Should return output size
        /// @return ? = 0
        virtual size_t GetOutputSize() const noexcept = 0;
        /// @brief Should return batch size
        /// @return ? = 0
        virtual size_t GetBatchSize() const noexcept = 0;

        /// @brief Calculates the internal state of the layer.
        /// @return energy used
        virtual float CalculateState() noexcept = 0;
        /// @brief Updates the states after calculation.
        virtual void UpdateState() noexcept = 0;
        /// @brief Updates the weights after state updates.
        virtual void UpdateWeights() noexcept = 0;
        /// @brief Flushes remaining batches.
        virtual void Flush() noexcept {}

    protected:
        /// @brief Size of input
        size_t size;
        /// @brief Size of output
        size_t nextSize;
    };
}
