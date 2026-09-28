#pragma once
#include <cstddef>
#include <sleef.h>
#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <map>
#include <vector>
#include <string>

/**
 * @file Layer.h
 * @brief Virtual Layer.
 *
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    /// @brief A named, shaped view into a layer's weight/bias data, for
    /// ModelIO's save/load (see Layer::GetStateDict()).
    struct TensorDescriptor
    {
        float *data;         ///< Pointer to the tensor's first element.
        std::vector<size_t> shape; ///< Tensor's dimensions, e.g. {out, in}.
    };
    /// @brief A deepity layer virtual class.
    class Layer
    {
    public:
        virtual ~Layer() = default;

        /// @brief Returns this layer's belief state `z`.
        virtual float *GetBeliefs() noexcept = 0;
        /// @brief Returns this layer's errors `e`.
        virtual const float *GetErrors() const noexcept = 0;
        /// @brief Returns this layer's input size.
        virtual size_t GetInputSize() const noexcept = 0;
        /// @brief Returns this layer's output size.
        virtual size_t GetOutputSize() const noexcept = 0;
        /// @brief Returns this layer's batch size.
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
        /// @brief Returns this layer's named, shaped weight/bias tensors,
        /// for ModelIO's save/load. Empty by default; override to expose
        /// a layer's actual parameters.
        virtual std::map<std::string, TensorDescriptor> GetStateDict() const { return {}; }

    protected:
        /// @brief Size of input
        size_t size;
        /// @brief Size of output
        size_t nextSize;
    };
}
