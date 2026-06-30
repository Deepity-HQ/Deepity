#pragma once
#include <cstddef>

namespace Deep
{
    class Layer {
    public:
        virtual ~Layer() = default;

        virtual float *GetBeliefs() noexcept = 0;
        virtual const float *GetErrors() const noexcept = 0;
        virtual size_t GetInputSize() const noexcept = 0;
        virtual size_t GetOutputSize() const noexcept = 0;
        virtual size_t GetBatchSize() const noexcept = 0;

        virtual float CalculateState() noexcept = 0;
        virtual void UpdateState() noexcept = 0;
        virtual void UpdateWeights() noexcept = 0;
        virtual void Flush() noexcept {}
    protected:
        /// @brief Size of input
        size_t size;
        /// @brief Size of output
        size_t nextSize;
    };
}