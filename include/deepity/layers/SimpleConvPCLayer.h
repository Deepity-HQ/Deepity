#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/backend/IComputeBackend.h>

/**
 * @file SimpleConvPCLayer.h
 * @brief ConvPCLayer with precision removed, AdamW/Adam support, routed
 * through IComputeBackend for GPU portability.
 *
 * @warning CPU correctness is verified against a from-scratch Python
 * im2col+GEMM reference (float32 precision). The GPU path has not been
 * tested.
 */

namespace Deep
{
    class SimpleConvPCNDiagnostics;

    /// @brief Convolutional PC layer, precision-free, AdamW-capable,
    /// routed through IComputeBackend for GPU portability.
    class SimpleConvPCLayer : public Layer
    {
    public:
        /// @param inChannels,outChannels Input/output channel counts.
        /// @param inHeight,inWidth Input spatial dimensions.
        /// @param kernelH,kernelW Convolution kernel size.
        /// @param strideH,strideW Convolution stride.
        /// @param padH,padW Zero-padding on each spatial dimension.
        /// @param batchSize Batch size.
        /// @param learningRate Learning rate for W/b.
        /// @param inferenceRate Inference rate (Euler integration step size).
        /// @param lmbda Weight decay (L2 regularization) coefficient.
        /// @param aType Activation type.
        /// @param dType Activation derivative type.
        /// @param backend Compute backend to run on; defaults to a new
        /// CPUBackend if nullptr.
        SimpleConvPCLayer(int inChannels, int outChannels,
                          int inHeight, int inWidth,
                          int kernelH, int kernelW,
                          int strideH = 1, int strideW = 1,
                          int padH = 0, int padW = 0,
                          int batchSize = 1,
                          float learningRate = 1e-6f, float inferenceRate = 0.1f,
                          float lmbda = 1e-2f,
                          ActivationType aType = ActivationType::RELU,
                          ActivationType dType = ActivationType::dRELU,
                          IComputeBackend *backend = nullptr);

        float CalculateState() noexcept override;
        void UpdateState() noexcept override;
        void UpdateWeights() noexcept override;

        void Flush() noexcept override {}

        /// @brief Clamps this layer's beliefs to `inputData`, fixing them
        /// against UpdateState() until UnclampState() is called.
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Releases a previous ClampState() call.
        void UnclampState() noexcept;

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        size_t GetInputSize() const noexcept override { return (size_t)inChannels * inHeight * inWidth; }
        size_t GetOutputSize() const noexcept override
        {
            return outChannels > 0 ? (size_t)outChannels * outHeight * outWidth : 0;
        }
        size_t GetBatchSize() const noexcept override { return batchSize; }

        /// @brief Returns the weight tensor, shape [outChannels, inChannels*kernelH*kernelW].
        const float *GetWeights() const noexcept { return W; }
        /// @brief Mutable overload of GetWeights() above.
        float *GetWeights() noexcept { return W; }
        /// @brief Returns the bias vector, length outChannels.
        const float *GetBiases() const noexcept { return b; }
        /// @brief Mutable overload of GetBiases() above.
        float *GetBiases() noexcept { return b; }

        /// @brief Returns the current learning rate.
        float GetLearningRate() const noexcept { return lr; }
        /// @brief Returns the current inference rate.
        float GetInferenceRate() const noexcept { return ir; }
        /// @brief Returns the current weight decay coefficient.
        float GetLambda() const noexcept { return lmbda; }

        /// @brief Sets the learning rate, propagating it to the optimizer state.
        void SetLearningRate(float lr) noexcept;
        /// @brief Sets the inference rate.
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        /// @brief Sets the weight decay coefficient.
        void SetLambda(float l) noexcept { this->lmbda = l; }
        /// @brief Sets the weight optimizer. Call BEFORE Compile().
        void SetOptimizer(const OptimizerType o) noexcept { opt = o; }
        /// @brief Whether ClampState() is currently active on this layer.
        bool IsClamped() const noexcept { return isClamped; }

        /// @brief Sets the layer immediately above this one in the network.
        void SetLayerAbove(SimpleConvPCLayer *above) noexcept { layerAbove = above; }
        /// @brief Sets the layer immediately below this one in the network.
        void SetLayerBelow(SimpleConvPCLayer *below) noexcept { layerBelow = below; }

        /// @brief Computes only mu (forward prediction), skipping error/energy.
        void ComputeMuOnly() noexcept;

        /// @brief Resets this layer's beliefs (z) without touching learned weights.
        void ResetState() noexcept;
        /// @brief Randomizes W/b.
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        /// @brief Returns this layer's activation type.
        ActivationType GetActivationType() const noexcept { return activationType; }
        /// @brief Returns this layer's activation derivative type.
        ActivationType GetDerivativeType() const noexcept { return derivativeType; }

        /// @brief Returns this layer's forward prediction buffer.
        const float *GetMu() const noexcept { return mu; }
        /// @brief Returns the input channel count.
        int GetInChannels() const noexcept { return inChannels; }
        /// @brief Returns the output channel count (0 for a terminal layer).
        int GetOutChannels() const noexcept { return outChannels; }
        /// @brief Returns the input height.
        int GetInHeight() const noexcept { return inHeight; }
        /// @brief Returns the input width.
        int GetInWidth() const noexcept { return inWidth; }
        /// @brief Returns the output height.
        int GetOutHeight() const noexcept { return outHeight; }
        /// @brief Returns the output width.
        int GetOutWidth() const noexcept { return outWidth; }
        /// @brief Returns the kernel height.
        int GetKernelH() const noexcept { return kernelH; }
        /// @brief Returns the kernel width.
        int GetKernelW() const noexcept { return kernelW; }

        /// @brief Total floats this layer needs from its MemoryArena.
        size_t GetRequiredFloats() const noexcept;

        /// @brief Binds this layer's buffers into a pre-allocated arena
        /// (MemoryArena for CPU, DeviceMemoryArena for GPU).
        /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
        template <typename ArenaT>
        void BindMemory(ArenaT &arena);

    private:
        std::unique_ptr<MemoryArena> localArena;

        int inChannels, outChannels;
        int inHeight, inWidth;
        int outHeight, outWidth;
        int kernelH, kernelW;
        int strideH, strideW;
        int padH, padW;
        int batchSize;

        float *W = nullptr;
        float *b = nullptr;

        float *z = nullptr;
        float *e = nullptr;
        float *dz_dt = nullptr;

        float *mu = nullptr;
        float *colBuffer = nullptr;
        float *feedbackScratch = nullptr;
        float *bottom_up_cols = nullptr;
        float *colsRepacked = nullptr;
        float *lgRepacked = nullptr;
        float *muRepacked = nullptr;

        // All-ones vector for the bias-gradient GEMM trick (grad_b =
        // lgRepacked @ ones), see UpdateWeights()'s implementation
        // comment for why this replaces the original per-row scalar
        // sum loop.
        float *onesVector = nullptr;

        float *cachedMu = nullptr;
        bool muCacheValid = false;

        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;

        // Device-resident t/lr, same reasoning as SimplePCLayer's own
        // port: graph capture (once this reaches GPU) can't re-record
        // for every changed learning rate or Adam step count, so both
        // must live in device memory the graph reads from directly.
        int *t_device = nullptr;
        float *lr_device = nullptr;

        float lr, ir, lmbda;
        bool isClamped = false;

        SimpleConvPCLayer *layerAbove = nullptr;
        SimpleConvPCLayer *layerBelow = nullptr;
        ActivationType activationType;
        ActivationType derivativeType;
        OptimizerType opt = OptimizerType::SGD;

        using BackendDeleter = void (*)(IComputeBackend *);
        std::unique_ptr<IComputeBackend, BackendDeleter> backend;

        friend class SimpleConvPCNDiagnostics;
    };

} // namespace Deep