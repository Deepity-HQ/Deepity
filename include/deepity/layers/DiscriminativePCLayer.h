#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <map>
#include <memory>
#include <random>
#include <vector>

/**
 * @file DiscriminativePCLayer.h
 * @brief The original, precision-weighted dense Predictive Coding layer,
 * now routed through IComputeBackend for GPU portability, mirroring
 * ConvPCLayer's own port (this layer's dense formulas are in fact the
 * ones ConvPCLayer's derivation was adapted from, see
 * @ref ConvPCLayer_math).
 *
 * @code{.cpp}
 * #include <deepity/layers/DiscriminativePCLayer.h>
 *
 * Deep::DiscriminativePCLayer layer(2, 3, 1, 1e-6);
 * layer.ClampState({ 0.1f, 0.75f });
 * layer.CalculateState();
 * @endcode
 *
 * @note All members are stored as pointers except for the input itself.
 * @version 2.0
 * @date 2026-09-27
 * @author Jack Rose
 */

namespace Deep
{
    class PCNDiagnostics;

    /// @brief Precision-weighted dense Predictive Coding layer, routed
    /// through IComputeBackend for GPU portability.
    ///
    /// To perform inference (prediction):
    /// 1. Clamp the input data to the bottom layer's latent state (z).
    /// 2. Enter a continuous loop across all layers, calling CalculateState()
    ///    to compute the local prediction errors (e).
    /// 3. Call UpdateState() to adjust the latent states (z) based on those errors.
    /// 4. Repeat steps 2 and 3 until the states settle into an equilibrium
    ///    (the energy is minimized and dz/dt approaches zero).
    /// 5. Read the final predictions from the latent states of the desired layers.
    /// 6. (For learning): Call UpdateWeights() on all layers simultaneously
    ///    once the states have fully settled.
    class DiscriminativePCLayer : public Layer
    {
    public:
        /// @brief Constructor for a Deepity Layer.
        /// @param size Size of this layer's own belief (z)
        /// @param nextSize Size of the layer above's belief (this layer's
        ///        outgoing prediction target); 0 marks a terminal layer
        /// @param batchSize Batch size (default to 1 for simplicity)
        /// @param learningRate Learning rate to update weights
        /// @param inferenceRate Learning rate to update state
        /// @param precisionRate Precision weight learning rate
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param aType Activation type
        /// @param dType Activation derivative type
        /// @param backend Compute backend to route all math through.
        ///        Defaults to nullptr, in which case this layer
        ///        constructs and owns its own CPUBackend internally,
        ///        existing callers don't need to change anything. Pass a
        ///        real backend (owned elsewhere, e.g. by the network) to
        ///        run this layer on GPU.
        DiscriminativePCLayer(int size, int nextSize, int batchSize = 1,
                              float learningRate = 1e-6f, float inferenceRate = 0.1f,
                              float precisionRate = 0.01f, float lmbda = 1e-2f,
                              ActivationType aType = ActivationType::RELU,
                              ActivationType dType = ActivationType::dRELU,
                              IComputeBackend *backend = nullptr);

        /// @brief Calculates the total network energy state.
        ///
        /// \f[
        /// E = \sum_l 1/2 p^{(l)} ||z^{(l)} - \mu^{(l)}||^2 - 1/2 \log p^{(l)}
        /// \f]
        /// @return This layer's energy contribution at the current state.
        float CalculateState() noexcept override;

        /// @brief Computes the state derivatives for inference.
        ///
        /// \f[
        /// \frac{dz^{(l)}}{dt} = -p^{(l)} e^{(l)} + \sigma'(z^{(l)}) \odot
        /// \left((e^{(l+1)} \odot p^{(l+1)}) W^{(l)}\right)
        /// \f]
        void UpdateState() noexcept override;

        /// @brief Computes weight updates via gradient descent, with L2 weight decay.
        ///
        /// \f[
        /// W^{(l)} \leftarrow (1 - \lambda) W^{(l)} + \eta (e^{(l+1)} \odot
        /// p^{(l+1)})^T \phi(z^{(l)})
        /// \f]
        void UpdateWeights() noexcept override;

        /// @brief Updates this layer's precision weighting based on the
        /// current prediction error.
        void UpdatePrecision() noexcept;

        /// @brief Does nothing; exists for class extension.
        void Flush() noexcept override {}

        /// @brief Recomputes log of precision if it falls behind.
        void ResyncLogPrecision() noexcept;

        /// @brief Clamps the layer to the input data.
        /// @param inputData Input
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Unclamps the layer from the input data.
        void UnclampState() noexcept;
        /// @brief Whether ClampState() is currently active on this layer.
        bool IsClamped() const noexcept { return isClamped; }

        /// @brief Returns beliefs.
        /// @return float *z
        float *GetBeliefs() noexcept override { return z; }
        /// @brief Returns errors.
        /// @return const float *e
        const float *GetErrors() const noexcept override { return e; }

        /// @brief Returns this layer's outgoing prediction buffer.
        /// @return const float *mu
        const float *GetMu() const noexcept { return mu; }
        /// @brief Returns size (input size).
        /// @return size_t size
        size_t GetInputSize() const noexcept override { return size; }
        /// @brief Returns nextSize (output size).
        /// @return size_t nextSize
        size_t GetOutputSize() const noexcept override { return nextSize; }
        /// @brief Returns batchSize.
        /// @return size_t batchSize
        size_t GetBatchSize() const noexcept override { return batchSize; }

        /// @brief Returns a read-only version of the stored weights.
        /// @return const float *W
        const float *GetWeights() const noexcept { return W; }
        /// @brief Returns a mutable version of the stored weights.
        /// @return float *W
        float *GetWeights() noexcept { return W; }
        /// @brief Returns a read-only version of the stored biases.
        /// @return const float *b
        const float *GetBiases() const noexcept { return b; }
        /// @brief Returns a mutable version of the stored biases.
        /// @return float *b
        float *GetBiases() noexcept { return b; }

        /// @brief Returns a read-only version of the stored precisions.
        /// @return const float *p
        const float *GetPrecisions() const noexcept { return p; }

        /// @brief Returns the learning rate used for weight updates.
        /// @return float lr
        float GetLearningRate() const noexcept { return lr; }
        /// @brief Returns the inference rate used for state updates.
        /// @return float ir
        float GetInferenceRate() const noexcept { return ir; }
        /// @brief Returns the learning rate used for precision updates.
        /// @return float pr
        float GetPrecisionRate() const noexcept { return pr; }
        /// @brief Returns the weight-decay (L2 regularization) coefficient.
        /// @return float lmbda
        float GetLambda() const noexcept { return lmbda; }

        /// @brief Sets the learning rate used for weight updates.
        /// @param learningRate The new learning rate.
        void SetLearningRate(float learningRate) noexcept;
        /// @brief Sets the inference rate used for state updates.
        /// @param inferenceRate The new inference rate.
        void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
        /// @brief Sets the learning rate used for precision updates.
        /// @param precisionRate The new precision rate.
        void SetPrecisionRate(float precisionRate) noexcept { pr = precisionRate; }
        /// @brief Sets the weight-decay (L2 regularization) coefficient.
        /// @param l The new lambda value.
        void SetLambda(float l) noexcept { lmbda = l; }
        /// @brief Selects the optimizer used for weight updates.
        /// @param o The optimizer type to use.
        void SetOptimizer(OptimizerType o) noexcept { opt = o; }

        /// @brief Ties this layer to one above it.
        /// @param above DiscriminativePCLayer*
        void SetLayerAbove(DiscriminativePCLayer *above) noexcept { layerAbove = above; }
        /// @brief Ties this layer to one below it.
        /// @param below DiscriminativePCLayer*
        void SetLayerBelow(DiscriminativePCLayer *below) noexcept { layerBelow = below; }

        /// @brief Resets 'z' (beliefs) to 0.
        void ResetState() noexcept;

        /// @brief Gets const reference to the layer above.
        /// @return layerAbove
        const DiscriminativePCLayer &GetLayerAbove() const noexcept { return *layerAbove; }
        /// @brief Gets const reference to the layer below.
        /// @return layerBelow
        const DiscriminativePCLayer &GetLayerBelow() const noexcept { return *layerBelow; }

        /// @brief Randomizes the weights W using a normal distribution
        /// scaled by fan-in/fan-out (see RandomizeWeights()'s .cpp).
        /// @param twister The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        /// @brief Returns this layer's configured activation type.
        /// @return ActivationType
        ActivationType GetActivationType() const noexcept { return activationType; }
        /// @brief Returns this layer's configured activation-derivative type.
        /// @return ActivationType
        ActivationType GetDerivativeType() const noexcept { return derivativeType; }

        /// @brief Calculates exact float count required by this layer's architecture.
        size_t GetRequiredFloats() const noexcept;

        /// @brief Binds all tensor pointers to the contiguous memory block.
        /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
        /// @param arena The arena to bind into.
        template <typename ArenaT>
        void BindMemory(ArenaT &arena);

        /// @brief Computes only mu (forward prediction), skipping error
        /// and energy computation entirely. Useful for callers that only
        /// need a forward pass through current weights (e.g. forward
        /// projection initialization) without the cost of the discarded
        /// error/energy values.
        void ComputeMuOnly() noexcept;

        std::map<std::string, TensorDescriptor> GetStateDict() const override;

    private:
        std::unique_ptr<MemoryArena> localArena;

        /// @brief The compute backend this layer routes all math through
        ///, see SimplePCLayer::backend's doc for the deleter reasoning.
        using BackendDeleter = void (*)(IComputeBackend *);
        std::unique_ptr<IComputeBackend, BackendDeleter> backend;

        /// @brief Weights: (nextSize, size)
        float *W = nullptr;
        /// @brief Biases: (nextSize,)
        float *b = nullptr;
        /// @brief Errors: (size,) per batch item
        float *e = nullptr;
        /// @brief Internal state: (size,) per batch item
        float *z = nullptr;
        /// @brief Precision: (size,)
        float *p = nullptr;
        /// @brief Log of precision: (size,)
        float *log_p = nullptr;

        /// @brief Cached copy of this layer's most recently computed
        /// outgoing prediction, used to skip recomputation while clamped.
        float *cachedMu = nullptr;
        /// @brief True once cachedMu holds a valid, up-to-date value.
        bool muCacheValid = false;

        /// @brief This layer's own activated belief, phi(z), used both
        /// for the forward GEMM (mu=zF@W^T+b) and the weight gradient.
        float *zF = nullptr;
        /// @brief Scratch buffer for the feedback term's raw GEMM output,
        /// prior to the elementwise activation-derivative multiply.
        float *feedbackScratch = nullptr;

        int batchSize;

        /// @brief This layer's outgoing prediction buffer.
        float *mu = nullptr;
        /// @brief Buffer holding the current state derivative (dz/dt).
        float *dz_dt = nullptr;
        /// @brief Scratch buffer for the precision-weighted bottom-up
        /// feedback term (e_above * p_above); reused as the weight-update
        /// local gradient in UpdateWeights().
        float *bottom_up = nullptr;
        /// @brief Row-summed bias gradient scratch (SGD path only,
        /// matches FullPCLayer::biasGradScratch).
        float *biasGradScratch = nullptr;

        /// @brief Learning rate for weights.
        float lr;
        /// @brief Learning rate for internal state.
        float ir;
        /// @brief Learning rate for precision.
        float pr;
        /// @brief Weight decay (L2 regularization) coefficient.
        float lmbda;
        /// @brief Flag to tell if `ClampState` was called.
        bool isClamped = false;

        /// @brief The optimizer currently selected for weight updates.
        OptimizerType opt = OptimizerType::SGD;
        /// @brief Device-resident Adam step count and learning rate,
        /// same reasoning as ConvPCLayer/SimplePCLayer's own ports.
        int *t_device = nullptr;
        float *lr_device = nullptr;
        /// @brief Scratch buffer for the weight gradient (Adam/AdamW only).
        float *grad_W = nullptr;
        /// @brief Adam/AdamW first-moment estimate for the weights.
        float *m_W = nullptr;
        /// @brief Adam/AdamW second-moment estimate for the weights.
        float *v_W = nullptr;
        /// @brief Scratch buffer for the bias gradient (Adam/AdamW only).
        float *grad_b = nullptr;
        /// @brief Adam/AdamW first-moment estimate for the biases.
        float *m_b = nullptr;
        /// @brief Adam/AdamW second-moment estimate for the biases.
        float *v_b = nullptr;

        /// @brief Pointer to next layer (or `nullptr` if last one).
        DiscriminativePCLayer *layerAbove = nullptr;
        /// @brief Pointer to previous layer (or `nullptr` if first one).
        DiscriminativePCLayer *layerBelow = nullptr;

        /// @brief The named activation type this layer was constructed with.
        ActivationType activationType;
        /// @brief The named activation-derivative type this layer was
        /// constructed with.
        ActivationType derivativeType;

        friend class PCNDiagnostics;
    };

} // namespace Deep

/*
 * MHSA USING PCNs:
 * In a standard Transformer, Multi-Head Self-Attention (MHSA) computes Queries, Keys,
 * and Values in a rigid, one-shot forward pass to route token context. If implemented
 * using a PCN, the attention scores and context vectors would become dynamic latent
 * states (z). When a sequence is input, the attention routing wouldn't be calculated
 * instantly; instead, it would iteratively settle over artificial time. The network
 * would adjust its attention states to minimize the prediction error between the
 * bottom-up token data and the top-down expectations from higher layers. The Q, K,
 * and V projection matrices would serve as the local weights (W) for this mechanism,
 * and would be updated entirely in parallel via local Hebbian rules once the attention
 * routing reached its minimum-energy equilibrium.
 */
