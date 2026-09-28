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
 * @file GaussSeidelPCLayer.h
 * @brief A PC layer whose settling dynamics follow a Gauss-Seidel
 * (sequential-sweep) update, not the Jacobi (fully-synchronous) update
 * SimplePCLayer uses. Routed through IComputeBackend for GPU
 * portability; feedback goes through the independent matrix E rather
 * than W (see below).
 *
 * KEY STRUCTURAL DIFFERENCE from SimplePCLayer: there is no single
 * CalculateState() that does everything. A full Gauss-Seidel timestep is
 * three separate sweeps across all layers, in this exact order:
 *
 *   1. UpdateState() on every layer: z updates using mu/e_above held
 *      over from the end of the previous timestep, not yet fresh for
 *      this one.
 *   2. ComputePrediction() on every layer: mu recomputes fresh, using
 *      the just-updated z from step 1. Order among layers doesn't
 *      matter here, since each layer's mu only depends on its own z.
 *   3. ComputeError() on every layer: e recomputes fresh, using the
 *      just-updated z (step 1) as the target and layerBelow's fresh mu
 *      (step 2) as the prediction. Order among layers doesn't matter
 *      here either.
 *
 * This differs from SimplePCLayer's single-call CalculateState(), which
 * computes error and prediction together, and whose settling loop is
 * simply (CalculateState(); UpdateState();) x N: fully synchronous
 * (Jacobi), where every layer only ever reads values from the end of the
 * previous full step, never a value updated earlier in the same step.
 *
 * A GaussSeidelPCNetwork class is required to actually orchestrate the
 * three sweeps above in the correct order across all layers. Calling
 * these methods directly, out of order, or on a single layer in
 * isolation will not reproduce the intended dynamics.
 *
 * IMPORTANT BUFFER-LIFETIME NOTE (the same class of bug mu-caching hit
 * elsewhere): mu must not be mutated in place into its own derivative
 * the way SimplePCLayer's UpdateState() does. It needs to stay in its
 * clean, activated form, since the layer above's ComputeError() reads it
 * later in this same timestep. Neither UpdateState() nor UpdateWeights()
 * ever writes to mu; they only read/derive z, via the shared `dz_dt`
 * scratch buffer, reused as a plain activation-derivative or
 * activated-z scratch across phases, never simultaneously.
 *
 * Does not implement mu-caching or activateBeforeTransform.
 *
 * @version 2.0
 * @date 2026-09-27
 */

namespace Deep
{
    class GaussSeidelPCNDiagnostics;

    /// @brief PC layer with Gauss-Seidel (sequential-sweep) settling
    /// dynamics, in place of the usual simultaneous update.
    class GaussSeidelPCLayer : public Layer
    {
    public:
        /// @brief Constructor for a GaussSeidelPCLayer.
        /// @param size Size of this layer's own belief (z)
        /// @param nextSize Size of the layer above's belief (this layer's
        ///        outgoing prediction target); 0 marks a terminal layer
        /// @param batchSize Batch size
        /// @param learningRate Learning rate for weight updates
        /// @param inferenceRate Inference rate (Euler integration step size)
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param aType Activation type
        /// @param dType Activation derivative type
        /// @param backend Compute backend to route all math through.
        ///        Defaults to nullptr, in which case this layer
        ///        constructs and owns its own CPUBackend internally.
        GaussSeidelPCLayer(int size, int nextSize, int batchSize = 1,
                           float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                           ActivationType aType = ActivationType::RELU, ActivationType dType = ActivationType::dRELU,
                           IComputeBackend *backend = nullptr);

        /// @brief Step 1 of a Gauss-Seidel timestep: updates z using
        /// mu/e_above HELD OVER from the end of the previous timestep.
        /// Does NOT mutate mu.
        /// @return Always 0.0f. Energy is only meaningful from
        /// ComputeError(); kept for Layer interface conformance.
        float CalculateState() noexcept override
        {
            UpdateState();
            return 0.0f;
        }

        /// @brief Step 1: the actual z-update. See class-level docs for
        /// the required three-sweep calling order.
        void UpdateState() noexcept override;

        /// @brief Step 2 of a Gauss-Seidel timestep: recomputes mu fresh,
        /// using the z JUST updated by UpdateState() in this same
        /// timestep. Call on EVERY layer before any layer's
        /// ComputeError().
        void ComputePrediction() noexcept;

        /// @brief Step 3 of a Gauss-Seidel timestep: recomputes e fresh,
        /// using this layer's own (just-updated) z as the target and
        /// layerBelow's FRESH mu (from its ComputePrediction() call) as
        /// the prediction.
        /// @return This layer's energy contribution at the current state.
        float ComputeError() noexcept;

        /// @brief Computes weight updates via gradient descent, with L2
        /// weight decay. Called once after the full settling loop
        /// completes.
        void UpdateWeights() noexcept override;

        /// @brief No-op; exists for Layer interface conformance.
        void Flush() noexcept override {}

        /// @brief Clamps this layer's beliefs to externally-provided data.
        /// @param inputData Flattened input data of length `size`.
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Releases a previous ClampState() call.
        void UnclampState() noexcept;
        /// @brief Whether ClampState() is currently active on this layer.
        bool IsClamped() const noexcept { return isClamped; }

        /// @brief Returns this layer's belief buffer.
        /// @return Pointer to this layer's beliefs.
        float *GetBeliefs() noexcept override { return z; }
        /// @brief Returns this layer's outgoing prediction buffer.
        /// @return const float *mu
        const float *GetMu() const noexcept { return mu; }
        /// @brief Returns this layer's prediction-error buffer.
        /// @return const float *e
        const float *GetErrors() const noexcept override { return e; }
        /// @brief Returns this layer's own belief size.
        /// @return size_t size
        size_t GetInputSize() const noexcept override { return size; }
        /// @brief Returns this layer's outgoing prediction size (0 for a
        /// terminal layer).
        /// @return size_t nextSize
        size_t GetOutputSize() const noexcept override { return nextSize; }
        /// @brief Returns the batch size this layer was constructed with.
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
        /// @brief Returns the feedback-alignment matrix used by
        /// UpdateState() in place of W.
        /// @return const float *E
        const float *GetFeedbackWeights() const noexcept { return E; }

        /// @brief Returns the learning rate used for weight updates.
        /// @return float lr
        float GetLearningRate() const noexcept { return lr; }
        /// @brief Returns the inference rate (Euler integration step size).
        /// @return float ir
        float GetInferenceRate() const noexcept { return ir; }
        /// @brief Returns the weight-decay (L2 regularization) coefficient.
        /// @return float lmbda
        float GetLambda() const noexcept { return lmbda; }

        /// @brief Sets the learning rate used for weight updates.
        /// @param learningRate The new learning rate.
        void SetLearningRate(float learningRate) noexcept;
        /// @brief Sets the inference rate (Euler integration step size).
        /// @param inferenceRate The new inference rate.
        void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
        /// @brief Sets the weight-decay (L2 regularization) coefficient.
        /// @param l The new lambda value.
        void SetLambda(float l) noexcept { lmbda = l; }
        /// @brief Selects the optimizer used for weight updates.
        /// @param o The optimizer type to use.
        void SetOptimizer(const OptimizerType o) noexcept { opt = o; }

        /// @brief Sets the layer immediately above this one in the network.
        /// @param above Pointer to the layer above; may be nullptr for a
        /// terminal layer.
        void SetLayerAbove(GaussSeidelPCLayer *above) noexcept { layerAbove = above; }
        /// @brief Sets the layer immediately below this one in the network.
        /// @param below Pointer to the layer below; may be nullptr for the
        /// input layer.
        void SetLayerBelow(GaussSeidelPCLayer *below) noexcept { layerBelow = below; }

        /// @brief Resets beliefs/errors/predictions back to their initial
        /// values, without touching learned weights.
        void ResetState() noexcept;

        /// @brief Returns the layer immediately above this one.
        /// @warning Dereferences layerAbove without a null check; only
        /// valid if SetLayerAbove() was previously called with a non-null
        /// pointer.
        const GaussSeidelPCLayer &GetLayerAbove() const noexcept { return *layerAbove; }
        /// @brief Returns the layer immediately below this one.
        /// @warning Dereferences layerBelow without a null check; only
        /// valid if SetLayerBelow() was previously called with a non-null
        /// pointer.
        const GaussSeidelPCLayer &GetLayerBelow() const noexcept { return *layerBelow; }

        /// @brief Randomizes this layer's weights (and the E
        /// feedback-alignment matrix) in place.
        /// @param twister The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        /// @brief Returns this layer's configured activation type.
        /// @return ActivationType
        ActivationType GetActivationType() const noexcept { return activationType; }
        /// @brief Returns this layer's configured activation-derivative type.
        /// @return ActivationType
        ActivationType GetDerivativeType() const noexcept { return derivativeType; }

        /// @brief Computes the total number of floats this layer requires
        /// from a MemoryArena.
        /// @return The required float count.
        size_t GetRequiredFloats() const noexcept;
        /// @brief Binds this layer's weight/state/scratch buffers into the
        /// supplied arena. Must be called before any other operation.
        /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
        /// @param arena The arena to bind into.
        template <typename ArenaT>
        void BindMemory(ArenaT &arena);

        std::map<std::string, TensorDescriptor> GetStateDict() const override;

    private:
        std::unique_ptr<MemoryArena> localArena;

        using BackendDeleter = void (*)(IComputeBackend *);
        std::unique_ptr<IComputeBackend, BackendDeleter> backend;

        /// @brief Weights.
        float *W = nullptr;
        /// @brief Biases.
        float *b = nullptr;
        /// @brief Errors.
        float *e = nullptr;
        /// @brief Internal state.
        float *z = nullptr;

        int batchSize;

        /// @brief This layer's own outgoing prediction. Stays clean and
        /// activated at all times, never mutated into a derivative.
        float *mu = nullptr;
        /// @brief Shared scratch buffer, own_state_size-length, reused
        /// across phases (never simultaneously): the feedback GEMM's
        /// output in UpdateState(), and the activated-z (phi(z)) input
        /// to the forward GEMM in ComputePrediction()/UpdateWeights().
        float *dz_dt = nullptr;
        /// @brief Feedback-alignment matrix. Separate from W, same
        /// shape, randomly initialized once, never updated again. This
        /// is feedback alignment (Lillicrap et al.), not backprop-style
        /// transposed-weight feedback, so it is not W transposed.
        float *E = nullptr;

        /// @brief Learning rate for weights.
        float lr;
        /// @brief Inference rate (Euler integration step size).
        float ir;
        /// @brief Weight decay (L2 regularization) coefficient.
        float lmbda;
        /// @brief Flag to tell if `ClampState` was called.
        bool isClamped = false;

        /// @brief Pointer to the layer above (or `nullptr` if terminal).
        GaussSeidelPCLayer *layerAbove = nullptr;
        /// @brief Pointer to the layer below (or `nullptr` if the input layer).
        GaussSeidelPCLayer *layerBelow = nullptr;
        /// @brief The named activation type this layer was constructed with.
        ActivationType activationType;
        /// @brief The named activation-derivative type this layer was
        /// constructed with.
        ActivationType derivativeType;
        /// @brief The optimizer currently selected for weight updates.
        OptimizerType opt = OptimizerType::SGD;

        /// @brief Device-resident Adam step count and learning rate,
        /// same reasoning as every other ported layer.
        int *t_device = nullptr;
        float *lr_device = nullptr;
        /// @brief Scratch buffer for the weight gradient (Adam/AdamW only).
        float *grad_W = nullptr;
        /// @brief Scratch buffer for the bias gradient (Adam/AdamW only).
        float *grad_b = nullptr;
        /// @brief Adam/AdamW first-moment estimate for the weights.
        float *m_W = nullptr;
        /// @brief Adam/AdamW second-moment estimate for the weights.
        float *v_W = nullptr;
        /// @brief Adam/AdamW first-moment estimate for the biases.
        float *m_b = nullptr;
        /// @brief Adam/AdamW second-moment estimate for the biases.
        float *v_b = nullptr;
        /// @brief Row-summed bias gradient scratch (SGD path only,
        /// matches FullPCLayer::biasGradScratch).
        float *biasGradScratch = nullptr;

        friend class GaussSeidelPCNDiagnostics;
    };

} // namespace Deep
