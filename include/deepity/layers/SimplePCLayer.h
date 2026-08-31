#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include <deepity/Activations.h>
#include <deepity/AdamOptimizer.h>
#include <deepity/layers/Layer.h>
#include <deepity/MemoryArena.h>

/**
 * @file SimplePCLayer.h
 * @brief DiscriminativePCLayer with precision entirely removed.
 *
 * Precision weighting (p, log_p, UpdatePrecision()) was found this session
 * to be an unused cost, not a benefit: every real training run used pr=0.0
 * (precision inert -- p stays permanently at 1, log_p at 0), yet
 * UpdatePrecision() was still called unconditionally every train_step(),
 * and every layer paid for the p buffer, the log_p buffer, and the
 * precision-weighted energy formula's extra multiply/log term on every
 * single CalculateState()/UpdateState() call, for zero behavioral benefit.
 *
 * This class removes all of that: energy reduces to plain
 * E = 0.5*sum(e^2) (no precision weighting, no -0.5*log(p) term), and
 * UpdateState()'s own term reduces to dz_dt = -e (no p multiplier, since
 * p=1 always made that multiplication a no-op anyway).
 *
 * Mirrors DiscriminativePCLayer's public interface as closely as possible
 * (same AddLayer-style construction pattern, same ActivationType-based
 * constructor, same GetBeliefs/GetErrors/GetWeights/GetBiases accessors)
 * so it's a near-drop-in swap for anyone not using precision -- which, per
 * the above, was everyone.
 *
 * @warning Not yet gradient-checked independently -- this is a direct
 * strip-down of DiscriminativePCLayer's already-verified math (same
 * CalculateState/UpdateState/UpdateWeights formulas, precision terms
 * removed), not new math, but the removal itself hasn't been re-verified
 * via finite-difference check. Do that before trusting this for real
 * training -- same discipline as every other change this session.
 *
 * @note Optionally uses Adam (see AdamOptimizer.h) instead of plain SGD
 * for weight updates, selected via SetOptimizer(); Adam's moment buffers
 * are only allocated when selected, keeping the SGD-only path free of
 * their memory/compute cost.
 *
 * @note Optionally caches a CLAMPED layer's outgoing prediction (mu)
 * across settling steps via SetMuCaching() -- a clamped layer's z is
 * fixed for the whole settling loop, and W/b don't change until
 * UpdateWeights() runs afterward, so mu=f(W@z+b) is PROVABLY IDENTICAL
 * every step while clamped. This is an EXACT optimization, not an
 * approximation -- default OFF so both behaviors coexist for direct
 * correctness/timing comparison before trusting it.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    class SimplePCNDiagnostics;

    class SimplePCLayer : public Layer
    {
    public:
        /// @brief Constructor for a SimplePCLayer.
        /// @param size Size of this layer's own belief (z)
        /// @param nextSize Size of the layer above's belief (this layer's
        ///        outgoing prediction target); 0 marks a terminal layer
        /// @param batchSize Batch size
        /// @param learningRate Learning rate for weight updates
        /// @param inferenceRate Inference rate (Euler integration step size)
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param act Activation function
        /// @param dAct Derivative of activation function
        SimplePCLayer(int size, int nextSize, int batchSize = 1,
                      float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                      void (*act)(float *, size_t) = relu,
                      void (*dAct)(float *, size_t, bool) = dRelu);

        /// @brief Constructor for a SimplePCLayer, using a named
        /// ActivationType instead of raw function pointers.
        /// @param size Size of this layer's own belief (z)
        /// @param nextSize Size of the layer above's belief (this layer's
        ///        outgoing prediction target); 0 marks a terminal layer
        /// @param batchSize Batch size
        /// @param learningRate Learning rate for weight updates
        /// @param inferenceRate Inference rate (Euler integration step size)
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param aType Activation type
        /// @param dType Activation derivative type
        SimplePCLayer(int size, int nextSize, int batchSize = 1,
                      float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                      ActivationType aType = ActivationType::RELU, ActivationType dType = ActivationType::dRELU);

        /// @brief Calculates the total network energy state, and this
        /// layer's outgoing prediction (mu).
        ///
        /// \f[
        /// E = \sum_l 1/2 ||z^{(l)} - \mu^{(l)}||^2
        /// \f]
        /// mu is now computed as mu = W @ phi(z) + b -- activation
        /// applied to z BEFORE the linear transform, matching
        /// ngc-learn's documented convention exactly (was previously
        /// mu = phi(W@z+b), activation AFTER the transform).
        /// (No precision weighting -- see file-level note.)
        /// @return This layer's energy contribution at the current state.
        float CalculateState() noexcept override;

        /// @brief Computes the state derivatives for inference.
        ///
        /// \f[
        /// \frac{dz^{(l)}}{dt} = -e^{(l)} + \sigma'(z^{(l)}) \odot (W^{(l-1)})^T e^{(l-1)}
        /// \f]
        /// The derivative multiply now applies AFTER the W transform,
        /// using f'(z) (this layer's OWN state derivative), not f'(mu)
        /// applied before -- matching the activate-before-transform
        /// convention.
        void UpdateState() noexcept override;

        /// @brief Computes weight updates via gradient descent, with L2 weight decay.
        void UpdateWeights() noexcept override;

        /// @brief No-op; exists for Layer interface conformance.
        void Flush() noexcept override {}

        /// @brief Clamps this layer's beliefs to externally-provided data,
        /// preventing them from being updated by UpdateState().
        /// @param inputData Flattened input data of length `size`.
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Releases a previous ClampState() call, allowing this
        /// layer's beliefs to update normally again.
        void UnclampState() noexcept;

        /// @brief Returns this layer's belief buffer.
        /// @return Pointer to this layer's `size`-length beliefs.
        float *GetBeliefs() noexcept override { return z; }

        /// @brief Returns a prediction buffer
        /// exposed specifically for forward-projection initialization (seeding
        /// the next layer's z from a genuine, current-weights forward pass,
        /// rather than zero-init). nullptr for a terminal layer (nextSize=0).
        /// @return Outgoing prediction buffer
        const float *GetMu() const noexcept { return mu; }

        /// @brief Returns this layer's prediction-error buffer.
        /// @return Pointer to this layer's `size`-length errors.
        const float *GetErrors() const noexcept override { return e; }
        /// @brief Returns this layer's own belief size.
        size_t GetInputSize() const noexcept override { return size; }
        /// @brief Returns this layer's outgoing prediction size (0 for a
        /// terminal layer).
        size_t GetOutputSize() const noexcept override { return nextSize; }
        /// @brief Returns the batch size this layer was constructed with.
        size_t GetBatchSize() const noexcept override { return batchSize; }

        /// @brief Returns this layer's weight buffer.
        const float *GetWeights() const noexcept { return W; }
        /// @brief Returns this layer's weight buffer.
        float *GetWeights() noexcept { return W; }
        /// @brief Returns this layer's bias buffer.
        const float *GetBiases() const noexcept { return b; }
        /// @brief Returns this layer's bias buffer.
        float *GetBiases() noexcept { return b; }

        /// @brief Returns the learning rate used for weight updates.
        float GetLearningRate() const noexcept { return lr; }
        /// @brief Returns the inference rate (Euler integration step size).
        float GetInferenceRate() const noexcept { return ir; }
        /// @brief Returns the weight-decay (L2 regularization) coefficient.
        float GetLambda() const noexcept { return lmbda; }

        /// @brief Sets the learning rate used for weight updates.
        /// @param lr The new learning rate.
        void SetLearningRate(float lr) noexcept { this->lr = lr; }
        /// @brief Sets the inference rate (Euler integration step size).
        /// @param ir The new inference rate.
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        /// @brief Sets the weight-decay (L2 regularization) coefficient.
        /// @param l The new lambda value.
        void SetLambda(float l) noexcept { this->lmbda = l; }

        /// @brief Selects the optimizer used for weight updates (SGD or
        /// Adam). Adam's moment buffers are lazily allocated on first use.
        /// @param o The optimizer type to use.
        void SetOptimizer(const OptimizerType o) noexcept { opt = o; }

        /// @brief Sets the mu-cache staleness threshold: mu is recomputed
        /// only when ||z - z_at_last_recompute|| / (||z_at_last_recompute||
        /// + eps) exceeds this value. threshold=-1 disables caching
        /// entirely (default). threshold=0 reproduces the original,
        /// EXACT clamped-only behavior (a clamped layer's z never changes,
        /// so its ratio is always exactly 0, always below any
        /// threshold>=0). threshold>0 extends caching to UNCLAMPED layers
        /// too, as a genuine approximation -- correctness there means
        /// "close enough for real training," not "bit-identical," and
        /// needs its own accuracy-impact validation, not just a trajectory
        /// diff.
        void SetMuCacheThreshold(float threshold) noexcept { muCacheThreshold = threshold; }
        float GetMuCacheThreshold() const noexcept { return muCacheThreshold; }

	/// @brief Computes only mu (forward prediction), skipping error/energy
	/// entirely. Extracted from CalculateState() for callers (like
	/// ProjectForward()) that don't need the discarded error/energy values.
	/// Computes zF=phi(z) internally and uses it for the GEMM, matching
	/// the activate-before-transform convention (mu=W@phi(z)+b).
	void ComputeMuOnly() noexcept;

        /// @brief Sets the layer immediately above this one in the network.
        /// @param above Pointer to the layer above; may be nullptr for a
        /// terminal layer.
        void SetLayerAbove(SimplePCLayer *above) noexcept { layerAbove = above; }
        /// @brief Sets the layer immediately below this one in the network.
        /// @param below Pointer to the layer below; may be nullptr for the
        /// input layer.
        void SetLayerBelow(SimplePCLayer *below) noexcept { layerBelow = below; }

        /// @brief Resets this layer's beliefs/errors back to their initial
        /// values, without touching learned weights.
        void ResetState() noexcept;

        /// @brief Returns the layer immediately above this one.
        /// @warning Dereferences layerAbove without a null check; only
        /// valid if SetLayerAbove() was previously called with a non-null
        /// pointer.
        const SimplePCLayer &GetLayerAbove() const noexcept { return *layerAbove; }
        /// @brief Returns the layer immediately below this one.
        /// @warning Dereferences layerBelow without a null check; only
        /// valid if SetLayerBelow() was previously called with a non-null
        /// pointer.
        const SimplePCLayer &GetLayerBelow() const noexcept { return *layerBelow; }

        /// @brief Randomizes this layer's weights (and biases) in place.
        /// @param twister The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        /// @brief Returns this layer's configured activation type.
        ActivationType GetActivationType() const noexcept { return To_AType(activation); }
        /// @brief Returns this layer's configured activation-derivative type.
        ActivationType GetDerivativeType() const noexcept { return To_AType(activationDerivative); }

        /// @brief Computes the total number of floats this layer requires
        /// from a MemoryArena (weights, biases, beliefs, errors, scratch
        /// buffers, and, if Adam is selected, its moment buffers).
        /// @return The required float count.
        size_t GetRequiredFloats() const noexcept;
        /// @brief Binds this layer's weight/state/scratch buffers into the
        /// supplied arena. Must be called before any other operation.
        /// @param arena The MemoryArena to bind into.
        void BindMemory(MemoryArena &arena);

    private:
        std::unique_ptr<MemoryArena> localArena;
        float *W;
        float *b;
        float *e;
        float *z;

        int batchSize;

        float *mu;
        float *cachedMu; // separate from mu -- mu gets mutated in-place by
                         // UpdateState() every step (converted to its
                         // derivative), so caching must copy a preserved
                         // value back INTO mu each skipped step, not just
                         // skip writing to mu entirely
       float *zF;              // this layer's OWN activated belief,
                                 // phi(z) -- needed both for the forward
                                 // GEMM (mu=zF@W+b) AND the weight
                                 // gradient (dW uses zF, not raw z),
                                 // matching ngc-learn's documented
                                 // convention: mu_l = W_l . phi(z_{l-1})
        float *zFDeriv;         // SEPARATE scratch for f'(z), computed
                                 // from RAW z (activated=false flag) --
                                 // zF itself must stay clean/activated
                                 // for the weight-gradient step
        float *feedbackScratch; // own_state_size scratch for the raw
                                 // feedback GEMM's output -- the f'(z)
                                 // multiply applies ONLY to the feedback
                                 // term, not the -e term already in
                                 // dz_dt, so it can't accumulate directly
                                 // into dz_dt via the GEMM itself
        float lr;
        float ir;
        float lmbda;
        bool isClamped = false;

        float muCacheThreshold = -1.0f; // -1 = disabled. 0 = exact clamped-only
                                        // (today's validated behavior). >0 =
                                        // approximate, extends to unclamped
                                        // layers too.
        bool muCacheValid = false;      // true once mu computed at least once
                                        // since the most recent ClampState()

        SimplePCLayer *layerAbove;
        SimplePCLayer *layerBelow;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        DerivativeFn2 activationDerivativeInto; // fused two-buffer derivative
                                                  // (dst, src, n), resolved once
                                                  // at construction alongside
                                                  // activation/activationDerivative
        ActivationType activationType;
        OptimizerType opt = OptimizerType::SGD;
        // @private Optional Adam weights
        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;
        int t = 0;

        friend class SimplePCNDiagnostics;
    };

} // namespace Deep
