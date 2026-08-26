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
 * @file GaussSeidelPCLayer.h
 * @brief A PC layer whose settling dynamics follow a Gauss-Seidel
 * (sequential-sweep) update, matching ngc-learn's actual execution
 * order -- NOT the Jacobi (fully-synchronous) update SimplePCLayer uses.
 *
 * @note THIS IS AN UNPROVEN EXPERIMENT, not a confirmed improvement.
 * Classical numerical-methods results suggest Gauss-Seidel typically
 * needs fewer iterations than Jacobi for diagonally-dominant/symmetric
 * systems, and our CPU/OpenMP implementation was never exploiting
 * cross-layer parallelism anyway (unlike a GPU/JAX backend, where
 * Gauss-Seidel's sequential dependency chain is a real cost) -- so the
 * downside risk looks smaller here than it would for a GPU
 * implementation. But this has NOT been empirically confirmed for this
 * specific nonlinear, learned-weight system. Gradient-check independently
 * before trusting it, same discipline as every other new formula this
 * session.
 *
 * KEY STRUCTURAL DIFFERENCE from SimplePCLayer: there is no single
 * CalculateState() that does everything. Traced directly from
 * ngc-learn's real advance_process chain (E2,E3 -> z0,z1,z2,z3 -> W1,W2,
 * W3 -> e1,e2,e3), a full Gauss-Seidel timestep is THREE separate sweeps
 * across ALL layers, in this exact order:
 *
 *   1. UpdateState() on every layer       -- z updates using mu/e_above
 *      HELD OVER from the end of the PREVIOUS timestep (not yet fresh
 *      for this timestep).
 *   2. ComputePrediction() on every layer -- mu recomputes fresh, using
 *      the JUST-updated z from step 1. Order among layers doesn't matter
 *      here (each layer's mu only depends on its OWN z).
 *   3. ComputeError() on every layer      -- e recomputes fresh, using
 *      the JUST-updated z (step 1) as the target and layerBelow's FRESH
 *      mu (step 2) as the prediction. Order among layers doesn't matter
 *      here either.
 *
 * This differs from SimplePCLayer's single-call CalculateState(), which
 * computes error and prediction together, and whose settling loop is
 * simply (CalculateState(); UpdateState();) x N -- fully synchronous
 * (Jacobi): every layer only ever reads values from the END of the
 * PREVIOUS full step, never a value updated earlier in the SAME step.
 *
 * A GaussSeidelPCNetwork class is required to actually orchestrate the
 * three sweeps above in the correct order across all layers -- calling
 * these methods directly, out of order, or on a single layer in
 * isolation, will not reproduce the intended dynamics.
 *
 * IMPORTANT BUFFER-LIFETIME NOTE (the same class of bug mu-caching hit
 * earlier): mu must NOT be mutated in place into its own derivative the
 * way SimplePCLayer's UpdateState() does -- it needs to stay in its
 * clean, activated form, since the layer ABOVE's ComputeError() reads it
 * later in this SAME timestep. UpdateState() and UpdateWeights() each
 * compute mu's derivative into a separate scratch buffer (muDeriv)
 * instead, leaving mu itself untouched.
 *
 * Deliberately does NOT include mu-caching or activateBeforeTransform in
 * this first version -- isolating the Gauss-Seidel restructuring itself
 * for independent verification before layering anything else on top.
 *
 * @warning Not yet gradient-checked. Do that before trusting this for
 * real training.
 * @version 1.0
 * @date 2026-08-26
 */

namespace Deep
{
    class GaussSeidelPCNDiagnostics;

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
        /// @param act Activation function
        /// @param dAct Derivative of activation function
        GaussSeidelPCLayer(int size, int nextSize, int batchSize = 1,
                           float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                           void (*act)(float *, size_t) = relu,
                           void (*dAct)(float *, size_t, bool) = dRelu);

        /// @brief Constructor using a named ActivationType instead of raw
        /// function pointers.
        GaussSeidelPCLayer(int size, int nextSize, int batchSize = 1,
                           float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                           ActivationType aType = ActivationType::RELU, ActivationType dType = ActivationType::dRELU);

        /// @brief Step 1 of a Gauss-Seidel timestep: updates z using
        /// mu/e_above HELD OVER from the end of the previous timestep.
        /// Does NOT mutate mu -- computes its derivative into a separate
        /// scratch buffer.
        /// @return Always 0.0f -- energy is only meaningful from
        /// ComputeError(), kept for Layer interface conformance.
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
        /// completes, matching ngc-learn's evolve_process.
        void UpdateWeights() noexcept override;

        /// @brief No-op; exists for Layer interface conformance.
        void Flush() noexcept override {}

        /// @brief Clamps this layer's beliefs to externally-provided data.
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Releases a previous ClampState() call.
        void UnclampState() noexcept;

        float *GetBeliefs() noexcept override { return z; }
        const float *GetMu() const noexcept { return mu; }
        const float *GetErrors() const noexcept override { return e; }
        size_t GetInputSize() const noexcept override { return size; }
        size_t GetOutputSize() const noexcept override { return nextSize; }
        size_t GetBatchSize() const noexcept override { return batchSize; }

        const float *GetWeights() const noexcept { return W; }
        float *GetWeights() noexcept { return W; }
        const float *GetBiases() const noexcept { return b; }
        float *GetBiases() noexcept { return b; }

        float GetLearningRate() const noexcept { return lr; }
        float GetInferenceRate() const noexcept { return ir; }
        float GetLambda() const noexcept { return lmbda; }

        void SetLearningRate(float lr) noexcept { this->lr = lr; }
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        void SetLambda(float l) noexcept { this->lmbda = l; }
        void SetOptimizer(const OptimizerType o) noexcept { opt = o; }

        void SetLayerAbove(GaussSeidelPCLayer *above) noexcept { layerAbove = above; }
        void SetLayerBelow(GaussSeidelPCLayer *below) noexcept { layerBelow = below; }

        /// @brief Resets beliefs/errors/predictions back to their initial
        /// values, without touching learned weights.
        void ResetState() noexcept;

        const GaussSeidelPCLayer &GetLayerAbove() const noexcept { return *layerAbove; }
        const GaussSeidelPCLayer &GetLayerBelow() const noexcept { return *layerBelow; }

        void RandomizeWeights(std::mt19937 &twister) noexcept;

        ActivationType GetActivationType() const noexcept { return To_AType(activation); }
        ActivationType GetDerivativeType() const noexcept { return To_AType(activationDerivative); }

        size_t GetRequiredFloats() const noexcept;
        void BindMemory(MemoryArena &arena);

    private:
        std::unique_ptr<MemoryArena> localArena;
        float *W;
        float *b;
        float *e;
        float *z;

        int batchSize;

        float *mu;      // this layer's OWN outgoing prediction -- stays
                        // CLEAN/activated at all times; never mutated
                        // into a derivative in place
        float *muDeriv; // scratch buffer for mu's derivative, computed
                        // fresh whenever needed (UpdateState(),
                        // UpdateWeights()) -- copy mu here, derive in
                        // place, leaving the real mu untouched
        float *dz_dt;
        float *bottom_up;

        float lr;
        float ir;
        float lmbda;
        bool isClamped = false;

        GaussSeidelPCLayer *layerAbove;
        GaussSeidelPCLayer *layerBelow;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        ActivationType activationType;
        OptimizerType opt = OptimizerType::SGD;

        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;
        int t = 0;

        friend class GaussSeidelPCNDiagnostics;
    };

} // namespace Deep