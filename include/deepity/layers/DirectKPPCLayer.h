#pragma once

#include <deepity/layers/Layer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/backend/IComputeBackend.h>
#include <memory>
#include <vector>
#include <random>

/**
 * @file DirectKPPCLayer.h
 * @brief Direct Kolen-Pollack predictive coding layer, now routed
 * through IComputeBackend rather than calling cblas_Deep::_ directly
 * -- same refactor discipline as SimplePCLayer's own backend port.
 *
 * @note Two independent optimizer timesteps/learning-rates exist here,
 * not one: `t`/`lr` for the forward weights W, `tPsi`/`fl` for the
 * direct-feedback weights Psi. Both need their own device-resident
 * mirrors (t_device/lr_device, tPsi_device/fl_device) for the same
 * reason SimplePCLayer's did -- IComputeBackend::AdamStep/AdamWStep take
 * device pointers so their values can live inside a captured CUDA graph
 * without freezing at capture time.
 *
 * @warning `backend` defaults to nullptr, in which case this layer
 * constructs and owns its own CPUBackend internally -- every existing
 * call site keeps working unchanged. Passing a real backend is only
 * needed for GPU-aware call sites.
 * @warning RandomizeWeights() is still host-only for the CPU fallback
 * path's seed-drawing logic, but the actual weight fill now goes through
 * backend->RandomizeNormal(), which is GPU-safe -- unlike
 * SimplePCLayer's still-open RandomizeWeights gap, this one is resolved.
 */

namespace Deep
{
    class DirectKPPCLayer : public Layer
    {
    protected:
        size_t size;
        size_t nextSize;
        size_t terminalSize; // The size of the final output layer (e.g., 10 for MNIST)
        size_t batchSize;

        float lr;
        float ir;
        float fl;
        float lmbda;

        bool isClamped = false;
        bool muCacheValid = false;

        DirectKPPCLayer *layerAbove = nullptr;
        DirectKPPCLayer *layerBelow = nullptr;
        DirectKPPCLayer *terminalLayer = nullptr; // Direct pathway to \epsilon_L

        ActivationType activationType;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        DerivativeFn2 activationDerivativeInto;

        OptimizerType opt = OptimizerType::SGD;
        OptimizerType optPsi = OptimizerType::SGD;
        int t = 0;
        int tPsi = 0;

        /// @brief The compute backend this layer routes all math
        /// through. unique_ptr with a swappable deleter: no-op when an
        /// external backend was supplied (network-owned, GPU case),
        /// real delete when this layer had to construct its own
        /// fallback CPUBackend. See SimplePCLayer.h for the identical
        /// pattern and rationale.
        using BackendDeleter = void (*)(IComputeBackend *);
        std::unique_ptr<IComputeBackend, BackendDeleter> backend;

        // --- Memory Pointers ---
        // State
        float *z = nullptr;
        float *e = nullptr;

        // Forward Weights
        float *W = nullptr;
        float *b = nullptr;
        float *mu = nullptr;
        float *cachedMu = nullptr;
        float *Psi = nullptr; // Maps \epsilon_L directly to this layer's state
        float *proj = nullptr;

        // Forward Optimizer Buffers
        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;

        // Direct Feedback Optimizer Buffers
        float *grad_Psi = nullptr;
        float *m_Psi = nullptr;
        float *v_Psi = nullptr;

        // Device-resident optimizer scalars -- see file-level note.
        // Only allocated when the corresponding optimizer is ADAM/ADAMW.
        int *t_device = nullptr;
        float *lr_device = nullptr;
        int *tPsi_device = nullptr;
        float *fl_device = nullptr;

        // Scratch
        float *zF = nullptr;
        float *zFDeriv = nullptr;
        float *feedbackScratch = nullptr;

        std::unique_ptr<MemoryArena> localArena;

    public:
        /// @param backend Compute backend to route all math through.
        ///        Defaults to nullptr, in which case this layer
        ///        constructs and owns its own CPUBackend internally.
        DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                        float learningRate, float inferenceRate, float feedback, float lmbda,
                        ActivationType aType, ActivationType dType,
                        IComputeBackend *backend = nullptr);

        ~DirectKPPCLayer() override = default;

        // Setup
        /// @brief Templated so either MemoryArena (CPU) or
        /// DeviceMemoryArena (GPU) can be bound, resolved at compile
        /// time -- see the .cpp's explicit instantiations.
        template <typename ArenaT>
        void BindMemory(ArenaT &arena);
        size_t GetRequiredFloats() const noexcept;
        void RandomizeWeights(std::mt19937 &seedGenerator) noexcept;

        // Topology
        void SetLayerAbove(DirectKPPCLayer *l) noexcept { layerAbove = l; }
        void SetLayerBelow(DirectKPPCLayer *l) noexcept { layerBelow = l; }
        void SetTerminalLayer(DirectKPPCLayer *l) noexcept { terminalLayer = l; }

        // Core DKP-PC Mechanics
        float CalculateState() noexcept override { return CalculateState(true); }
        float CalculateState(bool needEnergy) noexcept;
        void ComputeMuOnly() noexcept;
        void UpdateState() noexcept override;   // Will now pull from terminalLayer->GetErrors()
        void UpdateWeights() noexcept override; // Must compute \Delta W AND \Delta \Psi
        void DirectFeedbackUpdate() noexcept;

        // Getters / Setters
        void ClampState(const std::vector<float> &inputData) noexcept;
        void UnclampState() noexcept;
        void ResetState() noexcept;

        void SetOptimizer(OptimizerType o) noexcept { opt = o; }
        void SetPsiOptimizer(OptimizerType o) noexcept { optPsi = o; }
        /// @brief Sets lr AND keeps its device-resident mirror
        /// (lr_device) in sync -- required for a captured CUDA graph to
        /// see an updated learning rate on later replays; a plain
        /// member write alone would be invisible to already-captured
        /// kernel arguments.
        void SetLearningRate(float learningRate) noexcept;
        void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
        /// @brief Sets fl (feedback learning rate) AND keeps its
        /// device-resident mirror (fl_device) in sync -- see
        /// SetLearningRate's own note; fl has the identical requirement
        /// since it drives Psi's own Adam/AdamW step.
        void SetFeedbackRate(float feedbackRate) noexcept;
        /// @brief Fixed: the original had `lmbda = lmbda`, a
        /// parameter-shadows-member bug that silently never updated the
        /// actual member. Pre-existing, unrelated to the backend
        /// refactor -- fixed here since it was noticed while reading
        /// through the class closely.
        void SetLambda(float lmbda) noexcept { this->lmbda = lmbda; }

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        const float *GetMu() const noexcept { return mu; }
        const float *GetWeights() const noexcept { return W; }
        const float *GetDirectFeedbackWeights() const noexcept { return Psi; }
        const float *GetBiases() const noexcept { return b; }

        size_t GetBatchSize() const noexcept override { return batchSize; }
        size_t GetInputSize() const noexcept override { return size; }
        size_t GetOutputSize() const noexcept override { return nextSize; }
        size_t GetTerminalSize() const noexcept { return terminalSize; }
    };
}