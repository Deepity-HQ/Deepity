#pragma once

#include <deepity/layers/Layer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <memory>
#include <vector>
#include <random>

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

        // Scratch
        float *zF = nullptr;
        float *zFDeriv = nullptr;
        float *feedbackScratch = nullptr;

        std::unique_ptr<MemoryArena> localArena;

    public:
        DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                        float learningRate, float inferenceRate, float feedback, float lmbda,
                        ActivationType aType, ActivationType dType);

        ~DirectKPPCLayer() override = default;

        // Setup
        void BindMemory(MemoryArena &arena);
        size_t GetRequiredFloats() const noexcept;
        void RandomizeWeights(std::mt19937 &seedGenerator) noexcept;

        // Topology
        void SetLayerAbove(DirectKPPCLayer *l) noexcept { layerAbove = l; }
        void SetLayerBelow(DirectKPPCLayer *l) noexcept { layerBelow = l; }
        void SetTerminalLayer(DirectKPPCLayer *l) noexcept { terminalLayer = l; }

        // Core DKP-PC Mechanics
        float CalculateState() noexcept override;
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
        void SetLearningRate(float learningRate) noexcept { lr = learningRate; }
        void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
        void SetFeedbackRate(float feedbackRate) noexcept { fl = feedbackRate; }
        void SetLambda(float lmbda) noexcept { lmbda = lmbda; }

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
