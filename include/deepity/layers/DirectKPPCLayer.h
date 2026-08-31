#pragma once

#include <deepity/layers/Layer.h>
#include <deepity/MemoryArena.h>
#include <deepity/Activations.h>
#include <deepity/AdamOptimizer.h>
#include <memory>
#include <vector>
#include <random>

namespace Deep
{
    class DirectKPPCLayer : public Layer
    {
    protected:
        int size;
        int nextSize;
        int terminalSize; // NEW: The size of the final output layer (e.g., 10 for MNIST)
        int batchSize;

        float lr;
        float ir;
        float lmbda;

        bool isClamped = false;
        bool muCacheValid = false;

        Layer *layerAbove = nullptr;
        Layer *layerBelow = nullptr;
        Layer *terminalLayer = nullptr; // NEW: Direct pathway to \epsilon_L

        ActivationType activationType;
        void (*activation)(float *, size_t);
        void (*activationDerivative)(float *, size_t, bool);

        OptimizerType opt = OptimizerType::SGD;
        int t = 0;

        // --- Memory Pointers ---
        // State
        float *z = nullptr;
        float *e = nullptr;
        float *dz_dt = nullptr;

        // Forward Weights
        float *W = nullptr;
        float *b = nullptr;
        float *mu = nullptr;
        float *cachedMu = nullptr;

        // NEW: Direct Feedback Weights
        float *Psi = nullptr; // Maps \epsilon_L directly to this layer's state

        // Forward Optimizer Buffers
        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;

        // NEW: Direct Feedback Optimizer Buffers
        float *grad_Psi = nullptr;
        float *m_Psi = nullptr;
        float *v_Psi = nullptr;

        // Scratch
        float *zF = nullptr;
        float *zFDeriv = nullptr;
        float *feedbackScratch = nullptr;

        std::unique_ptr<MemoryArena> localArena;

    public:
        DirectKPPCLayer(int size, int nextSize, int terminalSize, int batchSize,
                        float learningRate, float inferenceRate, float lmbda,
                        ActivationType aType, ActivationType dType);

        ~DirectKPPCLayer() override = default;

        // Setup
        void BindMemory(MemoryArena &arena) override;
        size_t GetRequiredFloats() const noexcept override;
        void RandomizeWeights(std::mt19937 &seedGenerator) noexcept override;

        // Topology
        void SetLayerAbove(Layer *l) noexcept { layerAbove = l; }
        void SetLayerBelow(Layer *l) noexcept { layerBelow = l; }
        void SetTerminalLayer(Layer *l) noexcept { terminalLayer = l; } // NEW: Bind this during network compilation

        // Core DKP-PC Mechanics
        float CalculateState() noexcept override;
        void ComputeMuOnly() noexcept;
        void UpdateState() noexcept override;   // Will now pull from terminalLayer->GetErrors()
        void UpdateWeights() noexcept override; // Must compute \Delta W AND \Delta \Psi

        // Getters / Setters
        void ClampState(const std::vector<float> &inputData) noexcept override;
        void UnclampState() noexcept override;
        void ResetState() noexcept override;

        void SetOptimizer(OptimizerType o) noexcept { opt = o; }
        void SetLearningRate(float learningRate) noexcept override { lr = learningRate; }
        void SetInferenceRate(float inferenceRate) noexcept override { ir = inferenceRate; }

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        const float *GetMu() const noexcept { return mu; }
        const float *GetWeights() const noexcept override { return W; }
        const float *GetDirectFeedbackWeights() const noexcept { return Psi; }
        const float *GetBiases() const noexcept override { return b; }

        int GetBatchSize() const noexcept override { return batchSize; }
        int GetInputSize() const noexcept override { return size; }
        int GetOutputSize() const noexcept override { return nextSize; }
        int GetTerminalSize() const noexcept { return terminalSize; }
    };
}
