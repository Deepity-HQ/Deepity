#pragma once
#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/DirectKPPCLayer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/backend/Backend.h>

/**
 * @file DirectKPPCNetwork.h
 * @brief Orchestrates DirectKPPCLayer's Direct Kolen-Pollack predictive
 * coding (DKP-PC) phases in the correct order, the ORDER of the phases
 * across layers is what makes this DKP-PC rather than plain PC.
 *
 * Per Casnici, Lefebvre, Dauwels & Frenkel, "Accelerated Predictive
 * Coding Networks via Direct Kolen-Pollack Feedback Alignment" (2026),
 * Algorithm 1, a full DKP-PC train step is four phases, in order:
 *
 *   0. Forward initialization, clamp input, forward-project every
 *      hidden layer's z from current weights, clamp the target, then
 *      CalculateState() on the terminal layer alone to produce a real,
 *      non-zero epsilon_L. Every other layer's error stays zero here.
 *
 *   1. Direct feedback alignment update, DirectFeedbackUpdate() on
 *      every non-terminal layer, perturbing W using epsilon_L (via
 *      terminalLayer) and the layer above's Psi, before settling begins.
 *      Order among layers doesn't matter: each layer only reads Psi and
 *      epsilon_L, both already available.
 *
 *   2. Inference phase, ordinary PC settling (CalculateState() +
 *      UpdateState() across every layer) for inferenceSteps steps. Phase
 *      1 already perturbed every layer's W, so inferenceSteps=1 is
 *      sufficient per the paper; larger values trade that speed
 *      advantage for potentially higher accuracy.
 *
 *   3. Learning phase, UpdateWeights() on every non-terminal layer,
 *      updating both W (from the settled state) and Psi (from the
 *      settled state and epsilon_L).
 *
 * Routed through IComputeBackend; `device` defaults to DEVICE_CPU.
 *
 * @warning Not yet gradient-checked at the network level, and not yet
 * tested on GPU. Verify the individual layer's math independently before
 * trusting accuracy conclusions from training with this class.
 */

namespace Deep
{
    /// @brief Predictive Coding network built from DirectKPPCLayer, using
    /// Direct Kolen-Pollack feedback alignment (DFA) instead of symmetric
    /// weight transport. See the file-level warning above on its current
    /// verification status.
    class DirectKPPCNetwork
    {
    public:
        /// @brief Constructs an empty network with a predetermined batch size.
        /// @param batchSize Batch size
        /// @param device Which device this network's layers should run
        ///        on. Defaults to DEVICE_CPU, preserving existing
        ///        behavior exactly for anyone not explicitly requesting
        ///        DEVICE_GPU.
        explicit DirectKPPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;
        /// @brief Default destructor.
        ~DirectKPPCNetwork() = default;

        /// @brief Copy construction is disabled; a network owns its
        /// layers' memory arena and is not meant to be duplicated.
        DirectKPPCNetwork(const DirectKPPCNetwork &) = delete;
        /// @brief Copy assignment is disabled; see the copy constructor.
        DirectKPPCNetwork &operator=(const DirectKPPCNetwork &) = delete;

        /// @brief Adds a layer to the network.
        /// @param size input size
        /// @param nextSize output size (0 marks a terminal layer)
        /// @param terminalSize the size of the network's final output
        /// layer (e.g. 10 for MNIST), required on every AddLayer call,
        /// not inferred, since the true terminal layer isn't known until
        /// the whole network has been assembled. Compile() sanity-checks
        /// this against the actual last layer's size.
        /// @param lr learning rate for W
        /// @param ir inference rate (Euler integration step size)
        /// @param fl feedback rate (see Layer implementatio)
        /// @param lmbda weight decay (L2 regularization) coefficient,
        /// shared between W and Psi
        /// @param aType activation type
        /// @param dType activation derivative type
        void AddLayer(size_t size, size_t nextSize, size_t terminalSize,
                      float lr, float ir, float fl, float lmbda,
                      ActivationType aType, ActivationType dType);

        /// @brief Randomizes the weights (W and Psi) of each layer.
        /// @param rng The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &rng);
        /// @brief Resets each layer's state without touching learned weights.
        void ResetState() noexcept;
        /// @brief Clamps the input to the first layer.
        /// @param input Reference to the input vector.
        void Clamp(const std::vector<float> &input);

        /// @brief Seeds every hidden layer's z from a genuine forward
        /// pass through current weights. Assumes layers[0] is already
        /// clamped.
        void ProjectForward() noexcept;

        /// @brief Phase 0's final step: computes the terminal layer's
        /// error alone, producing a real, non-zero epsilon_L. Call
        /// AFTER clamping the target onto the terminal layer, and
        /// BEFORE DirectFeedbackUpdate().
        /// @return The terminal layer's energy contribution.
        float CalculateTerminalError() noexcept;

        /// @brief Phase 1: runs DirectFeedbackUpdate() on every
        /// non-terminal layer. Order among layers does not matter (see
        /// class-level docs), unlike GaussSeidelPCNetwork, no sweep
        /// ordering is required here. Still sequential in this
        /// revision; see class-level warning.
        void DirectFeedbackUpdate() noexcept;

        /// @brief Phase 2: runs one ordinary PC settling step
        /// (CalculateState() + UpdateState()) across every layer.
        /// @param computeEnergy asks for energy to be returned
        /// @return Total energy, summed from every layer's
        /// CalculateState().
        float Step(bool computeEnergy = true) noexcept;

        /// @brief Phase 3: updates every non-terminal layer's W and Psi.
        /// Called once after the settling loop completes.
        void UpdateWeights() noexcept;

        /// @brief Sets the optimizer used for W updates, on every layer.
        void SetOptimizer(OptimizerType o) noexcept
        {
            for (auto &layer : layers)
                layer->SetOptimizer(o);
        }
        /// @brief Sets the optimizer used for Psi updates, on every layer.
        void SetPsiOptimizer(OptimizerType o) noexcept
        {
            for (auto &layer : layers)
                layer->SetPsiOptimizer(o);
        }
        /// @brief Sets the learning rate used for W updates, on every layer.
        void SetLearningRate(float lr) noexcept
        {
            for (auto &layer : layers)
                layer->SetLearningRate(lr);
        }
        /// @brief Sets the learning rate used for Psi updates, on every layer.
        void SetFeedbackRate(float fl) noexcept
        {
            for (auto &layer : layers)
                layer->SetFeedbackRate(fl);
        }

        /// @brief Returns the network's terminal (final) layer.
        DirectKPPCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }
        /// @brief Returns every layer in the network, in the order they were added.
        std::vector<std::unique_ptr<DirectKPPCLayer>> &GetLayers() noexcept { return layers; }
        /// @brief Returns every layer in the network, in the order they were added.
        const std::vector<std::unique_ptr<DirectKPPCLayer>> &GetLayers() const noexcept { return layers; }
        /// @brief Returns the batch size for the network's layers.
        int GetBatchSize() const noexcept { return batchSize; }
        /// @brief Returns which device this network's layers run on.
        DeviceType GetDevice() const noexcept { return device; }

        /// @brief Full train step: reset, clamp input+target, run all
        /// four DKP-PC phases in order, unclamp.
        /// @param x Flattened input batch, clamped to the input layer.
        /// @param y Flattened target batch, clamped to the terminal layer.
        /// @param inferenceSteps Number of settling steps for phase 2.
        /// Defaults to 1, matching the paper's own headline result,
        /// unlike this codebase's other PC variants, DKP-PC's entire
        /// point is that a single step is enough BECAUSE of the DFA
        /// phase, so defaulting to a larger value here would silently
        /// waste most of the intended speed advantage.
        /// @return The final settling step's total energy, before
        /// weight updates.
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y,
                        int inferenceSteps = 1);

        /// @brief Forward prediction: clamp input, settle with no DFA
        /// perturbation and no target clamped, read the terminal's beliefs.
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        /// @brief Loads all layers into one contiguous block of memory
        /// (MemoryArena for DEVICE_CPU, DeviceMemoryArena for
        /// DEVICE_GPU), and wires layerAbove/layerBelow/terminalLayer
        /// across every layer.
        void Compile();

    private:
        /// @brief Every layer in the network, in the order they were added.
        std::vector<std::unique_ptr<DirectKPPCLayer>> layers;

        /// @brief The network's own compute backend, created once at
        /// construction and shared by every layer added afterward.
        std::unique_ptr<IComputeBackend> backend;
        /// @brief Which device `backend` actually is.
        DeviceType device;

        /// @brief Used when device == DEVICE_CPU.
        std::unique_ptr<MemoryArena> cpuArena;
        /// @brief Used when device == DEVICE_GPU. Only compiled when
        /// DEEPITY_USE_CUDA is defined.
#if defined(DEEPITY_USE_CUDA)
        std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif

        /// @brief The batch size shared by every layer in the network.
        int batchSize;

        // @brief CUDA Graph:
        bool graphCaptured = false;
        int capturedInferenceSteps = -1;
    };
}