#pragma once
#include <deepity/backend/Backend.h>
#include <deepity/layers/FullPCLayer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file FullPCNetwork.h
 * @brief Orchestrates FullPCLayer's phases, matching DirectKPPCNetwork's
 * four-phase DKP-PC structure (see that class's docs for the phase
 * breakdown and citation), plus network-level toggles for muPC scaling
 * and residual connections, both OFF by default, so a plain
 * FullPCNetwork reproduces DirectKPPCNetwork exactly (FullPCLayer with
 * a=1.0, useResidual=false is bit-identical to DirectKPPCLayer).
 */

namespace Deep
{
/// @brief Predictive Coding network combining muPC scaling, optional
/// residual connections, and DKP direct feedback.
class FullPCNetwork
{
public:
  /// @brief Constructs an empty network; add layers via AddLayer(),
  /// then Compile() before use.
  /// @param batchSize Fixed batch size for every layer.
  /// @param device Which device this network's layers run on.
  explicit FullPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;
  ~FullPCNetwork() = default;

  FullPCNetwork(const FullPCNetwork&) = delete;
  FullPCNetwork& operator=(const FullPCNetwork&) = delete;

  /// @brief Adds a layer to the network.
  /// @param size input size
  /// @param nextSize output size (0 marks a terminal layer)
  /// @param terminalSize the size of the network's final output layer
  /// (e.g. 10 for MNIST), required on every AddLayer call, not
  /// inferred, since the true terminal layer isn't known until the
  /// whole network has been assembled. Compile() sanity-checks this
  /// against the actual last layer's size.
  /// @param lr learning rate for W
  /// @param ir inference rate (Euler integration step size)
  /// @param fl feedback rate (see FullPCLayer's own docs)
  /// @param lmbda weight decay (L2 regularization) coefficient, shared
  /// between W and Psi
  /// @param aType activation type
  /// @param dType activation derivative type
  void AddLayer(size_t size, size_t nextSize, size_t terminalSize, float lr, float ir, float fl,
                float lmbda, ActivationType aType, ActivationType dType);

  /// @brief Enables muPC-style per-layer forward scaling (Table 1
  /// of the muPC paper). OFF by default. Must be called before
  /// Compile(), Compile() is what actually computes and applies
  /// each layer's `a`, once the full architecture (every AddLayer
  /// call) is known.
  void SetUseMuPCScaling(bool enabled) noexcept
  {
    useMuPCScaling = enabled;
  }

  /// @brief Enables residual/skip connections on middle hidden
  /// layers (muPC's "1-skip" ResNet formulation, see that
  /// paper's A.2.4). OFF by default. Requires every middle hidden
  /// layer to have the SAME width as its neighbor; Compile()
  /// throws std::invalid_argument if that's violated, rather than
  /// silently skipping the mismatched connection (a silent skip
  /// risks masking a real configuration mistake).
  void SetUseResidualConnections(bool enabled) noexcept
  {
    useResidualConnections = enabled;
  }

  /// @brief Randomizes the weights (W and Psi) of each layer.
  /// @param rng The classic Mersenne Twister
  void RandomizeWeights(std::mt19937& rng);
  /// @brief Resets each layer's state without touching learned weights.
  void ResetState() noexcept;
  /// @brief Clamps the input to the first layer.
  /// @param input Reference to the input vector.
  void Clamp(const std::vector<float>& input);

  /// @brief Seeds hidden layers from a genuine forward pass through
  /// current weights, instead of zero-init. Call after Clamp(), before
  /// the settling loop.
  void ProjectForward() noexcept;
  /// @brief Computes the terminal layer's error/energy against its
  /// clamped target. Call AFTER clamping the target onto the terminal
  /// layer, and BEFORE DirectFeedbackUpdate().
  /// @return The terminal layer's energy contribution.
  float CalculateTerminalError() noexcept;
  /// @brief Runs the DFA phase on every non-terminal layer: perturbs
  /// each layer's W using the layer above's Psi and the terminal
  /// layer's error.
  void DirectFeedbackUpdate() noexcept;
  /// @brief Runs one settling step on every layer.
  /// @param computeEnergy asks for energy to be returned
  /// @return Total energy, summed from every layer's CalculateState().
  float Step(bool computeEnergy = true) noexcept;
  /// @brief Applies weight updates to every layer.
  void UpdateWeights() noexcept;

  /// @brief Sets the weight optimizer (W) on every layer.
  void SetOptimizer(OptimizerType o) noexcept
  {
    for (auto& layer : layers)
      layer->SetOptimizer(o);
  }
  /// @brief Sets the feedback-weight optimizer (Psi) on every layer.
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    for (auto& layer : layers)
      layer->SetPsiOptimizer(o);
  }
  /// @brief Sets the learning rate (W) on every layer.
  void SetLearningRate(float lr) noexcept
  {
    for (auto& layer : layers)
      layer->SetLearningRate(lr);
  }
  /// @brief Sets the feedback rate (Psi) on every layer.
  void SetFeedbackRate(float fl) noexcept
  {
    for (auto& layer : layers)
      layer->SetFeedbackRate(fl);
  }

  /// @brief Returns the last layer (nextSize==0, the terminal one).
  FullPCLayer* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  /// @brief Returns every layer in the network, in the order they were added.
  std::vector<std::unique_ptr<FullPCLayer>>& GetLayers() noexcept
  {
    return layers;
  }
  /// @brief const overload of GetLayers() above.
  const std::vector<std::unique_ptr<FullPCLayer>>& GetLayers() const noexcept
  {
    return layers;
  }
  /// @brief Returns the batch size given at construction.
  int GetBatchSize() const noexcept
  {
    return batchSize;
  }
  /// @brief Returns which device this network's layers run on.
  DeviceType GetDevice() const noexcept
  {
    return device;
  }

  /// @brief Full train step: reset, clamp input+target, run all four
  /// DKP-PC phases in order, unclamp.
  /// @param x Flattened input batch, clamped to the input layer.
  /// @param y Flattened target batch, clamped to the terminal layer.
  /// @param inferenceSteps Number of settling steps for phase 2.
  /// @return The final settling step's total energy, before weight updates.
  float TrainStep(const std::vector<float>& x, const std::vector<float>& y, int inferenceSteps = 1);

  /// @brief Forward prediction: clamp input, settle with no DFA
  /// perturbation and no target clamped, read the terminal's beliefs.
  std::vector<float> Predict(const std::vector<float>& x, int inferenceSteps);

  /// @brief Loads all layers into one contiguous block of memory,
  /// wires layerAbove/layerBelow/terminalLayer across every
  /// layer, same as DirectKPPCNetwork::Compile(), PLUS: if
  /// useMuPCScaling, computes and applies each layer's `a` (Table
  /// 1) from the full, now-known architecture; if
  /// useResidualConnections, applies SetResidual(true) to every
  /// middle hidden layer after checking width match (throws
  /// std::invalid_argument on mismatch).
  void Compile();

  /// @brief Enables iPC: UpdateWeights() runs every settling
  /// step instead of once after settling completes. OFF by default,
  /// TrainStep() matches DirectKPPCNetwork's standard, two-phase
  /// behavior exactly when this is false.
  /// @see Salvatori et al., "Incremental Predictive Coding", https://arxiv.org/abs/2212.00720
  void SetUseIPC(bool enabled) noexcept
  {
    useIPC = enabled;
  }

  /// @brief Enables momentum (inertial) settling on every layer.
  /// Loops over layers and calls each one's own SetMomentum(),
  /// same pattern as SetLearningRate/SetFeedbackRate. OFF by default.
  void SetUseMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    for (auto& layer : layers)
      layer->SetMomentum(enabled, beta);
  }

  /// @brief Enables softmax cross-entropy energy on the TERMINAL layer
  /// only (unlike the other Set* toggles, this does NOT loop over every
  /// layer, cross-entropy only ever makes sense on the final output,
  /// against a one-hot/class-probability target). OFF by default
  /// (plain Gaussian energy everywhere, matching DirectKPPCNetwork).
  void SetUseCrossEntropy(bool enabled) noexcept
  {
    GetTerminalLayer()->SetCrossEntropy(enabled);
  }

private:
  std::vector<std::unique_ptr<FullPCLayer>> layers;

  std::unique_ptr<IComputeBackend> backend;
  DeviceType device;

  std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif

  int batchSize;
  bool useIPC = false;

  bool useMuPCScaling = false;
  bool useResidualConnections = false;

  bool graphCaptured = false;
  int capturedInferenceSteps = -1;
};
} // namespace Deep
