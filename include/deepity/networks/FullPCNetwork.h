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
 * @brief Orchestrates FullPCLayer's phases (the same four-phase DKP-PC
 * structure as DirectKPPCNetwork), plus network-level toggles for muPC
 * scaling and residual connections, both off by default.
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
  explicit FullPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU);
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

  /// @brief Enables each layer's forward-scaling factor `a`
  /// (width/depth-dependent) and unit-variance weight init, in place of
  /// plain PC's fan-scaled init. Off by default. Must be called before
  /// Compile() (which computes and applies `a` once the full
  /// architecture is known) and before RandomizeWeights() (the
  /// init-variance change only takes effect on the next call).
  /// @see Innocenti et al., "muPC: Scaling Predictive Coding to 100+ Layer
  /// Networks", https://arxiv.org/abs/2505.13124
  void SetUseMuPCScaling(bool enabled) noexcept
  {
    useMuPCScaling = enabled;
  }

  /// @brief Enables residual/skip connections on middle hidden layers.
  /// Off by default. Requires every middle hidden layer to have the
  /// same width as its neighbor; Compile() throws
  /// std::invalid_argument if that's violated.
  void SetUseResidualConnections(bool enabled) noexcept
  {
    useResidualConnections = enabled;
  }

  /// @brief Randomizes the weights (W and Psi) of each layer.
  /// @param rng The classic Mersenne Twister
  /// @param distribution Forwarded to every layer's RandomizeWeights();
  /// see FullPCLayer::RandomizeWeights()'s own doc comment.
  void RandomizeWeights(std::mt19937& rng, const char* distribution = "normal");
  /// @brief Resets each layer's state without touching learned weights.
  void ResetState() noexcept;
  /// @brief Clamps the input to the first layer.
  /// @param input Reference to the input vector.
  void Clamp(const std::vector<float>& input);
  /// @brief Same as Clamp(), but `deviceSrc` is already a pointer on
  /// this network's own backend device. For GPU interop only -- see
  /// FullPCLayer::ClampStateDevice().
  void ClampDevice(const float* deviceSrc, size_t n);

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
  /// @brief ePC's settling step, used by TrainStep() in place of Step()
  /// when useEPC is set: a full forward sweep (bottom to top, every
  /// hidden layer's belief reconstructed from the layer below's fresh
  /// prediction) followed by a full backward sweep (the terminal's
  /// error chained all the way down through every hidden layer). See
  /// SetUseEPC().
  void EPCStep() noexcept;
  /// @brief Applies weight updates to every layer.
  void UpdateWeights() noexcept;

  /// @brief Re-centers every non-terminal layer's W rows after each UpdateWeights().
  /// Off by default.
  void SetCenterWeights(bool enabled) noexcept
  {
    if (enabled == centerWeights)
      return;
    centerWeights = enabled;
    graphCaptured = false;
  }

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

  /// @brief Returns this network's backend compute stream as an opaque
  /// integer handle (0 on a backend with no stream concept, e.g. CPU).
  /// For cross-framework GPU interop only -- see
  /// IComputeBackend::GetStreamHandle().
  std::uintptr_t GetStreamHandle() const noexcept
  {
    return backend->GetStreamHandle();
  }

  /// @brief Full train step: reset, clamp input+target, run all four
  /// DKP-PC phases in order, unclamp.
  /// @param x Flattened input batch, clamped to the input layer.
  /// @param y Flattened target batch, clamped to the terminal layer.
  /// @param inferenceSteps Number of settling steps for phase 2.
  /// @return The final settling step's total energy, before weight updates.
  float TrainStep(const std::vector<float>& x, const std::vector<float>& y, int inferenceSteps = 1);
  /// @brief Same as TrainStep(), but x/y are already pointers on this
  /// network's own backend device. For GPU interop only: the caller is
  /// responsible for both being safe to read on this backend's stream
  /// (see GetStreamHandle()).
  float TrainStepDevice(const float* xDevice, size_t xN, const float* yDevice, size_t yN,
                        int inferenceSteps = 1);

  /// @brief Forward prediction: clamp input, settle with no DFA
  /// perturbation and no target clamped, read the terminal's beliefs.
  std::vector<float> Predict(const std::vector<float>& x, int inferenceSteps);
  /// @brief Same as Predict(), but xDevice is already a pointer on this
  /// network's own backend device. For GPU interop only -- see
  /// TrainStepDevice(). Still returns a host vector: this is the input
  /// side of GPU interop only, not the output side.
  std::vector<float> PredictDevice(const float* xDevice, size_t xN, int inferenceSteps);

  /// @brief Loads all layers into one contiguous block of memory and
  /// wires layerAbove/layerBelow/terminalLayer across every layer.
  /// If useMuPCScaling, computes and applies each layer's `a` and
  /// enables unit-variance init. If useResidualConnections, applies
  /// SetResidual(true) to every middle hidden layer after checking
  /// width match (throws std::invalid_argument on mismatch).
  void Compile();

  /// @brief Enables iPC: UpdateWeights() runs every settling step
  /// instead of once after settling completes. Off by default.
  /// @see Salvatori et al., "Incremental Predictive Coding", https://arxiv.org/abs/2212.00720
  void SetUseIPC(bool enabled) noexcept
  {
    useIPC = enabled;
    InvalidateCapturedGraphs();
  }

  /// @brief Enables momentum (inertial) settling on every layer. Off
  /// by default.
  void SetUseMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    for (auto& layer : layers)
      layer->SetMomentum(enabled, beta);
    InvalidateCapturedGraphs();
  }

  /// @brief Enables softmax cross-entropy energy on the terminal layer
  /// only. Off by default.
  void SetUseCrossEntropy(bool enabled) noexcept
  {
    GetTerminalLayer()->SetCrossEntropy(enabled);
    InvalidateCapturedGraphs();
  }

  /// @brief Enables ePC settling: each settling step becomes a full
  /// forward sweep (recompute every hidden layer's belief/prediction
  /// bottom-to-top) followed by a full backward sweep (chain the
  /// terminal's error down through every hidden layer), instead of the
  /// usual per-layer local CalculateState()/UpdateState() loop. Only
  /// affects hidden layers. Off by default.
  /// @see Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  /// Digital Simulation", https://arxiv.org/abs/2505.20137
  void SetUseEPC(bool enabled) noexcept
  {
    useEPC = enabled;
    for (size_t i = 1; i + 1 < layers.size(); ++i)
      layers[i]->SetUseEPC(enabled);
    InvalidateCapturedGraphs();
  }

private:
  /// @brief Forces TrainStep()/Predict() to recapture their CUDA graphs
  /// on their next call. Call after any toggle that changes behavior
  /// inside the captured region. No-op on CPU.
  void InvalidateCapturedGraphs() noexcept
  {
    graphCaptured = false;
    predictGraphCaptured = false;
  }
  /// @brief TrainStep()/TrainStepDevice()'s shared body, run after
  /// whichever Clamp variant the caller used: the settle loop, graph
  /// capture/replay, weight update, and final energy readout.
  float TrainStepSettle(int inferenceSteps);
  /// @brief Predict()/PredictDevice()'s shared body, run after whichever
  /// Clamp variant the caller used.
  std::vector<float> PredictSettle(int inferenceSteps);
  std::vector<std::unique_ptr<FullPCLayer>> layers;

  std::unique_ptr<IComputeBackend> backend;
  DeviceType device;

  std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif

  int batchSize;
  bool useIPC = false;
  bool useEPC = false;

  bool useMuPCScaling = false;
  bool useResidualConnections = false;

  bool centerWeights = false;
  bool graphCaptured = false;
  int capturedInferenceSteps = -1;

  // Predict()'s own graph, kept in a separate slot from TrainStep()'s
  // above since they capture different operations.
  bool predictGraphCaptured = false;
  int capturedPredictInferenceSteps = -1;
  static constexpr size_t kPredictGraphSlot = 1;
};
} // namespace Deep
