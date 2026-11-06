#pragma once
#include <deepity/backend/Backend.h>
#include <deepity/layers/DirectKPPCLayer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file DirectKPPCNetwork.h
 * @brief Orchestrates DirectKPPCLayer's Direct Kolen-Pollack predictive
 * coding (DKP-PC) phases, in order:
 *
 *   0. Forward initialization: clamp input, forward-project every
 *      hidden layer's z from current weights, clamp the target, then
 *      CalculateState() on the terminal layer alone to produce a real,
 *      non-zero epsilon_L. Every other layer's error stays zero here.
 *
 *   1. Direct feedback alignment update: DirectFeedbackUpdate() on
 *      every non-terminal layer, perturbing W using epsilon_L (via
 *      terminalLayer) and the layer above's Psi, before settling
 *      begins.
 *
 *   2. Inference phase: ordinary PC settling (CalculateState() +
 *      UpdateState() across every layer) for inferenceSteps steps.
 *
 *   3. Learning phase: UpdateWeights() on every non-terminal layer,
 *      updating both W (from the settled state) and Psi (from the
 *      settled state and epsilon_L).
 *
 * Routed through IComputeBackend; `device` defaults to DEVICE_CPU.
 *
 * @see Casnici, Lefebvre, Dauwels & Frenkel, "Accelerated Predictive
 * Coding Networks via Direct Kolen-Pollack Feedback Alignment" (2026)
 *
 * @warning Not yet gradient-checked at the network level, and not yet
 * tested on GPU.
 */

namespace Deep
{
/// @brief Predictive Coding network built from DirectKPPCLayer, using
/// Direct Kolen-Pollack feedback alignment (DFA) instead of symmetric
/// weight transport.
class DirectKPPCNetwork
{
public:
  /// @brief Constructs an empty network with a predetermined batch size.
  /// @param batchSize Batch size
  /// @param device Which device this network's layers should run
  ///        on. Defaults to DEVICE_CPU, preserving existing
  ///        behavior exactly for anyone not explicitly requesting
  ///        DEVICE_GPU.
  explicit DirectKPPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU);
  /// @brief Default destructor.
  ~DirectKPPCNetwork() = default;

  /// @brief Copy construction is disabled; a network owns its
  /// layers' memory arena and is not meant to be duplicated.
  DirectKPPCNetwork(const DirectKPPCNetwork&) = delete;
  /// @brief Copy assignment is disabled; see the copy constructor.
  DirectKPPCNetwork& operator=(const DirectKPPCNetwork&) = delete;

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
  void AddLayer(size_t size, size_t nextSize, size_t terminalSize, float lr, float ir, float fl,
                float lmbda, ActivationType aType, ActivationType dType);

  /// @brief Randomizes the weights (W and Psi) of each layer.
  /// @param rng The classic Mersenne Twister
  /// @param distribution Forwarded to every layer's RandomizeWeights();
  /// see DirectKPPCLayer::RandomizeWeights()'s own doc comment.
  void RandomizeWeights(std::mt19937& rng, const char* distribution = "normal");
  /// @brief Resets each layer's state without touching learned weights.
  void ResetState() noexcept;
  /// @brief Clamps the input to the first layer.
  /// @param input Reference to the input vector.
  void Clamp(const std::vector<float>& input);

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
  /// non-terminal layer. Order among layers does not matter.
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
    for (auto& layer : layers)
      layer->SetOptimizer(o);
  }
  /// @brief Sets the optimizer used for Psi updates, on every layer.
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    for (auto& layer : layers)
      layer->SetPsiOptimizer(o);
  }
  /// @brief Sets the learning rate used for W updates, on every layer.
  void SetLearningRate(float lr) noexcept
  {
    for (auto& layer : layers)
      layer->SetLearningRate(lr);
  }
  /// @brief Sets the learning rate used for Psi updates, on every layer.
  void SetFeedbackRate(float fl) noexcept
  {
    for (auto& layer : layers)
      layer->SetFeedbackRate(fl);
  }

  /// @brief Returns the network's terminal (final) layer.
  DirectKPPCLayer* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  /// @brief Returns every layer in the network, in the order they were added.
  std::vector<std::unique_ptr<DirectKPPCLayer>>& GetLayers() noexcept
  {
    return layers;
  }
  /// @brief Returns every layer in the network, in the order they were added.
  const std::vector<std::unique_ptr<DirectKPPCLayer>>& GetLayers() const noexcept
  {
    return layers;
  }
  /// @brief Returns the batch size for the network's layers.
  int GetBatchSize() const noexcept
  {
    return batchSize;
  }
  /// @brief Returns which device this network's layers run on.
  DeviceType GetDevice() const noexcept
  {
    return device;
  }

  /// @brief Returns the length of Compile()'s detected batched-GPU
  /// fast-path run (0 if none eligible, see the "Batched-GPU fast
  /// path" member comments below). Benchmark/debug use only.
  size_t GetBatchedRunCount() const noexcept
  {
    return batchedRunCount;
  }
  /// @brief Benchmark/debug-only override: forces UseBatchedPath() to
  /// report false regardless of Compile()'s own detection, by zeroing
  /// batchedRunCount, so a benchmark can isolate batching's effect on
  /// timing from "GPU is just fast." Pass GetBatchedRunCount()'s own
  /// earlier return value to restore the natural batched run.
  void SetBatchedRunCountForBenchmark(size_t count) noexcept
  {
    batchedRunCount = count;
  }

  /// @brief Full train step: reset, clamp input+target, run all
  /// four DKP-PC phases in order, unclamp.
  /// @param x Flattened input batch, clamped to the input layer.
  /// @param y Flattened target batch, clamped to the terminal layer.
  /// @param inferenceSteps Number of settling steps for phase 2.
  /// Defaults to 1.
  /// @return The final settling step's total energy, before
  /// weight updates.
  float TrainStep(const std::vector<float>& x, const std::vector<float>& y, int inferenceSteps = 1);

  /// @brief Forward prediction: clamp input, settle with no DFA
  /// perturbation and no target clamped, read the terminal's beliefs.
  std::vector<float> Predict(const std::vector<float>& x, int inferenceSteps);

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

  bool graphCaptured = false;
  int capturedInferenceSteps = -1;

  // Predict()'s own graph, kept in a separate slot from TrainStep()'s
  // above since they capture different operations.
  bool predictGraphCaptured = false;
  int capturedPredictInferenceSteps = -1;
  static constexpr size_t kPredictGraphSlot = 1;

  // --- Batched-GPU fast path (issue #74's DKP-PC depth-scaling result)
  // -------------------------------------------------------------------
  //
  // layers[batchedRunStart .. batchedRunStart+batchedRunCount) is the
  // maximal contiguous run of uniform-width (size==nextSize==
  // batchedRunWidth), uniform-activation, uniform-terminalSize middle
  // layers, always starting at index 1 (just past layers.front(), which
  // is never part of it, see Clamp()'s own call -- only it and the
  // terminal layer are ever clamped). The run does NOT necessarily
  // reach layers.back(): when the run's own width differs from
  // terminalSize (the common case -- a hidden width like 128 next to a
  // 10-wide classifier head), the one middle layer feeding directly
  // into the terminal has nextSize==terminalSize, not the run's width,
  // so Compile()'s detection stops one (or more, for a network with
  // several different widths stacked in its last few layers) layer
  // short of the terminal. Every method below treats that "suffix" of
  // zero or more trailing non-run layers, up to and including the
  // terminal, as ordinary per-layer work -- see each method's own
  // comment. Detected once in Compile(); batchedRunCount==0 means no
  // eligible run exists (non-uniform network, or fewer than 2 eligible
  // layers) or device != DEVICE_GPU, and every method below behaves
  // exactly as it always has.
  //
  // When the run exists, EVERY run layer's buffers are aliased (via
  // DirectKPPCLayer::BindExternalMemory()) into these flat, per-quantity
  // buffers spanning the whole run, instead of each layer's own
  // self-contained arena slice -- see BindExternalMemory()'s own doc
  // comment for why this is safe: every existing per-layer method
  // keeps working completely unchanged against them, run layers included,
  // since none of those methods know or care where their buffers live.
  // ProjectForward()/DirectFeedbackUpdate()/Step()/UpdateWeights() below
  // use that aliasing to additionally dispatch MatMulBatched()/flattened
  // elementwise calls across the whole run in place of their ordinary
  // per-layer loop, when UseBatchedPath() says it's safe to for this call.
  size_t batchedRunStart = 0;
  size_t batchedRunCount = 0;
  size_t batchedRunWidth = 0;

  float* flatZ = nullptr;
  float* flatE = nullptr;
  float* flatW = nullptr;
  float* flatB = nullptr;
  float* flatMu = nullptr;
  float* flatCachedMu = nullptr;
  float* flatPreact = nullptr;
  float* flatCachedPreact = nullptr;
  float* flatDerivScratch = nullptr;
  float* flatFeedbackError = nullptr;
  float* flatProj = nullptr;
  float* flatPsi = nullptr;
  float* flatFeedbackScratch = nullptr;
  float* flatBiasGradScratch = nullptr;
  float* flatGradW = nullptr;
  float* flatMW = nullptr;
  float* flatVW = nullptr;
  float* flatGradB = nullptr;
  float* flatMB = nullptr;
  float* flatVB = nullptr;
  float* flatGradPsi = nullptr;
  float* flatMPsi = nullptr;
  float* flatVPsi = nullptr;
  int* flatTDevice = nullptr;
  float* flatLrDevice = nullptr;
  int* flatTPsiDevice = nullptr;
  float* flatFlDevice = nullptr;

  // MatMulBatched()'s per-call pointer-array scratch, reused across
  // every batched GEMM dispatch (sized batchedRunCount, resized once in
  // Compile()).
  std::vector<const float*> batchedA, batchedB;
  std::vector<float*> batchedC;

  /// @brief Whether this call can use the batched run at all: a run
  /// exists AND every layer in it currently uses SGD for both W and
  /// Psi (checked fresh, not cached, since SetOptimizer()/
  /// SetPsiOptimizer() can be called after Compile()). Adam/AdamW's
  /// math isn't batched -- those calls fall back to the ordinary
  /// per-layer loop below, which still works correctly against the
  /// same aliased buffers either way.
  bool UseBatchedPath() const noexcept;

  // Batched counterparts to ProjectForward()/DirectFeedbackUpdate()/
  // Step()/UpdateWeights(), covering ONLY layers[batchedRunStart ..
  // batchedRunStart+batchedRunCount), whenever UseBatchedPath() is
  // true. layers.front() and the "suffix" (layers[batchedRunStart+
  // batchedRunCount ..], zero or more trailing non-run layers up to and
  // including the terminal, see the member comments above) are never
  // part of the run and keep calling their own ordinary per-layer
  // methods directly, at the call sites below, in the same relative
  // order the original all-sequential loops always used -- the couple
  // of layer-sized bits of redundant recomputation that fall out of
  // this (e.g. the run's own last layer's mu, computed once by the
  // batched call and once more by the ordinary per-layer call right
  // after it, for correct read/write ordering at that seam) are O(1)
  // extra work, not O(layers.size()).
  //
  // ComputeMuBatchedRun() batches ComputeMuOnly()'s math (GEMM + bias +
  // preact snapshot + activation) across every run layer; it underlies
  // ProjectForwardBatchedRun(), StepBatchedRunCalculateState(), and
  // UpdateWeights()'s own mu refresh, exactly like the ordinary
  // ComputeMuOnly() underlies ProjectForward(), CalculateState(), and
  // UpdateWeights() today.
  void ComputeMuBatchedRun() noexcept;
  void ProjectForwardBatchedRun() noexcept;
  void DirectFeedbackUpdateBatchedRun() noexcept;
  void StepBatchedRunCalculateState() noexcept;
  void StepBatchedRunUpdateState() noexcept;
  void UpdateWeightsBatchedRun() noexcept;
  /// @brief Batched counterpart to TrainStep()'s own trailing
  /// `for (auto& l : layers) finalEnergy += l->CalculateState(true);`
  /// loop, covering run-layers [batchedRunStart+1 ..
  /// batchedRunStart+batchedRunCount) -- the run's first layer still
  /// needs layers.front()'s mu (outside the run) and is left to the
  /// caller's own per-layer CalculateState(true) call, same split as
  /// every other batched-run method here. z is already final by this
  /// point (CalculateState() never writes z), so every run layer's mu
  /// depends only on its own already-fixed z -- safe to recompute all
  /// at once via ComputeMuBatchedRun(), same reasoning as
  /// ProjectForwardBatchedRun()'s own shift-and-copy.
  /// @return The summed energy contribution of every covered layer.
  float FinalEnergyBatchedRun() noexcept;
};
} // namespace Deep