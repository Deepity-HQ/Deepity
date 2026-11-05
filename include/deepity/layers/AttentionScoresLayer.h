#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>

/**
 * @file AttentionScoresLayer.h
 * @brief Causal single-head self-attention weights as a free-standing PC
 * node: a FullPCLayer toggle can't express this, since its mu depends on
 * two upstream beliefs (Q and K). Does not inherit Layer.
 *
 * Uses a categorical/KL energy, not the usual Gaussian PC energy, since
 * this layer's output is a softmax distribution.
 * @see Pinchetti et al., "Predictive Coding beyond Gaussian
 * Distributions" (arXiv:2211.03481)
 */

namespace Deep
{
/// @brief Causal attention-weights node: mu[b,i,j] = (Q[b,i,:] . K[b,j,:])
/// / sqrt(headDim), causally masked, settled via the categorical/KL
/// energy against its own belief z (the attention weights themselves).
class AttentionScoresLayer
{
public:
  /// @param batchSize Batch size.
  /// @param seqLen Sequence length (both query and key positions).
  /// @param headDim Per-head feature dimension of Q and K.
  /// @param backend Compute backend to use; nullptr constructs an owned
  /// CPUBackend.
  AttentionScoresLayer(size_t batchSize, size_t seqLen, size_t headDim,
                       IComputeBackend* backend = nullptr);

  AttentionScoresLayer(const AttentionScoresLayer&) = delete;
  AttentionScoresLayer& operator=(const AttentionScoresLayer&) = delete;

  /// @brief Computes mu from the current Q/K beliefs: scaled dot-product
  /// scores, causally masked (position j > i set to a large negative
  /// value so softmax assigns it ~0 probability).
  /// @param qBeliefs Q layer's belief buffer, shape [batchSize, seqLen, headDim].
  /// @param kBeliefs K layer's belief buffer, same shape.
  void ComputeMuOnly(const float* qBeliefs, const float* kBeliefs) noexcept;

  /// @brief Computes e = z - softmax(mu) (row-wise over the key
  /// dimension), and optionally this layer's total categorical energy.
  /// @param needEnergy Whether to compute and return the energy.
  /// @return Total energy if needEnergy, else 0.
  float CalculateState(bool needEnergy) noexcept;

  /// @brief Settles z one Euler step: z += ir * (feedback - e), where
  /// feedback is the top-down signal from AttentionOutputLayer above
  /// (feedback[b,i,j] = sum_d e_above[b,i,d] * V[b,j,d], i.e. e_above
  /// projected through V^T). With no layer above (vBeliefs == nullptr),
  /// reduces to plain z += -ir * e.
  /// @param ir Inference rate (Euler step size).
  /// @param eAbove AttentionOutputLayer's error buffer, shape
  /// [batchSize, seqLen, headDim], or nullptr if there's no layer above.
  /// @param vBeliefs V layer's belief buffer, same shape as eAbove, or
  /// nullptr if there's no layer above.
  void UpdateState(float ir, const float* eAbove, const float* vBeliefs) noexcept;

  float* GetBeliefs() noexcept
  {
    return z;
  }
  const float* GetErrors() const noexcept
  {
    return e;
  }
  size_t GetBatchSize() const noexcept
  {
    return batchSize;
  }
  size_t GetSeqLen() const noexcept
  {
    return seqLen;
  }
  size_t GetHeadDim() const noexcept
  {
    return headDim;
  }

private:
  size_t batchSize;
  size_t seqLen;
  size_t headDim;

  float* z = nullptr;  ///< Belief (settled attention weights), shape [batchSize, seqLen, seqLen].
  float* e = nullptr;  ///< Error buffer, same shape as z.
  float* mu = nullptr; ///< Causally-masked raw scores, same shape as z.
  float* rowEnergies = nullptr; ///< Per-row energy scratch, length batchSize*seqLen.
  float* feedbackScratch =
      nullptr; ///< Scratch for the e_above @ V^T feedback GEMM, same shape as z.

  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend;
  std::unique_ptr<MemoryArena> arena;
};
} // namespace Deep
