#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>

/**
 * @file AttentionScoresLayer.h
 * @brief Causal single-head self-attention weights as a free-standing PC
 * node -- NOT a FullPCLayer toggle, since its mu depends on TWO upstream
 * beliefs (Q and K), breaking the single-layerBelow assumption
 * FullPCLayer's state machinery relies on everywhere. Does not inherit
 * Layer for the same reason FullPCLayer<T> doesn't (see that class's own
 * doc comment): Layer's virtual interface has no notion of two
 * predecessors either.
 *
 * Energy function follows Pinchetti et al., "Predictive Coding beyond
 * Gaussian Distributions" (arXiv:2211.03481): every ordinary PC layer
 * keeps the usual Gaussian energy, but a layer whose output is a softmax
 * distribution (attention weights here) needs the categorical/KL energy
 * instead, or training collapses (their own naive-Gaussian-PC transformer
 * hit test perplexity 590 vs backprop's 162.64; the categorical-energy
 * version reached 175.90). Deepity already has this energy function --
 * FullPCLayer's SetCrossEntropy()/ComputeSoftmaxCrossEntropyError(AndEnergy)
 * -- this class reuses it directly rather than re-deriving it, by
 * treating each of the batchSize*seqLen query positions as one
 * independent softmax-over-seqLen row.
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
  /// CPUBackend (CPU-only for v1, matching the current proof-of-concept
  /// scope -- see the file doc comment).
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
  /// projected through V^T -- the Jacobian of mu_output = attnZ @ V
  /// w.r.t. attnZ, no activation derivative needed since that product
  /// is already linear in attnZ). With no layer above (vBeliefs ==
  /// nullptr), reduces to plain z += -ir * e.
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
