#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>

/**
 * @file AttentionOutputLayer.h
 * @brief Attention's weighted-sum-with-V step as a free-standing PC
 * node: mu depends on two upstream beliefs (the settled attention
 * weights and V), which a FullPCLayer toggle can't express.
 *
 * Has no softmax and no learnable weights; mu is a fixed bilinear
 * combination of its two inputs, settled via the ordinary Gaussian PC
 * energy.
 */

namespace Deep
{
/// @brief Attention-output node: mu[b,i,:] = sum_j attnZ[b,i,j] *
/// V[b,j,:], settled via the ordinary Gaussian energy.
class AttentionOutputLayer
{
public:
  /// @param batchSize Batch size.
  /// @param seqLen Sequence length.
  /// @param headDim Per-head feature dimension of V (and of this
  /// layer's own belief).
  /// @param backend Compute backend to use; nullptr constructs an owned
  /// CPUBackend.
  AttentionOutputLayer(size_t batchSize, size_t seqLen, size_t headDim,
                       IComputeBackend* backend = nullptr);

  AttentionOutputLayer(const AttentionOutputLayer&) = delete;
  AttentionOutputLayer& operator=(const AttentionOutputLayer&) = delete;

  /// @brief Computes mu[b,i,:] = sum_j attnBeliefs[b,i,j] * vBeliefs[b,j,:].
  /// @param attnBeliefs AttentionScoresLayer's belief buffer, shape
  /// [batchSize, seqLen, seqLen].
  /// @param vBeliefs V layer's belief buffer, shape [batchSize, seqLen, headDim].
  void ComputeMuOnly(const float* attnBeliefs, const float* vBeliefs) noexcept;

  /// @brief Computes e = z - mu (plain Gaussian), and optionally this
  /// layer's total energy.
  float CalculateState(bool needEnergy) noexcept;

  /// @brief Settles z one Euler step: z += ir * (eAbove - e) if there's
  /// a real layer above (eAbove non-null), else plain z += -ir * e.
  void UpdateState(float ir, const float* eAbove = nullptr) noexcept;

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

  float* z = nullptr;  ///< Belief, shape [batchSize, seqLen, headDim].
  float* e = nullptr;  ///< Error buffer, same shape as z.
  float* mu = nullptr; ///< Prediction buffer, same shape as z.

  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend;
  std::unique_ptr<MemoryArena> arena;
};
} // namespace Deep
