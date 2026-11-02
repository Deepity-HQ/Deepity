#include "deepity/layers/AttentionScoresLayer.h"
#include "deepity/backend/CPUBackend.h"
#include <cmath>

namespace Deep
{
namespace
{
void DeleteBackend(IComputeBackend* p)
{
  delete p;
}
void NoOpDeleter(IComputeBackend*) {}

// No existing backend primitive does triangular masking; this is a
// batch*seqLen^2-sized scalar loop, run once per settling step on a
// tiny buffer (seqLen is in the tens for the v1 toy scale), not worth a
// dedicated SIMD kernel.
constexpr float kMaskValue = -1.0e9f;
} // namespace

AttentionScoresLayer::AttentionScoresLayer(size_t batchSize, size_t seqLen, size_t headDim,
                                           IComputeBackend* backend)
    : batchSize(batchSize)
    , seqLen(seqLen)
    , headDim(headDim)
    , backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
{
  size_t scoreElems = batchSize * seqLen * seqLen;

  // +16 floats (one 64-byte block) of slack per buffer: Allocate()
  // rounds each individual call up to a 64-byte boundary independently,
  // so summing exact element counts here can undercount the arena's
  // real total requirement. 5 buffers below.
  arena = std::make_unique<MemoryArena>(scoreElems * 4 + batchSize * seqLen + 5 * 16);

  z = arena->AllocateFloats(scoreElems);
  e = arena->AllocateFloats(scoreElems);
  mu = arena->AllocateFloats(scoreElems);
  feedbackScratch = arena->AllocateFloats(scoreElems);
  rowEnergies = arena->AllocateFloats(batchSize * seqLen);

  this->backend->Zero(z, scoreElems);
  this->backend->Zero(e, scoreElems);
  this->backend->Zero(mu, scoreElems);
}

void AttentionScoresLayer::ComputeMuOnly(const float* qBeliefs, const float* kBeliefs) noexcept
{
  float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

  for (size_t b = 0; b < batchSize; ++b)
  {
    const float* qb = qBeliefs + b * seqLen * headDim;
    const float* kb = kBeliefs + b * seqLen * headDim;
    float* mub = mu + b * seqLen * seqLen;

    // mub (seqLen x seqLen) = qb (seqLen x headDim) @ kb^T (headDim x seqLen) * scale.
    backend->MatMul(/*transA=*/false, /*transB=*/true, (int)seqLen, (int)seqLen, (int)headDim,
                    scale, qb, (int)headDim, kb, (int)headDim, 0.0f, mub, (int)seqLen);

    // Causal mask: query i may only attend to key j <= i.
    for (size_t i = 0; i < seqLen; ++i)
      for (size_t j = i + 1; j < seqLen; ++j)
        mub[i * seqLen + j] = kMaskValue;
  }
}

float AttentionScoresLayer::CalculateState(bool needEnergy) noexcept
{
  // Each of the batchSize*seqLen query positions is an independent
  // softmax-over-seqLen row -- reuses FullPCLayer's own categorical
  // energy kernel directly (see this file's own doc comment).
  size_t rows = batchSize * seqLen;

  if (needEnergy)
    return backend->ComputeSoftmaxCrossEntropyErrorAndEnergy(e, z, mu, rows, seqLen, rowEnergies);

  backend->ComputeSoftmaxCrossEntropyError(e, z, mu, rows, seqLen);
  return 0.0f;
}

void AttentionScoresLayer::UpdateState(float ir, const float* eAbove,
                                       const float* vBeliefs) noexcept
{
  size_t N = batchSize * seqLen * seqLen;

  if (eAbove == nullptr || vBeliefs == nullptr)
  {
    backend->AxpyInto(z, e, N, -ir);
    return;
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    const float* eAboveB = eAbove + b * seqLen * headDim;
    const float* vb = vBeliefs + b * seqLen * headDim;
    float* fbB = feedbackScratch + b * seqLen * seqLen;

    // fbB (seqLen x seqLen) = eAboveB (seqLen x headDim) @ vb^T (headDim x seqLen),
    // the Jacobian of mu_output = attnZ @ V w.r.t. attnZ (see header doc).
    backend->MatMul(/*transA=*/false, /*transB=*/true, (int)seqLen, (int)seqLen, (int)headDim,
                    1.0f, eAboveB, (int)headDim, vb, (int)headDim, 0.0f, fbB, (int)seqLen);
  }

  // Same convention as FullPCLayer::UpdateState(): z += ir*(feedback - e).
  backend->AxpyInto(feedbackScratch, e, N, -1.0f);
  backend->AxpyInto(z, feedbackScratch, N, ir);
}
} // namespace Deep
