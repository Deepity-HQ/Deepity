#include "deepity/layers/AttentionOutputLayer.h"
#include "deepity/backend/CPUBackend.h"

namespace Deep
{
namespace
{
void DeleteBackend(IComputeBackend* p)
{
  delete p;
}
void NoOpDeleter(IComputeBackend*) {}
} // namespace

AttentionOutputLayer::AttentionOutputLayer(size_t batchSize, size_t seqLen, size_t headDim,
                                           IComputeBackend* backend)
    : batchSize(batchSize)
    , seqLen(seqLen)
    , headDim(headDim)
    , backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
{
  size_t elems = batchSize * seqLen * headDim;

  // +16 floats (one 64-byte block) of slack per buffer, same reasoning
  // as AttentionScoresLayer's constructor. 3 buffers below.
  arena = std::make_unique<MemoryArena>(elems * 3 + 3 * 16);

  z = arena->AllocateFloats(elems);
  e = arena->AllocateFloats(elems);
  mu = arena->AllocateFloats(elems);

  this->backend->Zero(z, elems);
  this->backend->Zero(e, elems);
  this->backend->Zero(mu, elems);
}

void AttentionOutputLayer::ComputeMuOnly(const float* attnBeliefs, const float* vBeliefs) noexcept
{
  for (size_t b = 0; b < batchSize; ++b)
  {
    const float* attnB = attnBeliefs + b * seqLen * seqLen;
    const float* vb = vBeliefs + b * seqLen * headDim;
    float* mub = mu + b * seqLen * headDim;

    // mub (seqLen x headDim) = attnB (seqLen x seqLen) @ vb (seqLen x headDim).
    backend->MatMul(/*transA=*/false, /*transB=*/false, (int)seqLen, (int)headDim, (int)seqLen,
                    1.0f, attnB, (int)seqLen, vb, (int)headDim, 0.0f, mub, (int)headDim);
  }
}

float AttentionOutputLayer::CalculateState(bool needEnergy) noexcept
{
  size_t N = batchSize * seqLen * headDim;

  if (needEnergy)
    return backend->ComputeErrorAndEnergy(e, z, mu, N);

  backend->ComputeError(e, z, mu, N);
  return 0.0f;
}

void AttentionOutputLayer::UpdateState(float ir, const float* eAbove) noexcept
{
  size_t N = batchSize * seqLen * headDim;

  if (eAbove == nullptr)
  {
    backend->AxpyInto(z, e, N, -ir);
    return;
  }

  // z += ir*(eAbove - e), same convention as FullPCLayer::UpdateState()'s
  // "no top-down projection needed" case (no activation nonlinearity or
  // weight matrix sits between this layer and whatever's above it).
  backend->AxpyInto(z, eAbove, N, ir);
  backend->AxpyInto(z, e, N, -ir);
}
} // namespace Deep
