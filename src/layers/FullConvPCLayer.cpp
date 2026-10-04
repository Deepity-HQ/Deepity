#include "deepity/layers/FullConvPCLayer.h"
#include "deepity/backend/CPUBackend.h"
#include <cmath>
#include <type_traits>

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

FullConvPCLayer::FullConvPCLayer(int inChannels, int outChannels, int inHeight, int inWidth,
                                 int kernelH, int kernelW, int strideH, int strideW, int padH,
                                 int padW, int terminalSize, int batchSize, float learningRate,
                                 float inferenceRate, float feedback, float lmbda,
                                 ActivationType aType, ActivationType dType,
                                 IComputeBackend* backend)
    : inChannels(inChannels)
    , outChannels(outChannels)
    , inHeight(inHeight)
    , inWidth(inWidth)
    , kernelH(kernelH)
    , kernelW(kernelW)
    , strideH(strideH)
    , strideW(strideW)
    , padH(padH)
    , padW(padW)
    , batchSize(batchSize)
    , terminalSize(terminalSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , fl(feedback)
    , lmbda(lmbda)
    , activationType(aType)
    , derivativeType(dType)
    , backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
{
  outHeight = (outChannels > 0) ? ConvOutDim(inHeight, kernelH, strideH, padH) : 0;
  outWidth = (outChannels > 0) ? ConvOutDim(inWidth, kernelW, strideW, padW) : 0;

  localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
  BindMemory(*localArena);
}

void FullConvPCLayer::SetLearningRate(float learningRate) noexcept
{
  lr = learningRate;
  if (lr_device)
    backend->CopyFromHost(lr_device, &lr, 1);
}

void FullConvPCLayer::SetFeedbackRate(float feedbackRate) noexcept
{
  fl = feedbackRate;
  if (fl_device)
    backend->CopyFromHost(fl_device, &fl, 1);
}

size_t FullConvPCLayer::GetRequiredFloats() const noexcept
{
  auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };

  size_t total = 0;
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  total += pad16(ownStateSize) * 3; // z, e, dz_dt
  total += pad16((size_t)batchSize);  // rowEnergies

  if (outChannels > 0)
  {
    size_t outSize = (size_t)outChannels * outHeight * outWidth;
    size_t outStateSize = (size_t)batchSize * outSize;
    size_t colRows = (size_t)inChannels * kernelH * kernelW;
    size_t colCols = (size_t)outHeight * outWidth;
    size_t colSize = colRows * colCols;
    size_t Wsize = (size_t)outChannels * colRows;
    size_t M = (size_t)batchSize * colCols;
    size_t Psisize = (size_t)outChannels * terminalSize;

    total += pad16(Wsize);
    total += pad16((size_t)outChannels); // b
    total += pad16(outStateSize) * 2;    // mu, cachedMu
    total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
    total += pad16(outStateSize);        // bottom_up_cols
    total += pad16((size_t)batchSize * colSize); // colsRepacked
    total += pad16(outStateSize);        // lgRepacked
    total += pad16(M);                   // onesVector
    total += pad16(ownStateSize);        // v (momentum)
    total += pad16(ownStateSize);        // adjoint

    total += pad16(Wsize) * 3;               // grad_W, m_W, v_W
    total += pad16((size_t)outChannels) * 3; // grad_b, m_b, v_b
    total += pad16(1) * 2;                   // t_device, lr_device

    total += pad16(Psisize) * 4;             // Psi, grad_Psi, m_Psi, v_Psi
    total += pad16((size_t)batchSize * outChannels); // projChannel
    total += pad16(outStateSize);            // proj
    total += pad16(1) * 2;                   // tPsi_device, fl_device
  }

  return total;
}

template <typename ArenaT> void FullConvPCLayer::BindMemory(ArenaT& arena)
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  z = arena.AllocateFloats(ownStateSize);
  e = arena.AllocateFloats(ownStateSize);
  dz_dt = arena.AllocateFloats(ownStateSize);
  rowEnergies = arena.AllocateFloats(batchSize);

  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  backend->Zero(dz_dt, ownStateSize);
  backend->Zero(rowEnergies, batchSize);

  if (outChannels > 0)
  {
    size_t colRows = (size_t)inChannels * kernelH * kernelW;
    size_t colCols = (size_t)outHeight * outWidth;
    size_t outStateSize = (size_t)batchSize * outChannels * colCols;
    size_t colSize = (size_t)batchSize * colRows * colCols;
    size_t Wsize = (size_t)outChannels * colRows;
    size_t M = (size_t)batchSize * colCols;
    size_t Psisize = (size_t)outChannels * terminalSize;

    W = arena.AllocateFloats(Wsize);
    b = arena.AllocateFloats(outChannels);
    mu = arena.AllocateFloats(outStateSize);
    cachedMu = arena.AllocateFloats(outStateSize);
    colBuffer = arena.AllocateFloats(colSize);
    feedbackScratch = arena.AllocateFloats(colSize);
    bottom_up_cols = arena.AllocateFloats(outStateSize);
    colsRepacked = arena.AllocateFloats(colSize);
    lgRepacked = arena.AllocateFloats(outStateSize);
    onesVector = arena.AllocateFloats(M);
    v = arena.AllocateFloats(ownStateSize);
    adjoint = arena.AllocateFloats(ownStateSize);

    backend->Zero(b, outChannels);
    backend->Zero(mu, outStateSize);
    backend->Zero(cachedMu, outStateSize);
    backend->Zero(colBuffer, colSize);
    backend->Zero(feedbackScratch, colSize);
    backend->Zero(bottom_up_cols, outStateSize);
    backend->Zero(colsRepacked, colSize);
    backend->Zero(lgRepacked, outStateSize);
    backend->Fill(onesVector, M, 1.0f);
    backend->Zero(v, ownStateSize);
    backend->Zero(adjoint, ownStateSize);

    grad_W = arena.AllocateFloats(Wsize);
    m_W = arena.AllocateFloats(Wsize);
    v_W = arena.AllocateFloats(Wsize);
    grad_b = arena.AllocateFloats(outChannels);
    m_b = arena.AllocateFloats(outChannels);
    v_b = arena.AllocateFloats(outChannels);

    backend->Zero(grad_W, Wsize);
    backend->Zero(m_W, Wsize);
    backend->Zero(v_W, Wsize);
    backend->Zero(grad_b, outChannels);
    backend->Zero(m_b, outChannels);
    backend->Zero(v_b, outChannels);

    t_device = reinterpret_cast<int*>(arena.AllocateFloats(1));
    lr_device = arena.AllocateFloats(1);
    int zero = 0;
    backend->CopyFromHost(reinterpret_cast<float*>(t_device), reinterpret_cast<float*>(&zero), 1);
    backend->CopyFromHost(lr_device, &lr, 1);

    Psi = arena.AllocateFloats(Psisize);
    grad_Psi = arena.AllocateFloats(Psisize);
    m_Psi = arena.AllocateFloats(Psisize);
    v_Psi = arena.AllocateFloats(Psisize);
    projChannel = arena.AllocateFloats((size_t)batchSize * outChannels);
    proj = arena.AllocateFloats(outStateSize);

    backend->Zero(Psi, Psisize);
    backend->Zero(grad_Psi, Psisize);
    backend->Zero(m_Psi, Psisize);
    backend->Zero(v_Psi, Psisize);
    backend->Zero(projChannel, (size_t)batchSize * outChannels);
    backend->Zero(proj, outStateSize);

    tPsi_device = reinterpret_cast<int*>(arena.AllocateFloats(1));
    fl_device = arena.AllocateFloats(1);
    int zeroPsi = 0;
    backend->CopyFromHost(
        reinterpret_cast<float*>(tPsi_device), reinterpret_cast<float*>(&zeroPsi), 1);
    backend->CopyFromHost(fl_device, &fl, 1);
  }

  if constexpr (std::is_same_v<ArenaT, MemoryArena>)
  {
    if (localArena && localArena.get() != &arena)
      localArena.reset();
  }
  else
  {
    localArena.reset();
  }
}

void FullConvPCLayer::RandomizeWeights(std::mt19937& seedGenerator) noexcept
{
  if (outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t Wsz = (size_t)outChannels * colRows;
  size_t Psisz = (size_t)outChannels * terminalSize;

  float limit = useMuPCInit ? 1.0f : std::sqrt(2.0f / (float)colRows);
  float limPsi = std::sqrt(2.0f / (float)(outChannels + terminalSize));

  std::uniform_int_distribution<uint32_t> seedDist;
  uint32_t seedW = seedDist(seedGenerator);
  uint32_t seedPsi = seedDist(seedGenerator);

  backend->RandomizeNormal(W, Wsz, 0.0f, limit, seedW);
  backend->RandomizeNormal(Psi, Psisz, 0.0f, limPsi, seedPsi);
}

float FullConvPCLayer::CalculateState(bool needEnergy) noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  if (layerBelow == nullptr)
  {
    backend->Zero(e, ownStateSize);
    if (outChannels > 0)
      ComputeMuOnly();
    return 0.0f;
  }

  float totalEnergy = 0.0f;
  if (useCrossEntropy)
  {
    if (needEnergy)
      totalEnergy = backend->ComputeSoftmaxCrossEntropyErrorAndEnergy(
          e, z, layerBelow->mu, batchSize, ownSize, rowEnergies);
    else
      backend->ComputeSoftmaxCrossEntropyError(e, z, layerBelow->mu, batchSize, ownSize);
  }
  else if (needEnergy)
  {
    totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, ownStateSize);
  }
  else
  {
    backend->ComputeError(e, z, layerBelow->mu, ownStateSize);
  }

  if (outChannels > 0)
    ComputeMuOnly();

  return needEnergy ? totalEnergy : 0.0f;
}

void FullConvPCLayer::ComputeMuOnly() noexcept
{
  if (outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t Nout = (size_t)batchSize * outChannels * colCols;

  if (isClamped && muCacheValid)
  {
    backend->Copy(mu, cachedMu, Nout);
    return;
  }

  // Everything after Im2Col was ALSO looping per-batch-item -- batchSize
  // separate small GEMMs instead of one big one, each paying its own
  // BLAS call overhead. Fixed by reusing the exact repack trick
  // UpdateWeights() already uses for its own GEMM: RepackForBatchedGemm
  // turns [batch,rows,cols] into [rows,batch*cols], and (since it's a
  // pure axis-swap) calling it AGAIN with `batch`/`rows` swapped
  // undoes it -- no new buffers needed, colsRepacked/lgRepacked are
  // pure scratch here exactly as they are in UpdateWeights().
  backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW, strideH, strideW,
                  padH, padW, colBuffer);

  backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
  backend->MatMul(
      /*transA=*/false,
      /*transB=*/false,
      outChannels,
      (int)(batchSize * colCols),
      (int)colRows,
      a,
      W,
      (int)colRows,
      colsRepacked,
      (int)(batchSize * colCols),
      0.0f,
      lgRepacked,
      (int)(batchSize * colCols));
  backend->AddBiasPerChannel(lgRepacked, b, outChannels, batchSize * colCols); // once, not per-batch
  backend->RepackForBatchedGemm(mu, lgRepacked, outChannels, batchSize, colCols); // un-repack

  backend->Activation(activationType, mu, Nout);

  // Residual (forward): mu += z, AFTER activation (the standard "F(x)+x"
  // placement), requires inChannels==outChannels and matching spatial
  // dims -- checked once at network Compile() time, not here.
  if (useResidual)
    backend->AxpyInto(mu, z, Nout, 1.0f);

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    muCacheValid = true;
  }
}

void FullConvPCLayer::UpdateState() noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;
  size_t colCols = (outChannels > 0) ? (size_t)outHeight * outWidth : 0;
  size_t colRows = (outChannels > 0) ? (size_t)inChannels * kernelH * kernelW : 0;

  bool canFuse = (layerAbove != nullptr && outChannels > 0 && !isClamped);

  // mu must end up holding phi'(mu) whenever outChannels > 0, regardless
  // of isClamped/layerAbove -- UpdateWeights() reads it unconditionally
  // later with no isClamped guard of its own. Fuse the derivative with
  // the immediately-following multiply only when this branch actually
  // reaches that multiply; fall back to the plain derivative-only call
  // otherwise so the invariant still holds (same pattern
  // SimpleConvPCLayer already established).
  if (outChannels > 0 && !canFuse)
  {
    size_t outTotal = (size_t)batchSize * outChannels * colCols;
    backend->ActivationDerivative(derivativeType, mu, outTotal, true);
  }

  if (isClamped)
    return;

  backend->Zero(dz_dt, ownStateSize);

  if (layerAbove != nullptr && outChannels > 0)
  {
    const float* e_above = layerAbove->GetErrors();
    size_t outSize = (size_t)outChannels * colCols;
    size_t outTotal = (size_t)batchSize * outSize;

    backend->FusedActivationDerivativeMultiply(bottom_up_cols, e_above, mu, derivativeType,
                                               outTotal);

    // Same batched-GEMM trick as ComputeMuOnly(): one big transA GEMM
    // instead of batchSize small ones, un-repacking the result back to
    // per-batch-contiguous layout Col2Im() expects.
    backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)colRows,
        (int)(batchSize * colCols),
        outChannels,
        a,
        W,
        (int)colRows,
        lgRepacked,
        (int)(batchSize * colCols),
        0.0f,
        colsRepacked,
        (int)(batchSize * colCols));
    backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

    backend->Col2Im(feedbackScratch, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW,
                    strideH, strideW, padH, padW, dz_dt);
  }

  backend->AxpyInto(dz_dt, e, ownStateSize, -1.0f);

  if (useMomentum)
  {
    backend->Scale(v, ownStateSize, momentumBeta);
    backend->AxpyInto(v, dz_dt, ownStateSize, 1.0f - momentumBeta);
    backend->AxpyInto(z, v, ownStateSize, ir);
  }
  else
  {
    backend->AxpyInto(z, dz_dt, ownStateSize, ir);
  }
}

void FullConvPCLayer::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t outSize = (size_t)outChannels * colCols;
  size_t Wsize = (size_t)outChannels * colRows;
  size_t outTotal = (size_t)batchSize * outSize;

  const float* e_above = layerAbove->GetErrors();

  // mu still holds the derivative left over from the last
  // UpdateState()/ComputeAdjoint() call -- see both methods' docs.
  backend->MultiplyInto(bottom_up_cols, e_above, mu, outTotal);

  backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
  backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);

  size_t M = (size_t)batchSize * colCols;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > 0.0f)
      backend->Scale(W, Wsize, 1.0f - lmbda);

    float lr_batch = a * lr / batchSize;

    backend->MatMul(
        /*transA=*/false,
        /*transB=*/true,
        outChannels,
        (int)colRows,
        (int)M,
        lr_batch,
        lgRepacked,
        (int)M,
        colsRepacked,
        (int)M,
        1.0f,
        W,
        (int)colRows);

    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false,
        outChannels,
        1,
        (int)M,
        lr_batch,
        lgRepacked,
        (int)M,
        onesVector,
        1,
        1.0f,
        b,
        1);
    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(t_device);

    float grad_scale = -1.0f * a / batchSize;

    backend->MatMul(
        /*transA=*/false,
        /*transB=*/true,
        outChannels,
        (int)colRows,
        (int)M,
        grad_scale,
        lgRepacked,
        (int)M,
        colsRepacked,
        (int)M,
        0.0f,
        grad_W,
        (int)colRows);

    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false,
        outChannels,
        1,
        (int)M,
        grad_scale,
        lgRepacked,
        (int)M,
        onesVector,
        1,
        0.0f,
        grad_b,
        1);

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W, m_W, v_W, Wsize, t_device, lr_device, lmbda);
    else
      backend->AdamStep(W, grad_W, m_W, v_W, Wsize, t_device, lr_device);

    backend->AdamStep(b, grad_b, m_b, v_b, outChannels, t_device, lr_device);
    break;
  }
  }
}

void FullConvPCLayer::DirectFeedbackUpdate() noexcept
{
  if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
    return;

  size_t colCols = (size_t)outHeight * outWidth;
  size_t colRows = (size_t)inChannels * kernelH * kernelW;

  // projChannel = terminalError @ Psi^T, shape [batch, outChannels].
  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      batchSize,
      outChannels,
      terminalSize,
      1.0f,
      terminalLayer->GetErrors(),
      terminalSize,
      layerAbove->GetDirectFeedbackWeights(),
      terminalSize,
      0.0f,
      projChannel,
      outChannels);

  // Broadcast projChannel across every spatial position within its
  // channel (same broadcast AddBiasPerChannel already does for `b`),
  // per batch item.
  backend->Zero(proj, (size_t)batchSize * outChannels * colCols);
  for (int batch = 0; batch < batchSize; ++batch)
  {
    float* proj_item = proj + (size_t)batch * outChannels * colCols;
    const float* projChannel_item = projChannel + (size_t)batch * outChannels;
    backend->AddBiasPerChannel(proj_item, projChannel_item, outChannels, colCols);
  }

  backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
  backend->RepackForBatchedGemm(lgRepacked, proj, batchSize, outChannels, colCols);

  size_t M = (size_t)batchSize * colCols;
  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      outChannels,
      (int)colRows,
      (int)M,
      fl / batchSize,
      lgRepacked,
      (int)M,
      colsRepacked,
      (int)M,
      1.0f,
      W,
      (int)colRows);
}

void FullConvPCLayer::ReconstructBelief() noexcept
{
  if (layerBelow == nullptr)
    return;

  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Copy(z, layerBelow->GetMu(), ownStateSize);
  backend->AxpyInto(z, e, ownStateSize, 1.0f);
}

void FullConvPCLayer::ComputeAdjoint(const float* adjointAbove, float adjointAboveScale) noexcept
{
  if (layerAbove == nullptr || outChannels == 0)
    return;

  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t outTotal = (size_t)batchSize * outChannels * colCols;

  // Same fused derivative+multiply UpdateState() uses, but the result is
  // materialized so the layer below can read it as ITS incoming signal
  // -- the "global via AD" chain, one hop at a time.
  backend->FusedActivationDerivativeMultiply(bottom_up_cols, adjointAbove, mu, derivativeType,
                                             outTotal);
  if (adjointAboveScale != 1.0f)
    backend->Scale(bottom_up_cols, outTotal, adjointAboveScale);

  // Same batched-GEMM trick as UpdateState()'s feedback path.
  backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
  backend->MatMul(
      /*transA=*/true,
      /*transB=*/false,
      (int)colRows,
      (int)(batchSize * colCols),
      outChannels,
      a,
      W,
      (int)colRows,
      lgRepacked,
      (int)(batchSize * colCols),
      0.0f,
      colsRepacked,
      (int)(batchSize * colCols));
  backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

  backend->Zero(adjoint, (size_t)batchSize * ownSize);
  backend->Col2Im(feedbackScratch, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW,
                  strideH, strideW, padH, padW, adjoint);
}

void FullConvPCLayer::EnsureMuHoldsDerivative() noexcept
{
  if (outChannels == 0)
    return;

  size_t outTotal = (size_t)batchSize * outChannels * outHeight * outWidth;
  backend->ActivationDerivative(derivativeType, mu, outTotal, true);
}

void FullConvPCLayer::UpdateErrorEPC() noexcept
{
  if (isClamped)
    return;

  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Scale(e, ownStateSize, 1.0f - ir);
  backend->AxpyInto(e, adjoint, ownStateSize, -ir);
}

void FullConvPCLayer::ClampState(const std::vector<float>& inputData) noexcept
{
  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  size_t copyFloats = (std::min)(inputData.size(), ownStateSize);
  backend->CopyFromHost(z, inputData.data(), copyFloats);
  isClamped = true;
  muCacheValid = false;
}

void FullConvPCLayer::UnclampState() noexcept
{
  isClamped = false;
}

void FullConvPCLayer::ResetState() noexcept
{
  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  if (v)
    backend->Zero(v, ownStateSize);
}

template void FullConvPCLayer::BindMemory<MemoryArena>(MemoryArena& arena);
#if defined(DEEPITY_USE_CUDA)
template void FullConvPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena& arena);
#endif
} // namespace Deep
