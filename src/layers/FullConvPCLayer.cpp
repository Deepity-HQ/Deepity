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
                                 ActivationType aType, ActivationType dType, int poolH, int poolW,
                                 int poolStrideH, int poolStrideW, IComputeBackend* backend)
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
    , poolH(poolH)
    , poolW(poolW)
    , poolStrideH(poolStrideH)
    , poolStrideW(poolStrideW)
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
  poolOutHeight = (outChannels > 0 && HasPooling()) ? PoolOutDim(outHeight, poolH, poolStrideH)
                                                    : outHeight;
  poolOutWidth = (outChannels > 0 && HasPooling()) ? PoolOutDim(outWidth, poolW, poolStrideW)
                                                   : outWidth;

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
  if ((feedbackRate == 0.0f) != (fl == 0.0f))
    flZeroDirty = true;

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

    if (HasPooling())
    {
      size_t poolOutStateSize = (size_t)batchSize * outChannels * poolOutHeight * poolOutWidth;
      total += pad16(poolOutStateSize) * 2; // pooledMu, poolArgmax (as floats)
      total += pad16(outStateSize);         // unpoolScratch
    }
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

    if (HasPooling())
    {
      size_t poolOutStateSize = (size_t)batchSize * outChannels * poolOutHeight * poolOutWidth;
      pooledMu = arena.AllocateFloats(poolOutStateSize);
      poolArgmax = reinterpret_cast<int*>(arena.AllocateFloats(poolOutStateSize));
      unpoolScratch = arena.AllocateFloats(outStateSize);

      backend->Zero(pooledMu, poolOutStateSize);
      backend->Zero(reinterpret_cast<float*>(poolArgmax), poolOutStateSize);
      backend->Zero(unpoolScratch, outStateSize);
    }
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

  float limPsi = std::sqrt(2.0f / (float)(outChannels + terminalSize));

  std::uniform_int_distribution<uint32_t> seedDist;
  uint32_t seedW = seedDist(seedGenerator);
  uint32_t seedPsi = seedDist(seedGenerator);

  if (usePCXInit)
  {
    float pcxLimit = std::sqrt(1.0f / (float)colRows);
    backend->RandomizeUniform(W, Wsz, -pcxLimit, pcxLimit, seedW);
  }
  else
  {
    float limit = useMuPCInit ? 1.0f : std::sqrt(2.0f / (float)colRows);
    backend->RandomizeNormal(W, Wsz, 0.0f, limit, seedW);
  }
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
          e, z, layerBelow->GetMu(), batchSize, ownSize, rowEnergies);
    else
      backend->ComputeSoftmaxCrossEntropyError(e, z, layerBelow->GetMu(), batchSize, ownSize);
  }
  else if (needEnergy)
  {
    totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->GetMu(), ownStateSize);
  }
  else
  {
    backend->ComputeError(e, z, layerBelow->GetMu(), ownStateSize);
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
    if (HasPooling())
      backend->MaxPool2DForward(mu, batchSize, outChannels, outHeight, outWidth, poolH, poolW,
                                poolStrideH, poolStrideW, pooledMu, poolArgmax);
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

  if (HasPooling())
    backend->MaxPool2DForward(mu, batchSize, outChannels, outHeight, outWidth, poolH, poolW,
                              poolStrideH, poolStrideW, pooledMu, poolArgmax);
}

const float* FullConvPCLayer::UnpoolError(const float* pooledError) noexcept
{
  if (!HasPooling())
    return pooledError;

  size_t outStateSize = (size_t)batchSize * outChannels * outHeight * outWidth;

  backend->Zero(unpoolScratch, outStateSize);
  backend->MaxPool2DBackward(pooledError, poolArgmax, batchSize, outChannels, outHeight, outWidth,
                             poolOutHeight, poolOutWidth, unpoolScratch);
  return unpoolScratch;
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
    const float* e_above = UnpoolError(layerAbove->GetErrors());
    size_t outSize = (size_t)outChannels * colCols;
    size_t outTotal = (size_t)batchSize * outSize;

    backend->FusedActivationDerivativeMultiply(bottom_up_cols, e_above, mu, derivativeType,
                                               outTotal);
    // Same muPC chain-rule factor as EnsureMuHoldsDerivative()'s own
    // comment: mu = Activation(a*preact+b), so the derivative needs an
    // extra factor of a. FusedActivationDerivativeMultiply wrote BOTH
    // bottom_up_cols (=e_above*Activation'(mu)) and mu itself
    // (=Activation'(mu), in place) without it; both need scaling. No-op
    // when a==1.0 (useMuPCScaling off).
    backend->Scale(bottom_up_cols, outTotal, a);
    backend->Scale(mu, outTotal, a);

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
    // v = dz_dt + momentumBeta*v (a "trace", matching optax's SGD(momentum=.)
    // exactly), NOT v = momentumBeta*v + (1-momentumBeta)*dz_dt (an EMA --
    // what this used to compute). The two are NOT the same hyperparameter:
    // the EMA's steady-state step is just ir, while the trace's is
    // ir/(1-momentumBeta). Porting a PCX ir/momentum pair verbatim under
    // the EMA silently under-drove settling by that factor (confirmed:
    // at momentumBeta=0.55, 2.2x), compounding exponentially with depth.
    // This makes ir/momentum mean exactly what PCX's own config means.
    backend->Scale(v, ownStateSize, momentumBeta);
    backend->AxpyInto(v, dz_dt, ownStateSize, 1.0f);
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

  const float* e_above = UnpoolError(layerAbove->GetErrors());

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

    // No `a` here: EnsureMuHoldsDerivative() now folds muPC's chain-rule
    // factor into mu itself (mu = Activation(a*preact+b), so the
    // derivative needs a*Activation'(.), not just Activation'(.)), which
    // flows into lgRepacked below. Multiplying by `a` again here was a
    // real, previously-uncaught bug: it double-counted the SAME factor,
    // confirmed via finite differences (consistent ~a-sized mismatch
    // across every weight, which cleared up once this line stopped
    // reapplying it). a defaults to 1.0f, so this is a no-op whenever
    // useMuPCScaling is off, unaffecting every already-verified config.
    float lr_batch = lr / batchSize;

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

    float grad_scale = -1.0f / batchSize;  // same reasoning as lr_batch's own comment above

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
      backend->AdamWStep(W, grad_W, m_W, v_W, Wsize, t_device, lr_device, lmbda, 0.9f, 0.999f,
                        adamEpsilon);
    else
      backend->AdamStep(W, grad_W, m_W, v_W, Wsize, t_device, lr_device, 0.9f, 0.999f,
                        adamEpsilon);

    backend->AdamStep(b, grad_b, m_b, v_b, outChannels, t_device, lr_device, 0.9f, 0.999f,
                      adamEpsilon);
    break;
  }
  }
}

void FullConvPCLayer::DirectFeedbackUpdate() noexcept
{
  if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
    return;

  // fl==0 makes this function's entire contribution to W exactly zero
  // (see the final MatMul's alpha=fl/batchSize below) -- skip the
  // broadcast, repack, and GEMM outright rather than computing a result
  // that just gets multiplied away. See flZeroDirty's own comment for
  // why FullConvPCNetwork::TrainStep() has to recapture if fl crosses
  // this boundary, instead of this skip silently sticking forever.
  if (fl == 0.0f)
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

  adjointAbove = UnpoolError(adjointAbove);

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
  // bottom_up_cols's own muPC chain-rule factor is already folded into
  // the MatMul below via alpha=a. mu itself is left holding the
  // UNSCALED derivative by the line above, though -- UpdateWeights()
  // reads mu directly later expecting the fully-correct (scaled) value,
  // same as the classic-settling path, so fix it here too. No-op when
  // a==1.0 (useMuPCScaling off).
  backend->Scale(mu, outTotal, a);

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

  // mu = Activation(a*preact + b), so d(mu)/d(preact) = a*Activation'(.),
  // not just Activation'(.) -- the outer muPC scale `a` is itself part
  // of the function being differentiated. Missing this was a real,
  // previously-uncaught bug: every existing gradient check ran with
  // useMuPCScaling off (a=1.0, where this line is a no-op), so nothing
  // ever exercised a != 1 here. a defaults to 1.0f, so this changes
  // nothing for any currently-working configuration.
  size_t outTotal = (size_t)batchSize * outChannels * outHeight * outWidth;
  backend->ActivationDerivative(derivativeType, mu, outTotal, true);
  backend->Scale(mu, outTotal, a);
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

std::vector<float> FullConvPCLayer::GetDiagnosticStats() const noexcept
{
  size_t eSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  std::vector<float> eHost(eSize);
  backend->CopyToHost(eHost.data(), e, eSize);

  double eSum = 0.0, eSqSum = 0.0;
  for (float v_ : eHost)
  {
    eSum += v_;
    eSqSum += (double)v_ * v_;
  }
  float errorMean = eSize > 0 ? (float)(eSum / (double)eSize) : 0.0f;
  float errorRMS = eSize > 0 ? (float)std::sqrt(eSqSum / (double)eSize) : 0.0f;

  float weightNorm = 0.0f;
  float deadFraction = 0.0f;
  if (outChannels > 0)
  {
    size_t Wsize = (size_t)outChannels * inChannels * kernelH * kernelW;
    std::vector<float> wHost(Wsize);
    backend->CopyToHost(wHost.data(), W, Wsize);
    double wSqSum = 0.0;
    for (float w : wHost)
      wSqSum += (double)w * w;
    weightNorm = (float)std::sqrt(wSqSum);

    size_t muSize = (size_t)batchSize * outChannels * outHeight * outWidth;
    std::vector<float> muHost(muSize);
    backend->CopyToHost(muHost.data(), mu, muSize);
    size_t deadCount = 0;
    for (float m : muHost)
      if (m == 0.0f)
        deadCount++;
    deadFraction = muSize > 0 ? (float)deadCount / (float)muSize : 0.0f;
  }

  return {errorMean, errorRMS, weightNorm, deadFraction};
}

template void FullConvPCLayer::BindMemory<MemoryArena>(MemoryArena& arena);
#if defined(DEEPITY_USE_CUDA)
template void FullConvPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena& arena);
#endif
} // namespace Deep
