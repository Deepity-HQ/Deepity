/**
 * @file CUDABackendFusedOps.cu
 * @brief CUDABackend's fused PC-specific ops: the settling-step state
 * update (plain and momentum variants), Gaussian error/energy, and softmax
 * cross-entropy error/energy. Mirrors the "Fused PC-specific ops" section of
 * IComputeBackend.h (minus TryFusedForwardPass, see CUDABackendGemm.cu).
 * Split out of the former monolithic CUDABackend.cu, see
 * CUDABackendCore.cu, CUDABackendGemm.cu, CUDABackendElementwise.cu,
 * CUDABackendActivations.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <cmath>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include "CUDACommon.cuh"

namespace Deep
{
__global__ void FusedStateUpdateKernel(float* z, const float* feedback, ActivationType dType,
                                       const float* e, size_t n, float ir)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float fb = feedback ? feedback[i] : 0.0f;
    float dev = ActivationDerivativeDevice(dType, z[i]);
    float err = e ? e[i] : 0.0f;
    z[i] += ir * ((fb * dev) - err);
  }
}

__global__ void FusedStateUpdateMomentumKernel(float* z, float* v, const float* feedback,
                                               ActivationType dType, const float* e, size_t n,
                                               float ir, float beta)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float fb = feedback ? feedback[i] : 0.0f;
    float dev = ActivationDerivativeDevice(dType, z[i]);
    float err = e ? e[i] : 0.0f;
    float update = (fb * dev) - err;
    float v_new = beta * v[i] + (1.0f - beta) * update;
    v[i] = v_new;
    z[i] += ir * v_new;
  }
}

__global__ void ComputeErrorKernel(float* e, const float* z, const float* mu, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    e[i] = z[i] - mu[i];
}

void CUDABackend::FusedStateUpdate(float* z, const float* feedback, ActivationType dType,
                                   const float* e, size_t n, float ir) noexcept
{
  if (!z || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedStateUpdateKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(z, feedback, dType, e, n, ir);
  CHECK_CUDA_LAUNCH();
}

void CUDABackend::FusedStateUpdateMomentum(float* z, float* v, const float* feedback,
                                           ActivationType dType, const float* e, size_t n, float ir,
                                           float beta) noexcept
{
  if (!z || !v || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedStateUpdateMomentumKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      z, v, feedback, dType, e, n, ir, beta);
  CHECK_CUDA_LAUNCH();
}

float CUDABackend::ComputeErrorAndEnergy(float* e, const float* z, const float* mu,
                                         size_t n) noexcept
{
  if (!e || !z || !mu || n == 0)
    return 0.0f;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  ComputeErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
  CHECK_CUDA_LAUNCH();

  float sum_of_squares = 0.0f;
  cublasSdot(handle, n, e, 1, e, 1, &sum_of_squares);
  cudaStreamSynchronize(stream);

  return 0.5f * sum_of_squares;
}

void CUDABackend::ComputeError(float* e, const float* z, const float* mu, size_t n) noexcept
{
  if (!e || !z || !mu || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  ComputeErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
  CHECK_CUDA_LAUNCH();
}

__global__ void SoftmaxCrossEntropyKernel(float* e, const float* z, const float* mu,
                                          size_t batchSize, size_t nextSize, float* rowEnergies)
{
  size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batchSize)
    return;

  size_t base = b * nextSize;
  const float eps = 1e-8f;

  float max_val = mu[base];
  for (size_t j = 1; j < nextSize; ++j)
    max_val = fmaxf(max_val, mu[base + j]);

  float sum_exp = 0.0f;
  for (size_t j = 0; j < nextSize; ++j)
  {
    float ex = expf(mu[base + j] - max_val);
    e[base + j] = ex;
    sum_exp += ex;
  }

  float row_energy = 0.0f;
  bool needRowEnergy = (rowEnergies != nullptr);
  for (size_t j = 0; j < nextSize; ++j)
  {
    float prob = e[base + j] / sum_exp;
    e[base + j] = prob;
    if (needRowEnergy)
      row_energy -= z[base + j] * logf(prob + eps);
  }
  if (needRowEnergy)
    rowEnergies[b] = row_energy;

  for (size_t j = 0; j < nextSize; ++j)
    e[base + j] = z[base + j] - e[base + j];
}

float CUDABackend::ComputeSoftmaxCrossEntropyErrorAndEnergy(float* e, const float* z,
                                                            const float* mu, size_t batchSize,
                                                            size_t nextSize,
                                                            float* rowEnergies) noexcept
{
  if (!e || !z || !mu || !rowEnergies || batchSize == 0 || nextSize == 0)
    return 0.0f;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
  SoftmaxCrossEntropyKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      e, z, mu, batchSize, nextSize, rowEnergies);
  CHECK_CUDA_LAUNCH();

  float total_energy = 0.0f;
  cublasSasum(handle, (int)batchSize, rowEnergies, 1, &total_energy);
  cudaStreamSynchronize(stream);

  return total_energy;
}

void CUDABackend::ComputeSoftmaxCrossEntropyError(float* e, const float* z, const float* mu,
                                                  size_t batchSize, size_t nextSize) noexcept
{
  if (!e || !z || !mu || batchSize == 0 || nextSize == 0)
    return;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
  SoftmaxCrossEntropyKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      e, z, mu, batchSize, nextSize, /*rowEnergies=*/nullptr);
  CHECK_CUDA_LAUNCH();
}

__global__ void FusedActivationDerivativeMultiplyKernel(float* dst, const float* a,
                                                         float* activatedInOut, ActivationType dType,
                                                         size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float deriv = ActivationDerivativeFromActivatedDevice(dType, activatedInOut[i]);
    dst[i] = a[i] * deriv;
    activatedInOut[i] = deriv;
  }
}

void CUDABackend::FusedActivationDerivativeMultiply(float* dst, const float* a, float* activatedInOut,
                                                     ActivationType dType, size_t n) noexcept
{
  if (!dst || !a || !activatedInOut || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedActivationDerivativeMultiplyKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, a, activatedInOut,
                                                                             dType, n);
  CHECK_CUDA_LAUNCH();
}

// One thread per element; each thread's energy contribution is folded into
// a single device accumulator via atomicAdd. Simple and obviously correct
// rather than a tree reduction, "port first, optimize" (see
// RepackForBatchedGemm's own doc comment for the same reasoning applied
// elsewhere in this file), and this isn't inside any graph-captured
// region: the ConvPCLayer/DiscriminativePCLayer family that needs this
// doesn't use CUDA graph capture at all (unlike SimplePCNetwork/
// FullPCNetwork/DirectKPPCNetwork), so a plain synchronous allocation
// here is safe.
__global__ void PrecisionWeightedErrorEnergyKernel(float* e, const float* z, const float* mu,
                                                    const float* p, size_t n, size_t width,
                                                    float* energyAccum)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    size_t i = idx % width;
    float err = z[idx] - mu[idx];
    e[idx] = err;
    float precision = fmaxf(p[i], 1e-8f);
    float contribution = 0.5f * precision * err * err - 0.5f * logf(precision);
    atomicAdd(energyAccum, contribution);
  }
}

float CUDABackend::ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                                          const float* p, size_t batchSize,
                                                          size_t width) noexcept
{
  size_t n = batchSize * width;
  if (!e || !z || !mu || !p || n == 0)
    return 0.0f;

  float* energyAccum = Allocate(1);
  if (!energyAccum)
    return 0.0f;
  cudaMemsetAsync(energyAccum, 0, sizeof(float), stream);

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  PrecisionWeightedErrorEnergyKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, p, n, width,
                                                                        energyAccum);
  CHECK_CUDA_LAUNCH();

  float total = 0.0f;
  cudaMemcpyAsync(&total, energyAccum, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  Free(energyAccum);

  return total;
}

__global__ void UpdatePrecisionFromErrorKernel(float* p, float* log_p, const float* e,
                                               size_t batchSize, size_t width, float pr)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width)
  {
    float grad = 0.0f;
    for (size_t b = 0; b < batchSize; ++b)
    {
      float err = e[b * width + i];
      grad += 0.5f * (p[i] * err * err - 1.0f);
    }
    grad /= (float)batchSize;

    float newLogP = log_p[i] - pr * grad;
    newLogP = fmaxf(-5.0f, fminf(newLogP, 5.0f));
    log_p[i] = newLogP;
    p[i] = expf(newLogP);
  }
}

void CUDABackend::UpdatePrecisionFromError(float* p, float* log_p, const float* e, size_t batchSize,
                                           size_t width, float pr) noexcept
{
  if (!p || !log_p || !e || batchSize == 0 || width == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((width + BLOCK_SIZE - 1) / BLOCK_SIZE);
  UpdatePrecisionFromErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(p, log_p, e, batchSize, width,
                                                                    pr);
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
