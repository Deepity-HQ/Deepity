/**
 * @file CUDABackendFusedOps.cu
 * @brief CUDABackend's fused PC-specific ops: the settling-step state
 * update (plain and momentum variants), Gaussian error/energy, and softmax
 * cross-entropy error/energy. Mirrors the "Fused PC-specific ops" section of
 * IComputeBackend.h (minus TryFusedForwardPass -- see CUDABackendGemm.cu).
 * Split out of the former monolithic CUDABackend.cu -- see
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
__global__ void FusedStateUpdateKernel(float* z, const float* feedback, const float* deriv,
                                       const float* e, size_t n, float ir)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float fb = feedback ? feedback[i] : 0.0f;
    float dev = deriv ? deriv[i] : 1.0f;
    float err = e ? e[i] : 0.0f;
    z[i] += ir * ((fb * dev) - err);
  }
}

__global__ void FusedStateUpdateMomentumKernel(float* z, float* v, const float* feedback,
                                               const float* deriv, const float* e, size_t n,
                                               float ir, float beta)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float fb = feedback ? feedback[i] : 0.0f;
    float dev = deriv ? deriv[i] : 1.0f;
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

void CUDABackend::FusedStateUpdate(float* z, const float* feedback, const float* deriv,
                                   const float* e, size_t n, float ir) noexcept
{
  if (!z || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedStateUpdateKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(z, feedback, deriv, e, n, ir);
  CHECK_CUDA_LAUNCH();
}

void CUDABackend::FusedStateUpdateMomentum(float* z, float* v, const float* feedback,
                                           const float* deriv, const float* e, size_t n, float ir,
                                           float beta) noexcept
{
  if (!z || !v || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedStateUpdateMomentumKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      z, v, feedback, deriv, e, n, ir, beta);
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
} // namespace Deep

#endif
