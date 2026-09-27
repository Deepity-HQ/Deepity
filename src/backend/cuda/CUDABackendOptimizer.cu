/**
 * @file CUDABackendOptimizer.cu
 * @brief CUDABackend's optimizer-step kernels: an atomic counter increment
 * (for device-resident step counting under graph replay) and the Adam/AdamW
 * parameter updates. Mirrors the "Optimizer" section of IComputeBackend.h.
 * Split out of the former monolithic CUDABackend.cu -- see
 * CUDABackendCore.cu, CUDABackendGemm.cu, CUDABackendElementwise.cu,
 * CUDABackendActivations.cu, CUDABackendFusedOps.cu, CUDABackendConv.cu for
 * the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <cmath>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include "CUDACommon.cuh"

namespace Deep
{
__global__ void IncrementCounterKernel(int* counter)
{
  if (counter)
    *counter += 1;
}
void CUDABackend::IncrementCounter(int* counter) noexcept
{
  if (!counter)
    return;
  IncrementCounterKernel<<<1, 1, 0, stream>>>(counter);
  CHECK_CUDA_LAUNCH();
}

__global__ void AdamStepKernel(float* param, const float* grad, float* m, float* v, size_t n,
                               const int* t_ptr, const float* lr_ptr, float beta1, float beta2,
                               float eps)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    int current_t = *t_ptr;
    if (current_t < 1)
      current_t = 1;
    float current_lr = *lr_ptr;

    float beta1_t = 1.0f - powf(beta1, static_cast<float>(current_t));
    float beta2_t = 1.0f - powf(beta2, static_cast<float>(current_t));
    float step_size = current_lr * sqrtf(beta2_t) / beta1_t;

    float g = grad[i];
    float m_val = beta1 * m[i] + (1.0f - beta1) * g;
    float v_val = beta2 * v[i] + (1.0f - beta2) * (g * g);

    m[i] = m_val;
    v[i] = v_val;
    param[i] -= step_size * m_val / (sqrtf(v_val) + eps);
  }
}

__global__ void AdamWStepKernel(float* param, const float* grad, float* m, float* v, size_t n,
                                const int* t_ptr, const float* lr_ptr, float weightDecay,
                                float beta1, float beta2, float eps)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    int current_t = *t_ptr;
    if (current_t < 1)
      current_t = 1;
    float current_lr = *lr_ptr;

    float beta1_t = 1.0f - powf(beta1, static_cast<float>(current_t));
    float beta2_t = 1.0f - powf(beta2, static_cast<float>(current_t));
    float step_size = current_lr * sqrtf(beta2_t) / beta1_t;

    float g = grad[i];
    float p = param[i];
    float m_val = beta1 * m[i] + (1.0f - beta1) * g;
    float v_val = beta2 * v[i] + (1.0f - beta2) * (g * g);

    m[i] = m_val;
    v[i] = v_val;
    p -= current_lr * weightDecay * p;
    p -= step_size * m_val / (sqrtf(v_val) + eps);
    param[i] = p;
  }
}

void CUDABackend::AdamStep(float* param, const float* grad, float* m, float* v, size_t n,
                           const int* t, const float* lr, float beta1, float beta2,
                           float eps) noexcept
{
  if (!param || !grad || !m || !v || !t || !lr || n == 0)
    return;

  // NO host copy -- t/lr stay as device pointers, dereferenced
  // inside the kernel itself, so graph capture/replay re-reads the
  // real, current value every time instead of baking in a
  // one-time snapshot from whenever capture happened to run.
  constexpr int blockSize = 256;
  int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);
  AdamStepKernel<<<numBlocks, blockSize, 0, stream>>>(
      param, grad, m, v, n, t, lr, beta1, beta2, eps);
  CHECK_CUDA_LAUNCH();
}

void CUDABackend::AdamWStep(float* param, const float* grad, float* m, float* v, size_t n,
                            const int* t, const float* lr, float weightDecay, float beta1,
                            float beta2, float eps) noexcept
{
  if (!param || !grad || !m || !v || !t || !lr || n == 0)
    return;

  constexpr int blockSize = 256;
  int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);
  AdamWStepKernel<<<numBlocks, blockSize, 0, stream>>>(
      param, grad, m, v, n, t, lr, weightDecay, beta1, beta2, eps);
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
