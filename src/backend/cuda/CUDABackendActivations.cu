/**
 * @file CUDABackendActivations.cu
 * @brief CUDABackend's activation kernels (forward + derivative, for every
 * ActivationType) and the Activation()/ActivationDerivative() dispatchers
 * that select among them. Mirrors the "Activation" section of
 * IComputeBackend.h. Split out of the former monolithic CUDABackend.cu --
 * see CUDABackendCore.cu, CUDABackendGemm.cu, CUDABackendElementwise.cu,
 * CUDABackendFusedOps.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <cmath>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include "CUDACommon.cuh"

namespace Deep
{
__global__ void ReluKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = fmaxf(0.0f, src[i]);
}

__global__ void GeluKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float xi = src[i];
    float inner = MAGIC_GELU_1 * xi * (1.0f + MAGIC_GELU_2 * xi * xi);
    float t;
#if __CUDA_ARCH__ >= 800
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
#else
    t = tanhf(inner);
#endif
    dst[i] = 0.5f * xi * (1.0f + t);
  }
}

__global__ void tanhKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(src[i]));
    dst[i] = y;
  }
}

__global__ void sigmoidKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * src[i]));
    dst[i] = fmaf(0.5f, t, 0.5f);
  }
}

__global__ void eSigmoidKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float s = src[i];
    dst[i] = 0.5f * (s / (1.0f + fabsf(s)) + 1.0f);
  }
}

__global__ void linearKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

__global__ void dReluKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (float)(src[i] > 0.0f);
}

constexpr float MAGIC_GELU_2_3 = 3.0f * MAGIC_GELU_2;

__global__ void dGeluKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float x = src[i];
    float xsq = x * x;
    float inner = MAGIC_GELU_1 * x * (1.0f + MAGIC_GELU_2 * xsq);
    float t;
#if __CUDA_ARCH__ >= 800
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
#else
    t = tanhf(inner);
#endif
    float gprime = MAGIC_GELU_1 * (1.0f + MAGIC_GELU_2_3 * xsq);
    float term1 = 0.5f * (1.0f + t);
    float term2 = 0.5f * x * gprime * (1.0f - t * t);
    dst[i] = term1 + term2;
  }
}

__global__ void dTanhKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(src[i]));
    dst[i] = fmaf(-t, t, 1.0f);
  }
}

__global__ void dSigmoidKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * src[i]));
    dst[i] = 0.25f * fmaf(-t, t, 1.0f);
  }
}

__global__ void d_eSigmoidKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float a = 1.0f + fabsf(src[i]);
    dst[i] = 0.5f / (a * a);
  }
}

__global__ void dLinearKernelInto(float* dst, const float* src, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = 1.0f;
}

void CUDABackend::Activation(ActivationType type, float* buf, size_t n) noexcept
{
  ActivationInto(type, buf, buf, n);
}

void CUDABackend::ActivationInto(ActivationType type, float* dst, const float* src,
                                 size_t n) noexcept
{
  if (!dst || !src || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  switch (type)
  {
  case ActivationType::RELU:
    ReluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::GELU:
    GeluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::SIGMOID:
    sigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::eSIGMOID:
    eSigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::TANH:
    tanhKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::LINEAR:
    linearKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::NONE:
  default:
    break;
  }
  CHECK_CUDA_LAUNCH();
}

void CUDABackend::ActivationDerivative(ActivationType type, float* buf, size_t n,
                                       bool activated) noexcept
{
  ActivationDerivativeInto(type, buf, buf, n);
}

void CUDABackend::ActivationDerivativeInto(ActivationType type, float* dst, const float* src,
                                           size_t n) noexcept
{
  if (!dst || !src || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  switch (type)
  {
  case ActivationType::dRELU:
    dReluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::dGELU:
    dGeluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::dSIGMOID:
    dSigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::d_eSIGMOID:
    d_eSigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::dTANH:
    dTanhKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::dLINEAR:
    dLinearKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  case ActivationType::NONE:
  default:
    dLinearKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n);
    break;
  }
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
