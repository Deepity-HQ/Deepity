#pragma once

#include <iostream>
#include <deepity/utils/ActivationType.h>

/**
 * @file CUDACommon.cuh
 * @brief Shared helper(s) for the CUDABackend*.cu translation units. Only
 * ever included from a .cu file compiled under DEEPITY_USE_CUDA, never
 * from host-only (.cpp) code.
 */

/// @brief Logs the most recent CUDA error (if any) to stderr, tagged with
/// the call site. Used right after every raw `<<<...>>>` kernel launch,
/// cuBLAS/CUTLASS calls check their own returned status instead.
#define CHECK_CUDA_LAUNCH()                                                                        \
  do                                                                                               \
  {                                                                                                \
    cudaError_t err = cudaGetLastError();                                                          \
    if (err != cudaSuccess)                                                                        \
    {                                                                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "                       \
                << cudaGetErrorString(err) << std::endl;                                           \
    }                                                                                              \
  } while (0)

namespace Deep
{
/// @brief Per-thread activation derivative, computed from a single
/// pre-activation value, the CUDA-kernel counterpart of
/// Deep::ActivationDerivativeScalar (see Activations.h) for host code.
/// Shared (via this header, not cross-TU device linkage) by the
/// standalone dXxxKernelInto kernels in CUDABackendActivations.cu and the
/// fused settling-step kernels in CUDABackendFusedOps.cu, so both compute
/// the exact same math from one place. Relies on MAGIC_GELU_1/MAGIC_GELU_2
/// (Deep::utils::activations::Gelu.h) already being visible via
/// CUDABackend.h -> IComputeBackend.h -> Activations.h, included before
/// this header in every CUDABackend*.cu translation unit.
/// @param dType A derivative-flavored ActivationType (e.g. dTANH, not
/// TANH). Anything else (including NONE) falls back to 1.0.
/// @param z The pre-activation value.
__device__ __forceinline__ float ActivationDerivativeDevice(ActivationType dType, float z)
{
  switch (dType)
  {
  case ActivationType::dRELU:
    return (float)(z > 0.0f);
  case ActivationType::dGELU:
  {
    float zsq = z * z;
    float inner = MAGIC_GELU_1 * z * (1.0f + MAGIC_GELU_2 * zsq);
    float t;
#if __CUDA_ARCH__ >= 800
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
#else
    t = tanhf(inner);
#endif
    float gprime = MAGIC_GELU_1 * (1.0f + 3.0f * MAGIC_GELU_2 * zsq);
    return 0.5f * (1.0f + t) + 0.5f * z * gprime * (1.0f - t * t);
  }
  case ActivationType::dSIGMOID:
  {
    float t;
#if __CUDA_ARCH__ >= 800
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * z));
#else
    t = tanhf(0.5f * z);
#endif
    return 0.25f * fmaf(-t, t, 1.0f);
  }
  case ActivationType::d_eSIGMOID:
  {
    float a = 1.0f + fabsf(z);
    return 0.5f / (a * a);
  }
  case ActivationType::dTANH:
  {
    float t;
#if __CUDA_ARCH__ >= 800
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(z));
#else
    t = tanhf(z);
#endif
    return fmaf(-t, t, 1.0f);
  }
  case ActivationType::dLINEAR:
  case ActivationType::NONE:
  default:
    return 1.0f;
  }
}

/// @brief Per-thread activation derivative computed FROM AN
/// ALREADY-ACTIVATED value, the device counterpart of
/// Deep::ActivationDerivativeFromActivatedScalar (see Activations.h).
/// @warning GELU has no closed form from its activated value alone;
/// see ActivationDerivativeFromActivatedScalar's warning, the dGELU
/// case here reuses ActivationDerivativeDevice's raw-z formula on
/// whatever was passed in, matching existing activated=true call sites'
/// pre-existing (wrong, for GELU) behavior rather than changing it.
/// @param dType A derivative-flavored ActivationType. Anything else
/// (including NONE) falls back to 1.0.
/// @param activated The already-activated value (or, for dGELU only,
/// see the warning above).
__device__ __forceinline__ float ActivationDerivativeFromActivatedDevice(ActivationType dType, float activated)
{
  switch (dType)
  {
  case ActivationType::dRELU:
    return (float)(activated > 0.0f);
  case ActivationType::dGELU:
    return ActivationDerivativeDevice(dType, activated);
  case ActivationType::dSIGMOID:
    return activated * (1.0f - activated);
  case ActivationType::d_eSIGMOID:
    return 2.0f * activated * (1.0f - activated);
  case ActivationType::dTANH:
    return fmaf(-activated, activated, 1.0f);
  case ActivationType::dLINEAR:
  case ActivationType::NONE:
  default:
    return 1.0f;
  }
}
} // namespace Deep
