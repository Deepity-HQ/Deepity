#pragma once
#include <deepity/backend/CPUBackendT.h>
#include <deepity/backend/CUDABackendT.h>
#include <deepity/backend/IComputeBackend.h>
#include <memory>
#include <stdexcept>

/**
 * @file BackendT.h
 * @brief Templated counterpart to Backend.h's CreateBackend(), for
 * FullPCNetworkT<T, AccumT>. Header-only (not CreateBackend()'s
 * declaration + CreateBackend.cpp's definition split): a function
 * template's definition has to be visible at every instantiation site
 * anyway, so splitting it into a .cpp would just mean explicitly
 * instantiating it there for every T/AccumT pair FullPCNetworkT
 * supports -- no simpler than defining it here once.
 */

namespace Deep
{
/// @brief Factory function to create the appropriate backend for a
/// given T/AccumT. DEVICE_GPU only has a real backend for T=AccumT=
/// float so far (issue #55; see CUDABackendT.h) -- every other T/AccumT
/// still throws, since no CUDA compute path (fp16 GEMM, sub-8-bit
/// decode/encode kernels) exists for them yet, on this path or the
/// non-templated one.
template <typename T, typename AccumT = T>
std::unique_ptr<IComputeBackendT<T, AccumT>> CreateBackendT(DeviceType device)
{
  if (device == DeviceType::DEVICE_CPU)
    return std::make_unique<CPUBackendT<T, AccumT>>();

  if (device == DeviceType::DEVICE_GPU)
  {
#ifdef DEEPITY_USE_CUDA
    if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
      return std::make_unique<CUDABackendT<float, float>>();
#endif
    throw std::runtime_error("CreateBackendT: CUDABackendT<T, AccumT> only supports T=AccumT="
                             "float so far (issue #55); no CUDA backend exists yet for this "
                             "T/AccumT, or CUDA support wasn't compiled into this build.");
  }

  throw std::invalid_argument("Unknown DeviceType requested from CreateBackendT.");
}
} // namespace Deep
