#pragma once
#include <deepity/backend/CPUBackendT.h>
#include <deepity/backend/CUDABackendT.h>
#include <deepity/backend/IComputeBackend.h>
#include <memory>
#include <stdexcept>

/**
 * @file BackendT.h
 * @brief Templated counterpart to Backend.h's CreateBackend(), for
 * FullPCNetworkT<T, AccumT>.
 */

namespace Deep
{
/// @brief Creates the appropriate backend for a given T/AccumT and device.
/// @throws std::runtime_error if `device` is DEVICE_GPU and no CUDA
/// backend exists for this T/AccumT (currently only T=AccumT=float is
/// supported on GPU), or if CUDA support wasn't compiled into this build.
/// @throws std::invalid_argument for an unrecognized DeviceType.
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
                             "float so far; no CUDA backend exists yet for this T/AccumT, or "
                             "CUDA support wasn't compiled into this build.");
  }

  throw std::invalid_argument("Unknown DeviceType requested from CreateBackendT.");
}
} // namespace Deep
