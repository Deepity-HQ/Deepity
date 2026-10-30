#pragma once
#include <deepity/backend/CPUBackendT.h>
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
/// given T/AccumT. No CUDABackendT<T, AccumT> exists yet (CUDA support
/// is still only on the non-templated CUDABackend) -- DEVICE_GPU always
/// throws here, not gated behind DEEPITY_USE_CUDA like CreateBackend()'s
/// own GPU branch, since the real gap is "not implemented yet," not
/// "not compiled in this build."
template <typename T, typename AccumT = T>
std::unique_ptr<IComputeBackendT<T, AccumT>> CreateBackendT(DeviceType device)
{
  if (device == DeviceType::DEVICE_CPU)
    return std::make_unique<CPUBackendT<T, AccumT>>();

  if (device == DeviceType::DEVICE_GPU)
    throw std::runtime_error("CreateBackendT: no CUDABackendT<T, AccumT> exists yet -- the "
                             "templated layer/network pass only has a CPU backend so far.");

  throw std::invalid_argument("Unknown DeviceType requested from CreateBackendT.");
}
} // namespace Deep
