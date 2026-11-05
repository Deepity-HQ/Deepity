#pragma once

namespace Deep
{
/// @brief Which physical device a Tensor/IComputeBackend pair runs on.
enum class DeviceType
{
  DEVICE_CPU,
  DEVICE_GPU
};
} // namespace Deep