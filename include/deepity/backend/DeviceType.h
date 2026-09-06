#pragma once

/// @brief This file exists to provide both Tensor.h and IComputeBackend.h with device types without introducing circular dependencies.
namespace Deep
{
    enum class DeviceType
    {
        DEVICE_CPU,
        DEVICE_GPU
    };
}