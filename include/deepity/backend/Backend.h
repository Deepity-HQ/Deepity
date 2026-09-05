#pragma once
#include <memory>
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/Tensor.h> 

namespace Deep
{
    /// @brief Factory function to create the appropriate backend.
    std::unique_ptr<IComputeBackend> CreateBackend(DeviceType device);
}