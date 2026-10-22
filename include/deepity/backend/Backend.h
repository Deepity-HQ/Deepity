#pragma once
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/Tensor.h>
#include <memory>

namespace Deep
{
/// @brief Factory function to create the appropriate backend.
std::unique_ptr<IComputeBackend> CreateBackend(DeviceType device);
} // namespace Deep