#include <deepity/backend/Backend.h>
#include <deepity/backend/CPUBackend.h>
#include <deepity/backend/CUDAPluginLoader.h>
#include <stdexcept>

namespace Deep
{
std::unique_ptr<IComputeBackend> CreateBackend(DeviceType device)
{
  if (device == DeviceType::DEVICE_CPU)
    return std::make_unique<CPUBackend>();

  if (device == DeviceType::DEVICE_GPU)
    return LoadCUDABackendPlugin();

  throw std::invalid_argument("Unknown DeviceType requested from CreateBackend.");
}
} // namespace Deep