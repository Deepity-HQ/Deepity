#include <deepity/backend/Backend.h>
#include <deepity/backend/CPUBackend.h>
#include <stdexcept>

#ifdef DEEPITY_ENABLE_CUDA
#include <deepity/backend/CUDABackend.h>
#endif

namespace Deep
{
    std::unique_ptr<IComputeBackend> CreateBackend(DeviceType device)
    {
        if (device == DeviceType::DEVICE_CPU)
        {
            return std::make_unique<CPUBackend>();
        }
        else if (device == DeviceType::DEVICE_GPU)
        {
#ifdef DEEPITY_ENABLE_CUDA
            return std::make_unique<CUDABackend>();
#else
            throw std::runtime_error("Deepity was compiled without CUDA support (-DDEEPITY_ENABLE_CUDA=OFF). Cannot create CUDABackend.");
#endif
        }
        
        throw std::invalid_argument("Unknown DeviceType requested from CreateBackend.");
    }
}