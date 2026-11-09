#include <deepity/backend/CUDABackend.h>

/**
 * @file CUDAPluginExports.cpp
 * @brief The DeepityCUDA plugin's entire public ABI: one factory
 * function, looked up by name at runtime by CUDAPluginLoader.cpp. Keep
 * this the only exported symbol -- everything else about CUDABackend
 * stays an implementation detail the host library never links against
 * directly.
 */

#if defined(_WIN32)
#define DEEPITY_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define DEEPITY_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

DEEPITY_PLUGIN_EXPORT Deep::IComputeBackend* Deepity_CreateCUDABackend()
{
  return new Deep::CUDABackend();
}
