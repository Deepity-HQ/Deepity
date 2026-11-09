#pragma once
#include <deepity/backend/IComputeBackend.h>
#include <memory>

/**
 * @file CUDAPluginLoader.h
 * @brief Runtime loader for the optional DeepityCUDA plugin library.
 *
 * Deepity's main library/extension module never links cuBLAS/cudart
 * directly -- that would make every `import deepity` (even on a
 * CPU-only machine with no CUDA installed at all) fail to load. The
 * actual CUDA backend lives in a separate shared library
 * (libDeepityCUDA.so / DeepityCUDA.dll), built alongside the main
 * library when a CUDA toolkit is available, and loaded here with
 * dlopen()/LoadLibrary() only at the moment a caller actually asks for
 * DeviceType::DEVICE_GPU. A CPU-only install (`pip install pydeepity`,
 * no CUDA extra) imports fine and simply gets a clear error the first
 * time GPU support is requested, instead of an import-time crash.
 */

namespace Deep
{
/// @brief Loads the DeepityCUDA plugin (if present) and constructs its
/// CUDABackend.
/// @throws std::runtime_error if the plugin library can't be found
/// (this build has no CUDA plugin) or can't be loaded (its own
/// dependencies -- the CUDA runtime, cuBLAS -- aren't resolvable; see
/// `pip install pydeepity[cuda]`), or if it was found but is missing
/// its expected entry point.
std::unique_ptr<IComputeBackend> LoadCUDABackendPlugin();
} // namespace Deep
