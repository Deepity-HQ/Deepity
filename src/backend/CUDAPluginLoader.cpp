#include <deepity/backend/CUDAPluginLoader.h>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Deep
{
namespace
{
using CreateFn = IComputeBackend* (*)();

#if defined(_WIN32)
constexpr const wchar_t* kPluginFileName = L"DeepityCUDA.dll";

/// @brief Directory containing whichever module (deepity_ext.pyd, or a
/// test executable's own statically-linked copy of Deepity) this
/// function's own code ends up in -- the plugin ships next to it either
/// way, since both land in the same place (see CMakeLists.txt's shared
/// RUNTIME_OUTPUT_DIRECTORY for build-tree executables, and the wheel's
/// install(... DESTINATION deepity) for the installed package).
std::wstring PluginPath()
{
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           reinterpret_cast<LPCWSTR>(&PluginPath), &module))
    return kPluginFileName;

  wchar_t path[MAX_PATH];
  DWORD len = GetModuleFileNameW(module, path, MAX_PATH);
  if (len == 0 || len == MAX_PATH)
    return kPluginFileName;

  std::wstring dir(path, len);
  size_t lastSlash = dir.find_last_of(L"\\/");
  if (lastSlash == std::wstring::npos)
    return kPluginFileName;
  return dir.substr(0, lastSlash + 1) + kPluginFileName;
}
#else
#if defined(__APPLE__)
constexpr const char* kPluginFileName = "libDeepityCUDA.dylib";
#else
constexpr const char* kPluginFileName = "libDeepityCUDA.so";
#endif

std::string PluginPath()
{
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&PluginPath), &info) || !info.dli_fname)
    return kPluginFileName;

  std::string path(info.dli_fname);
  size_t lastSlash = path.find_last_of('/');
  if (lastSlash == std::string::npos)
    return kPluginFileName;
  return path.substr(0, lastSlash + 1) + kPluginFileName;
}
#endif
} // namespace

std::unique_ptr<IComputeBackend> LoadCUDABackendPlugin()
{
  static const char* kHint =
      " Either this build has no CUDA plugin, or the plugin's own dependencies "
      "(the CUDA runtime, cuBLAS) aren't installed -- try `pip install pydeepity[cuda]`, "
      "or build from source with a CUDA toolkit on the PATH.";

#if defined(_WIN32)
  std::wstring path = PluginPath();
  HMODULE lib = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!lib)
    throw std::runtime_error("CUDA support isn't available: couldn't load DeepityCUDA.dll "
                              "(Windows error " +
                              std::to_string(GetLastError()) + ")." + kHint);

  auto create = reinterpret_cast<CreateFn>(GetProcAddress(lib, "Deepity_CreateCUDABackend"));
#else
  std::string path = PluginPath();
  void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib)
    throw std::runtime_error(std::string("CUDA support isn't available: couldn't load "
                                          "libDeepityCUDA.so (") +
                              dlerror() + ")." + kHint);

  auto create = reinterpret_cast<CreateFn>(dlsym(lib, "Deepity_CreateCUDABackend"));
#endif

  if (!create)
    throw std::runtime_error("CUDA support isn't available: the CUDA plugin is missing its "
                              "expected entry point (plugin/host version mismatch?).");

  // The plugin and the loading module are always built by this same
  // CMake project in one configure/build, so they share one C++ runtime
  // (one glibc heap on Linux/macOS; MSVC's dynamic CRT on Windows, which
  // this project doesn't override to static anywhere) -- std::unique_ptr's
  // default `delete`, running through CUDABackend's virtual destructor,
  // is safe to call from here even though `new` happened inside the
  // plugin's own translation unit.
  return std::unique_ptr<IComputeBackend>(create());
}
} // namespace Deep
