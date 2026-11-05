/**
 * @file tCUDARandomizeBenchmark.cpp
 * @brief Manual profiling tool (no pass/fail) for issue #67: times
 * CUDABackend's host-generate-and-copy path against its curand-kernel
 * path across a range of sizes straddling kGPURandomizeThreshold
 * (CUDABackendCore.cu), to find where the real crossover actually is on
 * real hardware. Not registered with ctest -- read the printed table and
 * move kGPURandomizeThreshold if it disagrees with the guessed 1<<16.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deepity/backend/Backend.h>
#include <deepity/backend/CUDABackend.h>
#include <tuple>
#include <vector>

using namespace Deep;
using Clock = std::chrono::steady_clock;

namespace
{
constexpr int kWarmup = 3;
constexpr int kReps = 20;

/// @brief Times `call` kReps times after kWarmup untimed warm-up calls
/// (so the CUDA allocator's sub-allocator is already warm, matching
/// how RandomizeWeights() is actually used -- many layers initialized
/// in the same process). Returns {coldFirstCallUs, warmMeanUs, warmMinUs}.
template <typename Fn>
std::tuple<double, double, double> Time(Fn&& call)
{
  auto t0 = Clock::now();
  call();
  double coldUs = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

  for (int i = 0; i < kWarmup; ++i)
    call();

  std::vector<double> samples(kReps);
  for (int i = 0; i < kReps; ++i)
  {
    auto t1 = Clock::now();
    call();
    samples[i] = std::chrono::duration<double, std::micro>(Clock::now() - t1).count();
  }
  double mean = 0.0;
  for (double s : samples)
    mean += s;
  mean /= samples.size();
  double minUs = *std::min_element(samples.begin(), samples.end());

  return {coldUs, mean, minUs};
}
} // namespace

int main()
{
  auto backendPtr = CreateBackend(DeviceType::DEVICE_GPU);
  auto* backend = dynamic_cast<CUDABackend*>(backendPtr.get());
  if (!backend)
  {
    printf("FAILED: CreateBackend(DEVICE_GPU) didn't return a CUDABackend.\n");
    return 1;
  }

  const size_t sizes[] = {256,    1024,   4096,   16384,  32768,  49152,
                          65536,  81920,  98304,  131072, 262144, 1048576, 4194304};

  printf("%12s | %10s %10s %10s | %10s %10s %10s | %s\n", "n", "host_cold", "host_mean",
        "host_min", "gpu_cold", "gpu_mean", "gpu_min", "faster (warm)");
  printf("-------------------------------------------------------------------------------------"
        "----------\n");

  for (size_t n : sizes)
  {
    Tensor buf(backend, DeviceType::DEVICE_GPU, n);

    auto [hostCold, hostMean, hostMin] =
        Time([&] { backend->RandomizeNormalForceHostPath(buf.Data(), n, 0.0f, 1.0f, 1); });
    auto [gpuCold, gpuMean, gpuMin] =
        Time([&] { backend->RandomizeNormalForceGPUPath(buf.Data(), n, 0.0f, 1.0f, 1); });

    printf("%12zu | %10.2f %10.2f %10.2f | %10.2f %10.2f %10.2f | %s\n", n, hostCold, hostMean,
          hostMin, gpuCold, gpuMean, gpuMin, (hostMean < gpuMean) ? "host" : "gpu");
  }

  printf("\nAll times in microseconds. 'faster (warm)' compares the steady-state means\n");
  printf("(post-warm-up), which is what matters once a process has initialized a few\n");
  printf("layers already. Set kGPURandomizeThreshold (CUDABackendCore.cu) to the\n");
  printf("smallest n where 'gpu' wins, if that differs from the current 1<<16=65536.\n");
  return 0;
}
