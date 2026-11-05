/**
 * @file tCUDARandomizeVerify.cpp
 * @brief Verifies CUDABackend::RandomizeNormal/RandomizeUniform on both
 * sides of issue #67's size threshold: the small-n host-generate-then-copy
 * path and the large-n curand-kernel path. Not a bit-exact differential
 * test against the CPU backend -- the two paths (and the CPU backend)
 * each use a different RNG, so this checks sampled mean/stddev land close
 * to what was requested instead.
 */
#include <cmath>
#include <cstdio>
#include <deepity/backend/Backend.h>
#include <vector>

using namespace Deep;

namespace
{
bool CheckDistribution(const std::vector<float>& samples, float expectedMean, float expectedStddev,
                       const char* label)
{
  double sum = 0.0;
  for (float v : samples)
    sum += v;
  double mean = sum / samples.size();

  double varSum = 0.0;
  for (float v : samples)
    varSum += (v - mean) * (v - mean);
  double stddev = std::sqrt(varSum / samples.size());

  printf("[%s] n=%zu mean=%f (expected %f) stddev=%f (expected %f)\n", label, samples.size(), mean,
        expectedMean, stddev, expectedStddev);

  // Loose tolerances: this is a sanity check that the kernel/host path
  // actually ran and produced the right distribution shape, not a
  // statistical-power exercise.
  bool ok = std::fabs(mean - expectedMean) < 0.1 * std::max(1.0, (double)expectedStddev) &&
           std::fabs(stddev - expectedStddev) < 0.15 * expectedStddev;
  if (!ok)
    printf("  FAILED: outside tolerance.\n");
  return ok;
}
} // namespace

int main()
{
  auto backend = CreateBackend(DeviceType::DEVICE_GPU);

  bool allOk = true;

  // Below kGPURandomizeThreshold (1<<16): exercises the new host-side
  // generate + cudaMemcpy path.
  {
    const size_t n = 4096;
    Tensor buf(backend.get(), DeviceType::DEVICE_GPU, n);
    backend->RandomizeNormal(buf.Data(), n, 0.0f, 1.0f, 123);
    std::vector<float> host(n);
    buf.CopyToHost(host.data());
    allOk &= CheckDistribution(host, 0.0f, 1.0f, "small normal");
  }
  {
    const size_t n = 4096;
    Tensor buf(backend.get(), DeviceType::DEVICE_GPU, n);
    backend->RandomizeUniform(buf.Data(), n, -3.0f, 3.0f, 124);
    std::vector<float> host(n);
    buf.CopyToHost(host.data());
    // Uniform(-3,3) has stddev = range / sqrt(12).
    allOk &= CheckDistribution(host, 0.0f, 6.0f / std::sqrt(12.0f), "small uniform");
  }

  // Above kGPURandomizeThreshold: exercises the original curand-kernel
  // path, unchanged by this PR, re-checked here so a future change to
  // the threshold can't silently stop testing it.
  {
    const size_t n = 1 << 18;
    Tensor buf(backend.get(), DeviceType::DEVICE_GPU, n);
    backend->RandomizeNormal(buf.Data(), n, 2.0f, 0.5f, 125);
    std::vector<float> host(n);
    buf.CopyToHost(host.data());
    allOk &= CheckDistribution(host, 2.0f, 0.5f, "large normal");
  }
  {
    const size_t n = 1 << 18;
    Tensor buf(backend.get(), DeviceType::DEVICE_GPU, n);
    backend->RandomizeUniform(buf.Data(), n, 0.0f, 10.0f, 126);
    std::vector<float> host(n);
    buf.CopyToHost(host.data());
    allOk &= CheckDistribution(host, 5.0f, 10.0f / std::sqrt(12.0f), "large uniform");
  }

  if (!allOk)
  {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
