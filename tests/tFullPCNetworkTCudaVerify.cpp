/**
 * @file tFullPCNetworkTCudaVerify.cpp
 * @brief CUDA-only differential test for issue #55: confirms
 * CreateBackendT<float,float>(DEVICE_GPU) (via CUDABackendT, see
 * include/deepity/backend/CUDABackendT.h) actually constructs and
 * runs a real FullPCNetworkT<float,float> end to end on a GPU, with
 * results close to the same network run on CPU.
 *
 * Not a bit-exact comparison like tFullPCLayerTemplatingVerify.cpp's
 * CPU-only checks -- cuBLAS's reduction order differs from OpenBLAS's,
 * so exact equality isn't the right bar here. Patterned after
 * tMatMulLargeAsymmetricVerify.cpp's own tolerance-based CPU-vs-GPU
 * comparison instead (CPU already trusted, GPU compared against it).
 *
 * Only runs when DEEPITY_CUDA_ENABLED (see CMakeLists.txt) -- requires
 * an actual GPU at runtime, not just the CUDA toolkit at build time,
 * so this is deliberately NOT registered with add_test()/ctest, same
 * as every other CUDA-only verify test in this suite. Run the built
 * binary directly: ./bin/FullPCNetworkTCudaVerify
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetworkT.h>
#include <limits>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
bool CloseEnough(float a, float b, const char* label)
{
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  if (diff > scale * 1e-2f)
  {
    printf("  MISMATCH in %s: cpu=%.6g gpu=%.6g\n", label, (double)a, (double)b);
    return false;
  }
  return true;
}

void BuildNetwork(FullPCNetworkT<float, float>& net, size_t inSize, size_t hidSize,
                  size_t outSize, float lr, float ir, float fl, float lmbda, std::mt19937& rng)
{
  net.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
              ActivationType::dSIGMOID);
  net.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.SetUseMuPCScaling(true);
  net.Compile();
  net.RandomizeWeights(rng);
}
} // namespace

int main()
{
  const size_t inSize = 8, hidSize = 8, outSize = 4, batchSize = 4;
  const float lr = 0.01f, ir = 0.1f, fl = 0.001f, lmbda = 0.0f;
  const int settlingSteps = 5;
  const int trainSteps = 10;

  std::vector<float> input(batchSize * inSize), target(batchSize * outSize);
  std::mt19937 dataRng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);
  for (auto& v : target)
    v = dist(dataRng);

  FullPCNetworkT<float, float> cpuNet(batchSize, DeviceType::DEVICE_CPU);
  std::mt19937 cpuSeed(99);
  BuildNetwork(cpuNet, inSize, hidSize, outSize, lr, ir, fl, lmbda, cpuSeed);

  FullPCNetworkT<float, float> gpuNet(batchSize, DeviceType::DEVICE_GPU);
  std::mt19937 gpuSeed(99);
  BuildNetwork(gpuNet, inSize, hidSize, outSize, lr, ir, fl, lmbda, gpuSeed);

  bool ok = true;
  float firstCpuEnergy = std::numeric_limits<float>::infinity();
  float firstGpuEnergy = std::numeric_limits<float>::infinity();
  float lastCpuEnergy = std::numeric_limits<float>::infinity();
  float lastGpuEnergy = std::numeric_limits<float>::infinity();

  for (int step = 0; step < trainSteps; ++step)
  {
    float cpuEnergy = cpuNet.TrainStep(input, target, settlingSteps);
    float gpuEnergy = gpuNet.TrainStep(input, target, settlingSteps);

    printf("  step %d: cpu energy=%.6f gpu energy=%.6f\n", step, cpuEnergy, gpuEnergy);

    if (!std::isfinite(cpuEnergy) || !std::isfinite(gpuEnergy))
    {
      printf("  FAIL: non-finite energy at step %d\n", step);
      ok = false;
      break;
    }

    char label[64];
    snprintf(label, sizeof(label), "network energy (step %d)", step);
    ok &= CloseEnough(cpuEnergy, gpuEnergy, label);

    if (step == 0)
    {
      firstCpuEnergy = cpuEnergy;
      firstGpuEnergy = gpuEnergy;
    }
    lastCpuEnergy = cpuEnergy;
    lastGpuEnergy = gpuEnergy;
  }

  // Both devices should show the same qualitative training behavior:
  // energy lower at the end than at the start, not just finite every step.
  if (ok && lastCpuEnergy >= firstCpuEnergy)
  {
    printf("  FAIL: CPU energy did not decrease over %d steps\n", trainSteps);
    ok = false;
  }
  if (ok && lastGpuEnergy >= firstGpuEnergy)
  {
    printf("  FAIL: GPU energy did not decrease over %d steps\n", trainSteps);
    ok = false;
  }

  // Predict() output agreement, same tolerance as the energy checks above.
  if (ok)
  {
    std::vector<float> cpuPred = cpuNet.Predict(input, settlingSteps);
    std::vector<float> gpuPred = gpuNet.Predict(input, settlingSteps);
    for (size_t i = 0; i < cpuPred.size() && ok; ++i)
    {
      char label[64];
      snprintf(label, sizeof(label), "Predict() output[%zu]", i);
      ok &= CloseEnough(cpuPred[i], gpuPred[i], label);
    }
  }

  printf(ok ? "PASS: FullPCNetworkT<float,float> on CUDABackendT matches the CPU reference\n"
            : "FAIL: see mismatches above\n");
  return ok ? 0 : 1;
}
