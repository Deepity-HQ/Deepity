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
 * Both networks' W/Psi are pushed from one set of host-generated values
 * (see InitIdenticalWeights() below), not RandomizeWeights() with "the
 * same seed" -- CPUBackend::RandomizeNormal seeds a std::mt19937,
 * CUDABackend::RandomizeNormal seeds a curandState, and those produce
 * unrelated sequences from an identical seed value, so that would silently
 * start the two networks from different weights.
 *
 * Only runs when DEEPITY_CUDA_ENABLED (see CMakeLists.txt) -- requires
 * an actual GPU at runtime, not just the CUDA toolkit at build time,
 * so this is deliberately NOT registered with add_test()/ctest, same
 * as every other CUDA-only verify test in this suite. Run the built
 * binary directly: ./bin/FullPCNetworkTCudaVerify
 */
#include <algorithm>
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
                  size_t outSize, float lr, float ir, float fl, float lmbda)
{
  net.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
              ActivationType::dSIGMOID);
  net.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.SetUseMuPCScaling(true);
  net.Compile();
}

/**
 * CPUBackend::RandomizeNormal seeds a std::mt19937, CUDABackend::RandomizeNormal
 * seeds a curandState -- same seed VALUE, two unrelated PRNG algorithms, so
 * calling RandomizeWeights(sameSeed) on each network independently does NOT
 * give them the same actual weights. Generate the values once on the host
 * instead and write the identical bytes into each layer's GetStateDict()
 * staging buffers directly (a plain host-side copy -- those pointers are
 * always host-resident now, see FullPCLayerT's own GetStateDict()), then
 * SyncStateDictToDevice() pushes them on to the real (possibly
 * device-resident) W/Psi.
 */
void InitIdenticalWeights(FullPCNetworkT<float, float>& cpuNet,
                          FullPCNetworkT<float, float>& gpuNet, std::mt19937& rng)
{
  std::normal_distribution<float> dist(0.0f, 0.1f);
  auto& cpuLayers = cpuNet.GetLayers();
  auto& gpuLayers = gpuNet.GetLayers();
  for (size_t i = 0; i < cpuLayers.size(); ++i)
  {
    auto cpuDict = cpuLayers[i]->GetStateDict();
    auto gpuDict = gpuLayers[i]->GetStateDict();
    for (const char* name : {"W", "Psi"})
    {
      auto& cpuTensor = cpuDict.at(name);
      size_t n = 1;
      for (size_t d : cpuTensor.shape)
        n *= d;
      if (n == 0)
        continue;

      std::vector<float> hostVals(n);
      for (auto& v : hostVals)
        v = dist(rng);

      // tensor.data is always a host staging pointer now (see FullPCLayerT's
      // own GetStateDict()), even for a GPU layer -- a plain host-side copy,
      // not backend->CopyFromHost(), which would treat it as a device
      // pointer on CUDABackendT and corrupt the stream.
      auto& gpuTensor = gpuDict.at(name);
      std::copy(hostVals.begin(), hostVals.end(), cpuTensor.data);
      std::copy(hostVals.begin(), hostVals.end(), gpuTensor.data);
    }
    // Pushes what was just written above on to the actual (possibly
    // device-resident) weights.
    cpuLayers[i]->SyncStateDictToDevice();
    gpuLayers[i]->SyncStateDictToDevice();
  }
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
  BuildNetwork(cpuNet, inSize, hidSize, outSize, lr, ir, fl, lmbda);

  FullPCNetworkT<float, float> gpuNet(batchSize, DeviceType::DEVICE_GPU);
  BuildNetwork(gpuNet, inSize, hidSize, outSize, lr, ir, fl, lmbda);

  std::mt19937 weightRng(99);
  InitIdenticalWeights(cpuNet, gpuNet, weightRng);

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
