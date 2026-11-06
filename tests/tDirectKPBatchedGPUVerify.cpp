/**
 * @file tDirectKPBatchedGPUVerify.cpp
 * @brief CUDA-only differential test for the DKP-PC batched-GPU fast
 * path (issue #74): confirms a DirectKPPCNetwork whose topology is
 * eligible for it (see DirectKPPCNetwork.h's "Batched-GPU fast path"
 * member comments) trains and predicts the same as an identical
 * network on CPU, which never takes that path at all.
 *
 * The topology below is deliberately chosen to exercise BOTH of
 * Compile()'s eligibility-detection edge cases at once: a uniform-width
 * run of 3 middle layers (hidden width 8), followed by ONE trailing
 * non-run middle layer whose own width (8) differs from terminalSize
 * (3) -- the case that stops the run one layer short of the terminal,
 * fixed after an earlier version of this feature assumed the run
 * always reaches layers.back() directly (see git history). If that
 * fix regresses, Compile() either detects no run at all (this test
 * would then be comparing two copies of the SAME unbatched code path,
 * silently not testing anything) or, worse, detects a run that
 * mis-sizes the flat buffers -- this test's shape/energy/weight checks
 * are there to catch either.
 *
 * Not a bit-exact comparison -- cuBLAS's reduction order differs from
 * OpenBLAS's -- patterned after tFullPCNetworkTCudaVerify.cpp's own
 * tolerance-based CPU-vs-GPU comparison and its InitIdenticalWeights()
 * technique (CPUBackend::RandomizeNormal and CUDABackend::RandomizeNormal
 * are unrelated PRNG algorithms, so "same seed" alone does not give two
 * networks the same actual weights -- generate once on the host and
 * push the identical bytes into both via backend->CopyFromHost()).
 *
 * Only runs when DEEPITY_CUDA_ENABLED (see CMakeLists.txt) -- requires
 * an actual GPU at runtime, not just the CUDA toolkit at build time, so
 * this is deliberately NOT registered with add_test()/ctest, same as
 * every other CUDA-only verify test in this suite. Run the built binary
 * directly: ./bin/DirectKPBatchedGPUVerify
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deepity/networks/DirectKPPCNetwork.h>
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

// 6 layers: input(6->8), three uniform-width middle layers(8->8), one
// trailing middle layer(8->3) whose own nextSize is forced to equal
// terminalSize (not the run's width 8), terminal(3->0). Compile() should
// detect a 3-layer run (indices 1,2,3), leaving index 4 and the
// terminal as an ordinary two-layer "suffix".
void BuildNetwork(DirectKPPCNetwork& net, float lr, float ir, float fl, float lmbda)
{
  const size_t terminalSize = 3;
  net.AddLayer(6, 8, terminalSize, lr, ir, fl, lmbda, ActivationType::SIGMOID, ActivationType::dSIGMOID);
  net.AddLayer(8, 8, terminalSize, lr, ir, fl, lmbda, ActivationType::SIGMOID, ActivationType::dSIGMOID);
  net.AddLayer(8, 8, terminalSize, lr, ir, fl, lmbda, ActivationType::SIGMOID, ActivationType::dSIGMOID);
  net.AddLayer(8, 8, terminalSize, lr, ir, fl, lmbda, ActivationType::SIGMOID, ActivationType::dSIGMOID);
  net.AddLayer(8, terminalSize, terminalSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
              ActivationType::dSIGMOID);
  net.AddLayer(terminalSize, 0, terminalSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
              ActivationType::dSIGMOID);
  net.Compile();
}

void InitIdenticalWeights(DirectKPPCNetwork& cpuNet, DirectKPPCNetwork& gpuNet, std::mt19937& rng)
{
  std::normal_distribution<float> dist(0.0f, 0.1f);
  auto& cpuLayers = cpuNet.GetLayers();
  auto& gpuLayers = gpuNet.GetLayers();
  for (size_t i = 0; i < cpuLayers.size(); ++i)
  {
    size_t size = cpuLayers[i]->GetInputSize();
    size_t nextSize = cpuLayers[i]->GetOutputSize();
    size_t terminalSize = cpuLayers[i]->GetTerminalSize();
    if (nextSize == 0)
      continue; // terminal layer: no W/b/Psi to seed

    std::vector<float> hostW(size * nextSize), hostB(nextSize), hostPsi(size * terminalSize);
    for (auto& v : hostW)
      v = dist(rng);
    for (auto& v : hostB)
      v = dist(rng);
    for (auto& v : hostPsi)
      v = dist(rng);

    cpuLayers[i]->GetBackend()->CopyFromHost(cpuLayers[i]->GetWeights(), hostW.data(), hostW.size());
    cpuLayers[i]->GetBackend()->CopyFromHost(cpuLayers[i]->GetBiases(), hostB.data(), hostB.size());
    cpuLayers[i]->GetBackend()->CopyFromHost(cpuLayers[i]->GetDirectFeedbackWeights(), hostPsi.data(),
                                             hostPsi.size());

    gpuLayers[i]->GetBackend()->CopyFromHost(gpuLayers[i]->GetWeights(), hostW.data(), hostW.size());
    gpuLayers[i]->GetBackend()->CopyFromHost(gpuLayers[i]->GetBiases(), hostB.data(), hostB.size());
    gpuLayers[i]->GetBackend()->CopyFromHost(gpuLayers[i]->GetDirectFeedbackWeights(), hostPsi.data(),
                                             hostPsi.size());
  }
}
} // namespace

int main()
{
  const size_t batchSize = 4;
  const float lr = 0.02f, ir = 0.1f, fl = 0.01f, lmbda = 0.001f;
  const int settlingSteps = 5;
  const int trainSteps = 10;

  std::vector<float> input(batchSize * 6), target(batchSize * 3);
  std::mt19937 dataRng(42);
  std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dataDist(dataRng);
  for (auto& v : target)
    v = dataDist(dataRng);

  DirectKPPCNetwork cpuNet((int)batchSize, DeviceType::DEVICE_CPU);
  BuildNetwork(cpuNet, lr, ir, fl, lmbda);

  DirectKPPCNetwork gpuNet((int)batchSize, DeviceType::DEVICE_GPU);
  BuildNetwork(gpuNet, lr, ir, fl, lmbda);

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

  // Predict() output agreement.
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

  // Trained weight agreement, directly -- a stronger check than energy/
  // Predict() alone, since a batched-dispatch bug could, in principle,
  // leave the right ENERGY while scrambling individual weights (e.g. a
  // transposed or mis-offset flat write). Checks every run layer (the
  // batched path's own territory) plus the trailing non-run layer and
  // layers.front(), so a boundary-handling bug shows up too.
  if (ok)
  {
    auto& cpuLayers = cpuNet.GetLayers();
    auto& gpuLayers = gpuNet.GetLayers();
    for (size_t i = 0; i < cpuLayers.size() && ok; ++i)
    {
      size_t size = cpuLayers[i]->GetInputSize();
      size_t nextSize = cpuLayers[i]->GetOutputSize();
      if (nextSize == 0)
        continue;

      std::vector<float> cpuW(size * nextSize), gpuW(size * nextSize);
      cpuLayers[i]->GetBackend()->CopyToHost(cpuW.data(), cpuLayers[i]->GetWeights(), cpuW.size());
      gpuLayers[i]->GetBackend()->CopyToHost(gpuW.data(), gpuLayers[i]->GetWeights(), gpuW.size());
      for (size_t j = 0; j < cpuW.size() && ok; ++j)
      {
        char label[80];
        snprintf(label, sizeof(label), "layer %zu W[%zu]", i, j);
        ok &= CloseEnough(cpuW[j], gpuW[j], label);
      }
    }
  }

  printf(ok ? "PASS: DirectKPPCNetwork's batched-GPU fast path matches the CPU reference\n"
            : "FAIL: see mismatches above\n");
  return ok ? 0 : 1;
}
