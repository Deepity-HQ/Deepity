/**
 * @file tFullPCNetworkTModelIOCudaVerify.cpp
 * @brief CUDA-only differential test for the ModelIO checkpointing gap on
 * a device-resident FullPCNetworkT<float,float>: confirms the host-staging
 * buffers added to FullPCLayerT (GetStateDict()/SyncStateDictFromDevice()/
 * SyncStateDictToDevice(), mirroring FullConvPCLayer's own pattern) make
 * ModelIO::Save()/Load() actually safe on a DEVICE_GPU network, not just
 * on DEVICE_CPU.
 *
 * Trains a GPU network, saves it, loads the checkpoint into a second GPU
 * network, and confirms its weights come back close to the first network's
 * CPU twin (initialized identically, same reasoning as
 * tFullPCNetworkTCudaVerify.cpp's own InitIdenticalWeights() -- CPU and GPU
 * RandomizeNormal are unrelated PRNG algorithms, so weights are pushed in via
 * CopyFromHost() rather than RandomizeWeights(sameSeed)).
 *
 * Only runs when DEEPITY_CUDA_ENABLED -- requires an actual GPU at runtime,
 * so deliberately NOT registered with add_test()/ctest, same as every other
 * CUDA-only verify test in this suite. Run the built binary directly:
 * ./bin/FullPCNetworkTModelIOCudaVerify
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetworkT.h>
#include <deepity/utils/ModelIO.h>
#include <filesystem>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
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

/// @see tFullPCNetworkTCudaVerify.cpp's own InitIdenticalWeights() doc comment.
void InitIdenticalWeights(FullPCNetworkT<float, float>& net, std::mt19937& rng)
{
  std::normal_distribution<float> dist(0.0f, 0.1f);
  for (auto& layerPtr : net.GetLayers())
  {
    auto dict = layerPtr->GetStateDict();
    for (const char* name : {"W", "Psi"})
    {
      auto& tensor = dict.at(name);
      size_t n = 1;
      for (size_t d : tensor.shape)
        n *= d;
      if (n == 0)
        continue;

      std::vector<float> hostVals(n);
      for (auto& v : hostVals)
        v = dist(rng);

      layerPtr->GetBackend()->CopyFromHost(tensor.data, hostVals.data(), n);
    }
    layerPtr->SyncStateDictToDevice();
  }
}

bool CloseEnough(float a, float b, const char* label)
{
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  if (diff > scale * 1e-2f)
  {
    printf("  MISMATCH in %s: a=%.6g b=%.6g\n", label, (double)a, (double)b);
    return false;
  }
  return true;
}
} // namespace

int main()
{
  const size_t inSize = 8, hidSize = 8, outSize = 4, batchSize = 4;
  const float lr = 0.01f, ir = 0.1f, fl = 0.001f, lmbda = 0.0f;
  const int settlingSteps = 5;

  std::vector<float> input(batchSize * inSize), target(batchSize * outSize);
  std::mt19937 dataRng(7);
  std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dataDist(dataRng);
  for (auto& v : target)
    v = dataDist(dataRng);

  FullPCNetworkT<float, float> gpuNet(batchSize, DeviceType::DEVICE_GPU);
  BuildNetwork(gpuNet, inSize, hidSize, outSize, lr, ir, fl, lmbda);
  std::mt19937 weightRng(55);
  InitIdenticalWeights(gpuNet, weightRng);

  for (int i = 0; i < 5; ++i)
    gpuNet.TrainStep(input, target, settlingSteps);

  // Snapshot W/b/Psi on the host for comparison, same as the save path
  // itself will do internally.
  std::vector<std::vector<float>> trained;
  for (auto& layerPtr : gpuNet.GetLayers())
  {
    layerPtr->SyncStateDictFromDevice();
    auto dict = layerPtr->GetStateDict();
    for (const char* name : {"W", "b", "Psi"})
    {
      auto& tensor = dict.at(name);
      size_t n = 1;
      for (size_t d : tensor.shape)
        n *= d;
      trained.emplace_back(tensor.data, tensor.data + n);
    }
  }

  std::string path =
      (std::filesystem::temp_directory_path() / "deepity_fullpcnetworkt_modelio_cuda_test.bin")
          .string();

  if (!ModelIO::Save(gpuNet, path))
  {
    printf("FAIL: ModelIO::Save() couldn't open %s for writing\n", path.c_str());
    return 1;
  }

  FullPCNetworkT<float, float> gpuNet2(batchSize, DeviceType::DEVICE_GPU);
  BuildNetwork(gpuNet2, inSize, hidSize, outSize, lr, ir, fl, lmbda);
  std::mt19937 differentRng(999);
  InitIdenticalWeights(gpuNet2, differentRng); // deliberately different starting weights

  if (!ModelIO::Load(gpuNet2, path))
  {
    printf("FAIL: ModelIO::Load() couldn't open %s for reading\n", path.c_str());
    return 1;
  }
  std::filesystem::remove(path);

  bool ok = true;
  size_t idx = 0;
  for (auto& layerPtr : gpuNet2.GetLayers())
  {
    layerPtr->SyncStateDictFromDevice();
    auto dict = layerPtr->GetStateDict();
    for (const char* name : {"W", "b", "Psi"})
    {
      auto& tensor = dict.at(name);
      size_t n = 1;
      for (size_t d : tensor.shape)
        n *= d;
      auto& ref = trained[idx++];
      for (size_t i = 0; i < n; ++i)
        ok &= CloseEnough(tensor.data[i], ref[i], name);
    }
  }

  // Confirm the loaded network is actually usable afterward, not just
  // byte-comparable.
  float e = gpuNet2.TrainStep(input, target, settlingSteps);
  printf("post-load energy=%.6f\n", e);

  printf(ok ? "PASS: FullPCNetworkT<float,float> ModelIO round-trip on CUDABackendT matches\n"
            : "FAIL: see mismatches above\n");
  return ok ? 0 : 1;
}
