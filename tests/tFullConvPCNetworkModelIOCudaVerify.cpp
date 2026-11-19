/**
 * @file tFullConvPCNetworkModelIOCudaVerify.cpp
 * @brief CUDA-only check that a device-resident FullConvPCNetwork checkpoints and resumes: train on the
 * GPU, save a checkpoint, load it into a second GPU network, and confirm both keep training alike while a
 * weights-only load does not (so the Adam moments and step counters really round-trip through the device).
 *
 * Needs a GPU at runtime, so it is not registered with ctest. Run the built binary directly:
 * ./bin/FullConvPCNetworkModelIOCudaVerify
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/utils/ModelIO.h>
#include <filesystem>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
constexpr int kBatch = 8;

std::unique_ptr<FullConvPCNetwork> Make(unsigned seed)
{
  auto n = std::make_unique<FullConvPCNetwork>(kBatch, DeviceType::DEVICE_GPU);
  const auto act = ActivationType::SIGMOID, dact = ActivationType::dSIGMOID;
  const auto lin = ActivationType::LINEAR, dlin = ActivationType::dLINEAR;
  // in, out, H, W, kH, kW, strideH, strideW, padH, padW, terminal, lr, ir, fl, lmbda, act, dact
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, act, dact);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, act, dact);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, lin, dlin);
  n->SetOptimizer(OptimizerType::ADAM);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

/// Largest relative difference over every layer's W and b, read back from the device.
double MaxDifference(FullConvPCNetwork& a, FullConvPCNetwork& b)
{
  double worst = 0;
  for (size_t l = 0; l < a.GetLayers().size(); ++l)
  {
    a.GetLayers()[l]->SyncStateDictFromDevice();
    b.GetLayers()[l]->SyncStateDictFromDevice();
    auto da = a.GetLayers()[l]->GetStateDict(), db = b.GetLayers()[l]->GetStateDict();
    for (const char* name : {"W", "b"})
    {
      const auto &ta = da.at(name), &tb = db.at(name);
      size_t n = 1;
      for (size_t d : ta.shape)
        n *= d;
      for (size_t i = 0; i < n; ++i)
      {
        const double x = ta.data[i], y = tb.data[i];
        worst = std::max(worst, std::fabs(x - y) / std::max(1.0, std::max(std::fabs(x), std::fabs(y))));
      }
    }
  }
  return worst;
}
} // namespace

int main()
{
  std::vector<float> x(kBatch * 64), y(kBatch * 10, 0.1f);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = 0.1f * (static_cast<float>(i % 7) - 3.f) / 3.f;

  auto ref = Make(11), resumed = Make(12), weightsOnly = Make(13);
  for (int i = 0; i < 5; ++i)
    ref->TrainStep(x, y, 2);

  const std::string path = (std::filesystem::temp_directory_path() / "deepity_fullconv_cuda_checkpoint.safetensors").string();
  ModelIO::SaveCheckpoint(*ref, path);
  ModelIO::LoadCheckpoint(*resumed, path);
  ModelIO::Load(*weightsOnly, path);
  std::filesystem::remove(path);

  const double loaded = MaxDifference(*ref, *resumed);
  for (int i = 0; i < 3; ++i)
  {
    ref->TrainStep(x, y, 2);
    resumed->TrainStep(x, y, 2);
    weightsOnly->TrainStep(x, y, 2);
  }
  const double resumedGap = MaxDifference(*ref, *resumed), weightsOnlyGap = MaxDifference(*ref, *weightsOnly);
  printf("  after load: max difference %.3g\n", loaded);
  printf("  after 3 more steps: resumed %.3g, weights-only %.3g\n", resumedGap, weightsOnlyGap);

  const bool ok = loaded < 1e-6 && resumedGap < 1e-3 && weightsOnlyGap > 10 * resumedGap;
  printf(ok ? "PASS: FullConvPCNetwork checkpoint resume on CUDABackend\n" : "FAIL: see the numbers above\n");
  return ok ? 0 : 1;
}
