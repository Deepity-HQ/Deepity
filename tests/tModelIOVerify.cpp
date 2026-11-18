/**
 * @file tModelIOVerify.cpp
 * @brief Round-trip regression test for ModelIO::Save()/Load().
 *
 * Saves a randomly initialised DirectKPPCNetwork, loads the file into a
 * differently initialised twin, and checks every W, b and Psi value matches.
 * The layer sizes deliberately differ from one layer to the next: a Psi
 * buffer is [size, terminal], so a Save that treats it as [nextSize, terminal]
 * drops rows and only shows up when size != nextSize.
 */
#include <cstdio>
#include <deepity/networks/DirectKPPCNetwork.h>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/networks/FullPCNetwork.h>
#include <deepity/networks/FullPCNetworkT.h>
#include <deepity/utils/ModelIO.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>

using namespace Deep;

namespace
{
constexpr int kBatch = 2;
const ActivationType kAct = ActivationType::RELU, kDAct = ActivationType::dRELU;
const ActivationType kLin = ActivationType::LINEAR, kDLin = ActivationType::dLINEAR;

std::unique_ptr<DirectKPPCNetwork> MakeNet(unsigned seed)
{
  auto n = std::make_unique<DirectKPPCNetwork>(kBatch);
  // size, next, terminal, lr, ir, fl, lmbda, act, dact
  n->AddLayer(8, 6, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

template <typename T>
bool SameRows(const T* a, const T* b, size_t rows, size_t cols, size_t stride, size_t* firstBad)
{
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < cols; ++c)
      if (a[r * stride + c] != b[r * stride + c])
      {
        *firstBad = r;
        return false;
      }
  return true;
}
std::unique_ptr<FullPCNetwork> MakeFullNet(unsigned seed)
{
  auto n = std::make_unique<FullPCNetwork>(kBatch);
  n->AddLayer(8, 6, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<FullPCNetworkT<double, double>> MakeDoubleNet(unsigned seed)
{
  auto n = std::make_unique<FullPCNetworkT<double, double>>(kBatch);
  n->AddLayer(8, 6, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
  n->AddLayer(6, 4, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
  n->AddLayer(4, 0, 4, 1e-3, 0.1, 1e-4, 1e-4, kLin, kDLin);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

std::unique_ptr<FullConvPCNetwork> MakeConvNet(unsigned seed)
{
  auto n = std::make_unique<FullConvPCNetwork>(kBatch);
  // in, out, H, W, kH, kW, strideH, strideW, padH, padW, terminal, lr, ir, fl, lmbda, act, dact
  n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, kAct, kDAct);
  n->AddLayer(4, 10, 8, 8, 8, 8, 1, 1, 0, 0, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, kAct, kDAct);
  n->AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 1e-3f, 0.1f, 0.0f, 1e-4f, kLin, kDLin);
  n->Compile();
  std::mt19937 rng(seed);
  n->RandomizeWeights(rng);
  return n;
}

/// @brief Compares every tensor of every layer's state dict, honouring row padding.
template <typename Net> bool SameState(const Net& a, const Net& b)
{
  bool ok = true;
  for (size_t i = 0; i < a.GetLayers().size(); ++i)
  {
    const auto da = a.GetLayers()[i]->GetStateDict();
    const auto db = b.GetLayers()[i]->GetStateDict();
    for (const auto& [name, t] : da)
    {
      const auto& u = db.at(name);
      const auto [rows, width] = detail::RowsAndWidth(t.shape);
      const size_t stride = t.rowStride ? t.rowStride : width;
      size_t bad = 0;
      const bool same = t.shape == u.shape && SameRows(t.data, u.data, rows, width, stride, &bad);
      if (!same)
        printf("  layer %zu tensor %s differs\n", i, name.c_str());
      ok &= same;
    }
  }
  return ok;
}

bool Throws(auto&& fn)
{
  try
  {
    fn();
  }
  catch (const std::exception&)
  {
    return true;
  }
  return false;
}
} // namespace

/// @brief Resuming from a checkpoint must continue training bit-for-bit like the uninterrupted run:
/// same moments, same step counters. A weights-only Load() must not.
template <typename Net, typename Make> bool CheckpointResumes(const char* name, Make make, const std::string& path)
{
  const std::vector<float> x(kBatch * 8, 0.5f), y(kBatch * 4, 0.25f);
  auto setup = [&](unsigned seed)
  {
    auto n = make(seed);
    n->SetOptimizer(OptimizerType::ADAM);
    n->SetPsiOptimizer(OptimizerType::ADAM);
    return n;
  };
  auto ref = setup(11), resumed = setup(12), weightsOnly = setup(13);
  for (int i = 0; i < 5; ++i)
    ref->TrainStep(x, y, 2);
  ModelIO::SaveCheckpoint(*ref, path);
  ModelIO::LoadCheckpoint(*resumed, path);
  ModelIO::Load(*weightsOnly, path);
  weightsOnly->SetOptimizer(OptimizerType::ADAM);
  weightsOnly->SetPsiOptimizer(OptimizerType::ADAM);
  for (int i = 0; i < 3; ++i)
  {
    ref->TrainStep(x, y, 2);
    resumed->TrainStep(x, y, 2);
    weightsOnly->TrainStep(x, y, 2);
  }
  const bool same = SameState(*ref, *resumed);
  const bool differs = !SameState(*ref, *weightsOnly);
  const bool notCheckpoint = Throws([&] { ModelIO::Save(*ref, path); ModelIO::LoadCheckpoint(*resumed, path); });
  printf("%s\n", same ? "PASSED" : "FAILED");
  printf("  %s: training continues bit-identically after LoadCheckpoint: %s\n", name, same ? "yes" : "NO");
  printf("  %s: a weights-only Load() does not reproduce it (so the state really matters): %s\n", name,
         differs ? "yes" : "NO");
  printf("  %s: LoadCheckpoint rejects a non-checkpoint file: %s\n", name, notCheckpoint ? "yes" : "NO");
  return same && differs && notCheckpoint;
}

int main()
{
  const auto path = (std::filesystem::temp_directory_path() / "deepity_modelio_verify.safetensors").string();
  auto src = MakeNet(1), dst = MakeNet(2);

  if (!ModelIO::Save(*src, path))
  {
    printf("FAILED: Save() could not write %s\n", path.c_str());
    return 1;
  }
  ModelIO::Load(*dst, path);
  std::filesystem::remove(path);

  bool ok = true;
  const auto& a = src->GetLayers();
  const auto& b = dst->GetLayers();
  for (size_t i = 0; i + 1 < a.size(); ++i) // the terminal layer owns no weights
  {
    const size_t size = a[i]->GetInputSize(), next = a[i]->GetOutputSize();
    const size_t term = a[i]->GetTerminalSize(), stride = a[i]->GetPsiStride();
    size_t bad = 0;

    const bool wOk = SameRows(a[i]->GetWeights(), b[i]->GetWeights(), next, size, size, &bad);
    const bool bOk = SameRows(a[i]->GetBiases(), b[i]->GetBiases(), 1, next, next, &bad);
    const bool pOk =
        SameRows(a[i]->GetDirectFeedbackWeights(), b[i]->GetDirectFeedbackWeights(), size, term, stride, &bad);
    printf("  layer %zu: W %s  b %s  Psi[%zux%zu] %s", i, wOk ? "ok" : "MISMATCH", bOk ? "ok" : "MISMATCH",
           size, term, pOk ? "ok" : "MISMATCH");
    if (!pOk)
      printf(" (first differing row %zu)", bad);
    printf("\n");
    ok &= wOk && bOk && pOk;
  }
  printf("%s\n", ok ? "PASSED: DirectKPPCNetwork save/load round trip" : "FAILED: save/load round trip");

  // FullPCNetwork: same generic check over every tensor in the state dict.
  auto fsrc = MakeFullNet(3), fdst = MakeFullNet(4);
  const bool fullOk = ModelIO::Save(*fsrc, path) && ModelIO::Load(*fdst, path) && SameState(*fsrc, *fdst);
  printf("%s\n", fullOk ? "PASSED: FullPCNetwork save/load round trip" : "FAILED: FullPCNetwork round trip");
  ok &= fullOk;

  auto dsrc = MakeDoubleNet(5), ddst = MakeDoubleNet(6);
  const bool dblOk = ModelIO::Save(*dsrc, path) && ModelIO::Load(*ddst, path) && SameState(*dsrc, *ddst);
  printf("%s\n", dblOk ? "PASSED: FullPCNetworkT<double> save/load round trip" : "FAILED: double round trip");
  ok &= dblOk;
  // A float32 file must not load into a double network.
  ModelIO::Save(*fsrc, path);
  const bool dtypeOk = Throws([&] { ModelIO::Load(*ddst, path); });
  printf("%s\n", dtypeOk ? "PASSED: dtype mismatch is rejected" : "FAILED: dtype mismatch was loaded");
  ok &= dtypeOk;

  auto csrc = MakeConvNet(7), cdst = MakeConvNet(8);
  const bool convOk = ModelIO::Save(*csrc, path) && ModelIO::Load(*cdst, path) && SameState(*csrc, *cdst);
  printf("%s\n", convOk ? "PASSED: FullConvPCNetwork save/load round trip" : "FAILED: conv round trip");
  ok &= convOk;
  ModelIO::Save(*fsrc, path); // fresh float file for the corruption check below

  // A single flipped byte in the data region must be rejected by the content hash.
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(-5, std::ios::end);
    char c = 0;
    f.get(c);
    f.seekp(-5, std::ios::end);
    f.put(static_cast<char>(c ^ 0x01));
  }
  const bool corruptOk = Throws([&] { ModelIO::Load(*fdst, path); });
  printf("%s\n", corruptOk ? "PASSED: corrupted file is rejected" : "FAILED: corrupted file was loaded");
  ok &= corruptOk;
  ok &= CheckpointResumes<DirectKPPCNetwork>("DirectKPPCNetwork", MakeNet, path);
  ok &= CheckpointResumes<FullPCNetwork>("FullPCNetwork", MakeFullNet, path);
  const bool gpuFree = !Throws([&] { ModelIO::SaveCheckpoint(*fsrc, path); });
  ok &= gpuFree;

  const bool missingOk = Throws([&] { ModelIO::Load(*fdst, path + ".does_not_exist"); }) &&
                         Throws([&] { ModelIO::Save(*fsrc, path + "_no_such_dir/x.safetensors"); });
  printf("%s\n", missingOk ? "PASSED: missing file and unwritable path throw" : "FAILED: no exception for bad paths");
  ok &= missingOk;
  std::filesystem::remove(path);
  return ok ? 0 : 1;
}
