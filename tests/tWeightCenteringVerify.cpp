/**
 * @file tWeightCenteringVerify.cpp
 * @brief Regression test for weight centering (SetCenterWeights()).
 *
 * Part 1 checks IComputeBackend::CenterRows() against a double-precision
 * reference, including a matrix large enough to take the OpenMP path.
 * Part 2 trains every network type a few steps with centering off and on.
 * Off must leave hidden-layer W rows with a visibly non-zero mean (so the
 * check can actually fail); on must leave every row mean near zero. A wrong
 * (rows, cols) choice in any layer's CenterWeights() would fail part 2.
 */
#include <cmath>
#include <cstdio>
#include <deepity/backend/Backend.h>
#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/networks/DirectKPPCNetwork.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/networks/FullPCNetwork.h>
#include <deepity/networks/FullPCNetworkT.h>
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <random>
#include <utility>
#include <vector>

using namespace Deep;

namespace
{
using Dims = std::vector<std::pair<size_t, size_t>>; // (rows, cols) per hidden layer

bool CheckBackend(DeviceType device, const char* name, size_t rows, size_t cols)
{
  auto backend = CreateBackend(device);
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.5f, 1.0f);
  std::vector<float> host(rows * cols);
  for (float& v : host)
    v = dist(rng);

  std::vector<float> expected(host);
  for (size_t r = 0; r < rows; ++r)
  {
    double sum = 0.0;
    for (size_t c = 0; c < cols; ++c)
      sum += host[r * cols + c];
    const float mean = (float)(sum / (double)cols);
    for (size_t c = 0; c < cols; ++c)
      expected[r * cols + c] = host[r * cols + c] - mean;
  }

  Tensor buf(backend.get(), device, host);
  backend->CenterRows(buf.Data(), rows, cols);
  std::vector<float> out;
  buf.CopyToHost(out);

  float worst = 0.0f;
  for (size_t i = 0; i < out.size(); ++i)
    worst = std::fmax(worst, std::fabs(out[i] - expected[i]));
  const bool ok = worst < 1e-4f;
  printf("  [%s] CenterRows %zux%zu: max err %.2e %s\n", name, rows, cols, worst,
         ok ? "PASSED" : "FAILED");
  return ok;
}

/// Largest |mean| over the rows of every listed hidden layer's W.
template <typename Net> double MaxRowMean(const Net& net, const Dims& dims)
{
  const auto& layers = net.GetLayers();
  double worst = 0.0;
  for (size_t i = 0; i < dims.size(); ++i)
  {
    const auto* W = layers[i]->GetWeights();
    for (size_t r = 0; r < dims[i].first; ++r)
    {
      double sum = 0.0;
      for (size_t c = 0; c < dims[i].second; ++c)
        sum += (double)W[r * dims[i].second + c];
      worst = std::fmax(worst, std::fabs(sum / (double)dims[i].second));
    }
  }
  return worst;
}

/// Trains `steps` batches of positive inputs and returns the worst row mean.
template <typename Net, typename Scalar = float>
double TrainAndMeasure(Net& net, bool center, const Dims& dims, size_t xSize, size_t ySize)
{
  std::mt19937 rng(3);
  net.Compile();
  net.RandomizeWeights(rng);
  net.SetCenterWeights(center);

  std::uniform_real_distribution<float> pos(0.1f, 1.0f);
  for (int s = 0; s < 4; ++s)
  {
    std::vector<Scalar> x(xSize), y(ySize, Scalar(0));
    for (auto& v : x)
      v = (Scalar)pos(rng);
    for (size_t b = 0; b < ySize; b += ySize / 2)
      y[b + (size_t)(rng() % (ySize / 2))] = Scalar(1);
    net.TrainStep(x, y, 3);
  }
  return MaxRowMean(net, dims);
}

bool Report(const char* name, double off, double on)
{
  // `off` must be clearly non-zero or the check could not catch a no-op.
  const bool ok = off > 1e-3 && on < 1e-5;
  printf("  %-28s off=%.2e  on=%.2e  %s\n", name, off, on, ok ? "PASSED" : "FAILED");
  return ok;
}

constexpr int kBatch = 2;
const ActivationType kAct = ActivationType::RELU, kDAct = ActivationType::dRELU;
const ActivationType kLin = ActivationType::LINEAR, kDLin = ActivationType::dLINEAR;
const Dims kDenseDims = {{12, 16}, {12, 12}, {4, 12}};

template <typename Net> void AddDenseDirect(Net& n) // size, next, terminal, lr, ir, fl, lmbda
{
  n.AddLayer(16, 12, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n.AddLayer(12, 12, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
  n.AddLayer(12, 4, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
  n.AddLayer(4, 0, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
}

bool TestDense()
{
  bool ok = true;
  const size_t x = kBatch * 16, y = kBatch * 4;
  auto run = [&](const char* name, auto make)
  {
    double off, on;
    {
      auto net = make();
      off = TrainAndMeasure(*net, false, kDenseDims, x, y);
    }
    {
      auto net = make();
      on = TrainAndMeasure(*net, true, kDenseDims, x, y);
    }
    ok &= Report(name, off, on);
  };

  run("DirectKPPCNetwork", []
      { auto n = std::make_unique<DirectKPPCNetwork>(kBatch); AddDenseDirect(*n); return n; });
  run("FullPCNetwork", []
      { auto n = std::make_unique<FullPCNetwork>(kBatch); AddDenseDirect(*n); return n; });
  run("SimplePCNetwork", []
      {
        auto n = std::make_unique<SimplePCNetwork>(kBatch);
        n->AddLayer(16, 12, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
        n->AddLayer(12, 12, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
        n->AddLayer(12, 4, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
        n->AddLayer(4, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
        return n;
      });
  run("GaussSeidelPCNetwork", []
      {
        auto n = std::make_unique<GaussSeidelPCNetwork>(kBatch);
        n->AddLayer(16, 12, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
        n->AddLayer(12, 12, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
        n->AddLayer(12, 4, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
        n->AddLayer(4, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
        return n;
      });
  run("DiscriminativePCNetwork", []
      {
        auto n = std::make_unique<DiscriminativePCNetwork>(kBatch);
        n->AddLayer(16, 12, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
        n->AddLayer(12, 12, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
        n->AddLayer(12, 4, 1e-3f, 0.1f, 0.01f, 1e-4f, kLin, kDLin);
        n->AddLayer(4, 0, 1e-3f, 0.1f, 0.01f, 1e-4f, kLin, kDLin);
        return n;
      });

  // Templated network, double precision.
  {
    auto make = [](bool center)
    {
      FullPCNetworkT<double, double> n(kBatch);
      n.AddLayer(16, 12, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
      n.AddLayer(12, 12, 4, 1e-3, 0.1, 1e-4, 1e-4, kAct, kDAct);
      n.AddLayer(12, 4, 4, 1e-3, 0.1, 1e-4, 1e-4, kLin, kDLin);
      n.AddLayer(4, 0, 4, 1e-3, 0.1, 1e-4, 1e-4, kLin, kDLin);
      return TrainAndMeasure<FullPCNetworkT<double, double>, double>(n, center, kDenseDims, x, y);
    };
    ok &= Report("FullPCNetworkT<double>", make(false), make(true));
  }
  return ok;
}

bool TestConv()
{
  bool ok = true;
  // Hidden conv layers: rows = outChannels, cols = inChannels * kH * kW.
  const Dims dims = {{4, 9}, {4, 36}, {4, 256}};
  const size_t x = kBatch * 64, y = kBatch * 4;
  auto run = [&](const char* name, auto make)
  {
    double off, on;
    {
      auto net = make();
      off = TrainAndMeasure(*net, false, dims, x, y);
    }
    {
      auto net = make();
      on = TrainAndMeasure(*net, true, dims, x, y);
    }
    ok &= Report(name, off, on);
  };

  run("ConvPCNetwork", []
      {
        auto n = std::make_unique<ConvPCNetwork>(kBatch);
        n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
        n->AddLayer(4, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3f, 0.1f, 0.01f, 1e-4f, kAct, kDAct);
        n->AddLayer(4, 4, 8, 8, 8, 8, 1, 1, 0, 0, 1e-3f, 0.1f, 0.01f, 1e-4f, kLin, kDLin);
        n->AddLayer(4, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1e-3f, 0.1f, 0.01f, 1e-4f, kLin, kDLin);
        return n;
      });
  run("SimpleConvPCNetwork", []
      {
        auto n = std::make_unique<SimpleConvPCNetwork>(kBatch);
        n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
        n->AddLayer(4, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-3f, 0.1f, 1e-4f, kAct, kDAct);
        n->AddLayer(4, 4, 8, 8, 8, 8, 1, 1, 0, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
        n->AddLayer(4, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1e-3f, 0.1f, 1e-4f, kLin, kDLin);
        return n;
      });
  run("FullConvPCNetwork", []
      {
        auto n = std::make_unique<FullConvPCNetwork>(kBatch);
        n->AddLayer(1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 0, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
        n->AddLayer(4, 4, 8, 8, 3, 3, 1, 1, 1, 1, 0, 1e-3f, 0.1f, 1e-4f, 1e-4f, kAct, kDAct);
        n->AddLayer(4, 4, 8, 8, 8, 8, 1, 1, 0, 0, 0, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
        n->AddLayer(4, 0, 1, 1, 1, 1, 1, 1, 0, 0, 4, 1e-3f, 0.1f, 1e-4f, 1e-4f, kLin, kDLin);
        return n;
      });
  return ok;
}
} // namespace

int main()
{
  bool allPassed = true;

  printf("--- CenterRows backend op ---\n");
  allPassed &= CheckBackend(DeviceType::DEVICE_CPU, "CPU", 1, 1);
  allPassed &= CheckBackend(DeviceType::DEVICE_CPU, "CPU", 3, 7);
  allPassed &= CheckBackend(DeviceType::DEVICE_CPU, "CPU", 256, 384);
  allPassed &= CheckBackend(DeviceType::DEVICE_CPU, "CPU", 600, 701); // OpenMP path
#ifdef DEEPITY_USE_CUDA
  allPassed &= CheckBackend(DeviceType::DEVICE_GPU, "CUDA", 3, 7);
  allPassed &= CheckBackend(DeviceType::DEVICE_GPU, "CUDA", 256, 384);
  allPassed &= CheckBackend(DeviceType::DEVICE_GPU, "CUDA", 600, 701);
#else
  printf("  CUDABackend skipped (DEEPITY_USE_CUDA not defined)\n");
#endif

  printf("--- Dense networks: worst hidden-layer row mean after 4 steps ---\n");
  allPassed &= TestDense();
  printf("--- Conv networks ---\n");
  allPassed &= TestConv();

  if (!allPassed)
  {
    printf("\nFAILED: weight centering is wrong for at least one backend or network.\n");
    return 1;
  }
  printf("\nPASSED: weight centering correct on every tested backend and network.\n");
  return 0;
}
