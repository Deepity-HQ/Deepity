// Standalone timing harness matching tGflopsBenchmark.cpp's exact
// workload (no Google Benchmark dependency, to avoid its unrelated
// build issue on a fresh reconfigure). Used to measure a full train
// step's wall-clock time independent of arch flags/BLAS linkage.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deepity/networks/SimplePCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
  const int batchSize = 256;
  const int numSteps = 20;
  const std::vector<int> layerSizes = {784, 512, 256, 64, 10};
  const int warmup = 5;
  const int timed = 50;

  SimplePCNetwork net(batchSize);
  for (size_t i = 0; i + 1 < layerSizes.size(); ++i)
    net.AddLayer(layerSizes[i], layerSizes[i + 1], 0.001f, 0.08f, 0.0f,
                 ActivationType::SIGMOID, ActivationType::dSIGMOID);
  net.AddLayer(layerSizes.back(), 0, 0.001f, 0.08f, 0.0f,
               ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();

  std::mt19937 rng(42);
  net.RandomizeWeights(rng);

  std::vector<float> X(batchSize * layerSizes.front());
  std::vector<float> Y(batchSize * layerSizes.back(), 0.001f);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& v : X)
    v = dist(rng);
  for (int b = 0; b < batchSize; ++b)
    Y[b * layerSizes.back() + (b % layerSizes.back())] = 0.999f;

  for (int i = 0; i < warmup; ++i)
    net.TrainStepWithProjection(X, Y, numSteps);

  std::vector<double> times;
  for (int i = 0; i < timed; ++i)
  {
    auto t0 = std::chrono::steady_clock::now();
    net.TrainStepWithProjection(X, Y, numSteps);
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  std::sort(times.begin(), times.end());
  double median = times[times.size() / 2];
  double best = times.front();

  double flopsPerStep = 0.0;
  for (size_t i = 0; i + 1 < layerSizes.size(); ++i)
    flopsPerStep += (double)batchSize * layerSizes[i] * layerSizes[i + 1] * (4.0 * numSteps + 2.0);

  printf("Timed iterations: %d\n", timed);
  printf("Median time/step: %.3f ms\n", median);
  printf("Best time/step:   %.3f ms\n", best);
  printf("Median GFLOPS/s:  %.3f\n", flopsPerStep / (median * 1e-3) / 1e9);
  return 0;
}
