/**
 * @file tDiscriminativePCNetworkTrainStepSanity.cpp
 * @brief Basic sanity check for DiscriminativePCNetwork::TrainStep() and
 * TrainStepWithProjection(), verifying the CUDA-graph-capture
 * restructuring of both (CPU path only, no GPU hardware available here)
 * preserved correct CPU behavior: energy decreases over repeated calls
 * on a fixed input/target pair, no crashes or non-finite values.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
  DiscriminativePCNetwork net(1);
  net.AddLayer(4, 8, 1e-3f, 0.1f, 0.01f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(8, 3, 1e-3f, 0.1f, 0.01f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.AddLayer(3, 0, 1e-3f, 0.1f, 0.01f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();

  std::mt19937 rng(11);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.3f, -0.6f, 0.1f, 0.9f};
  std::vector<float> y = {0.4f, -0.2f, 0.7f};

  printf("=== TrainStep() ===\n");
  bool pass = true;
  float firstTrainStepEnergy = -1.0f, lastTrainStepEnergy = -1.0f;
  for (int i = 0; i < 20; ++i)
  {
    float e = net.TrainStep(x, y, 5);
    printf("  step %d: energy=%.6f\n", i, e);
    if (!std::isfinite(e))
    {
      printf("  FAIL: non-finite energy\n");
      pass = false;
      break;
    }
    if (i == 0)
      firstTrainStepEnergy = e;
    lastTrainStepEnergy = e;
  }

  printf("=== TrainStepWithProjection() ===\n");
  float firstProjEnergy = -1.0f, lastProjEnergy = -1.0f;
  for (int i = 0; i < 20; ++i)
  {
    float e = net.TrainStepWithProjection(x, y, 5);
    printf("  step %d: energy=%.6f\n", i, e);
    if (!std::isfinite(e))
    {
      printf("  FAIL: non-finite energy\n");
      pass = false;
      break;
    }
    if (i == 0)
      firstProjEnergy = e;
    lastProjEnergy = e;
  }

  printf("TrainStep(): %.6f -> %.6f, TrainStepWithProjection(): %.6f -> %.6f\n",
         firstTrainStepEnergy, lastTrainStepEnergy, firstProjEnergy, lastProjEnergy);

  // The real check: energy actually decreased over 20 calls, not an
  // arbitrary absolute threshold.
  pass = pass && lastTrainStepEnergy < firstTrainStepEnergy && lastProjEnergy < firstProjEnergy;

  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
