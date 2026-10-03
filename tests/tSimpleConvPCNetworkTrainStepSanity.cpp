/**
 * @file tSimpleConvPCNetworkTrainStepSanity.cpp
 * @brief Basic sanity check for SimpleConvPCNetwork::TrainStep() and
 * TrainStepWithProjection(), verifying the CUDA-graph-capture
 * restructuring of both (CPU path only, no GPU hardware available here)
 * preserved correct CPU behavior: energy decreases over repeated calls
 * on a fixed input/target pair, no crashes or non-finite values.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
  SimpleConvPCNetwork net(1);
  // 1x4x4 -> conv 3x3 stride1 -> 2 channels, 2x2 -> flatten to terminal.
  net.AddLayer(1, 2, 4, 4, 3, 3, 1, 1, 0, 0, 1e-3f, 0.1f, 0.0f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(2, 0, 2, 2, 1, 1, 1, 1, 0, 0, 1e-3f, 0.1f, 0.0f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();

  std::mt19937 rng(11);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.2f, -0.5f, 0.8f, 0.1f, -0.3f, 0.6f, -0.7f, 0.4f,
                          0.9f, -0.2f, 0.3f, -0.6f, 0.5f, -0.9f, 0.1f, -0.1f};
  std::vector<float> y = {0.4f, -0.2f, 0.1f, 0.3f, -0.5f, 0.2f, 0.1f, -0.3f};

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
