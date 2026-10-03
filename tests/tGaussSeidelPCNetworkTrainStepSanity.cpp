/**
 * @file tGaussSeidelPCNetworkTrainStepSanity.cpp
 * @brief Basic sanity check for GaussSeidelPCNetwork::TrainStep() and
 * TrainStepWithProjection(), neither of which any existing test actually
 * exercises (tGaussSeidelPCLayerVerify.cpp's own SettleNetwork() helper
 * bypasses both, calling ResetState/Clamp/ProjectForward/Step() by hand).
 * Written to verify the CUDA-graph-capture restructuring of both methods
 * (CPU path only, no GPU hardware available here) didn't change their
 * CPU behavior: confirms energy decreases over repeated calls on a fixed
 * input/target pair, and that nothing crashes or produces non-finite
 * values across many calls (the graph-capture code paths are gated on
 * device==DEVICE_GPU and never run here, but the restructuring also
 * changed the CPU path's own energy-readout timing, see
 * ConvPCNetwork::TrainStep()'s comment for why).
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
  GaussSeidelPCNetwork net(1);
  net.AddLayer(4, 8, 1e-3f, 0.1f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(8, 3, 1e-3f, 0.1f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.AddLayer(3, 0, 1e-3f, 0.1f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
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
  // arbitrary absolute threshold (lr here is conservative, so the exact
  // magnitude reached in 20 steps isn't a meaningful bar on its own).
  pass = pass && lastTrainStepEnergy < firstTrainStepEnergy && lastProjEnergy < firstProjEnergy;

  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
