/**
 * @file tSimplePCLayerVerify.cpp
 * @brief Finite-difference gradient check for SimplePCLayer's weight
 * update (dE/dW) under the new mu=phi(W@z+b) convention, deliberately
 * including an extra CalculateState()-driven energy read between the
 * settling loop and UpdateWeights() (exactly the pattern that exposed
 * DiscriminativePCLayer's mu-reuse fragility). Ad hoc verification
 * written for this convention change, not part of the permanent suite.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/SimplePCNetwork.h>
#include <random>

using namespace Deep;

namespace
{
void SettleNetwork(SimplePCNetwork& net, const std::vector<float>& x, const std::vector<float>& y,
                   int steps)
{
  net.ResetState();
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);

  for (int t = 0; t < steps; ++t)
  {
    net.CalculateState(false);
    net.UpdateState();
  }
}

float TotalEnergy(SimplePCNetwork& net)
{
  return net.CalculateState(true);
}
} // namespace

int main()
{
  const int batchSize = 1;
  const float eps = 3e-3f;
  const float tolerance = 1e-2f;

  SimplePCNetwork net(batchSize);
  net.AddLayer(3, 4, 0.0f, 0.08f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 2, 1.0f, 0.08f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(2, 0, 0.0f, 0.08f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();

  std::mt19937 rng(42);
  net.RandomizeWeights(rng);

  net.SetOptimizer(OptimizerType::SGD);

  std::vector<float> x = {0.2f, -0.5f, 0.8f};
  std::vector<float> y = {1.0f, -1.0f};

  SimplePCLayer* testLayer = net.GetLayers()[1].get();
  float* W = const_cast<float*>(testLayer->GetWeights());
  const size_t testIdx = 0;

  SettleNetwork(net, x, y, /*steps=*/40);

  float wOriginal = W[testIdx];

  W[testIdx] = wOriginal + eps;
  float energyPlus = TotalEnergy(net);

  W[testIdx] = wOriginal - eps;
  float energyMinus = TotalEnergy(net);

  W[testIdx] = wOriginal;

  // Deliberately an EXTRA energy read here, after W is restored but
  // before UpdateWeights() -- this recomputes mu as the clean,
  // activated value via ComputeMuOnly(), stress-testing that
  // UpdateWeights() does NOT depend on mu having been left holding
  // f'(preact) by the settling loop's last UpdateState() call.
  printf("Baseline energy: %.6f, energyPlus: %.6f, energyMinus: %.6f\n",
         TotalEnergy(net),
         energyPlus,
         energyMinus);

  float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

  net.UpdateWeights();
  float wAfter = W[testIdx];
  float analyticalGrad = -(wAfter - wOriginal);

  printf("Numerical dE/dW[0]:  %.6f\n", numericalGrad);
  printf("Analytical dE/dW[0]: %.6f\n", analyticalGrad);

  float absDiff = std::fabs(numericalGrad - analyticalGrad);
  float relErr = absDiff / std::max(std::fabs(numericalGrad), 1e-6f);
  printf("Absolute difference: %.6f, relative error: %.6f\n", absDiff, relErr);

  if (relErr > tolerance)
  {
    printf("FAILED: relative error %.6f exceeds tolerance %.6f\n", relErr, tolerance);
    return 1;
  }

  printf("PASSED: SimplePCLayer's UpdateWeights() matches the true dE/dW, "
         "even with an intervening CalculateState() call.\n");
  return 0;
}
