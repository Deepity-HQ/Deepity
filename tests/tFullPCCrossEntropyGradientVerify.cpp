/**
 * @file tFullPCCrossEntropyGradientVerify.cpp
 * @brief Finite-difference gradient check for FullPCLayer's weight update
 * (dE/dW) THROUGH a softmax cross-entropy terminal, matching
 * tGaussSeidelPCLayerVerify.cpp/tDirectKPVerify.cpp's exact methodology.
 *
 * tCrossEntropyVerify.cpp already checks that the terminal's own e/energy
 * formula matches an independent softmax-CE reference; it stops there.
 * This file checks the NEXT link in the chain: that UpdateWeights() on the
 * layer BELOW the cross-entropy terminal (i.e. the one actually holding
 * W) implements the true gradient of that cross-entropy energy, not just
 * the plain-Gaussian one. Nothing in the codebase exercised this before.
 *
 * SIGN CONVENTION (identical derivation to the two files above): with
 * lr=1, batchSize=1, UpdateWeights() applies W += lr/batchSize * (grad),
 * so the implied gradient is -(W_after - W_before).
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
  const float eps = 3e-3f;
  const float tolerance = 1e-2f; // FD error is O(eps^2), see the two
                                  // sibling files for the same choice

  FullPCNetwork net(1);
  net.AddLayer(5, 4, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 0, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetUseCrossEntropy(true); // the thing under test
  net.SetOptimizer(OptimizerType::SGD);

  std::mt19937 rng(55);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.3f, -0.6f, 0.1f, 0.9f, -0.2f};
  std::vector<float> target = {0.001f, 0.001f, 0.997f, 0.001f}; // one-hot-ish

  FullPCLayer* l0 = net.GetLayers()[0].get();
  FullPCLayer* l1 = net.GetTerminalLayer();

  auto Clamp = [&]()
  {
    net.Clamp(x);
    l1->ClampState(target);
  };

  // Read-only: recomputes l0's forward pass and l1's cross-entropy
  // error/energy against the CURRENT W, without touching z (both layers
  // stay clamped throughout, so there's no settling to redo).
  auto TotalEnergy = [&]()
  {
    l0->InvalidateMuCache(); // otherwise ComputeMuOnly() would just
                              // replay the stale cached mu from before
                              // W was perturbed, see FullPCLayer.h's
                              // own docs on this exact hazard
    l0->CalculateState(false);
    return l1->CalculateState(true);
  };

  float* W = const_cast<float*>(l0->GetWeights());
  const size_t testIdx = 0;

  Clamp();
  float wOriginal = W[testIdx];

  W[testIdx] = wOriginal + eps;
  float energyPlus = TotalEnergy();

  W[testIdx] = wOriginal - eps;
  float energyMinus = TotalEnergy();

  W[testIdx] = wOriginal;

  float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

  // --- Analytical gradient (via the real UpdateWeights() code) ----
  Clamp();
  TotalEnergy(); // refresh mu/e to match the unperturbed W exactly

  float wBefore = W[testIdx];
  l0->UpdateWeights();
  float wAfter = W[testIdx];

  float impliedGrad = -(wAfter - wBefore);

  printf("Numerical dE/dW[%zu]:  % .6f\n", testIdx, numericalGrad);
  printf("Analytical dE/dW[%zu]: % .6f\n", testIdx, impliedGrad);

  float diff = std::fabs(numericalGrad - impliedGrad);
  float relDenom = std::fabs(numericalGrad) + 1e-6f;
  float relError = diff / relDenom;

  printf("Absolute difference: %.6f, relative error: %.6f\n", diff, relError);

  if (relError > tolerance)
  {
    printf("FAILED: gradient mismatch exceeds tolerance (%.4f)\n", tolerance);
    return 1;
  }

  printf("PASSED: UpdateWeights() matches the true dE/dW through a softmax "
         "cross-entropy terminal.\n");
  return 0;
}
