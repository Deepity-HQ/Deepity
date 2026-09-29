/**
 * @file tFullPCEPCGradientVerify.cpp
 * @brief Finite-difference gradient check for FullPCLayer's weight update
 * (dE/dW) after ePC settling, same convention as every other
 * gradient-check test this session (settle, perturb W[0] +-eps, compare
 * to -(W_after-W_before) at lr=1/batchSize=1).
 *
 * The point of this file: ePC's paper (arXiv:2505.20137, Theorem in
 * Appendix C.2) claims the converged weight gradient is IDENTICAL to
 * plain PC's, meaning FullPCLayer::UpdateWeights() needed ZERO changes
 * to support ePC -- this test is what actually confirms that claim is
 * true in this codebase, not just true in the paper.
 *
 * TotalEnergy() replays ePC's own FORWARD sweep only (ReconstructBelief +
 * ComputeMuOnly bottom-to-top, then the terminal's CalculateState), never
 * the BACKWARD sweep -- under ePC, `e` is the settled variable, and this
 * FD check needs it held fixed while only mu/z react to a perturbed W,
 * exactly like every other gradient-check file in this suite freezes
 * its own settled variable during the perturbation.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
void Settle(FullPCNetwork& net, const std::vector<float>& x, const std::vector<float>& y,
            int steps)
{
  net.ResetState();
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);
  net.ProjectForward();
  net.CalculateTerminalError();
  net.DirectFeedbackUpdate(); // no-op, fl=0
  for (int t = 0; t < steps; ++t)
    net.EPCStep();
}

// Read-only: ePC's forward sweep only, then every layer's own energy
// contribution (0.5*sum(e^2) for hidden layers, the terminal's own loss
// for the last one) -- matches tGaussSeidelPCLayerVerify.cpp's own
// "sum every layer's energy, not just the perturbed one's neighbor"
// convention, for correctness regardless of which W is under test.
float TotalEnergy(FullPCNetwork& net)
{
  auto& layers = net.GetLayers();

  layers[0]->InvalidateMuCache();
  layers[0]->ComputeMuOnly();
  for (size_t i = 1; i + 1 < layers.size(); ++i)
  {
    layers[i]->ReconstructBelief();
    layers[i]->ComputeMuOnly();
  }

  float total = 0.0f;
  for (size_t i = 1; i + 1 < layers.size(); ++i)
  {
    const float* e = layers[i]->GetErrors();
    size_t n = layers[i]->GetBatchSize() * layers[i]->GetInputSize();
    for (size_t k = 0; k < n; ++k)
      total += 0.5f * e[k] * e[k];
  }
  total += net.GetTerminalLayer()->CalculateState(true);
  return total;
}
} // namespace

int main()
{
  const float eps = 3e-3f;
  const float tolerance = 1e-2f; // FD error is O(eps^2), same as every
                                  // sibling gradient-check test

  FullPCNetwork net(1);
  net.AddLayer(4, 4, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 3, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(3, 0, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetUseEPC(true); // the thing under test
  net.SetOptimizer(OptimizerType::SGD);

  std::mt19937 rng(41);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.3f, -0.6f, 0.1f, 0.9f};
  std::vector<float> y = {0.4f, -0.2f, 0.7f};

  FullPCLayer* l0 = net.GetLayers()[0].get();
  float* W = const_cast<float*>(l0->GetWeights());
  const size_t testIdx = 0;

  Settle(net, x, y, /*steps=*/300);
  float wOriginal = W[testIdx];

  W[testIdx] = wOriginal + eps;
  float energyPlus = TotalEnergy(net);

  W[testIdx] = wOriginal - eps;
  float energyMinus = TotalEnergy(net);

  W[testIdx] = wOriginal;
  float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

  // --- Analytical gradient (via the real UpdateWeights() code) ----
  Settle(net, x, y, /*steps=*/300);
  TotalEnergy(net); // refresh mu/e to match the unperturbed W exactly

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

  printf("PASSED: UpdateWeights() still matches the true dE/dW after ePC "
         "settling, unchanged code, as the paper's equilibrium-equivalence "
         "theorem predicts.\n");
  return 0;
}
