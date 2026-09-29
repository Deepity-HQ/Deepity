/**
 * @file tFullConvPCEPCGradientVerify.cpp
 * @brief Finite-difference gradient check for FullConvPCLayer's weight
 * update (dE/dW) after ePC settling -- the conv analog of
 * tFullPCEPCGradientVerify.cpp, same convention (settle, perturb W[0]
 * +-eps, compare to -(W_after-W_before) at lr=1/batchSize=1).
 *
 * Conv layers activate `mu` directly (see FullConvPCLayer.h's own docs),
 * unlike the dense layer's raw-mu/activated-zF split, so the derivative
 * placement in ComputeAdjoint()/UpdateWeights() differs structurally
 * from the dense case even though the settling recursion is the same
 * shape -- this test is what actually confirms that adaptation is
 * correct, not just argued from the ConvPCLayer/SimpleConvPCLayer
 * precedent.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullConvPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
void Settle(FullConvPCNetwork& net, const std::vector<float>& x, const std::vector<float>& y,
            int steps)
{
  net.ResetState();
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);
  net.CalculateTerminalError();
  net.DirectFeedbackUpdate(); // no-op, fl=0
  for (int t = 0; t < steps; ++t)
    net.EPCStep();
}

float TotalEnergy(FullConvPCNetwork& net)
{
  auto& layers = net.GetLayers();

  layers[0]->InvalidateMuCache();
  layers[0]->ComputeMuOnly();
  for (size_t i = 1; i + 1 < layers.size(); ++i)
  {
    layers[i]->ReconstructBelief();
    layers[i]->ComputeMuOnly();
  }
  for (size_t i = 0; i + 1 < layers.size(); ++i)
    if (layers[i]->IsClamped())
      layers[i]->EnsureMuHoldsDerivative();

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
  const float tolerance = 1e-2f;

  FullConvPCNetwork net(1);
  // input(2ch,4x4) -> hidden(4ch,4x4, same-padded) -> hidden(3ch,1x1,
  // kernel covers remaining extent) -> terminal(3ch,1x1).
  net.AddLayer(2, 4, 4, 4, 3, 3, 1, 1, 1, 1, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(4, 3, 4, 4, 4, 4, 1, 1, 0, 0, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(3, 0, 1, 1, 1, 1, 1, 1, 0, 0, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
  net.SetUseEPC(true);
  net.SetOptimizer(OptimizerType::SGD);

  std::mt19937 rng(23);
  net.RandomizeWeights(rng);

  std::mt19937 dataRng(5);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x(2 * 4 * 4), y(3);
  for (auto& v : x)
    v = dist(dataRng);
  for (auto& v : y)
    v = dist(dataRng);

  FullConvPCLayer* l0 = net.GetLayers()[0].get();
  float* W = const_cast<float*>(l0->GetWeights());
  const size_t testIdx = 0;

  Settle(net, x, y, /*steps=*/400);
  float wOriginal = W[testIdx];

  W[testIdx] = wOriginal + eps;
  float energyPlus = TotalEnergy(net);

  W[testIdx] = wOriginal - eps;
  float energyMinus = TotalEnergy(net);

  W[testIdx] = wOriginal;
  float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

  Settle(net, x, y, /*steps=*/400);
  TotalEnergy(net);

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

  printf("PASSED: FullConvPCLayer::UpdateWeights() matches the true dE/dW after ePC settling.\n");
  return 0;
}
