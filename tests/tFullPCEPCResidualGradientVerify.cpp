/**
 * @file tFullPCEPCResidualGradientVerify.cpp
 * @brief Finite-difference gradient check for ComputeAdjoint() when both
 * useEPC and useResidual are enabled on the same layer -- same FD
 * methodology as tFullPCEPCErrorGradientVerify.cpp, with SetResidual(true)
 * added on the test layer(s).
 *
 * ComputeMuOnly() computes mu = a*(W@phi(z))+b [+z if useResidual], so
 * d(mu)/dz picks up an extra +I term under residual, on top of the usual
 * a*W*diag(phi'(z)) term. ComputeAdjoint() originally missed this: it
 * only ever computed the W-path, silently giving the wrong adjoint (and
 * therefore the wrong settled error) for any layer combining ePC with a
 * residual connection, confirmed by this test failing (~99.8% relative
 * error -- not just imprecise, the wrong direction entirely) before the
 * fix that added the missing `AxpyInto(adjoint, adjointAbove, N,
 * adjointAboveScale)` term. Checked at both a chained (non-firstHop) and
 * a firstHop-itself layer, since the fix needed to apply correctly to
 * both of ComputeAdjoint()'s branches.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
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

bool CheckLayer(FullPCNetwork& net, size_t testLayerIdx, const char* label, float eps,
                float tolerance)
{
  auto& layers = net.GetLayers();
  float* e = const_cast<float*>(layers[testLayerIdx]->GetErrors());
  const size_t testIdx = 0;
  float eOriginal = e[testIdx];

  e[testIdx] = eOriginal + eps;
  float energyPlus = TotalEnergy(net);
  e[testIdx] = eOriginal - eps;
  float energyMinus = TotalEnergy(net);
  e[testIdx] = eOriginal;
  float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

  TotalEnergy(net);
  const float* adjointAbove = net.GetTerminalLayer()->GetErrors();
  bool firstHop = true;
  for (size_t i = layers.size() - 2; i >= testLayerIdx; --i)
  {
    layers[i]->ComputeAdjoint(adjointAbove, firstHop ? -1.0f : 1.0f);
    firstHop = false;
    adjointAbove = layers[i]->GetAdjoint();
  }
  float impliedGrad = eOriginal + layers[testLayerIdx]->GetAdjoint()[testIdx];

  float diff = std::fabs(numericalGrad - impliedGrad);
  float relError = diff / (std::fabs(numericalGrad) + 1e-6f);

  printf("[%s] Numerical dE/d(eps):  % .6f\n", label, numericalGrad);
  printf("[%s] Analytical (eps+adjoint): % .6f\n", label, impliedGrad);
  printf("[%s] Relative error: %.6f -- %s\n", label, relError,
         relError <= tolerance ? "MATCH" : "MISMATCH");
  return relError <= tolerance;
}
} // namespace

int main()
{
  const float eps = 3e-3f;
  const float tolerance = 1e-2f;

  // All-4-wide (including terminal) so every hidden layer, including the
  // topmost, legitimately supports residual (requires size == nextSize).
  FullPCNetwork net(1);
  net.AddLayer(4, 4, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 4, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 4, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 4, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 0, 4, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetUseEPC(true);

  auto& layers = net.GetLayers();
  layers[1]->SetResidual(true); // 4->4, chained (non-firstHop) case
  layers[3]->SetResidual(true); // topmost hidden layer itself: firstHop case

  std::mt19937 rng(41);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.3f, -0.6f, 0.1f, 0.9f};
  std::vector<float> y = {0.4f, -0.2f, 0.7f, 0.1f};

  net.ResetState();
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);
  net.ProjectForward();
  net.CalculateTerminalError();
  net.DirectFeedbackUpdate();
  for (int t = 0; t < 10; ++t)
    net.EPCStep();

  bool pass1 = CheckLayer(net, 1, "residual layer, chained", eps, tolerance);
  bool pass2 = CheckLayer(net, 3, "residual layer, firstHop itself", eps, tolerance);

  bool allPass = pass1 && pass2;
  printf("%s\n", allPass ? "PASS" : "FAIL");
  return allPass ? 0 : 1;
}
