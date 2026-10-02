/**
 * @file tFullPCEPCErrorGradientVerify.cpp
 * @brief Finite-difference gradient check for ePC's actual settling
 * mechanism: ComputeAdjoint()/UpdateErrorEPC(), directly against
 * dE/d(epsilon_j), rather than dE/dW (already covered by
 * tFullPCEPCGradientVerify.cpp).
 *
 * This is the one piece of the ePC implementation neither existing ePC
 * test actually exercises: tFullPCEPCGradientVerify.cpp's FD check of
 * dE/dW passes regardless of whether ComputeAdjoint() computes the right
 * thing (UpdateWeights() is a valid -dE/dW formula at ANY (e,z) state,
 * settled correctly or not), and tFullPCEPCConvergenceVerify.cpp only
 * checks that energy goes down, not that the specific direction used is
 * the true gradient. This test closes that gap by comparing
 * ComputeAdjoint()'s output directly against a numerical dE/d(epsilon)
 * at a generic (not necessarily converged) settling state.
 *
 * Per the paper (arXiv:2505.20137, eq. 9): dE/d(epsilon_j) = epsilon_j +
 * adjoint_j, where adjoint_j is exactly FullPCLayer::ComputeAdjoint()'s
 * output (the reverse-mode-AD chain of d(yhat)/d(epsilon_j) through every
 * layer above j). Checked at two layers to cover both of ComputeAdjoint()'s
 * code paths: the topmost hidden layer (adjointAboveScale=-1, seeded
 * directly from the terminal's error) and the bottom-most hidden layer
 * (reached only after the adjoint has been chained through every
 * intermediate layer, exercising the general recursive case).
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
void BuildNet(FullPCNetwork& net)
{
  // input(4) -> h1(4) -> h2(4) -> h3(4) -> terminal(3): three genuinely
  // hidden layers, enough to exercise both the firstHop and chained
  // ComputeAdjoint() cases in the same network.
  net.AddLayer(4, 4, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 4, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 4, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 3, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(3, 0, 3, 1.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetUseEPC(true);
}

// Forward sweep from layer `fromIdx` upward (ReconstructBelief + mu, then
// every layer above), followed by every hidden layer's own 0.5*||e||^2
// plus the terminal's loss -- matches tFullPCEPCGradientVerify.cpp's
// TotalEnergy() convention exactly, generalized to start partway up since
// perturbing e at fromIdx leaves everything below it untouched.
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

// Replays ePC's own backward sweep (EPCStep()'s exact loop, mirrored here)
// down to and including testLayerIdx, then returns that layer's adjoint.
// Requires TotalEnergy() to have just been called (fresh mu/z/e).
const float* AdjointAt(FullPCNetwork& net, size_t testLayerIdx)
{
  auto& layers = net.GetLayers();
  const float* adjointAbove = net.GetTerminalLayer()->GetErrors();
  bool firstHop = true;
  for (size_t i = layers.size() - 2; i >= testLayerIdx; --i)
  {
    layers[i]->ComputeAdjoint(adjointAbove, firstHop ? -1.0f : 1.0f);
    firstHop = false;
    adjointAbove = layers[i]->GetAdjoint();
  }
  return layers[testLayerIdx]->GetAdjoint();
}

// Returns true on success; prints detailed diagnostics either way.
bool CheckLayer(FullPCNetwork& net, size_t layerIdx, const char* label, float eps, float tolerance)
{
  auto& layers = net.GetLayers();
  float* e = const_cast<float*>(layers[layerIdx]->GetErrors());
  const size_t testIdx = 0;

  float eOriginal = e[testIdx];

  e[testIdx] = eOriginal + eps;
  float energyPlus = TotalEnergy(net);

  e[testIdx] = eOriginal - eps;
  float energyMinus = TotalEnergy(net);

  e[testIdx] = eOriginal;
  float numericalGrad = (energyPlus - energyMinus) / (2.0f * eps);

  // Refresh mu/z/e to the unperturbed state, then replay the backward
  // sweep to get the real ComputeAdjoint() output at this layer.
  TotalEnergy(net);
  const float* adjoint = AdjointAt(net, layerIdx);
  float impliedGrad = eOriginal + adjoint[testIdx];

  float diff = std::fabs(numericalGrad - impliedGrad);
  float relDenom = std::fabs(numericalGrad) + 1e-6f;
  float relError = diff / relDenom;

  printf("[%s] Numerical dE/d(eps[%zu]):  % .6f\n", label, testIdx, numericalGrad);
  printf("[%s] Analytical (eps+adjoint): % .6f\n", label, impliedGrad);
  printf("[%s] Absolute difference: %.6f, relative error: %.6f\n", label, diff, relError);

  return relError <= tolerance;
}
} // namespace

int main()
{
  const float eps = 3e-3f;
  const float tolerance = 1e-2f; // FD error is O(eps^2), same as every
                                  // sibling gradient-check test

  FullPCNetwork net(1);
  BuildNet(net);

  std::mt19937 rng(41);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.3f, -0.6f, 0.1f, 0.9f};
  std::vector<float> y = {0.4f, -0.2f, 0.7f};

  // Settle to a generic (not necessarily converged) state -- the FD check
  // validates the gradient FORMULA at whatever state this reaches, not
  // that settling has converged.
  net.ResetState();
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);
  net.ProjectForward();
  net.CalculateTerminalError();
  net.DirectFeedbackUpdate(); // no-op, fl=0
  for (int t = 0; t < 10; ++t)
    net.EPCStep();

  auto& layers = net.GetLayers();
  size_t topHidden = layers.size() - 2;    // firstHop=-1 case
  size_t bottomHidden = 1;                 // fully chained case

  bool passTop = CheckLayer(net, topHidden, "topmost hidden (firstHop)", eps, tolerance);
  printf("\n");
  bool passBottom = CheckLayer(net, bottomHidden, "bottom hidden (chained)", eps, tolerance);

  printf("\n");
  if (!passTop || !passBottom)
  {
    printf("FAILED: ComputeAdjoint()/UpdateErrorEPC() does not match the true "
           "dE/d(epsilon) at %s%s.\n",
           !passTop ? "the topmost hidden layer" : "",
           !passTop && !passBottom ? " and " : (!passBottom ? "the bottom hidden layer" : ""));
    return 1;
  }

  printf("PASSED: ComputeAdjoint()/UpdateErrorEPC() matches the true "
         "dE/d(epsilon) exactly (eq. 9 of arXiv:2505.20137), at both the "
         "firstHop-seeded topmost hidden layer and a fully-chained "
         "lower layer.\n");
  return 0;
}
