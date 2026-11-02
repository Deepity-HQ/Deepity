/**
 * @file tSimplePCLayerTemplatingVerify.cpp
 * @brief TEMP (issue #51's SimplePCLayer/SimplePCNetwork pass):
 * numerical comparison of SimplePCLayer vs. SimplePCLayerT<float,
 * float>, identically seeded/constructed, run through several real
 * TrainStep-shaped iterations.
 *
 * Deliberately a TOLERANCE comparison, not the bit-exact (==) one
 * tFullPCLayerTemplatingVerify.cpp uses for FullPCLayerT: the
 * non-templated SimplePCLayer routes its feedback GEMM and forward
 * GEMM+bias through oneDNN's fused primitives (TryFusedMatMul{,Bias},
 * TryFusedFeedbackUpdate -- this build has DEEPITY_USE_ONEDNN=ON, so
 * these are live, not stubs), while SimplePCLayerT<T, AccumT>
 * deliberately never calls them (see SimplePCLayerT.h's own class
 * comment) and always takes the plain cblas_sgemm-based fallback path
 * instead. oneDNN's GEMM microkernel uses different accumulation
 * blocking/order than cblas_sgemm, so the two are expected to differ in
 * the last few bits of a float even when both are computing the
 * mathematically identical thing -- this test checks for that
 * (numerical agreement), not bit-identity.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deepity/layers/SimplePCLayerT.h>
#include <deepity/layers/SimplePCLayer.h>
#include <deepity/networks/SimplePCNetworkT.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <deepity/utils/QuantFormat.h>
#include <hwy/base.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
bool CloseEqual(const float* a, const float* b, size_t n, const char* label, float tol = 1e-4f)
{
  for (size_t i = 0; i < n; ++i)
  {
    float diff = std::fabs(a[i] - b[i]);
    float scale = std::max(std::fabs(a[i]), std::fabs(b[i]));
    if (diff > tol && diff > tol * scale)
    {
      printf("  MISMATCH in %s at index %zu: old=%.9g new=%.9g (diff=%.3g)\n", label, i,
             (double)a[i], (double)b[i], (double)diff);
      return false;
    }
  }
  return true;
}

/// @brief Training smoke test shared by the quantized/half-precision
/// parts: SimplePCNetworkT<QuantT, float> trains for 100 steps with no
/// NaN/Inf/divergence -- no accuracy bar expected at low precision.
template <typename QuantT>
bool VerifyTrainingSmoke(const std::vector<float>& input, const std::vector<float>& target,
                        size_t inSize, size_t outSize, size_t batchSize, float lr, float ir,
                        float lmbda, int settlingSteps, OptimizerType opt, unsigned seed,
                        const char* label)
{
  SimplePCNetworkT<QuantT, float> net(batchSize);
  net.AddLayer(inSize, outSize, lr, ir, lmbda, ActivationType::SIGMOID, ActivationType::dSIGMOID);
  net.AddLayer(outSize, 0, lr, ir, lmbda, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(opt);
  std::mt19937 netSeed(seed);
  net.RandomizeWeights(netSeed);

  std::vector<QuantT> inputQ(input.begin(), input.end());
  std::vector<QuantT> targetQ(target.begin(), target.end());

  for (int step = 0; step < 100; ++step)
  {
    float e = net.TrainStep(inputQ, targetQ, settlingSteps);
    if (std::isnan(e) || std::isinf(e))
    {
      printf("  FAIL: %s energy is NaN/Inf at step %d\n", label, step);
      return false;
    }
    if (std::fabs(e) > 1e6f)
    {
      printf("  FAIL: %s energy diverged to %.9g at step %d\n", label, (double)e, step);
      return false;
    }
  }
  return true;
}
} // namespace

int main()
{
  const size_t inSize = 8, outSize = 4, batchSize = 4;
  const float lr = 0.01f, ir = 0.1f, lmbda = 0.0f;

  std::vector<float> input(batchSize * inSize), target(batchSize * outSize);
  std::mt19937 dataRng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);
  for (auto& v : target)
    v = dist(dataRng);

  // --- Part 1: old SimplePCLayer vs. new SimplePCLayerT<float, float>,
  // two layers (input + terminal), run through several TrainStep-shaped
  // iterations. ---
  bool ok = true;
  {
    SimplePCLayer oldIn(inSize, outSize, batchSize, lr, ir, lmbda, ActivationType::SIGMOID,
                        ActivationType::dSIGMOID);
    SimplePCLayer oldOut(outSize, 0, batchSize, lr, ir, lmbda, ActivationType::LINEAR,
                         ActivationType::dLINEAR);
    oldIn.SetLayerAbove(&oldOut);
    oldOut.SetLayerBelow(&oldIn);

    SimplePCLayerT<float, float> newIn(inSize, outSize, batchSize, lr, ir, lmbda,
                                       ActivationType::SIGMOID, ActivationType::dSIGMOID);
    SimplePCLayerT<float, float> newOut(outSize, 0, batchSize, lr, ir, lmbda,
                                        ActivationType::LINEAR, ActivationType::dLINEAR);
    newIn.SetLayerAbove(&newOut);
    newOut.SetLayerBelow(&newIn);

    std::mt19937 seedRng1(7);
    oldIn.RandomizeWeights(seedRng1);
    std::mt19937 seedRng2(7);
    newIn.RandomizeWeights(seedRng2);

    oldIn.SetOptimizer(OptimizerType::ADAM);
    newIn.SetOptimizer(OptimizerType::ADAM);

    ok &= CloseEqual(oldIn.GetWeights(), newIn.GetWeights(), inSize * outSize,
                     "W (input layer, init)");

    const int settlingSteps = 5;
    const int trainSteps = 10;

    for (int step = 0; step < trainSteps; ++step)
    {
      oldIn.ClampState(input);
      newIn.ClampState(input);
      oldOut.ClampState(target);
      newOut.ClampState(target);

      for (int s = 0; s < settlingSteps; ++s)
      {
        oldOut.CalculateState(false);
        oldIn.CalculateState(false);
        newOut.CalculateState(false);
        newIn.CalculateState(false);

        oldIn.UpdateState();
        newIn.UpdateState();
      }

      oldIn.UpdateWeights();
      newIn.UpdateWeights();

      oldIn.UnclampState();
      newIn.UnclampState();
      oldOut.UnclampState();
      newOut.UnclampState();

      char label[64];
      snprintf(label, sizeof(label), "z (input, step %d)", step);
      ok &= CloseEqual(oldIn.GetBeliefs(), newIn.GetBeliefs(), batchSize * inSize, label);
      snprintf(label, sizeof(label), "W (input layer, step %d)", step);
      ok &= CloseEqual(oldIn.GetWeights(), newIn.GetWeights(), inSize * outSize, label);
      snprintf(label, sizeof(label), "b (input layer, step %d)", step);
      ok &= CloseEqual(oldIn.GetBiases(), newIn.GetBiases(), outSize, label);
    }
  }

  printf(ok ? "PASS: SimplePCLayerT<float,float> numerically matches SimplePCLayer\n"
            : "FAIL: see mismatches above\n");

  // --- Part 2: SimplePCNetwork vs. SimplePCNetworkT<float, float>, a
  // real 2-layer network built through AddLayer()/Compile(), run
  // through several TrainStep()s. ---
  bool ok2 = true;
  {
    SimplePCNetwork oldNet(batchSize);
    oldNet.AddLayer((int)inSize, (int)outSize, lr, ir, lmbda, ActivationType::SIGMOID,
                   ActivationType::dSIGMOID);
    oldNet.AddLayer((int)outSize, 0, lr, ir, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    oldNet.Compile();
    std::mt19937 netSeed1(99);
    oldNet.RandomizeWeights(netSeed1);

    SimplePCNetworkT<float, float> newNet(batchSize);
    newNet.AddLayer(inSize, outSize, lr, ir, lmbda, ActivationType::SIGMOID,
                   ActivationType::dSIGMOID);
    newNet.AddLayer(outSize, 0, lr, ir, lmbda, ActivationType::LINEAR, ActivationType::dLINEAR);
    newNet.Compile();
    std::mt19937 netSeed2(99);
    newNet.RandomizeWeights(netSeed2);

    const int settlingSteps = 5;
    const int trainSteps = 10;

    for (int step = 0; step < trainSteps; ++step)
    {
      float oldEnergy = oldNet.TrainStep(input, target, settlingSteps);
      float newEnergy = newNet.TrainStep(input, target, settlingSteps);

      char label[64];
      snprintf(label, sizeof(label), "network energy (step %d)", step);
      ok2 &= CloseEqual(&oldEnergy, &newEnergy, 1, label, 1e-2f);

      snprintf(label, sizeof(label), "network input-layer W (step %d)", step);
      ok2 &= CloseEqual(oldNet.GetLayers()[0]->GetWeights(), newNet.GetLayers()[0]->GetWeights(),
                        inSize * outSize, label);
    }

    std::vector<float> oldPred = oldNet.Predict(input, settlingSteps);
    std::vector<float> newPred = newNet.Predict(input, settlingSteps);
    ok2 &= CloseEqual(oldPred.data(), newPred.data(), oldPred.size(), "Predict() output");
  }

  printf(ok2 ? "PASS: SimplePCNetworkT<float,float> numerically matches SimplePCNetwork\n"
             : "FAIL: see mismatches above\n");

  // --- Part 3: SimplePCNetworkT<double, double> qualitative
  // convergence smoke test -- double has no bit-exact/tolerance
  // reference to diff against (SimplePCNetwork is float-only), so this
  // just confirms training runs and the energy trends down. ---
  bool ok3 = true;
  {
    std::vector<double> inputD(input.begin(), input.end());
    std::vector<double> targetD(target.begin(), target.end());

    SimplePCNetworkT<double, double> netD(batchSize);
    netD.AddLayer(inSize, outSize, lr, ir, lmbda, ActivationType::SIGMOID,
                 ActivationType::dSIGMOID);
    netD.AddLayer(outSize, 0, lr, ir, lmbda, ActivationType::LINEAR, ActivationType::dLINEAR);
    netD.Compile();
    std::mt19937 netSeedD(99);
    netD.RandomizeWeights(netSeedD);

    double firstEnergy = 0.0, lastEnergy = 0.0;
    for (int step = 0; step < 200; ++step)
    {
      double e = netD.TrainStep(inputD, targetD, 5);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: SimplePCNetworkT<double,double> energy is NaN/Inf at step %d\n", step);
        ok3 = false;
        break;
      }
      if (step == 0)
        firstEnergy = e;
      lastEnergy = e;
    }

    if (ok3 && !(lastEnergy < firstEnergy))
    {
      printf("  FAIL: SimplePCNetworkT<double,double> energy did not decrease (first=%.9g "
             "last=%.9g)\n",
             firstEnergy, lastEnergy);
      ok3 = false;
    }
  }

  printf(ok3 ? "PASS: SimplePCNetworkT<double,double> trains without NaN/Inf and converges\n"
             : "FAIL: see mismatches above\n");

  // --- Parts 4-8: quantized/half-precision training smoke tests, same
  // scope as tFullPCLayerTemplatingVerify.cpp's own Parts 5b/6b/7b/8b/9b. ---
  bool ok4 = VerifyTrainingSmoke<Fp4E2M1Value>(input, target, inSize, outSize, batchSize, lr, ir,
                                               lmbda, 5, OptimizerType::ADAM, 7,
                                               "SimplePCNetworkT<Fp4E2M1Value,float>");
  printf(ok4 ? "PASS: SimplePCNetworkT<Fp4E2M1Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok5 = VerifyTrainingSmoke<hwy::float16_t>(input, target, inSize, outSize, batchSize, lr, ir,
                                                 lmbda, 5, OptimizerType::ADAMW, 11,
                                                 "SimplePCNetworkT<hwy::float16_t,float>");
  printf(ok5 ? "PASS: SimplePCNetworkT<hwy::float16_t,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok6 = VerifyTrainingSmoke<Posit4_0Value>(input, target, inSize, outSize, batchSize, lr, ir,
                                                lmbda, 5, OptimizerType::ADAM, 13,
                                                "SimplePCNetworkT<Posit4_0Value,float>");
  printf(ok6 ? "PASS: SimplePCNetworkT<Posit4_0Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok7 = VerifyTrainingSmoke<Posit4_1Value>(input, target, inSize, outSize, batchSize, lr, ir,
                                                lmbda, 5, OptimizerType::ADAMW, 17,
                                                "SimplePCNetworkT<Posit4_1Value,float>");
  printf(ok7 ? "PASS: SimplePCNetworkT<Posit4_1Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok8 = VerifyTrainingSmoke<Posit4_2Value>(input, target, inSize, outSize, batchSize, lr, ir,
                                                lmbda, 5, OptimizerType::ADAM, 19,
                                                "SimplePCNetworkT<Posit4_2Value,float>");
  printf(ok8 ? "PASS: SimplePCNetworkT<Posit4_2Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool allOk = ok && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8;
  return allOk ? 0 : 1;
}
