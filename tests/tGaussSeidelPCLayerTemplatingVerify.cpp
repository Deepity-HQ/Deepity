/**
 * @file tGaussSeidelPCLayerTemplatingVerify.cpp
 * @brief TEMP (issue #51's GaussSeidelPCLayer/GaussSeidelPCNetwork
 * pass): numerical comparison of GaussSeidelPCLayer vs.
 * GaussSeidelPCLayerT<float, float>, identically seeded/constructed,
 * run through several real three-sweep-settling TrainStep-shaped
 * iterations.
 *
 * Unlike tSimplePCLayerTemplatingVerify.cpp, GaussSeidelPCLayer never
 * calls any of CPUBackend's oneDNN-fused Try* paths (confirmed by
 * reading GaussSeidelPCLayer.cpp -- every op goes through plain
 * MatMul/AddBiasBroadcast/ActivationDerivativeInto/etc.), so there is
 * no fused-vs-plain accumulation-order divergence to account for here.
 * A tolerance comparison is still used rather than bit-exact (==),
 * defensively, for one real restructuring this port makes in
 * UpdateWeights()'s SGD branch: the original bakes lr_batch directly
 * into a single fused GEMM (C = alpha*A@B + 1.0*C), while the
 * templated version computes the raw product into grad_W first (beta
 * = 0) and applies the lr scale via a separate AxpyIntoFromAccum call
 * -- see GaussSeidelPCLayerT.h's own class comment, this is the same
 * restructuring FullPCLayerT::UpdateWeights() already uses, required
 * because local_grad is AccumT now and MatMulAccumMixed() never blends
 * into an existing T* accumulator.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deepity/layers/GaussSeidelPCLayerT.h>
#include <deepity/layers/GaussSeidelPCLayer.h>
#include <deepity/networks/GaussSeidelPCNetworkT.h>
#include <deepity/networks/GaussSeidelPCNetwork.h>
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
/// parts: GaussSeidelPCNetworkT<QuantT, float> trains for 100 steps
/// with no NaN/Inf/divergence -- no accuracy bar expected at low
/// precision.
template <typename QuantT>
bool VerifyTrainingSmoke(const std::vector<float>& input, const std::vector<float>& target,
                        size_t inSize, size_t outSize, size_t batchSize, float lr, float ir,
                        float lmbda, int settlingSteps, OptimizerType opt, unsigned seed,
                        const char* label)
{
  GaussSeidelPCNetworkT<QuantT, float> net(batchSize);
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

  // --- Part 1: old GaussSeidelPCLayer vs. new
  // GaussSeidelPCLayerT<float, float>, two layers (input + terminal),
  // run through several TrainStep-shaped iterations, manually
  // replaying the required three-sweep order at layer granularity. ---
  bool ok = true;
  {
    GaussSeidelPCLayer oldIn(inSize, outSize, batchSize, lr, ir, lmbda, ActivationType::SIGMOID,
                            ActivationType::dSIGMOID);
    GaussSeidelPCLayer oldOut(outSize, 0, batchSize, lr, ir, lmbda, ActivationType::LINEAR,
                              ActivationType::dLINEAR);
    oldIn.SetLayerAbove(&oldOut);
    oldOut.SetLayerBelow(&oldIn);

    GaussSeidelPCLayerT<float, float> newIn(inSize, outSize, batchSize, lr, ir, lmbda,
                                            ActivationType::SIGMOID, ActivationType::dSIGMOID);
    GaussSeidelPCLayerT<float, float> newOut(outSize, 0, batchSize, lr, ir, lmbda,
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
    ok &= CloseEqual(oldIn.GetFeedbackWeights(), newIn.GetFeedbackWeights(), inSize * outSize,
                     "E (input layer, init)");

    const int settlingSteps = 5;
    const int trainSteps = 10;

    for (int step = 0; step < trainSteps; ++step)
    {
      oldIn.ResetState();
      oldOut.ResetState();
      newIn.ResetState();
      newOut.ResetState();

      oldIn.ClampState(input);
      newIn.ClampState(input);
      oldOut.ClampState(target);
      newOut.ClampState(target);

      for (int s = 0; s < settlingSteps; ++s)
      {
        oldIn.UpdateState();
        oldOut.UpdateState();
        newIn.UpdateState();
        newOut.UpdateState();

        oldIn.ComputePrediction();
        oldOut.ComputePrediction();
        newIn.ComputePrediction();
        newOut.ComputePrediction();

        oldIn.ComputeError(false);
        oldOut.ComputeError(false);
        newIn.ComputeError(false);
        newOut.ComputeError(false);
      }

      oldIn.UpdateWeights();
      newIn.UpdateWeights();

      // Fresh prediction/error readout, same as
      // GaussSeidelPCNetwork::TrainStep()'s own final recompute.
      oldIn.ComputePrediction();
      oldOut.ComputePrediction();
      newIn.ComputePrediction();
      newOut.ComputePrediction();
      oldIn.ComputeError();
      oldOut.ComputeError();
      newIn.ComputeError();
      newOut.ComputeError();

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

  printf(ok ? "PASS: GaussSeidelPCLayerT<float,float> numerically matches GaussSeidelPCLayer\n"
            : "FAIL: see mismatches above\n");

  // --- Part 2: GaussSeidelPCNetwork vs. GaussSeidelPCNetworkT<float,
  // float>, a real 2-layer network built through AddLayer()/Compile(),
  // run through several TrainStep()s. ---
  bool ok2 = true;
  {
    GaussSeidelPCNetwork oldNet(batchSize);
    oldNet.AddLayer((int)inSize, (int)outSize, lr, ir, lmbda, ActivationType::SIGMOID,
                   ActivationType::dSIGMOID);
    oldNet.AddLayer((int)outSize, 0, lr, ir, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    oldNet.Compile();
    std::mt19937 netSeed1(99);
    oldNet.RandomizeWeights(netSeed1);

    GaussSeidelPCNetworkT<float, float> newNet(batchSize);
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

  printf(ok2 ? "PASS: GaussSeidelPCNetworkT<float,float> numerically matches "
              "GaussSeidelPCNetwork\n"
             : "FAIL: see mismatches above\n");

  // --- Part 3: GaussSeidelPCNetworkT<double, double> qualitative
  // convergence smoke test -- double has no bit-exact/tolerance
  // reference to diff against (GaussSeidelPCNetwork is float-only), so
  // this just confirms training runs and the energy trends down. ---
  bool ok3 = true;
  {
    std::vector<double> inputD(input.begin(), input.end());
    std::vector<double> targetD(target.begin(), target.end());

    GaussSeidelPCNetworkT<double, double> netD(batchSize);
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
        printf("  FAIL: GaussSeidelPCNetworkT<double,double> energy is NaN/Inf at step %d\n",
               step);
        ok3 = false;
        break;
      }
      if (step == 0)
        firstEnergy = e;
      lastEnergy = e;
    }

    if (ok3 && !(lastEnergy < firstEnergy))
    {
      printf("  FAIL: GaussSeidelPCNetworkT<double,double> energy did not decrease (first=%.9g "
             "last=%.9g)\n",
             firstEnergy, lastEnergy);
      ok3 = false;
    }
  }

  printf(ok3 ? "PASS: GaussSeidelPCNetworkT<double,double> trains without NaN/Inf and converges\n"
             : "FAIL: see mismatches above\n");

  // --- Parts 4-8: quantized/half-precision training smoke tests, same
  // scope as tSimplePCLayerTemplatingVerify.cpp's own Parts 4-8. ---
  bool ok4 = VerifyTrainingSmoke<Fp4E2M1Value>(
      input, target, inSize, outSize, batchSize, lr, ir, lmbda, 5, OptimizerType::ADAM, 7,
      "GaussSeidelPCNetworkT<Fp4E2M1Value,float>");
  printf(ok4
             ? "PASS: GaussSeidelPCNetworkT<Fp4E2M1Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok5 = VerifyTrainingSmoke<hwy::float16_t>(
      input, target, inSize, outSize, batchSize, lr, ir, lmbda, 5, OptimizerType::ADAMW, 11,
      "GaussSeidelPCNetworkT<hwy::float16_t,float>");
  printf(ok5
             ? "PASS: GaussSeidelPCNetworkT<hwy::float16_t,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok6 = VerifyTrainingSmoke<Posit4_0Value>(
      input, target, inSize, outSize, batchSize, lr, ir, lmbda, 5, OptimizerType::ADAM, 13,
      "GaussSeidelPCNetworkT<Posit4_0Value,float>");
  printf(ok6
             ? "PASS: GaussSeidelPCNetworkT<Posit4_0Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok7 = VerifyTrainingSmoke<Posit4_1Value>(
      input, target, inSize, outSize, batchSize, lr, ir, lmbda, 5, OptimizerType::ADAMW, 17,
      "GaussSeidelPCNetworkT<Posit4_1Value,float>");
  printf(ok7
             ? "PASS: GaussSeidelPCNetworkT<Posit4_1Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok8 = VerifyTrainingSmoke<Posit4_2Value>(
      input, target, inSize, outSize, batchSize, lr, ir, lmbda, 5, OptimizerType::ADAM, 19,
      "GaussSeidelPCNetworkT<Posit4_2Value,float>");
  printf(ok8
             ? "PASS: GaussSeidelPCNetworkT<Posit4_2Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool allOk = ok && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8;
  return allOk ? 0 : 1;
}
