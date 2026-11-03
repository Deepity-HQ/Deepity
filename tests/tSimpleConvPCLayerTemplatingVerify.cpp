/**
 * @file tSimpleConvPCLayerTemplatingVerify.cpp
 * @brief TEMP (issue #51's conv-family pass): bit-exact comparison of
 * SimpleConvPCLayer vs. SimpleConvPCLayerT<float, float>, identically
 * seeded/constructed, run through several real TrainStep-shaped
 * iterations. Confirmed by reading CPUBackend.h/CPUBackendT.h that
 * TryFusedConvForward/BackwardData/BackwardFilter always return false
 * on CPU (no fused conv path exists there at all, templated or not),
 * so both the old and new class take the identical Im2Col()+
 * RepackForBatchedGemm()+MatMul() fallback -- making a genuine
 * bit-exact (==) match achievable, same bar as
 * tDirectKPPCLayerTemplatingVerify.cpp's own Part 1/Part 2.
 */
#include <cmath>
#include <cstdio>
#include <deepity/layers/SimpleConvPCLayerT.h>
#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/networks/SimpleConvPCNetworkT.h>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/utils/QuantFormat.h>
#include <hwy/base.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
bool ExactEqual(const float* a, const float* b, size_t n, const char* label)
{
  for (size_t i = 0; i < n; ++i)
  {
    if (a[i] != b[i])
    {
      printf("  MISMATCH in %s at index %zu: old=%.9g new=%.9g\n", label, i, (double)a[i],
             (double)b[i]);
      return false;
    }
  }
  return true;
}

/// @brief Training smoke test shared by the quantized/half-precision
/// parts: SimpleConvPCNetworkT<QuantT, float> trains for 30 steps with
/// no NaN/Inf/divergence -- no accuracy bar expected at low precision.
template <typename QuantT>
bool VerifyTrainingSmoke(const std::vector<float>& input, const std::vector<float>& target,
                        int inC, int outC, int h, int w, int kernel, int batchSize, float lr,
                        float ir, float lmbda, int settlingSteps, OptimizerType opt, unsigned seed,
                        const char* label)
{
  SimpleConvPCNetworkT<QuantT, float> net(batchSize);
  net.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, lr, ir, lmbda,
              ActivationType::RELU, ActivationType::dRELU);
  net.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, lr, ir, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(opt);
  std::mt19937 netSeed(seed);
  net.RandomizeWeights(netSeed);

  std::vector<QuantT> inputQ(input.begin(), input.end());
  std::vector<QuantT> targetQ(target.begin(), target.end());

  for (int step = 0; step < 30; ++step)
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
  const int inC = 2, outC = 3, h = 6, w = 6, kernel = 3, batchSize = 2;
  const float lr = 0.01f, ir = 0.1f, lmbda = 0.0f;
  const size_t inSize = (size_t)batchSize * inC * h * w;
  const size_t outSize = (size_t)batchSize * outC * h * w;

  std::vector<float> input(inSize), target(outSize);
  std::mt19937 dataRng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);
  for (auto& v : target)
    v = dist(dataRng);

  // --- Part 1: old SimpleConvPCLayer vs. new
  // SimpleConvPCLayerT<float, float>, two layers (input conv/terminal,
  // outChannels=0), run through several TrainStep-shaped iterations. ---
  bool ok = true;
  {
    SimpleConvPCLayer oldIn(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2,
                            batchSize, lr, ir, lmbda, ActivationType::RELU, ActivationType::dRELU);
    SimpleConvPCLayer oldOut(outC, 0, h, w, 1, 1, 1, 1, 0, 0, batchSize, lr, ir, lmbda,
                             ActivationType::LINEAR, ActivationType::dLINEAR);
    oldIn.SetLayerAbove(&oldOut);
    oldOut.SetLayerBelow(&oldIn);

    SimpleConvPCLayerT<float, float> newIn(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2,
                                           kernel / 2, batchSize, lr, ir, lmbda,
                                           ActivationType::RELU, ActivationType::dRELU);
    SimpleConvPCLayerT<float, float> newOut(outC, 0, h, w, 1, 1, 1, 1, 0, 0, batchSize, lr, ir,
                                            lmbda, ActivationType::LINEAR,
                                            ActivationType::dLINEAR);
    newIn.SetLayerAbove(&newOut);
    newOut.SetLayerBelow(&newIn);

    std::mt19937 seedRng1(7);
    oldIn.RandomizeWeights(seedRng1);
    std::mt19937 seedRng2(7);
    newIn.RandomizeWeights(seedRng2);

    oldIn.SetOptimizer(OptimizerType::ADAM);
    newIn.SetOptimizer(OptimizerType::ADAM);

    ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(),
                     (size_t)outC * inC * kernel * kernel, "W (input layer, init)");

    const int settlingSteps = 4;
    const int trainSteps = 6;

    for (int step = 0; step < trainSteps; ++step)
    {
      oldIn.ClampState(input);
      newIn.ClampState(input);
      oldOut.ClampState(target);
      newOut.ClampState(target);

      for (int s = 0; s < settlingSteps; ++s)
      {
        oldOut.CalculateState();
        oldIn.CalculateState();
        newOut.CalculateState();
        newIn.CalculateState();

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
      ok &= ExactEqual(oldIn.GetBeliefs(), newIn.GetBeliefs(), inSize, label);
      snprintf(label, sizeof(label), "W (input layer, step %d)", step);
      ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(), (size_t)outC * inC * kernel * kernel,
                       label);
      snprintf(label, sizeof(label), "b (input layer, step %d)", step);
      ok &= ExactEqual(oldIn.GetBiases(), newIn.GetBiases(), outC, label);
    }
  }

  printf(ok ? "PASS: SimpleConvPCLayerT<float,float> is bit-exact against SimpleConvPCLayer\n"
            : "FAIL: see mismatches above\n");

  // --- Part 2: SimpleConvPCNetwork vs. SimpleConvPCNetworkT<float,
  // float>, a real 2-layer network built through AddLayer()/Compile(),
  // run through several TrainStep()s. ---
  bool ok2 = true;
  {
    SimpleConvPCNetwork oldNet(batchSize);
    oldNet.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, lr, ir, lmbda,
                   ActivationType::RELU, ActivationType::dRELU);
    oldNet.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, lr, ir, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    oldNet.Compile();
    std::mt19937 netSeed1(99);
    oldNet.RandomizeWeights(netSeed1);

    SimpleConvPCNetworkT<float, float> newNet(batchSize);
    newNet.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, lr, ir, lmbda,
                   ActivationType::RELU, ActivationType::dRELU);
    newNet.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, lr, ir, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    newNet.Compile();
    std::mt19937 netSeed2(99);
    newNet.RandomizeWeights(netSeed2);

    const int settlingSteps = 3;
    const int trainSteps = 6;

    for (int step = 0; step < trainSteps; ++step)
    {
      float oldEnergy = oldNet.TrainStep(input, target, settlingSteps);
      float newEnergy = newNet.TrainStep(input, target, settlingSteps);

      char label[64];
      snprintf(label, sizeof(label), "network energy (step %d)", step);
      ok2 &= ExactEqual(&oldEnergy, &newEnergy, 1, label);

      snprintf(label, sizeof(label), "network input-layer W (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetLayers()[0]->GetWeights(), newNet.GetLayers()[0]->GetWeights(),
                        (size_t)outC * inC * kernel * kernel, label);
    }

    std::vector<float> oldPred = oldNet.Predict(input, settlingSteps);
    std::vector<float> newPred = newNet.Predict(input, settlingSteps);
    ok2 &= ExactEqual(oldPred.data(), newPred.data(), oldPred.size(), "Predict() output");
  }

  printf(ok2 ? "PASS: SimpleConvPCNetworkT<float,float> is bit-exact against SimpleConvPCNetwork\n"
             : "FAIL: see mismatches above\n");

  // --- Part 3: SimpleConvPCNetworkT<double, double> qualitative
  // convergence smoke test -- double has no bit-exact reference to
  // diff against (SimpleConvPCNetwork is float-only), so this just
  // confirms training runs and the energy trends down. ---
  bool ok3 = true;
  {
    std::vector<double> inputD(input.begin(), input.end());
    std::vector<double> targetD(target.begin(), target.end());

    SimpleConvPCNetworkT<double, double> netD(batchSize);
    netD.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, lr, ir, lmbda,
                 ActivationType::RELU, ActivationType::dRELU);
    netD.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, lr, ir, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netD.Compile();
    std::mt19937 netSeedD(99);
    netD.RandomizeWeights(netSeedD);

    double firstEnergy = 0.0, lastEnergy = 0.0;
    for (int step = 0; step < 100; ++step)
    {
      double e = netD.TrainStep(inputD, targetD, 3);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: SimpleConvPCNetworkT<double,double> energy is NaN/Inf at step %d\n", step);
        ok3 = false;
        break;
      }
      if (step == 0)
        firstEnergy = e;
      lastEnergy = e;
    }

    if (ok3 && !(lastEnergy < firstEnergy))
    {
      printf("  FAIL: SimpleConvPCNetworkT<double,double> energy did not decrease (first=%.9g "
             "last=%.9g)\n",
             firstEnergy, lastEnergy);
      ok3 = false;
    }
  }

  printf(ok3 ? "PASS: SimpleConvPCNetworkT<double,double> trains without NaN/Inf and converges\n"
             : "FAIL: see mismatches above\n");

  // --- Parts 4-8: quantized/half-precision training smoke tests. ---
  bool ok4 = VerifyTrainingSmoke<Fp4E2M1Value>(input, target, inC, outC, h, w, kernel, batchSize,
                                               lr, ir, lmbda, 3, OptimizerType::ADAM, 7,
                                               "SimpleConvPCNetworkT<Fp4E2M1Value,float>");
  printf(ok4 ? "PASS: SimpleConvPCNetworkT<Fp4E2M1Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok5 = VerifyTrainingSmoke<hwy::float16_t>(input, target, inC, outC, h, w, kernel, batchSize,
                                                 lr, ir, lmbda, 3, OptimizerType::ADAMW, 11,
                                                 "SimpleConvPCNetworkT<hwy::float16_t,float>");
  printf(ok5 ? "PASS: SimpleConvPCNetworkT<hwy::float16_t,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok6 = VerifyTrainingSmoke<Posit4_0Value>(input, target, inC, outC, h, w, kernel, batchSize,
                                                lr, ir, lmbda, 3, OptimizerType::ADAM, 13,
                                                "SimpleConvPCNetworkT<Posit4_0Value,float>");
  printf(ok6 ? "PASS: SimpleConvPCNetworkT<Posit4_0Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok7 = VerifyTrainingSmoke<Posit4_1Value>(input, target, inC, outC, h, w, kernel, batchSize,
                                                lr, ir, lmbda, 3, OptimizerType::ADAMW, 17,
                                                "SimpleConvPCNetworkT<Posit4_1Value,float>");
  printf(ok7 ? "PASS: SimpleConvPCNetworkT<Posit4_1Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok8 = VerifyTrainingSmoke<Posit4_2Value>(input, target, inC, outC, h, w, kernel, batchSize,
                                                lr, ir, lmbda, 3, OptimizerType::ADAM, 19,
                                                "SimpleConvPCNetworkT<Posit4_2Value,float>");
  printf(ok8 ? "PASS: SimpleConvPCNetworkT<Posit4_2Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool allOk = ok && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8;
  return allOk ? 0 : 1;
}
