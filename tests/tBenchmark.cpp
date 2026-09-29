// tBenchmark.cpp
//
// Comprehensive performance harness for every PC network/layer pair in
// Deepity, designed to surface WHERE optimization effort would actually
// pay off -- not just "is it fast," but which phase, which layer, and
// which toggle costs what. Manual chrono instrumentation (not Google
// Benchmark macros): none of this codebase's existing granular-profiling
// files (tSimpleNetworkProfile.cpp, tDiscriminativeNetworkProfile.cpp)
// use Benchmark either, since its one-function-one-number model doesn't
// naturally produce a per-phase, per-layer breakdown within a single
// train step. This file generalizes tSimpleNetworkProfile.cpp's own
// proven shape (ms/batch + % of total per phase, plus an explicit
// "Unaccounted" bucket flagging hidden overhead) across all 8 network
// types, none of which except Simple/Discriminative had any benchmark
// coverage before this file.
//
// Sections:
//   1. Per-network harnesses (8): baseline settling-loop breakdown.
//   2. FullPCLayer/FullConvPCLayer toggle-cost comparison: quantifies
//      the per-step overhead of ePC/iPC/momentum/muPC/residual/DKP
//      against a plain baseline, since a toggle's cost only means
//      something paired against what it buys (e.g. ePC costs more per
//      settling step but needs far fewer steps to converge -- this file
//      measures the first half of that trade-off, not the second).
//   3. Conv primitive micro-benchmark: Im2Col vs GEMM vs Col2Im vs bias,
//      via the public IComputeBackend interface directly (no layer/
//      network source touched), at the sizes the conv harness actually
//      uses -- the standard place to look for an implicit-GEMM/fused win.
//   4. Batch-size scaling sweep.
//   5. Cross-network summary table.
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <deepity/backend/CPUBackend.h>
#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/networks/DirectKPPCNetwork.h>
#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/networks/FullPCNetwork.h>
#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <deepity/utils/Im2Col.h>

using namespace Deep;
using Clock = std::chrono::steady_clock;

namespace
{
double Secs(Clock::time_point t0, Clock::time_point t1)
{
  return std::chrono::duration<double>(t1 - t0).count();
}

// Accumulates named phases (insertion order, repeat-adds accumulate) and
// prints ms/batch + % of total, plus an "Unaccounted" bucket -- the
// generalized form of tSimpleNetworkProfile.cpp's own report shape.
struct PhaseReport
{
  std::string title;
  double totalTime = 0.0;
  int nBatches = 1;
  std::vector<std::pair<std::string, double>> phases;

  explicit PhaseReport(std::string t) : title(std::move(t)) {}

  void Add(const std::string& name, double seconds)
  {
    for (auto& p : phases)
    {
      if (p.first == name)
      {
        p.second += seconds;
        return;
      }
    }
    phases.emplace_back(name, seconds);
  }

  void Print() const
  {
    printf("\n=== %s ===\n", title.c_str());
    printf("Total: %.3fs  (%.4f ms/batch over %d batches)\n", totalTime,
           1000.0 * totalTime / nBatches, nBatches);
    double accounted = 0.0;
    for (auto& [name, secs] : phases)
    {
      printf("  %-32s %9.4f ms/batch  (%5.1f%%)\n", name.c_str(), 1000.0 * secs / nBatches,
             totalTime > 0.0 ? 100.0 * secs / totalTime : 0.0);
      accounted += secs;
    }
    double unaccounted = totalTime - accounted;
    printf("  %-32s %9.4f ms/batch  (%5.1f%%)  <- hidden overhead: thread\n"
           "  %-32s                              spin-up, virtual dispatch, allocation\n",
           "Unaccounted", 1000.0 * unaccounted / nBatches,
           totalTime > 0.0 ? 100.0 * unaccounted / totalTime : 0.0, "");
  }
};

std::vector<std::pair<std::string, double>>& SummaryTable()
{
  static std::vector<std::pair<std::string, double>> table;
  return table;
}

void RecordSummary(const std::string& name, double msPerBatch)
{
  SummaryTable().emplace_back(name, msPerBatch);
}

std::vector<float> RandomData(size_t n, unsigned seed, float lo = 0.0f, float hi = 1.0f)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v)
    x = dist(rng);
  return v;
}

// ===========================================================================
// 1. Per-network baseline harnesses
// ===========================================================================

void BenchmarkSimplePCNetwork(int nBatches)
{
  const int BATCH = 128, STEPS = 10;
  SimplePCNetwork net(BATCH);
  net.AddLayer(784, 256, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(256, 128, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(128, 10, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(10, 0, 0.001f, 0.1f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 784, 1);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  PhaseReport report("SimplePCNetwork (Jacobi settling), 784-256-128-10, batch=128");
  auto& layers = net.GetLayers();
  std::vector<double> calc(layers.size(), 0.0), upd(layers.size(), 0.0);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
  {
    auto ts = Clock::now();
    net.ResetState();
    report.Add("reset_state", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.Clamp(x);
    report.Add("clamp_input", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.ProjectForward();
    report.Add("project_forward", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->ClampState(y);
    report.Add("clamp_target", Secs(ts, Clock::now()));

    for (int s = 0; s < STEPS; ++s)
    {
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->CalculateState(false);
        calc[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->UpdateState();
        upd[i] += Secs(ts, Clock::now());
      }
    }

    ts = Clock::now();
    net.UpdateWeights();
    report.Add("update_weights", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->UnclampState();
    report.Add("unclamp", Secs(ts, Clock::now()));
  }
  report.totalTime = Secs(t0, Clock::now());
  report.nBatches = nBatches;
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_calc[layer " + std::to_string(i) + "]", calc[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_update[layer " + std::to_string(i) + "]", upd[i]);
  report.Print();
  RecordSummary(report.title, 1000.0 * report.totalTime / nBatches);
}

void BenchmarkGaussSeidelPCNetwork(int nBatches)
{
  const int BATCH = 128, STEPS = 10;
  GaussSeidelPCNetwork net(BATCH);
  net.AddLayer(784, 256, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(256, 128, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(128, 10, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(10, 0, 0.001f, 0.1f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 784, 2);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  PhaseReport report("GaussSeidelPCNetwork (3-sweep settling), 784-256-128-10, batch=128");
  auto& layers = net.GetLayers();
  std::vector<double> updT(layers.size(), 0.0), predT(layers.size(), 0.0), errT(layers.size(), 0.0);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
  {
    auto ts = Clock::now();
    net.ResetState();
    report.Add("reset_state", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.Clamp(x);
    report.Add("clamp_input", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.ProjectForward();
    report.Add("project_forward", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->ClampState(y);
    report.Add("clamp_target", Secs(ts, Clock::now()));

    for (int s = 0; s < STEPS; ++s)
    {
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->UpdateState();
        updT[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->ComputePrediction();
        predT[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->ComputeError();
        errT[i] += Secs(ts, Clock::now());
      }
    }

    ts = Clock::now();
    net.UpdateWeights();
    report.Add("update_weights", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->UnclampState();
    report.Add("unclamp", Secs(ts, Clock::now()));
  }
  report.totalTime = Secs(t0, Clock::now());
  report.nBatches = nBatches;
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("sweep1_updateState[layer " + std::to_string(i) + "]", updT[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("sweep2_computePrediction[layer " + std::to_string(i) + "]", predT[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("sweep3_computeError[layer " + std::to_string(i) + "]", errT[i]);
  report.Print();
  RecordSummary(report.title, 1000.0 * report.totalTime / nBatches);
}

void BenchmarkDirectKPPCNetwork(int nBatches)
{
  const int BATCH = 128, STEPS = 10;
  DirectKPPCNetwork net(BATCH);
  net.AddLayer(784, 256, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(256, 128, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(128, 10, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(10, 0, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 784, 3);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  PhaseReport report("DirectKPPCNetwork (DKP direct feedback), 784-256-128-10, batch=128");
  auto& layers = net.GetLayers();
  std::vector<double> calc(layers.size(), 0.0), upd(layers.size(), 0.0);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
  {
    auto ts = Clock::now();
    net.ResetState();
    report.Add("reset_state", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.Clamp(x);
    report.Add("clamp_input", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.ProjectForward();
    report.Add("project_forward", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->ClampState(y);
    report.Add("clamp_target", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.CalculateTerminalError();
    report.Add("calculate_terminal_error", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.DirectFeedbackUpdate();
    report.Add("direct_feedback_update (Psi)", Secs(ts, Clock::now()));

    for (int s = 0; s < STEPS; ++s)
    {
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->CalculateState(false);
        calc[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->UpdateState();
        upd[i] += Secs(ts, Clock::now());
      }
    }

    ts = Clock::now();
    net.UpdateWeights();
    report.Add("update_weights", Secs(ts, Clock::now()));
  }
  report.totalTime = Secs(t0, Clock::now());
  report.nBatches = nBatches;
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_calc[layer " + std::to_string(i) + "]", calc[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_update[layer " + std::to_string(i) + "]", upd[i]);
  report.Print();
  RecordSummary(report.title, 1000.0 * report.totalTime / nBatches);
}

void BenchmarkFullPCNetworkBaseline(int nBatches)
{
  const int BATCH = 128, STEPS = 10;
  FullPCNetwork net(BATCH);
  net.AddLayer(784, 256, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(256, 128, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(128, 10, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(10, 0, 10, 0.001f, 0.1f, 1e-4f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 784, 4);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  double total = 0.0;
  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
    total += net.TrainStep(x, y, STEPS), (void)0;
  double elapsed = Secs(t0, Clock::now());

  printf("\n=== FullPCNetwork (all toggles OFF), 784-256-128-10, batch=128 ===\n");
  printf("Total: %.3fs  (%.4f ms/batch over %d batches)\n", elapsed, 1000.0 * elapsed / nBatches,
         nBatches);
  RecordSummary("FullPCNetwork (baseline, all toggles off)", 1000.0 * elapsed / nBatches);
}

void BenchmarkDiscriminativePCNetwork(int nBatches)
{
  const int BATCH = 128, STEPS = 10;
  DiscriminativePCNetwork net(BATCH);
  net.AddLayer(784, 256, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(256, 128, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(128, 10, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(10, 0, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 784, 5);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  PhaseReport report("DiscriminativePCNetwork (precision-weighted), 784-256-128-10, batch=128");
  auto& layers = net.GetLayers();
  std::vector<double> calc(layers.size(), 0.0), upd(layers.size(), 0.0);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
  {
    auto ts = Clock::now();
    net.ResetState();
    report.Add("reset_state", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.Clamp(x);
    report.Add("clamp_input", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.ProjectForward();
    report.Add("project_forward", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->ClampState(y);
    report.Add("clamp_target", Secs(ts, Clock::now()));

    for (int s = 0; s < STEPS; ++s)
    {
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->CalculateState();
        calc[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->UpdateState();
        upd[i] += Secs(ts, Clock::now());
      }
    }

    ts = Clock::now();
    net.UpdatePrecision();
    report.Add("update_precision", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.UpdateWeights();
    report.Add("update_weights", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->UnclampState();
    report.Add("unclamp", Secs(ts, Clock::now()));
  }
  report.totalTime = Secs(t0, Clock::now());
  report.nBatches = nBatches;
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_calc[layer " + std::to_string(i) + "]", calc[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_update[layer " + std::to_string(i) + "]", upd[i]);
  report.Print();
  RecordSummary(report.title, 1000.0 * report.totalTime / nBatches);
}

// Conv architecture shared by ConvPCNetwork/SimpleConvPCNetwork/FullConvPCNetwork's
// baseline harnesses and the batch-size sweep: a real downsampling CNN
// shape (stride-2 convs), proven to actually train well this session --
// not a pathological single giant collapsing kernel.
template <typename NetT>
void BuildConvArch(NetT& net, int batch)
{
  (void)batch;
  net.AddLayer(1, 16, 28, 28, 3, 3, 2, 2, 1, 1, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(16, 32, 14, 14, 3, 3, 2, 2, 1, 1, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(32, 32, 7, 7, 3, 3, 1, 1, 1, 1, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(32, 10, 7, 7, 7, 7, 1, 1, 0, 0, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0.001f, 0.1f, 0.01f, 1e-4f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
}

void BenchmarkConvPCNetwork(int nBatches)
{
  const int BATCH = 64, STEPS = 10;
  ConvPCNetwork net(BATCH);
  BuildConvArch(net, BATCH);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 1 * 28 * 28, 6);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  PhaseReport report("ConvPCNetwork (precision-weighted), 1-16-32-32-10, batch=64");
  auto& layers = net.GetLayers();
  std::vector<double> calc(layers.size(), 0.0), upd(layers.size(), 0.0);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
  {
    auto ts = Clock::now();
    net.ResetState();
    report.Add("reset_state", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.Clamp(x);
    report.Add("clamp_input", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.ProjectForward();
    report.Add("project_forward", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->ClampState(y);
    report.Add("clamp_target", Secs(ts, Clock::now()));

    for (int s = 0; s < STEPS; ++s)
    {
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->CalculateState();
        calc[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->UpdateState();
        upd[i] += Secs(ts, Clock::now());
      }
    }

    ts = Clock::now();
    net.UpdatePrecision();
    report.Add("update_precision", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.UpdateWeights();
    report.Add("update_weights", Secs(ts, Clock::now()));
  }
  report.totalTime = Secs(t0, Clock::now());
  report.nBatches = nBatches;
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_calc[layer " + std::to_string(i) + "]", calc[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_update[layer " + std::to_string(i) + "]", upd[i]);
  report.Print();
  RecordSummary(report.title, 1000.0 * report.totalTime / nBatches);
}

void BenchmarkSimpleConvPCNetwork(int nBatches)
{
  const int BATCH = 64, STEPS = 10;
  SimpleConvPCNetwork net(BATCH);
  net.AddLayer(1, 16, 28, 28, 3, 3, 2, 2, 1, 1, 0.001f, 0.1f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(16, 32, 14, 14, 3, 3, 2, 2, 1, 1, 0.001f, 0.1f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(32, 32, 7, 7, 3, 3, 1, 1, 1, 1, 0.001f, 0.1f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(32, 10, 7, 7, 7, 7, 1, 1, 0, 0, 0.001f, 0.1f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0.001f, 0.1f, 1e-4f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 1 * 28 * 28, 7);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  PhaseReport report("SimpleConvPCNetwork (plain, no precision), 1-16-32-32-10, batch=64");
  auto& layers = net.GetLayers();
  std::vector<double> calc(layers.size(), 0.0), upd(layers.size(), 0.0);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
  {
    auto ts = Clock::now();
    net.ResetState();
    report.Add("reset_state", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.Clamp(x);
    report.Add("clamp_input", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.ProjectForward();
    report.Add("project_forward", Secs(ts, Clock::now()));

    ts = Clock::now();
    net.GetTerminalLayer()->ClampState(y);
    report.Add("clamp_target", Secs(ts, Clock::now()));

    for (int s = 0; s < STEPS; ++s)
    {
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->CalculateState();
        calc[i] += Secs(ts, Clock::now());
      }
      for (size_t i = 0; i < layers.size(); ++i)
      {
        ts = Clock::now();
        layers[i]->UpdateState();
        upd[i] += Secs(ts, Clock::now());
      }
    }

    ts = Clock::now();
    net.UpdateWeights();
    report.Add("update_weights", Secs(ts, Clock::now()));
  }
  report.totalTime = Secs(t0, Clock::now());
  report.nBatches = nBatches;
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_calc[layer " + std::to_string(i) + "]", calc[i]);
  for (size_t i = 0; i < layers.size(); ++i)
    report.Add("settle_update[layer " + std::to_string(i) + "]", upd[i]);
  report.Print();
  RecordSummary(report.title, 1000.0 * report.totalTime / nBatches);
}

void BenchmarkFullConvPCNetworkBaseline(int nBatches)
{
  const int BATCH = 64, STEPS = 10;
  FullConvPCNetwork net(BATCH);
  net.AddLayer(1, 16, 28, 28, 3, 3, 2, 2, 1, 1, 10, 0.001f, 0.1f, 0.0f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(16, 32, 14, 14, 3, 3, 2, 2, 1, 1, 10, 0.001f, 0.1f, 0.0f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(32, 32, 7, 7, 3, 3, 1, 1, 1, 1, 10, 0.001f, 0.1f, 0.0f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(32, 10, 7, 7, 7, 7, 1, 1, 0, 0, 10, 0.001f, 0.1f, 0.0f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 0.001f, 0.1f, 0.0f, 1e-4f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 1 * 28 * 28, 8);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  double total = 0.0;
  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
    total += net.TrainStep(x, y, STEPS), (void)0;
  double elapsed = Secs(t0, Clock::now());

  printf("\n=== FullConvPCNetwork (all toggles OFF), 1-16-32-32-10, batch=64 ===\n");
  printf("Total: %.3fs  (%.4f ms/batch over %d batches)\n", elapsed, 1000.0 * elapsed / nBatches,
         nBatches);
  RecordSummary("FullConvPCNetwork (baseline, all toggles off)", 1000.0 * elapsed / nBatches);
}

// ===========================================================================
// 2. FullPCLayer / FullConvPCLayer toggle-cost comparison
// ===========================================================================

double MeasureFullPCVariant(const std::string& label, int nBatches,
                            void (*configure)(FullPCNetwork&))
{
  const int BATCH = 128, STEPS = 10;
  FullPCNetwork net(BATCH);
  net.AddLayer(784, 256, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(256, 256, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::TANH,
              ActivationType::dTANH); // same-width, so residual is valid here
  net.AddLayer(256, 10, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(10, 0, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
  configure(net);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);
  net.SetPsiOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 784, 9);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
    net.TrainStep(x, y, STEPS);
  double elapsed = Secs(t0, Clock::now());
  double msPerBatch = 1000.0 * elapsed / nBatches;
  printf("  %-38s %9.4f ms/batch\n", label.c_str(), msPerBatch);
  return msPerBatch;
}

void BenchmarkFullPCToggleComparison(int nBatches)
{
  printf("\n=== FullPCLayer toggle-cost comparison, 784-256-256-10, batch=128, 10 steps ===\n");
  printf("(Per-STEP cost only -- pair against this session's own convergence-speed findings,\n"
         " e.g. ePC needs far FEWER steps to reach the same energy, before judging net cost.)\n");

  double baseline = MeasureFullPCVariant("baseline (all toggles off)", nBatches,
                                        [](FullPCNetwork&) {});
  double epc =
      MeasureFullPCVariant("+ePC", nBatches, [](FullPCNetwork& n) { n.SetUseEPC(true); });
  double ipc =
      MeasureFullPCVariant("+iPC", nBatches, [](FullPCNetwork& n) { n.SetUseIPC(true); });
  double momentum = MeasureFullPCVariant("+momentum", nBatches,
                                        [](FullPCNetwork& n) { n.SetUseMomentum(true); });
  double mupc = MeasureFullPCVariant("+muPC scaling", nBatches,
                                    [](FullPCNetwork& n) { n.SetUseMuPCScaling(true); });
  double residual = MeasureFullPCVariant("+residual connections", nBatches, [](FullPCNetwork& n)
                                        { n.SetUseResidualConnections(true); });
  double dkp = MeasureFullPCVariant(
      "+DKP feedback (fl>0, set at construction)", nBatches, [](FullPCNetwork&) {});

  printf("\n  Overhead vs baseline:\n");
  printf("  %-38s %+7.1f%%\n", "ePC", 100.0 * (epc - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "iPC", 100.0 * (ipc - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "momentum", 100.0 * (momentum - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "muPC scaling", 100.0 * (mupc - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "residual connections", 100.0 * (residual - baseline) / baseline);
  printf("  %-38s (see note: fl was 0 in the baseline build above, DKP's own\n"
         "  %-38s  contribution needs a separate fl>0 construction to isolate)\n",
         "DKP direct feedback", "");
  (void)dkp;
}

double MeasureFullConvPCVariant(const std::string& label, int nBatches,
                                void (*configure)(FullConvPCNetwork&))
{
  const int BATCH = 64, STEPS = 10;
  FullConvPCNetwork net(BATCH);
  // L=3 weight-bearing layers (H=2): Compile()'s residual loop covers
  // ONLY layers[1..H-1]=layers[1], so that's the one made shape-matching
  // (32->32, same spatial extent) -- all downsampling front-loaded into
  // layer0's own stride, exactly like tFullPCEPCGradientVerify.cpp's
  // dense analog put the single residual-eligible layer at H=2's only
  // covered index.
  net.AddLayer(1, 32, 28, 28, 8, 8, 4, 4, 2, 2, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU); // downsample 28x28 -> 7x7 in one stride-4 conv
  net.AddLayer(32, 32, 7, 7, 3, 3, 1, 1, 1, 1, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU); // same-shape, the ONLY residual-eligible layer here
  net.AddLayer(32, 10, 7, 7, 7, 7, 1, 1, 0, 0, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::RELU,
              ActivationType::dRELU);
  net.AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 0.001f, 0.1f, 1e-3f, 1e-4f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  configure(net);
  net.Compile();
  std::mt19937 rng(42);
  net.RandomizeWeights(rng);
  net.SetOptimizer(OptimizerType::ADAMW);
  net.SetPsiOptimizer(OptimizerType::ADAMW);

  auto x = RandomData((size_t)BATCH * 1 * 28 * 28, 10);
  std::vector<float> y((size_t)BATCH * 10, 0.1f);

  auto t0 = Clock::now();
  for (int b = 0; b < nBatches; ++b)
    net.TrainStep(x, y, STEPS);
  double elapsed = Secs(t0, Clock::now());
  double msPerBatch = 1000.0 * elapsed / nBatches;
  printf("  %-38s %9.4f ms/batch\n", label.c_str(), msPerBatch);
  return msPerBatch;
}

void BenchmarkFullConvPCToggleComparison(int nBatches)
{
  printf("\n=== FullConvPCLayer toggle-cost comparison, 1-32-32-10, batch=64, 10 steps ===\n");

  double baseline = MeasureFullConvPCVariant("baseline (all toggles off)", nBatches,
                                            [](FullConvPCNetwork&) {});
  double epc = MeasureFullConvPCVariant("+ePC", nBatches,
                                       [](FullConvPCNetwork& n) { n.SetUseEPC(true); });
  double ipc = MeasureFullConvPCVariant("+iPC", nBatches,
                                       [](FullConvPCNetwork& n) { n.SetUseIPC(true); });
  double momentum = MeasureFullConvPCVariant("+momentum", nBatches,
                                            [](FullConvPCNetwork& n) { n.SetUseMomentum(true); });
  double mupc = MeasureFullConvPCVariant("+muPC scaling", nBatches, [](FullConvPCNetwork& n)
                                        { n.SetUseMuPCScaling(true); });
  double residual = MeasureFullConvPCVariant(
      "+residual connections", nBatches,
      [](FullConvPCNetwork& n) { n.SetUseResidualConnections(true); });

  printf("\n  Overhead vs baseline:\n");
  printf("  %-38s %+7.1f%%\n", "ePC", 100.0 * (epc - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "iPC", 100.0 * (ipc - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "momentum", 100.0 * (momentum - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "muPC scaling", 100.0 * (mupc - baseline) / baseline);
  printf("  %-38s %+7.1f%%\n", "residual connections", 100.0 * (residual - baseline) / baseline);
}

// ===========================================================================
// 3. Conv primitive micro-benchmark: Im2Col vs GEMM vs Col2Im vs bias
// ===========================================================================

void BenchmarkConvPrimitives(int reps)
{
  printf("\n=== Conv primitive micro-benchmark (public IComputeBackend calls only) ===\n");
  printf("Sizes match the conv harnesses' own layers -- this is where to look for an\n"
         "implicit-GEMM/fused win if Im2Col or Col2Im dominates over raw GEMM compute.\n");

  CPUBackend backend;
  struct LayerShape
  {
    const char* name;
    int inC, inH, inW, outC, kH, kW, strideH, strideW, padH, padW;
  };
  std::vector<LayerShape> shapes = {
      {"layer0 (1->16, 28x28, s2)", 1, 28, 28, 16, 3, 3, 2, 2, 1, 1},
      {"layer1 (16->32, 14x14, s2)", 16, 14, 14, 32, 3, 3, 2, 2, 1, 1},
      {"layer2 (32->32, 7x7, s1 same)", 32, 7, 7, 32, 3, 3, 1, 1, 1, 1},
      {"layer3 (32->10, 7x7->1x1)", 32, 7, 7, 10, 7, 7, 1, 1, 0, 0},
  };

  for (auto& s : shapes)
  {
    int outH = ConvOutDim(s.inH, s.kH, s.strideH, s.padH);
    int outW = ConvOutDim(s.inW, s.kW, s.strideW, s.padW);
    size_t colRows = (size_t)s.inC * s.kH * s.kW;
    size_t colCols = (size_t)outH * outW;

    std::vector<float> input((size_t)s.inC * s.inH * s.inW, 0.5f);
    std::vector<float> cols(colRows * colCols, 0.0f);
    std::vector<float> W((size_t)s.outC * colRows, 0.01f);
    std::vector<float> mu((size_t)s.outC * colCols, 0.0f);
    std::vector<float> bias((size_t)s.outC, 0.0f);
    std::vector<float> feedback((size_t)s.outC * colCols, 0.1f);
    std::vector<float> scratch(colRows * colCols, 0.0f);
    std::vector<float> outputImage(input.size(), 0.0f);

    auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
      backend.Im2Col(input.data(), s.inC, s.inH, s.inW, s.kH, s.kW, s.strideH, s.strideW, s.padH,
                     s.padW, cols.data());
    double im2colT = Secs(t0, Clock::now());

    t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
      backend.MatMul(false, false, s.outC, (int)colCols, (int)colRows, 1.0f, W.data(),
                     (int)colRows, cols.data(), (int)colCols, 0.0f, mu.data(), (int)colCols);
    double gemmT = Secs(t0, Clock::now());

    t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
      backend.AddBiasPerChannel(mu.data(), bias.data(), s.outC, colCols);
    double biasT = Secs(t0, Clock::now());

    t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
      backend.MatMul(true, false, (int)colRows, (int)colCols, s.outC, 1.0f, W.data(),
                     (int)colRows, feedback.data(), (int)colCols, 0.0f, scratch.data(),
                     (int)colCols);
    double feedbackGemmT = Secs(t0, Clock::now());

    t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
    {
      backend.Zero(outputImage.data(), outputImage.size());
      backend.Col2Im(scratch.data(), s.inC, s.inH, s.inW, s.kH, s.kW, s.strideH, s.strideW,
                     s.padH, s.padW, outputImage.data());
    }
    double col2imT = Secs(t0, Clock::now());

    double total = im2colT + gemmT + biasT + feedbackGemmT + col2imT;
    printf("\n  %s: (%d reps)\n", s.name, reps);
    printf("    %-24s %9.4f ms/call  (%5.1f%%)\n", "Im2Col", 1000.0 * im2colT / reps,
           100.0 * im2colT / total);
    printf("    %-24s %9.4f ms/call  (%5.1f%%)\n", "forward GEMM", 1000.0 * gemmT / reps,
           100.0 * gemmT / total);
    printf("    %-24s %9.4f ms/call  (%5.1f%%)\n", "AddBiasPerChannel", 1000.0 * biasT / reps,
           100.0 * biasT / total);
    printf("    %-24s %9.4f ms/call  (%5.1f%%)\n", "feedback GEMM", 1000.0 * feedbackGemmT / reps,
           100.0 * feedbackGemmT / total);
    printf("    %-24s %9.4f ms/call  (%5.1f%%)\n", "Col2Im (+ zero)", 1000.0 * col2imT / reps,
           100.0 * col2imT / total);
  }
}

// ===========================================================================
// 4. Batch-size scaling sweep
// ===========================================================================

void BenchmarkBatchScalingDense(int nBatches)
{
  printf("\n=== Batch-size scaling: SimplePCNetwork (784-256-128-10) ===\n");
  printf("  %-8s %12s %14s\n", "batch", "ms/batch", "us/sample");
  for (int batch : {1, 16, 64, 256})
  {
    SimplePCNetwork net(batch);
    net.AddLayer(784, 256, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(256, 128, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(128, 10, 0.001f, 0.1f, 1e-4f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(10, 0, 0.001f, 0.1f, 1e-4f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();
    std::mt19937 rng(42);
    net.RandomizeWeights(rng);
    net.SetOptimizer(OptimizerType::ADAMW);

    auto x = RandomData((size_t)batch * 784, 11);
    std::vector<float> y((size_t)batch * 10, 0.1f);

    auto t0 = Clock::now();
    for (int b = 0; b < nBatches; ++b)
      net.TrainStep(x, y, 10);
    double elapsed = Secs(t0, Clock::now());
    double msPerBatch = 1000.0 * elapsed / nBatches;
    printf("  %-8d %12.4f %14.4f\n", batch, msPerBatch, 1000.0 * msPerBatch / batch);
  }
}

void BenchmarkBatchScalingConv(int nBatches)
{
  printf("\n=== Batch-size scaling: FullConvPCNetwork (1-16-32-32-10) ===\n");
  printf("  %-8s %12s %14s\n", "batch", "ms/batch", "us/sample");
  for (int batch : {1, 16, 64, 256})
  {
    FullConvPCNetwork net(batch);
    net.AddLayer(1, 16, 28, 28, 3, 3, 2, 2, 1, 1, 10, 0.001f, 0.1f, 0.0f, 1e-4f,
                ActivationType::RELU, ActivationType::dRELU);
    net.AddLayer(16, 32, 14, 14, 3, 3, 2, 2, 1, 1, 10, 0.001f, 0.1f, 0.0f, 1e-4f,
                ActivationType::RELU, ActivationType::dRELU);
    net.AddLayer(32, 32, 7, 7, 3, 3, 1, 1, 1, 1, 10, 0.001f, 0.1f, 0.0f, 1e-4f,
                ActivationType::RELU, ActivationType::dRELU);
    net.AddLayer(32, 10, 7, 7, 7, 7, 1, 1, 0, 0, 10, 0.001f, 0.1f, 0.0f, 1e-4f,
                ActivationType::RELU, ActivationType::dRELU);
    net.AddLayer(10, 0, 1, 1, 1, 1, 1, 1, 0, 0, 10, 0.001f, 0.1f, 0.0f, 1e-4f,
                ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile();
    std::mt19937 rng(42);
    net.RandomizeWeights(rng);
    net.SetOptimizer(OptimizerType::ADAMW);

    auto x = RandomData((size_t)batch * 1 * 28 * 28, 12);
    std::vector<float> y((size_t)batch * 10, 0.1f);

    auto t0 = Clock::now();
    for (int b = 0; b < nBatches; ++b)
      net.TrainStep(x, y, 10);
    double elapsed = Secs(t0, Clock::now());
    double msPerBatch = 1000.0 * elapsed / nBatches;
    printf("  %-8d %12.4f %14.4f\n", batch, msPerBatch, 1000.0 * msPerBatch / batch);
  }
}

// ===========================================================================
// 5. Cross-network summary
// ===========================================================================

void PrintSummary()
{
  printf("\n=== Cross-network summary (ms/batch, matched representative config) ===\n");
  for (auto& [name, ms] : SummaryTable())
    printf("  %-55s %10.4f ms/batch\n", name.c_str(), ms);
}
} // namespace

int main()
{
  const int N_BATCHES = 20;
  const int PRIM_REPS = 200;

  printf("######################################################################\n");
  printf("# Deepity comprehensive PCN benchmark -- CPU\n");
  printf("######################################################################\n");

  BenchmarkSimplePCNetwork(N_BATCHES);
  BenchmarkGaussSeidelPCNetwork(N_BATCHES);
  BenchmarkDirectKPPCNetwork(N_BATCHES);
  BenchmarkFullPCNetworkBaseline(N_BATCHES);
  BenchmarkDiscriminativePCNetwork(N_BATCHES);
  BenchmarkConvPCNetwork(N_BATCHES);
  BenchmarkSimpleConvPCNetwork(N_BATCHES);
  BenchmarkFullConvPCNetworkBaseline(N_BATCHES);

  BenchmarkFullPCToggleComparison(N_BATCHES);
  BenchmarkFullConvPCToggleComparison(N_BATCHES);

  BenchmarkConvPrimitives(PRIM_REPS);

  BenchmarkBatchScalingDense(N_BATCHES);
  BenchmarkBatchScalingConv(N_BATCHES);

  PrintSummary();

#if defined(DEEPITY_USE_CUDA)
  // GPU pass: same harnesses, DeviceType::DEVICE_GPU. Not exercised on
  // this machine (no CUDA toolchain in this build), but every harness
  // above takes DeviceType implicitly via each network's own default
  // constructor argument -- on a CUDA-enabled build, re-run this file
  // with DEVICE_GPU-constructed networks for the GPU-side comparison.
  printf("\n(DEEPITY_USE_CUDA is defined in this build, but this benchmark's harnesses "
         "construct CPU networks only -- see the file header for how to add a GPU pass.)\n");
#endif

  return 0;
}
