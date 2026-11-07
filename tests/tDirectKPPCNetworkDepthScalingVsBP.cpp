/**
 * @file tDirectKPPCNetworkDepthScalingVsBP.cpp
 * @brief Real-network timing benchmark (issue #74 follow-up): does
 * DirectKPPCNetwork's batched-GPU fast path actually pay off in
 * wall-clock terms at the NETWORK level (TrainStep(), not raw GEMM
 * calls), and how does DKP-PC compare against a hand-rolled backprop
 * baseline at the same depth/width?
 *
 * Three columns per depth:
 *   dkp_batched_ms   - real DirectKPPCNetwork::TrainStep(), GPU, the
 *                      batched fast path engaged naturally (Compile()'s
 *                      own eligibility detection, uniform-width run).
 *   dkp_unbatched_ms - a SEPARATE, otherwise-identical network with the
 *                      batched path forced off via
 *                      SetBatchedRunCountForBenchmark(0) BEFORE its
 *                      first TrainStep() call -- isolates batching's
 *                      own contribution from "GPU is just fast." Must
 *                      be a distinct network instance, not the same one
 *                      toggled mid-run: once a network's graph is
 *                      captured, CUDA graph replay re-runs only the
 *                      recorded GPU ops, never the host-side
 *                      UseBatchedPath() branch that chose them, so
 *                      flipping the flag on an already-captured network
 *                      would silently keep replaying the OLD path.
 *   bp_ms            - hand-rolled backprop forward+backward through an
 *                      equivalent MLP (same IN_DIM/HIDDEN/OUT_DIM/BATCH,
 *                      same depth), raw backend calls, no DKP-PC
 *                      machinery at all -- adapted from
 *                      tCUDABatchedGemmDepthScaling.cpp's own
 *                      ForwardSequential()/BPBackwardSequential().
 *
 * Not a correctness test -- see tDirectKPBatchedGPUVerify.cpp for that.
 * Manual profiling tool, no pass/fail criterion, not registered with
 * ctest, same as CUDARandomizeBenchmark/CUDABatchedGemmDepthScaling.
 *
 * DKP-PC's own per-step cost includes `SETTLING_STEPS` settling
 * iterations plus DFA/weight-update phases BP has no analogue for --
 * deliberate, not an oversight: each algorithm is timed exactly as it
 * is actually used, so this compares "DKP-PC at its real per-step
 * cost, parallel-depth advantage included" against "BP at its real
 * (cheaper-per-step, but strictly sequential-depth) cost."
 *
 * @note Written and reviewed with no local GPU/nvcc to compile-check
 * against (see CONTRIBUTING.md's portability rules) -- must be built
 * and run on real CUDA hardware (e.g. OSC) before trusting its numbers.
 */
#include <chrono>
#include <cstdio>
#include <deepity/backend/Backend.h>
#include <deepity/backend/CUDABackend.h>
#include <deepity/networks/DirectKPPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;
using Clock = std::chrono::steady_clock;

namespace
{
constexpr int IN_DIM = 64;
constexpr int HIDDEN = 128;
constexpr int OUT_DIM = 10;
constexpr int BATCH = 128;
constexpr int WARMUP = 3;
constexpr int TIMED = 10;
constexpr int SETTLING_STEPS = 5;
// Isolates how much of DKP-PC's gap against BP is just "5 settling
// iterations' worth of GEMMs" vs. remaining per-dispatch overhead --
// see this file's own header comment on why DKP-PC's per-TrainStep()
// cost already includes work (ProjectForward/DFA/5x settling/
// UpdateWeights) BP has no equivalent of.
constexpr int SETTLING_STEPS_ABLATION = 1;
constexpr int DEPTHS[] = {2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048};
constexpr int N_DEPTHS = sizeof(DEPTHS) / sizeof(DEPTHS[0]);

std::vector<float> RandomVec(size_t n, std::mt19937& rng)
{
  std::normal_distribution<float> dist(0.0f, 0.1f);
  std::vector<float> v(n);
  for (auto& x : v)
    x = dist(rng);
  return v;
}

/// @brief Times `call` after WARMUP untimed calls, over TIMED timed
/// reps, returning the mean in milliseconds. See
/// tCUDABatchedGemmDepthScaling.cpp's identical helper for why `call`
/// must itself Synchronize() (or, for a DirectKPPCNetwork call, already
/// does internally) before returning.
template <typename Fn>
double TimeMsFn(Fn&& call)
{
  for (int i = 0; i < WARMUP; ++i)
    call();
  double total = 0.0;
  for (int i = 0; i < TIMED; ++i)
  {
    auto t0 = Clock::now();
    call();
    total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }
  return total / TIMED;
}

// `depth` uniform-width (HIDDEN->HIDDEN) middle layers, matching
// Compile()'s own "extend while uniform" scan exactly: layers[1..depth]
// become the batched run, layers[depth+1] is the one trailing layer
// whose nextSize==OUT_DIM breaks uniformity (every realistic network
// has exactly this shape -- the hidden width almost never equals
// terminalSize), and layers[depth+2] is the terminal. See
// DirectKPPCNetwork.h's own "Batched-GPU fast path" member comments.
void BuildNetwork(DirectKPPCNetwork& net, int depth)
{
  const float lr = 0.02f, ir = 0.1f, fl = 0.01f, lmbda = 0.0f;
  net.AddLayer(IN_DIM, HIDDEN, OUT_DIM, lr, ir, fl, lmbda, ActivationType::TANH,
               ActivationType::dTANH);
  for (int l = 0; l < depth; ++l)
    net.AddLayer(HIDDEN, HIDDEN, OUT_DIM, lr, ir, fl, lmbda, ActivationType::TANH,
                 ActivationType::dTANH);
  net.AddLayer(HIDDEN, OUT_DIM, OUT_DIM, lr, ir, fl, lmbda, ActivationType::TANH,
               ActivationType::dTANH);
  net.AddLayer(OUT_DIM, 0, OUT_DIM, lr, ir, fl, lmbda, ActivationType::TANH, ActivationType::dTANH);
  net.Compile();
}

// Hand-rolled BP baseline's per-depth device-side state, allocated once
// outside every timed call -- see tCUDABatchedGemmDepthScaling.cpp's own
// DepthState comment for why (fresh per-call allocation inside a hot
// timed loop swamps the signal with allocator noise, worst exactly at
// the large depths this benchmark cares about).
struct BPState
{
  int depth;
  Tensor x, W_in, W_out, target;
  Tensor Wflat;
  std::vector<Tensor> acts, actsPre;
  Tensor bpDelta, bpGradWOut, bpDeltaHidden, bpDeriv, bpCurDelta, bpGradW, bpNextDelta, bpGradWIn;

  BPState(CUDABackend* backend, int d, std::mt19937& rng)
      : depth(d)
      , x(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * IN_DIM, rng))
      , W_in(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)IN_DIM * HIDDEN, rng))
      , W_out(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)HIDDEN * OUT_DIM, rng))
      , target(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * OUT_DIM, rng))
      , Wflat(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)d * HIDDEN * HIDDEN, rng))
      , bpDelta(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * OUT_DIM)
      , bpGradWOut(backend, DeviceType::DEVICE_GPU, (size_t)HIDDEN * OUT_DIM)
      , bpDeltaHidden(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpDeriv(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpCurDelta(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpGradW(backend, DeviceType::DEVICE_GPU, (size_t)HIDDEN * HIDDEN)
      , bpNextDelta(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpGradWIn(backend, DeviceType::DEVICE_GPU, (size_t)IN_DIM * HIDDEN)
  {
    for (int l = 0; l < d + 2; ++l)
    {
      acts.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      actsPre.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
    }
  }

  float* W(int l) noexcept
  {
    return Wflat.Data() + (size_t)l * HIDDEN * HIDDEN;
  }
};

void ForwardSequential(CUDABackend* backend, BPState& s, float* out)
{
  backend->MatMul(false, false, BATCH, HIDDEN, IN_DIM, 1.0f, s.x.Data(), IN_DIM, s.W_in.Data(),
                   HIDDEN, 0.0f, s.actsPre[1].Data(), HIDDEN);
  backend->ActivationInto(ActivationType::TANH, s.acts[1].Data(), s.actsPre[1].Data(),
                           (size_t)BATCH * HIDDEN);

  for (int l = 0; l < s.depth; ++l)
  {
    backend->MatMul(false, false, BATCH, HIDDEN, HIDDEN, 1.0f, s.acts[l + 1].Data(), HIDDEN, s.W(l),
                     HIDDEN, 0.0f, s.actsPre[l + 2].Data(), HIDDEN);
    backend->ActivationInto(ActivationType::TANH, s.acts[l + 2].Data(), s.actsPre[l + 2].Data(),
                             (size_t)BATCH * HIDDEN);
  }

  backend->MatMul(false, false, BATCH, OUT_DIM, HIDDEN, 1.0f, s.acts[s.depth + 1].Data(), HIDDEN,
                   s.W_out.Data(), OUT_DIM, 0.0f, out, OUT_DIM);
}

// Real reverse-mode backprop through the SAME forward chain -- see
// tCUDABatchedGemmDepthScaling.cpp's identical function for why this is
// the honest, can't-be-batched-away sequential baseline.
void BPBackwardSequential(CUDABackend* backend, BPState& s, const float* pred)
{
  backend->ComputeError(s.bpDelta.Data(), pred, s.target.Data(), (size_t)BATCH * OUT_DIM);
  backend->Scale(s.bpDelta.Data(), (size_t)BATCH * OUT_DIM, 2.0f / BATCH);

  backend->MatMul(true, false, HIDDEN, OUT_DIM, BATCH, 1.0f, s.acts[s.depth + 1].Data(), HIDDEN,
                   s.bpDelta.Data(), OUT_DIM, 0.0f, s.bpGradWOut.Data(), OUT_DIM);

  backend->MatMul(false, true, BATCH, HIDDEN, OUT_DIM, 1.0f, s.bpDelta.Data(), OUT_DIM,
                   s.W_out.Data(), OUT_DIM, 0.0f, s.bpDeltaHidden.Data(), HIDDEN);

  backend->Copy(s.bpCurDelta.Data(), s.bpDeltaHidden.Data(), (size_t)BATCH * HIDDEN);
  for (int l = s.depth - 1; l >= 0; --l)
  {
    backend->ActivationDerivativeInto(ActivationType::dTANH, s.bpDeriv.Data(),
                                       s.actsPre[l + 2].Data(), (size_t)BATCH * HIDDEN);
    backend->MultiplyInto(s.bpCurDelta.Data(), s.bpCurDelta.Data(), s.bpDeriv.Data(),
                           (size_t)BATCH * HIDDEN);

    backend->MatMul(true, false, HIDDEN, HIDDEN, BATCH, 1.0f, s.acts[l + 1].Data(), HIDDEN,
                     s.bpCurDelta.Data(), HIDDEN, 0.0f, s.bpGradW.Data(), HIDDEN);

    backend->MatMul(false, true, BATCH, HIDDEN, HIDDEN, 1.0f, s.bpCurDelta.Data(), HIDDEN, s.W(l),
                     HIDDEN, 0.0f, s.bpNextDelta.Data(), HIDDEN);
    backend->Copy(s.bpCurDelta.Data(), s.bpNextDelta.Data(), (size_t)BATCH * HIDDEN);
  }

  backend->ActivationDerivativeInto(ActivationType::dTANH, s.bpDeriv.Data(), s.actsPre[1].Data(),
                                     (size_t)BATCH * HIDDEN);
  backend->MultiplyInto(s.bpCurDelta.Data(), s.bpCurDelta.Data(), s.bpDeriv.Data(),
                         (size_t)BATCH * HIDDEN);
  backend->MatMul(true, false, IN_DIM, HIDDEN, BATCH, 1.0f, s.x.Data(), IN_DIM, s.bpCurDelta.Data(),
                   HIDDEN, 0.0f, s.bpGradWIn.Data(), HIDDEN);
}
} // namespace

int main()
{
  auto backendPtr = CreateBackend(DeviceType::DEVICE_GPU);
  auto* backend = dynamic_cast<CUDABackend*>(backendPtr.get());
  if (!backend)
  {
    printf("FAILED: CreateBackend(DEVICE_GPU) didn't return a CUDABackend.\n");
    return 1;
  }

  std::mt19937 rng(42);
  std::vector<float> input((size_t)BATCH * IN_DIM), target((size_t)BATCH * OUT_DIM);
  {
    std::uniform_real_distribution<float> dataDist(-1.0f, 1.0f);
    for (auto& v : input)
      v = dataDist(rng);
    for (auto& v : target)
      v = dataDist(rng);
  }

  printf("IN_DIM=%d HIDDEN=%d OUT_DIM=%d BATCH=%d settlingSteps=%d (ablation column: %d)\n\n",
         IN_DIM, HIDDEN, OUT_DIM, BATCH, SETTLING_STEPS, SETTLING_STEPS_ABLATION);
  printf("%6s%12s%12s%14s%10s%10s%10s\n", "depth", "dkp5_ms", "dkp1_ms", "dkp_unb_ms", "bp_ms",
         "dkp5/bp", "dkp1/bp");

  std::vector<double> dkp5Ms, dkp1Ms, dkpUnbatchedMs, bpMs;
  for (int depth : DEPTHS)
  {
    DirectKPPCNetwork netBatched((int)BATCH, DeviceType::DEVICE_GPU);
    BuildNetwork(netBatched, depth);
    netBatched.RandomizeWeights(rng);

    size_t naturalRun = netBatched.GetBatchedRunCount();
    if (naturalRun != (size_t)depth)
    {
      printf("  depth %d: SKIPPED -- Compile() detected a batched run of %zu layers, expected "
             "%d. Eligibility detection or this test's topology regressed.\n",
             depth, naturalRun, depth);
      continue;
    }

    double dkp5 = TimeMsFn([&] { netBatched.TrainStep(input, target, SETTLING_STEPS); });
    // Reuses netBatched: TrainStep() recaptures its graph whenever
    // inferenceSteps changes from what's currently captured (see its
    // own capturedInferenceSteps check), so this is a real, fresh
    // 1-settling-step graph, not a stale 5-step one.
    double dkp1 = TimeMsFn([&] { netBatched.TrainStep(input, target, SETTLING_STEPS_ABLATION); });

    // TEMP: isolate the batched-only profile (dkp5/dkp1), skipping the
    // deliberately-unbatched and BP baselines -- revert before committing.
#if 1
    double unbatched = 0.0, bp = 1.0;
#else
    // A SEPARATE network, not netBatched with the flag flipped -- see
    // this file's header comment for why that would silently replay
    // the wrong graph.
    DirectKPPCNetwork netUnbatched((int)BATCH, DeviceType::DEVICE_GPU);
    BuildNetwork(netUnbatched, depth);
    netUnbatched.RandomizeWeights(rng);
    netUnbatched.SetBatchedRunCountForBenchmark(0);
    double unbatched = TimeMsFn([&] { netUnbatched.TrainStep(input, target, SETTLING_STEPS); });

    BPState s(backend, depth, rng);
    Tensor out(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * OUT_DIM);
    double bp = TimeMsFn(
        [&]
        {
          ForwardSequential(backend, s, out.Data());
          BPBackwardSequential(backend, s, out.Data());
          backend->Synchronize();
        });
#endif

    dkp5Ms.push_back(dkp5);
    dkp1Ms.push_back(dkp1);
    dkpUnbatchedMs.push_back(unbatched);
    bpMs.push_back(bp);
    printf("%6d%12.4f%12.4f%14.4f%10.4f%10.3f%10.3f\n", depth, dkp5, dkp1, unbatched, bp, dkp5 / bp,
           dkp1 / bp);
  }

  size_t n = dkp5Ms.size();
  if (n >= 2)
  {
    printf("\nGrowth from depth=%d to depth=%d (%.1fx more layers):\n", DEPTHS[0],
           DEPTHS[N_DEPTHS - 1], (double)DEPTHS[N_DEPTHS - 1] / DEPTHS[0]);
    printf("  DKP-PC, batched, %d settling steps:     %.2fx slower\n", SETTLING_STEPS,
           dkp5Ms[n - 1] / dkp5Ms[0]);
    printf("  DKP-PC, batched, %d settling step:      %.2fx slower\n", SETTLING_STEPS_ABLATION,
           dkp1Ms[n - 1] / dkp1Ms[0]);
    printf("  DKP-PC, batched path forced off:       %.2fx slower\n",
           dkpUnbatchedMs[n - 1] / dkpUnbatchedMs[0]);
    printf("  BP (hand-rolled, same forward shape):  %.2fx slower\n", bpMs[n - 1] / bpMs[0]);
  }
  return 0;
}
