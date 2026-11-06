/**
 * @file tCUDABatchedGemmDepthScaling.cpp
 * @brief GPU counterpart of tests/tBatchedGemmDepthScaling.cpp (CPU): does
 * CUDABackend::MatMulBatched() (cublasSgemmBatched(), issue #74) recover
 * DKP-PC's theoretical O(1) cross-layer dependency-depth advantage over
 * BP's O(L) sequential chain, on a GPU wide enough (thousands of cores)
 * that Brent's theorem actually favors it? The CPU version answered "no"
 * -- but for CPU/OpenBLAS-specific reasons (a libgomp nested-thread-pool
 * deadlock, a confirmed cblas_sgemm_batch crash bug below M/N/K=128), not
 * because the underlying theorem is wrong. This is the untested,
 * theoretically-favorable regime: real cuBLAS batched GEMM, real GPU
 * core count, no OpenMP/OpenBLAS involved at all.
 *
 * Scope: uniform hidden width across all layers -- cublasSgemmBatched()
 * itself requires identical M/N/K across the batch (see
 * CUDABackend::MatMulBatched()'s own doc comment); a ragged-width network
 * isn't covered yet.
 *
 * Manual profiling tool, no pass/fail criterion -- not registered with
 * ctest, same as CUDARandomizeBenchmark.
 *
 * @note Written and reviewed with no local GPU/nvcc to compile-check
 * against (see CONTRIBUTING.md's portability rules) -- matches this
 * codebase's existing cuBLAS/Tensor/benchmark conventions as closely as
 * possible, but must be built and run on real CUDA hardware (e.g. OSC)
 * before trusting its numbers, or even its exit code.
 */
#include <chrono>
#include <cstdio>
#include <deepity/backend/Backend.h>
#include <deepity/backend/CUDABackend.h>
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
constexpr int WARMUP = 5;
constexpr int TIMED = 30;
constexpr int DEPTHS[] = {2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};

std::vector<float> RandomVec(size_t n, std::mt19937& rng)
{
  std::normal_distribution<float> dist(0.0f, 0.1f);
  std::vector<float> v(n);
  for (auto& x : v)
    x = dist(rng);
  return v;
}

/// @brief Times `call` after WARMUP untimed calls, over TIMED timed reps,
/// returning the mean in milliseconds. `call` itself must Synchronize()
/// the backend before returning -- MatMul()/MatMulBatched()/Activation()
/// etc. are all async on the backend's own stream, so timing without a
/// Synchronize() would just measure "how long it took to enqueue
/// kernels," not real GPU execution time.
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

/// @brief Per-depth device-side state. Every Tensor is allocated once,
/// outside every timed call, and reused in place on every subsequent
/// call -- see tBatchedGemmDepthScaling.cpp's own DepthState for why:
/// fresh per-call allocation inside a hot timed loop swamps the real
/// compute/dispatch signal with allocator noise, worst exactly at the
/// large depths where any real asymptotic difference would show up.
struct DepthState
{
  int depth;

  Tensor x, W_in, W_out, err, target;
  std::vector<Tensor> W, Psi, z;

  /// @brief ForwardSequential's stored activations, depth+2 of them,
  /// indices 1..depth+1 used (index 0 is unused -- ForwardSequential's
  /// own first GEMM reads s.x directly, not acts[0]; kept for 1:1 index
  /// correspondence with the CPU version rather than renumbering).
  /// `acts` holds the POST-activation (tanh'd) value at each index, used
  /// as the next layer's GEMM input and as each weight-gradient GEMM's
  /// activated-input operand. `actsPre` holds the matching PRE-activation
  /// (raw GEMM output) value, needed separately because
  /// ActivationDerivativeInto() recomputes tanh(z) from z itself (see
  /// Activations.h's ActivationDerivativeScalar) -- it cannot be fed the
  /// already-activated value.
  std::vector<Tensor> acts, actsPre;

  // BPBackwardSequential's scratch.
  Tensor bpDelta, bpGradWOut, bpDeltaHidden, bpDeriv, bpCurDelta, bpGradW, bpNextDelta, bpGradWIn;

  // IndependentPhasesBatched's scratch (each depth entries, each own-layer-sized).
  std::vector<Tensor> proj, fgrad, mu, e, zUpdate, wgrad;
  std::vector<const float*> Aarr, Barr;
  std::vector<float*> Carr;

  DepthState(CUDABackend* backend, int d, std::mt19937& rng)
      : depth(d)
      , x(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * IN_DIM, rng))
      , W_in(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)IN_DIM * HIDDEN, rng))
      , W_out(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)HIDDEN * OUT_DIM, rng))
      , err(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * OUT_DIM, rng))
      , target(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * OUT_DIM, rng))
      , bpDelta(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * OUT_DIM)
      , bpGradWOut(backend, DeviceType::DEVICE_GPU, (size_t)HIDDEN * OUT_DIM)
      , bpDeltaHidden(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpDeriv(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpCurDelta(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpGradW(backend, DeviceType::DEVICE_GPU, (size_t)HIDDEN * HIDDEN)
      , bpNextDelta(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN)
      , bpGradWIn(backend, DeviceType::DEVICE_GPU, (size_t)IN_DIM * HIDDEN)
      , Aarr(d)
      , Barr(d)
      , Carr(d)
  {
    for (int l = 0; l < d; ++l)
    {
      W.emplace_back(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)HIDDEN * HIDDEN, rng));
      Psi.emplace_back(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)OUT_DIM * HIDDEN, rng));
      z.emplace_back(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * HIDDEN, rng));
      proj.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      fgrad.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)OUT_DIM * HIDDEN);
      mu.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      e.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      zUpdate.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      wgrad.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)HIDDEN * HIDDEN);
    }
    for (int l = 0; l < d + 2; ++l)
    {
      acts.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      actsPre.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
    }
  }
};

// Strictly sequential across layers -- the honest baseline for "no
// batching scheme can remove this part." Writes every intermediate
// activation into s.acts (needed by BPBackwardSequential() below) and
// the final output into `out`.
void ForwardSequential(CUDABackend* backend, DepthState& s, float* out)
{
  backend->MatMul(false, false, BATCH, HIDDEN, IN_DIM, 1.0f, s.x.Data(), IN_DIM, s.W_in.Data(),
                  HIDDEN, 0.0f, s.actsPre[1].Data(), HIDDEN);
  backend->ActivationInto(ActivationType::TANH, s.acts[1].Data(), s.actsPre[1].Data(),
                          (size_t)BATCH * HIDDEN);

  for (int l = 0; l < s.depth; ++l)
  {
    backend->MatMul(false, false, BATCH, HIDDEN, HIDDEN, 1.0f, s.acts[l + 1].Data(), HIDDEN,
                    s.W[l].Data(), HIDDEN, 0.0f, s.actsPre[l + 2].Data(), HIDDEN);
    backend->ActivationInto(ActivationType::TANH, s.acts[l + 2].Data(), s.actsPre[l + 2].Data(),
                            (size_t)BATCH * HIDDEN);
  }

  backend->MatMul(false, false, BATCH, OUT_DIM, HIDDEN, 1.0f, s.acts[s.depth + 1].Data(), HIDDEN,
                  s.W_out.Data(), OUT_DIM, 0.0f, out, OUT_DIM);
}

// Real reverse-mode backprop through the SAME forward chain: computes
// dL/dW for every layer via the chain rule, strictly output-to-input,
// because each layer's gradient needs the NEXT layer's upstream delta --
// a genuine sequential dependency BP cannot batch away, unlike
// IndependentPhasesBatched()'s cross-layer-independent phases below.
void BPBackwardSequential(CUDABackend* backend, DepthState& s, const float* pred)
{
  // delta = 2*(pred - target)/BATCH, built from existing generic
  // backend primitives (e = z - mu, then scale) rather than a raw
  // elementwise loop, since pred/target are device buffers here.
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

    backend->MatMul(false, true, BATCH, HIDDEN, HIDDEN, 1.0f, s.bpCurDelta.Data(), HIDDEN,
                    s.W[l].Data(), HIDDEN, 0.0f, s.bpNextDelta.Data(), HIDDEN);
    backend->Copy(s.bpCurDelta.Data(), s.bpNextDelta.Data(), (size_t)BATCH * HIDDEN);
  }

  backend->ActivationDerivativeInto(ActivationType::dTANH, s.bpDeriv.Data(), s.actsPre[1].Data(),
                                    (size_t)BATCH * HIDDEN);
  backend->MultiplyInto(s.bpCurDelta.Data(), s.bpCurDelta.Data(), s.bpDeriv.Data(),
                        (size_t)BATCH * HIDDEN);
  backend->MatMul(true, false, IN_DIM, HIDDEN, BATCH, 1.0f, s.x.Data(), IN_DIM, s.bpCurDelta.Data(),
                  HIDDEN, 0.0f, s.bpGradWIn.Data(), HIDDEN);
}

// DKP-PC's three cross-layer-independent phases, each dispatched as ONE
// cublasSgemmBatched() call across all L layers instead of a per-layer
// loop -- the thing this whole benchmark exists to test.
void IndependentPhasesBatched(CUDABackend* backend, DepthState& s)
{
  const int L = s.depth;
  std::vector<const float*>& Aarr = s.Aarr;
  std::vector<const float*>& Barr = s.Barr;
  std::vector<float*>& Carr = s.Carr;

  // proj[l] = err @ Psi[l]^T (err is the SAME shared pointer for every
  // layer -- each layer reads the shared, already-frozen terminal
  // error, never another layer's output).
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.err.Data();
    Barr[l] = s.Psi[l].Data();
    Carr[l] = s.proj[l].Data();
  }
  backend->MatMulBatched(false, true, BATCH, HIDDEN, OUT_DIM, 1.0f, Aarr.data(), OUT_DIM,
                        Barr.data(), OUT_DIM, 0.0f, Carr.data(), HIDDEN, L);

  // feedback_grad[l] = proj[l]^T @ err ; Psi[l] -= lr * feedback_grad[l]
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.err.Data();
    Barr[l] = s.proj[l].Data();
    Carr[l] = s.fgrad[l].Data();
  }
  backend->MatMulBatched(true, false, OUT_DIM, HIDDEN, BATCH, 1.0f, Aarr.data(), OUT_DIM,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);
  for (int l = 0; l < L; ++l)
    backend->AxpyInto(s.Psi[l].Data(), s.fgrad[l].Data(), (size_t)OUT_DIM * HIDDEN, -1e-3f);

  // mu[l] = z[l] @ W[l] ; e[l] = proj[l] - mu[l] ; z[l] += 0.1 * e[l] @ W[l]^T
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.z[l].Data();
    Barr[l] = s.W[l].Data();
    Carr[l] = s.mu[l].Data();
  }
  backend->MatMulBatched(false, false, BATCH, HIDDEN, HIDDEN, 1.0f, Aarr.data(), HIDDEN,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);

  for (int l = 0; l < L; ++l)
    backend->ComputeError(s.e[l].Data(), s.proj[l].Data(), s.mu[l].Data(), (size_t)BATCH * HIDDEN);

  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.e[l].Data();
    Barr[l] = s.W[l].Data();
    Carr[l] = s.zUpdate[l].Data();
  }
  backend->MatMulBatched(false, true, BATCH, HIDDEN, HIDDEN, 1.0f, Aarr.data(), HIDDEN,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);
  for (int l = 0; l < L; ++l)
    backend->AxpyInto(s.z[l].Data(), s.zUpdate[l].Data(), (size_t)BATCH * HIDDEN, 0.1f);

  // weight_grad[l] = z[l]^T @ e[l] ; W[l] -= lr * weight_grad[l]
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.z[l].Data();
    Barr[l] = s.e[l].Data();
    Carr[l] = s.wgrad[l].Data();
  }
  backend->MatMulBatched(true, false, HIDDEN, HIDDEN, BATCH, 1.0f, Aarr.data(), HIDDEN,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);
  for (int l = 0; l < L; ++l)
    backend->AxpyInto(s.W[l].Data(), s.wgrad[l].Data(), (size_t)HIDDEN * HIDDEN, -1e-3f);
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

  printf("IN_DIM=%d HIDDEN=%d OUT_DIM=%d BATCH=%d  (Deepity CUDABackend, cublasSgemmBatched)\n\n",
        IN_DIM, HIDDEN, OUT_DIM, BATCH);
  printf("%6s%12s%14s%10s%10s\n", "depth", "phases_ms", "full_step_ms", "bp_ms", "dkp/bp");

  std::vector<double> phaseMs, fullMs, bpMs;
  for (int depth : DEPTHS)
  {
    DepthState s(backend, depth, rng);
    Tensor out(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * OUT_DIM);

    double ph = TimeMsFn(
        [&]
        {
          IndependentPhasesBatched(backend, s);
          backend->Synchronize();
        });

    double full = TimeMsFn(
        [&]
        {
          ForwardSequential(backend, s, out.Data());
          IndependentPhasesBatched(backend, s);
          backend->Synchronize();
        });

    double bp = TimeMsFn(
        [&]
        {
          ForwardSequential(backend, s, out.Data());
          BPBackwardSequential(backend, s, out.Data());
          backend->Synchronize();
        });

    phaseMs.push_back(ph);
    fullMs.push_back(full);
    bpMs.push_back(bp);
    printf("%6d%12.4f%14.4f%10.4f%10.3f\n", depth, ph, full, bp, full / bp);
  }

  size_t n = phaseMs.size();
  printf("\nGrowth from depth=%d to depth=%d (%.1fx more layers):\n", DEPTHS[0], DEPTHS[n - 1],
        (double)DEPTHS[n - 1] / DEPTHS[0]);
  printf("  independent phases only (batched cuBLAS, zero per-layer dispatch): %.2fx slower\n",
        phaseMs[n - 1] / phaseMs[0]);
  printf("  full step (forward + phases):                                     %.2fx slower\n",
        fullMs[n - 1] / fullMs[0]);
  printf("  BP (hand-written backward, same forward pass):                    %.2fx slower\n",
        bpMs[n - 1] / bpMs[0]);
  return 0;
}
