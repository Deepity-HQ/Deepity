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

  /// @brief Every per-layer quantity below is ONE contiguous Tensor
  /// holding all `depth` layers back-to-back (layer l's slice starts at
  /// l*perLayerSize), not a std::vector of `depth` separate allocations.
  /// This matters beyond memory layout: it lets elementwise updates
  /// (AxpyInto/ComputeError) run as a SINGLE O(1)-dispatch call over
  /// `depth*perLayerSize` elements instead of `depth` separate per-layer
  /// calls -- without it, IndependentPhasesBatched() below would still
  /// batch its GEMMs but stay O(depth) in total dispatch count anyway,
  /// defeating the point of this whole benchmark (this is exactly the
  /// gap a first version of this file had, found via a flattening-out
  /// ratio plateau that didn't budge from depth=96 to depth=2048).
  Tensor Wflat, Psiflat, zflat;
  Tensor projflat, fgradflat, muflat, eflat, zUpdateflat, wgradflat;

  /// @brief ForwardSequential's stored activations, depth+2 of them,
  /// indices 1..depth+1 used (index 0 is unused -- ForwardSequential's
  /// own first GEMM reads s.x directly, not acts[0]; kept for 1:1 index
  /// correspondence with the CPU version rather than renumbering). Left
  /// as a std::vector<Tensor>, not flattened like the above -- there is
  /// no cross-layer-independent batching to do here at all, ever: each
  /// entry depends on the previous one by construction (the whole point
  /// of ForwardSequential/BPBackwardSequential as the honest sequential
  /// baseline).
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

  // IndependentPhasesBatched's per-call GEMM pointer-array scratch --
  // cublasSgemmBatched (via MatMulBatched()) still wants an array of
  // per-layer pointers even though those pointers are now just
  // base+l*stride into one of the flat Tensors above, rather than `depth`
  // independently-allocated Tensors' own Data() pointers.
  std::vector<const float*> Aarr, Barr;
  std::vector<float*> Carr;

  DepthState(CUDABackend* backend, int d, std::mt19937& rng)
      : depth(d)
      , x(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * IN_DIM, rng))
      , W_in(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)IN_DIM * HIDDEN, rng))
      , W_out(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)HIDDEN * OUT_DIM, rng))
      , err(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * OUT_DIM, rng))
      , target(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)BATCH * OUT_DIM, rng))
      , Wflat(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)d * HIDDEN * HIDDEN, rng))
      , Psiflat(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)d * OUT_DIM * HIDDEN, rng))
      , zflat(backend, DeviceType::DEVICE_GPU, RandomVec((size_t)d * BATCH * HIDDEN, rng))
      , projflat(backend, DeviceType::DEVICE_GPU, (size_t)d * BATCH * HIDDEN)
      , fgradflat(backend, DeviceType::DEVICE_GPU, (size_t)d * OUT_DIM * HIDDEN)
      , muflat(backend, DeviceType::DEVICE_GPU, (size_t)d * BATCH * HIDDEN)
      , eflat(backend, DeviceType::DEVICE_GPU, (size_t)d * BATCH * HIDDEN)
      , zUpdateflat(backend, DeviceType::DEVICE_GPU, (size_t)d * BATCH * HIDDEN)
      , wgradflat(backend, DeviceType::DEVICE_GPU, (size_t)d * HIDDEN * HIDDEN)
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
    for (int l = 0; l < d + 2; ++l)
    {
      acts.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
      actsPre.emplace_back(backend, DeviceType::DEVICE_GPU, (size_t)BATCH * HIDDEN);
    }
  }

  float* W(int l) noexcept { return Wflat.Data() + (size_t)l * HIDDEN * HIDDEN; }
  float* Psi(int l) noexcept { return Psiflat.Data() + (size_t)l * OUT_DIM * HIDDEN; }
  float* z(int l) noexcept { return zflat.Data() + (size_t)l * BATCH * HIDDEN; }
  float* proj(int l) noexcept { return projflat.Data() + (size_t)l * BATCH * HIDDEN; }
  float* fgrad(int l) noexcept { return fgradflat.Data() + (size_t)l * OUT_DIM * HIDDEN; }
  float* mu(int l) noexcept { return muflat.Data() + (size_t)l * BATCH * HIDDEN; }
  float* e(int l) noexcept { return eflat.Data() + (size_t)l * BATCH * HIDDEN; }
  float* zUpdate(int l) noexcept { return zUpdateflat.Data() + (size_t)l * BATCH * HIDDEN; }
  float* wgrad(int l) noexcept { return wgradflat.Data() + (size_t)l * HIDDEN * HIDDEN; }
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
                    s.W(l), HIDDEN, 0.0f, s.actsPre[l + 2].Data(), HIDDEN);
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
                    s.W(l), HIDDEN, 0.0f, s.bpNextDelta.Data(), HIDDEN);
    backend->Copy(s.bpCurDelta.Data(), s.bpNextDelta.Data(), (size_t)BATCH * HIDDEN);
  }

  backend->ActivationDerivativeInto(ActivationType::dTANH, s.bpDeriv.Data(), s.actsPre[1].Data(),
                                    (size_t)BATCH * HIDDEN);
  backend->MultiplyInto(s.bpCurDelta.Data(), s.bpCurDelta.Data(), s.bpDeriv.Data(),
                        (size_t)BATCH * HIDDEN);
  backend->MatMul(true, false, IN_DIM, HIDDEN, BATCH, 1.0f, s.x.Data(), IN_DIM, s.bpCurDelta.Data(),
                  HIDDEN, 0.0f, s.bpGradWIn.Data(), HIDDEN);
}

// DKP-PC's three cross-layer-independent phases. Every GEMM group is
// dispatched as ONE cublasSgemmBatched() call across all L layers
// (MatMulBatched()); every elementwise update is ONE call over all L
// layers' flattened buffers at once (AxpyInto/ComputeError, called with
// L*perLayerSize elements, not looped per layer) -- O(1) host-side
// dispatches total, except the two GEMM groups below whose OUT_DIM=10
// shape falls below MatMulBatched()'s kMinSafeDim and so still cost
// O(L) dispatches internally (see that function's own doc comment). This
// is the thing this whole benchmark exists to test.
void IndependentPhasesBatched(CUDABackend* backend, DepthState& s)
{
  const int L = s.depth;
  std::vector<const float*>& Aarr = s.Aarr;
  std::vector<const float*>& Barr = s.Barr;
  std::vector<float*>& Carr = s.Carr;

  // proj[l] = err @ Psi[l]^T (err is the SAME shared pointer for every
  // layer -- each layer reads the shared, already-frozen terminal
  // error, never another layer's output). OUT_DIM=10 < kMinSafeDim, so
  // MatMulBatched() itself falls back to a per-layer loop here.
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.err.Data();
    Barr[l] = s.Psi(l);
    Carr[l] = s.proj(l);
  }
  backend->MatMulBatched(false, true, BATCH, HIDDEN, OUT_DIM, 1.0f, Aarr.data(), OUT_DIM,
                        Barr.data(), OUT_DIM, 0.0f, Carr.data(), HIDDEN, L);

  // feedback_grad[l] = proj[l]^T @ err ; Psi[l] -= lr * feedback_grad[l]
  // (same OUT_DIM=10 fallback as above for the GEMM itself.)
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.err.Data();
    Barr[l] = s.proj(l);
    Carr[l] = s.fgrad(l);
  }
  backend->MatMulBatched(true, false, OUT_DIM, HIDDEN, BATCH, 1.0f, Aarr.data(), OUT_DIM,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);
  backend->AxpyInto(s.Psiflat.Data(), s.fgradflat.Data(), (size_t)L * OUT_DIM * HIDDEN, -1e-3f);

  // mu[l] = z[l] @ W[l] ; e[l] = proj[l] - mu[l] ; z[l] += 0.1 * e[l] @ W[l]^T
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.z(l);
    Barr[l] = s.W(l);
    Carr[l] = s.mu(l);
  }
  backend->MatMulBatched(false, false, BATCH, HIDDEN, HIDDEN, 1.0f, Aarr.data(), HIDDEN,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);

  backend->ComputeError(s.eflat.Data(), s.projflat.Data(), s.muflat.Data(),
                        (size_t)L * BATCH * HIDDEN);

  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.e(l);
    Barr[l] = s.W(l);
    Carr[l] = s.zUpdate(l);
  }
  backend->MatMulBatched(false, true, BATCH, HIDDEN, HIDDEN, 1.0f, Aarr.data(), HIDDEN,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);
  backend->AxpyInto(s.zflat.Data(), s.zUpdateflat.Data(), (size_t)L * BATCH * HIDDEN, 0.1f);

  // weight_grad[l] = z[l]^T @ e[l] ; W[l] -= lr * weight_grad[l]
  for (int l = 0; l < L; ++l)
  {
    Aarr[l] = s.z(l);
    Barr[l] = s.e(l);
    Carr[l] = s.wgrad(l);
  }
  backend->MatMulBatched(true, false, HIDDEN, HIDDEN, BATCH, 1.0f, Aarr.data(), HIDDEN,
                        Barr.data(), HIDDEN, 0.0f, Carr.data(), HIDDEN, L);
  backend->AxpyInto(s.Wflat.Data(), s.wgradflat.Data(), (size_t)L * HIDDEN * HIDDEN, -1e-3f);
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
