#pragma once
#include <cstddef>
#include <deepity/utils/ActivationType.h>

/**
 * @file CPUBackendHighway.h
 * @brief Declares the Google Highway (github.com/google/highway) runtime-
 * dispatched implementations backing a handful of CPUBackend's elementwise
 * kernels. Deliberately a separate CMake target (`DeepityHighwayKernels`,
 * see CMakeLists.txt) compiled WITHOUT the project's hard-pinned -march
 * flag -- Highway's own foreach_target.h mechanism compiles multiple
 * per-ISA-target variants of CPUBackendHighway.cpp and picks the best one
 * at runtime; a project-wide fixed ISA would defeat that entirely.
 */

namespace Deep
{
/// @copydoc Deep::IComputeBackend::AddBiasPerChannel
void AddBiasPerChannelDispatch(float* buf, const float* bias, size_t channels,
                               size_t spatialSize) noexcept;

/// @copydoc Deep::IComputeBackend::FusedActivationDerivativeMultiply
/// @note dGELU is NOT handled here -- its derivative needs a transcendental
/// (SLEEF tanh) that isn't in scope for this pass, see
/// ActivationDerivativeFromActivatedScalar's own documented GELU caveat.
/// Callers must route dGELU to the existing scalar path instead.
void FusedActivationDerivativeMultiplyDispatch(float* dst, const float* a, float* activatedInOut,
                                               ActivationType dType, size_t n) noexcept;

/// @copydoc Deep::IComputeBackend::Fill
void FillDispatch(float* buf, size_t n, float value) noexcept;

/// @copydoc Deep::IComputeBackend::MultiplyInto
void MultiplyIntoDispatch(float* dst, const float* a, const float* b, size_t n) noexcept;

/// @copydoc Deep::IComputeBackend::FusedStateUpdate
/// @note Unlike FusedActivationDerivativeMultiplyDispatch, dType here is
/// evaluated from the raw pre-activation `z` (see
/// Deep::ActivationDerivativeScalar), not an already-activated value --
/// dGELU, dSIGMOID and dTANH all need a transcendental (SLEEF tanh or
/// std::exp) in that form, so only dRELU, d_eSIGMOID, dLINEAR and NONE are
/// handled here. Callers must route the other three to the existing scalar
/// path instead.
void FusedStateUpdateDispatch(float* z, const float* feedback, ActivationType dType, const float* e,
                              size_t n, float ir) noexcept;

/// @copydoc Deep::IComputeBackend::FusedStateUpdateMomentum
/// @note Same dType restriction as FusedStateUpdateDispatch() above.
void FusedStateUpdateMomentumDispatch(float* z, float* v, const float* feedback,
                                      ActivationType dType, const float* e, size_t n, float ir,
                                      float beta) noexcept;

/// @copydoc Deep::IComputeBackend::FusedActivationDerivativeMultiplyFromPreact
/// @note Unlike FusedActivationDerivativeMultiplyDispatch above, dType here
/// is evaluated from the raw pre-activation value (see
/// Deep::ActivationDerivativeScalar), not an already-activated one --
/// dGELU, dSIGMOID and dTANH all need a transcendental in that form. Unlike
/// FusedStateUpdateDispatch, this one DOES handle all three: see this
/// function's own file comment on hwy/contrib/math's portable vectorized
/// Exp/Tanh. No caller-side routing needed; every ActivationType is
/// handled here.
void FusedActivationDerivativeMultiplyFromPreactDispatch(float* dst, const float* a,
                                                          const float* preact, ActivationType dType,
                                                          size_t n) noexcept;

/// @brief Row-granular vectorized kernel backing
/// Deep::IComputeBackend::ComputeSoftmaxCrossEntropyError: softmax over
/// `muRow` (nextSize logits), then `eRow := zRow - softmax(muRow)`. Takes
/// ONE row (not the whole batch) since this file's DeepityHighwayKernels
/// target links no OpenMP of its own -- CPUBackend.cpp's existing
/// `#pragma omp parallel for` over the batch dimension calls this once
/// per row, same call count/grain as the scalar loop it replaces.
/// @note Unlike the other transcendental cases in this file, this calls
/// hn::Exp directly (not hn::Tanh) -- see this file's own math-inl.h Exp
/// note: the argument here is always muRow[j]-max_val <= 0 (max_val is
/// this row's own max), so it can only ever hit Exp's explicitly-guarded
/// lower bound (returns 0 for x<-104, correct), never its unguarded,
/// unsafe upper bound that bit dSIGMOID/dTANH.
void ComputeSoftmaxCrossEntropyErrorRowDispatch(float* eRow, const float* zRow,
                                                 const float* muRow, size_t nextSize) noexcept;

/// @brief Vectorized chunk kernel backing Deep::IComputeBackend::AdamStep
/// (over [start, end), not the whole [0, n) range -- this file's
/// DeepityHighwayKernels target links no OpenMP of its own, so
/// CPUBackend.cpp's own `#pragma omp parallel` picks each thread's
/// [start, end) slice and calls this once per thread, same split the
/// scalar loop it replaces already used via `schedule(static)`. No
/// transcendental involved (hn::Sqrt is a native op on every target,
/// unlike hn::Exp/Tanh elsewhere in this file) so there's no domain-
/// safety note needed here: v[i] is a sum of squares, never negative.
void AdamStepChunkDispatch(float* param, const float* grad, float* m, float* v, size_t start,
                           size_t end, float stepSize, float beta1, float beta2,
                           float eps) noexcept;

/// @brief Vectorized chunk kernel backing
/// Deep::IComputeBackend::TryComputeErrorAndActivation, over [start, end)
/// for the same reason as AdamStepChunkDispatch above (no OpenMP linked
/// into this target). Unlike that scalar function's own doc comment
/// (CPUBackend.cpp), this keeps BOTH the fusion (one pass over z, one
/// parallel region) AND vectorizes the activation -- the scalar version
/// traded vectorization away specifically to keep the fusion, not
/// because the two are mutually exclusive; this is that same fusion
/// with hn::Tanh-based forward formulas (ActivationForwardHwy, mirroring
/// Deep::ActivationScalar) doing the per-type math instead.
void ComputeErrorAndActivationChunkDispatch(float* e, float* zF, const float* z, const float* mu,
                                             ActivationType activationType, size_t start,
                                             size_t end) noexcept;

/// @copydoc Deep::AdamStepChunkDispatch
/// @note Same as AdamStepChunkDispatch but with the AdamW weight-decay
/// term folded in, matching Deep::IComputeBackend::AdamWStep.
void AdamWStepChunkDispatch(float* param, const float* grad, float* m, float* v, size_t start,
                            size_t end, float stepSize, float beta1, float beta2, float eps,
                            float lrWeightDecay) noexcept;
} // namespace Deep
