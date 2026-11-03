#pragma once
#ifdef DEEPITY_USE_CUDA
#include <deepity/backend/CUDABackend.h>

/**
 * @file CUDABackendT.h
 * @brief Templated counterpart to the plain CUDABackend, for issue #55
 * (CUDA backend for the templated FullPCNetworkT<T, AccumT> path).
 *
 * IComputeBackend.h's own closing line is `using IComputeBackend =
 * IComputeBackendT<float, float>;` -- so the existing, already-shipping
 * CUDABackend class ALREADY fully implements IComputeBackendT<float,
 * float>, just never spelled that way. CUDABackendT<float, float> is
 * therefore nothing more than that same class under a templated name,
 * so CreateBackendT<T, AccumT> (BackendT.h) can construct it generically.
 *
 * No other (T, AccumT) pair is supported yet: no fp16 cuBLAS GEMM and
 * no device-side decode/encode kernels for the QuantFormat.h sub-8-bit
 * structs exist anywhere in this codebase today, even on the
 * non-templated path -- that's genuinely new engineering, deliberately
 * out of scope here. The primary template below is intentionally left
 * incomplete (never defined) for every pair but float/float, so naming
 * e.g. CUDABackendT<double, double> anywhere is a compile error, not a
 * runtime throw.
 */

namespace Deep
{
template <typename T, typename AccumT = T>
class CUDABackendT; // intentionally incomplete for every pair but float,float

template <>
class CUDABackendT<float, float> final : public CUDABackend
{
};
} // namespace Deep
#endif
