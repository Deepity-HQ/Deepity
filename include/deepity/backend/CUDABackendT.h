#pragma once
#ifdef DEEPITY_USE_CUDA
#include <deepity/backend/CUDABackend.h>

/**
 * @file CUDABackendT.h
 * @brief Templated counterpart to CUDABackend, implementing
 * IComputeBackendT<T, AccumT> on CUDA.
 *
 * Only CUDABackendT<float, float> is defined. Naming any other (T,
 * AccumT) pair, e.g. CUDABackendT<double, double>, is a compile error.
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
