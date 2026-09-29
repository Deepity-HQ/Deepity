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
} // namespace Deep
