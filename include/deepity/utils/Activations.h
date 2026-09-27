#pragma once

/**
 * @file Activations.h
 * @brief Defines the activation functions of a predictive coding model.
 *
 * This is an umbrella header: the actual implementations live in
 * deepity/utils/activations/ (ActivationDispatch.h for the ActivationType
 * dispatch tables, then one header per activation family -- Relu.h, Gelu.h,
 * Tanh.h, Sigmoid.h, Linear.h -- plus the currently-unused VectorMath.h).
 * Split out so each activation's SIMD implementation is easier to find and
 * review in isolation; existing `#include <deepity/utils/Activations.h>`
 * call sites are unaffected.
 *
 * @code{.cpp}
 * #include <deepity/utils/Activations.h>
 *
 * Deep::tanh(array, arraysize)
 * @endcode
 *
 * @note Separate implementations exist for AVX512F, AVX2, SSE, and naive.
 *       Memory does NOT need to be aligned.
 * @version 1.1
 * @date 2026-08-23
 * @author Jack Rose
 */

#include <deepity/utils/activations/ActivationDispatch.h>
#include <deepity/utils/activations/VectorMath.h>
#include <deepity/utils/activations/Relu.h>
#include <deepity/utils/activations/Gelu.h>
#include <deepity/utils/activations/Tanh.h>
#include <deepity/utils/activations/Sigmoid.h>
#include <deepity/utils/activations/Linear.h>
