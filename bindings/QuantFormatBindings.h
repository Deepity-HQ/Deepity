#pragma once

#include <nanobind/nanobind.h>

/**
 * @file QuantFormatBindings.h
 * @brief Entry point for the sub-8-bit quantization format bindings
 * (Fp4E2M1Value, Posit4_0/1/2Value, and their free Decode/Encode
 * functions). See QuantFormatBindings.cpp.
 */

/// @brief Binds QuantFormat.h's storage types and Decode/Encode
/// functions into `m`.
void bind_quant_formats(nanobind::module_& m);
