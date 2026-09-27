#pragma once

#include <nanobind/nanobind.h>

/**
 * @file UtilityBindings.h
 * @brief Entry point for the free-standing utility nanobind bindings
 * (StreamAlignedBatcher, activation functions, OpenMP/cache introspection).
 * See UtilityBindings.cpp.
 */

/// @brief Binds StreamAlignedBatcher and the free-standing utility
/// functions/activations into `m`.
void bind_utilities(nanobind::module_ &m);
