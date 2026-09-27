#pragma once

#include <nanobind/nanobind.h>

/**
 * @file LayerBindings.h
 * @brief Entry point for the Layer-side nanobind bindings (Layer and every
 * concrete PC layer type). See LayerBindings.cpp.
 */

/// @brief Binds Deep::Layer and every concrete layer subclass into `m`.
void bind_layers(nanobind::module_ &m);
