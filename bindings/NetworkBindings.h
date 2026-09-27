#pragma once

#include <nanobind/nanobind.h>

/**
 * @file NetworkBindings.h
 * @brief Entry point for the Network-side nanobind bindings (every concrete
 * PC network type). See NetworkBindings.cpp.
 */

/// @brief Binds every concrete PC network type into `m`.
void bind_networks(nanobind::module_ &m);
