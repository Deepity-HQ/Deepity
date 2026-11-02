#pragma once

#include <nanobind/nanobind.h>

/**
 * @file AttentionBindings.h
 * @brief Entry point for the AttentionScoresLayer/AttentionOutputLayer
 * nanobind bindings. See AttentionBindings.cpp.
 */

/// @brief Binds Deep::AttentionScoresLayer and Deep::AttentionOutputLayer into `m`.
void bind_attention_layers(nanobind::module_& m);
