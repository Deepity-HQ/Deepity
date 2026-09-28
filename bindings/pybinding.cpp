/**
 * @file pybinding.cpp
 * @brief Entry point for the `pydeepity` nanobind module. The actual
 * bindings live in LayerBindings.cpp, NetworkBindings.cpp, and
 * UtilityBindings.cpp, this file only wires them together.
 */
#include <nanobind/nanobind.h>

#include "LayerBindings.h"
#include "NetworkBindings.h"
#include "UtilityBindings.h"

NB_MODULE(pydeepity, m)
{
    m.doc() = "Deepity: A high-performance Predictive Coding library.";

    bind_layers(m);
    bind_networks(m);
    bind_utilities(m);
}
