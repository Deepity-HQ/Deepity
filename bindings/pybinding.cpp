/**
 * @file pybinding.cpp
 * @brief Entry point for the `deepity` nanobind module (distributed on
 * PyPI as the `pydeepity` package, see pyproject.toml -- distribution
 * name and import name are deliberately different, same convention as
 * e.g. scikit-learn/sklearn or opencv-python/cv2). The actual bindings
 * live in LayerBindings.cpp, NetworkBindings.cpp, and UtilityBindings.cpp,
 * this file only wires them together.
 */
#include <nanobind/nanobind.h>

#include "LayerBindings.h"
#include "NetworkBindings.h"
#include "UtilityBindings.h"

NB_MODULE(deepity, m)
{
  m.doc() = "Deepity: A high-performance Predictive Coding library.";

  bind_layers(m);
  bind_networks(m);
  bind_utilities(m);
}
