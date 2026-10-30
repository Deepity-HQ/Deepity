#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <initializer_list>
#include <string>

#include <deepity/utils/ActivationType.h>

/**
 * @file BindingHelpers.h
 * @brief Small helpers shared by more than one of the bind_*() translation
 * units (LayerBindings.cpp, NetworkBindings.cpp, UtilityBindings.cpp).
 * Anything used by only one of them lives locally in that .cpp instead.
 */

namespace nb = nanobind;

/// @brief "Any-shape, contiguous, CPU, float32" array, the nanobind
/// equivalent of pybind11's `py::array_t<float, py::array::c_style |
/// py::array::forcecast>`.
using FloatArray = nb::ndarray<float, nb::c_contig, nb::device::cpu>;

/// @brief Same as FloatArray, for int32 label/index arrays.
using IntArray = nb::ndarray<int, nb::c_contig, nb::device::cpu>;

/// @brief Generic counterpart to FloatArray/DoubleArray, for the
/// templated FullPCLayerT<T, AccumT>/FullPCNetworkT<T, AccumT>
/// bindings (BindFullPCLayerT/BindFullPCNetworkT), which bind once per
/// storage type T rather than being hardcoded to float.
template <typename T> using TArray = nb::ndarray<T, nb::c_contig, nb::device::cpu>;

namespace
{

/// @brief Allocates a new numpy array and copies `n` elements from `src`
/// into it, used where the source buffer doesn't outlive the call
/// (e.g. a local std::vector), so a zero-copy view isn't safe.
template <typename T>
nb::ndarray<nb::numpy, T> CopyToNewArray(const T* src, std::initializer_list<size_t> shape)
{
  size_t n = 1;
  for (auto s : shape)
    n *= s;
  T* data = new T[n];
  std::memcpy(data, src, n * sizeof(T));
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T>(data, shape, owner);
}

/// @brief Same as CopyToNewArray(), but for a layer/network whose
/// storage type T is lossy (e.g. Fp4E2M1Value) and whose Python-facing
/// wire type is AccumT instead: decodes each element via `AccumT(x)`
/// while copying, rather than a raw memcpy -- used for a quantized
/// FullPCLayerT<T, AccumT>/FullPCNetworkT<T, AccumT>'s .beliefs/.errors/
/// .weights/predict() results, which can't be a zero-copy view over T
/// (numpy has no dtype for these custom structs) and shouldn't expose
/// T's raw bits to Python even if it did.
template <typename T, typename AccumT>
nb::ndarray<nb::numpy, AccumT> DecodeCopyToNewArray(const T* src,
                                                    std::initializer_list<size_t> shape)
{
  size_t n = 1;
  for (auto s : shape)
    n *= s;
  AccumT* data = new AccumT[n];
  for (size_t i = 0; i < n; ++i)
    data[i] = AccumT(src[i]);
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<AccumT*>(p); });
  return nb::ndarray<nb::numpy, AccumT>(data, shape, owner);
}

/// @brief Maps a Python-facing activation name (e.g. "relu", "dgelu") to
/// its Deep::ActivationType enumerator, for layers/networks that
/// dispatch on the enum rather than a raw function pointer.
/// @return Deep::ActivationType::LINEAR if `act` matches nothing known.
Deep::ActivationType resolveActEnum(const std::string& act)
{
  if (act == "tanh")
    return Deep::ActivationType::TANH;
  if (act == "dtanh")
    return Deep::ActivationType::dTANH;
  if (act == "relu")
    return Deep::ActivationType::RELU;
  if (act == "drelu")
    return Deep::ActivationType::dRELU;
  if (act == "gelu")
    return Deep::ActivationType::GELU;
  if (act == "dgelu")
    return Deep::ActivationType::dGELU;
  if (act == "sigmoid")
    return Deep::ActivationType::SIGMOID;
  if (act == "dsigmoid")
    return Deep::ActivationType::dSIGMOID;
  if (act == "esigmoid")
    return Deep::ActivationType::eSIGMOID;
  if (act == "d_esigmoid")
    return Deep::ActivationType::d_eSIGMOID;
  if (act == "dlinear")
    return Deep::ActivationType::dLINEAR;
  if (act == "hard_tanh")
    return Deep::ActivationType::HARD_TANH;
  if (act == "dhard_tanh")
    return Deep::ActivationType::dHARD_TANH;
  return Deep::ActivationType::LINEAR;
}

} // namespace
