/**
 * @file QuantFormatBindings.cpp
 * @brief nanobind bindings for QuantFormat.h's sub-8-bit storage types
 * and their free Decode/Encode functions -- previously only reachable
 * transitively through a FullPCNetwork_fp4_e2m1/FullPCLayer_fp4_e2m1
 * instance's train_step()/predict(), never directly. Split out like
 * LayerBindings.cpp/NetworkBindings.cpp/UtilityBindings.cpp.
 */
#include "QuantFormatBindings.h"
#include "BindingHelpers.h"

#include <deepity/utils/QuantFormat.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace
{
/// @brief Binds one of the four 1-byte quantization structs (all share
/// the same shape: a raw `bits` field, a from-float constructor that
/// rounds to the nearest grid point, and float/double conversion), so
/// the four calls below don't each repeat this by hand.
template <typename QuantT>
void BindQuantType(nb::module_& m, const char* className, const char* formatDoc)
{
  nb::class_<QuantT>(m, className, formatDoc)
      .def(
          "__init__",
          [](QuantT* self, float value) { new (self) QuantT(value); },
          nb::arg("value"),
          "Rounds value to the nearest representable grid point (saturating at the "
          "format's max magnitude).")
      .def_rw("bits",
              &QuantT::bits,
              "Raw packed bit pattern (bit 3 = sign, bits 2:0 = "
              "magnitude index).")
      .def("__float__", [](const QuantT& self) { return (float)self; })
      .def("__repr__",
           [className](const QuantT& self)
           { return std::string(className) + "(" + std::to_string((float)self) + ")"; });
}
} // namespace

void bind_quant_formats(nb::module_& m)
{
  BindQuantType<Deep::Fp4E2M1Value>(
      m,
      "Fp4E2M1Value",
      "FP4 E2M1 (1 sign + 2 exponent + 1 mantissa bit) quantization grid value.");
  BindQuantType<Deep::Posit4_0Value>(
      m, "Posit4_0Value", "Posit<4,0> (es=0) quantization grid value.");
  BindQuantType<Deep::Posit4_1Value>(
      m, "Posit4_1Value", "Posit<4,1> (es=1) quantization grid value.");
  BindQuantType<Deep::Posit4_2Value>(
      m, "Posit4_2Value", "Posit<4,2> (es=2) quantization grid value.");

  m.def("decode_fp4_e2m1",
        &Deep::DecodeFp4E2M1,
        nb::arg("bits"),
        "Decodes a raw Fp4E2M1Value bit pattern (0-15) to its real value.");
  m.def("encode_fp4_e2m1",
        &Deep::EncodeFp4E2M1,
        nb::arg("value"),
        "Encodes a float to its nearest Fp4E2M1Value bit pattern.");
  m.def("decode_posit4_0",
        &Deep::DecodePosit4_0,
        nb::arg("bits"),
        "Decodes a raw Posit4_0Value bit pattern (0-15) to its real value.");
  m.def("encode_posit4_0",
        &Deep::EncodePosit4_0,
        nb::arg("value"),
        "Encodes a float to its nearest Posit4_0Value bit pattern.");
  m.def("decode_posit4_1",
        &Deep::DecodePosit4_1,
        nb::arg("bits"),
        "Decodes a raw Posit4_1Value bit pattern (0-15) to its real value.");
  m.def("encode_posit4_1",
        &Deep::EncodePosit4_1,
        nb::arg("value"),
        "Encodes a float to its nearest Posit4_1Value bit pattern.");
  m.def("decode_posit4_2",
        &Deep::DecodePosit4_2,
        nb::arg("bits"),
        "Decodes a raw Posit4_2Value bit pattern (0-15) to its real value.");
  m.def("encode_posit4_2",
        &Deep::EncodePosit4_2,
        nb::arg("value"),
        "Encodes a float to its nearest Posit4_2Value bit pattern.");

  // The grids themselves, for introspection -- each format's 8
  // non-negative magnitude levels (index 0 is always 0), mirroring
  // QuantFormat.h's own kFp4E2M1Magnitudes/kPosit4_*Magnitudes tables.
  m.attr("FP4_E2M1_MAGNITUDES") = std::vector<float>(std::begin(Deep::detail::kFp4E2M1Magnitudes),
                                                     std::end(Deep::detail::kFp4E2M1Magnitudes));
  m.attr("POSIT4_0_MAGNITUDES") = std::vector<float>(std::begin(Deep::detail::kPosit4_0Magnitudes),
                                                     std::end(Deep::detail::kPosit4_0Magnitudes));
  m.attr("POSIT4_1_MAGNITUDES") = std::vector<float>(std::begin(Deep::detail::kPosit4_1Magnitudes),
                                                     std::end(Deep::detail::kPosit4_1Magnitudes));
  m.attr("POSIT4_2_MAGNITUDES") = std::vector<float>(std::begin(Deep::detail::kPosit4_2Magnitudes),
                                                     std::end(Deep::detail::kPosit4_2Magnitudes));
}
