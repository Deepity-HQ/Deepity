#pragma once
#include <deepity/utils/QuantFormat.h>
#include <hwy/base.h>

/**
 * @file DTypeTag.h
 * @brief Maps a C++ storage type to its safetensors-style dtype string,
 * for ModelIO's save/load -- see that class's own doc comment on why
 * the dtype now gets authentically derived instead of hardcoded "F32".
 * Add a new specialization here as each new templated T lands (half,
 * the FP4/posit formats, ...); ModelIO itself needs no changes to pick
 * it up, since it reads this trait off the type it already has in hand
 * (a layer's GetStateDict() tensor's own element type).
 */

namespace Deep
{
template <typename T> struct DTypeTag;

template <> struct DTypeTag<float>
{
  static constexpr const char* Name() noexcept
  {
    return "F32";
  }
};

template <> struct DTypeTag<double>
{
  static constexpr const char* Name() noexcept
  {
    return "F64";
  }
};

template <> struct DTypeTag<Fp4E2M1Value>
{
  static constexpr const char* Name() noexcept
  {
    return "FP4_E2M1";
  }
};

template <> struct DTypeTag<hwy::float16_t>
{
  static constexpr const char* Name() noexcept
  {
    return "F16";
  }
};

template <> struct DTypeTag<Posit4_0Value>
{
  static constexpr const char* Name() noexcept
  {
    return "POSIT4_0";
  }
};

template <> struct DTypeTag<Posit4_1Value>
{
  static constexpr const char* Name() noexcept
  {
    return "POSIT4_1";
  }
};

template <> struct DTypeTag<Posit4_2Value>
{
  static constexpr const char* Name() noexcept
  {
    return "POSIT4_2";
  }
};
} // namespace Deep
