#pragma once
#include <cmath>
#include <cstdint>
#include <type_traits>

/**
 * @file QuantFormat.h
 * @brief Sub-8-bit storage types for the quantization plan's Phase 2+:
 * each format is its own 1-byte struct (no bit-packing yet, see the
 * plan's own "Known, accepted catch" section), with a free Decode/Encode
 * pair using the paper's grid table. No native CPU/GPU ALU exists for
 * these, so every use goes through a decode-to-AccumT/compute/encode-
 * back-to-T wrap -- CPUBackendT's own generic (non-float/double) code
 * paths already do exactly that (every op decodes via `AccumT(x)`
 * before computing and re-encodes via `T(result)` afterward), so these
 * structs only need to support that round-trip conversion themselves,
 * not any arithmetic of their own.
 *
 * Four separate structs are planned here (Fp4E2M1Value first, then the
 * three posit variants), not a `template<int nbits, int es>` -- each
 * format needs its own hand-tuned grid, and with only four callers a
 * shared abstraction isn't worth it (see the plan's own reasoning).
 */

namespace Deep
{
namespace detail
{
/// @brief FP4 E2M1's 8 non-negative magnitude levels (1 sign bit + 2
/// exponent bits + 1 mantissa bit, exponent bias 1): index 0 is the
/// denormal/zero row (0, 0.5), indices 1-3 are exponent 01 (1, 1.5),
/// indices 4-5 are exponent 10 (2, 3), indices 6-7 are exponent 11 (4,
/// 6) -- i.e. the magnitude bits read directly as a 0-7 table index.
inline constexpr float kFp4E2M1Magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

/// @brief Posit<4,0>'s 8 non-negative magnitude levels (es=0: no
/// exponent bits, so all 3 remaining bits go to the mantissa -- the
/// most uniform spacing near 1.0 of the three posit variants, at the
/// cost of the narrowest dynamic range).
inline constexpr float kPosit4_0Magnitudes[8] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 4.0f};

/// @brief Posit<4,1>'s 8 non-negative magnitude levels (es=1: one
/// exponent bit traded away from the mantissa for range -- every level
/// here is a power of two, unlike Posit4_0's).
inline constexpr float kPosit4_1Magnitudes[8] = {0.0f, 0.0625f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
                                                 16.0f};

/// @brief Posit<4,2>'s 8 non-negative magnitude levels (es=2: the
/// widest dynamic range of the three, at the cost of almost no
/// resolution near 1.0 -- only {1, 2} fall between its two widest gaps).
inline constexpr float kPosit4_2Magnitudes[8] = {0.0f, 1.0f / 256.0f, 0.0625f, 0.125f, 1.0f, 2.0f,
                                                 16.0f, 256.0f};
} // namespace detail

/// @brief Decodes an Fp4E2M1Value's raw 4-bit pattern (bit 3 = sign,
/// bits 2:0 = magnitude index into kFp4E2M1Magnitudes) to its real value.
inline float DecodeFp4E2M1(uint8_t bits) noexcept
{
  bool negative = (bits & 0x8) != 0;
  float magnitude = detail::kFp4E2M1Magnitudes[bits & 0x7];
  return negative ? -magnitude : magnitude;
}

/// @brief Encodes a float to the nearest Fp4E2M1Value grid point
/// (round-to-nearest over the 8 magnitude levels, saturating at ±6 --
/// the format's max representable magnitude -- for anything beyond
/// it). A genuine zero always encodes as bits=0 (never the sign-bit-set
/// "negative zero" pattern), so every all-zero buffer -- e.g. from
/// MemoryArenaT<Fp4E2M1Value>'s own Zero()/memset -- already is the
/// correct zero value with no separate init needed.
inline uint8_t EncodeFp4E2M1(float value) noexcept
{
  bool negative = value < 0.0f;
  float magnitude = std::fabs(value);

  uint8_t bestIdx = 0;
  float bestDist = std::fabs(magnitude - detail::kFp4E2M1Magnitudes[0]);
  for (uint8_t i = 1; i < 8; ++i)
  {
    float dist = std::fabs(magnitude - detail::kFp4E2M1Magnitudes[i]);
    if (dist < bestDist)
    {
      bestDist = dist;
      bestIdx = i;
    }
  }

  return (negative && bestIdx != 0) ? uint8_t(bestIdx | 0x8) : bestIdx;
}

/// @brief Storage type for the FP4 E2M1 format (see arXiv:2610.15933's
/// W4A4E4 QAT setup). Converts to/from any arithmetic AccumT via
/// Decode/EncodeFp4E2M1 (always through a `float` intermediate -- the
/// grid itself is only ever specified to float precision, so routing a
/// double AccumT through it loses nothing beyond what the 4-bit grid
/// already discards). The conversion constructor is explicit, since an
/// implicit float/double -> 4-bit-grid narrowing happening by accident
/// (rather than at a deliberate CPUBackendT decode/encode point) would
/// be exactly the kind of silent precision bug this whole templating
/// pass exists to make impossible to miss.
struct Fp4E2M1Value
{
  uint8_t bits = 0;

  Fp4E2M1Value() noexcept = default;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  explicit Fp4E2M1Value(U value) noexcept : bits(EncodeFp4E2M1(static_cast<float>(value)))
  {
  }

  explicit operator float() const noexcept
  {
    return DecodeFp4E2M1(bits);
  }
  explicit operator double() const noexcept
  {
    return double(DecodeFp4E2M1(bits));
  }
};

/// @brief Decodes a Posit4_0Value's raw 4-bit pattern (bit 3 = sign,
/// bits 2:0 = magnitude index into kPosit4_0Magnitudes) to its real value.
inline float DecodePosit4_0(uint8_t bits) noexcept
{
  bool negative = (bits & 0x8) != 0;
  float magnitude = detail::kPosit4_0Magnitudes[bits & 0x7];
  return negative ? -magnitude : magnitude;
}

/// @brief Encodes a float to the nearest Posit4_0Value grid point --
/// same round-to-nearest-with-clamping and canonical-zero rules as
/// EncodeFp4E2M1() above.
inline uint8_t EncodePosit4_0(float value) noexcept
{
  bool negative = value < 0.0f;
  float magnitude = std::fabs(value);

  uint8_t bestIdx = 0;
  float bestDist = std::fabs(magnitude - detail::kPosit4_0Magnitudes[0]);
  for (uint8_t i = 1; i < 8; ++i)
  {
    float dist = std::fabs(magnitude - detail::kPosit4_0Magnitudes[i]);
    if (dist < bestDist)
    {
      bestDist = dist;
      bestIdx = i;
    }
  }

  return (negative && bestIdx != 0) ? uint8_t(bestIdx | 0x8) : bestIdx;
}

/// @brief Storage type for the Posit<4,0> format (es=0). See
/// Fp4E2M1Value's own doc comment for the conversion-through-float,
/// explicit-constructor reasoning -- identical here.
struct Posit4_0Value
{
  uint8_t bits = 0;

  Posit4_0Value() noexcept = default;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  explicit Posit4_0Value(U value) noexcept : bits(EncodePosit4_0(static_cast<float>(value)))
  {
  }

  explicit operator float() const noexcept
  {
    return DecodePosit4_0(bits);
  }
  explicit operator double() const noexcept
  {
    return double(DecodePosit4_0(bits));
  }
};

/// @brief Decodes a Posit4_1Value's raw 4-bit pattern (bit 3 = sign,
/// bits 2:0 = magnitude index into kPosit4_1Magnitudes) to its real value.
inline float DecodePosit4_1(uint8_t bits) noexcept
{
  bool negative = (bits & 0x8) != 0;
  float magnitude = detail::kPosit4_1Magnitudes[bits & 0x7];
  return negative ? -magnitude : magnitude;
}

/// @brief Encodes a float to the nearest Posit4_1Value grid point --
/// same round-to-nearest-with-clamping and canonical-zero rules as
/// EncodeFp4E2M1() above.
inline uint8_t EncodePosit4_1(float value) noexcept
{
  bool negative = value < 0.0f;
  float magnitude = std::fabs(value);

  uint8_t bestIdx = 0;
  float bestDist = std::fabs(magnitude - detail::kPosit4_1Magnitudes[0]);
  for (uint8_t i = 1; i < 8; ++i)
  {
    float dist = std::fabs(magnitude - detail::kPosit4_1Magnitudes[i]);
    if (dist < bestDist)
    {
      bestDist = dist;
      bestIdx = i;
    }
  }

  return (negative && bestIdx != 0) ? uint8_t(bestIdx | 0x8) : bestIdx;
}

/// @brief Storage type for the Posit<4,1> format (es=1). See
/// Fp4E2M1Value's own doc comment for the conversion-through-float,
/// explicit-constructor reasoning -- identical here.
struct Posit4_1Value
{
  uint8_t bits = 0;

  Posit4_1Value() noexcept = default;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  explicit Posit4_1Value(U value) noexcept : bits(EncodePosit4_1(static_cast<float>(value)))
  {
  }

  explicit operator float() const noexcept
  {
    return DecodePosit4_1(bits);
  }
  explicit operator double() const noexcept
  {
    return double(DecodePosit4_1(bits));
  }
};

/// @brief Decodes a Posit4_2Value's raw 4-bit pattern (bit 3 = sign,
/// bits 2:0 = magnitude index into kPosit4_2Magnitudes) to its real value.
inline float DecodePosit4_2(uint8_t bits) noexcept
{
  bool negative = (bits & 0x8) != 0;
  float magnitude = detail::kPosit4_2Magnitudes[bits & 0x7];
  return negative ? -magnitude : magnitude;
}

/// @brief Encodes a float to the nearest Posit4_2Value grid point --
/// same round-to-nearest-with-clamping and canonical-zero rules as
/// EncodeFp4E2M1() above.
inline uint8_t EncodePosit4_2(float value) noexcept
{
  bool negative = value < 0.0f;
  float magnitude = std::fabs(value);

  uint8_t bestIdx = 0;
  float bestDist = std::fabs(magnitude - detail::kPosit4_2Magnitudes[0]);
  for (uint8_t i = 1; i < 8; ++i)
  {
    float dist = std::fabs(magnitude - detail::kPosit4_2Magnitudes[i]);
    if (dist < bestDist)
    {
      bestDist = dist;
      bestIdx = i;
    }
  }

  return (negative && bestIdx != 0) ? uint8_t(bestIdx | 0x8) : bestIdx;
}

/// @brief Storage type for the Posit<4,2> format (es=2). See
/// Fp4E2M1Value's own doc comment for the conversion-through-float,
/// explicit-constructor reasoning -- identical here.
struct Posit4_2Value
{
  uint8_t bits = 0;

  Posit4_2Value() noexcept = default;

  template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
  explicit Posit4_2Value(U value) noexcept : bits(EncodePosit4_2(static_cast<float>(value)))
  {
  }

  explicit operator float() const noexcept
  {
    return DecodePosit4_2(bits);
  }
  explicit operator double() const noexcept
  {
    return double(DecodePosit4_2(bits));
  }
};
} // namespace Deep
