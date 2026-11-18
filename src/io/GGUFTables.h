#pragma once
#include <cstdint>

/// Lookup tables of the IQ (importance) GGUF quant formats, one row of values per grid index.
namespace Deep::io::iq
{
extern const uint8_t kGridIQ2XXS[256 * 8];
extern const uint8_t kGridIQ2XS[512 * 8];
extern const uint8_t kGridIQ2S[1024 * 8];
extern const uint8_t kGridIQ3XXS[256 * 4];
extern const uint8_t kGridIQ3S[512 * 4];
extern const int8_t kGridIQ1S[2048 * 8];
extern const uint8_t kSignsIQ2[128];
} // namespace Deep::io::iq
