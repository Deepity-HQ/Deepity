#include "deepity/io/DType.h"
#include <algorithm>
#include <cstring>

namespace Deep::io
{
namespace
{
// Two 4-bit codes per byte, the first element in the low nibble.
void PackNibbles(const uint8_t* codes, size_t count, uint8_t* packed)
{
  for (size_t i = 0; i < count; i += 2)
    packed[i / 2] = static_cast<uint8_t>((codes[i] & 0x0F) | (i + 1 < count ? (codes[i + 1] & 0x0F) << 4 : 0));
}

void UnpackNibbles(const uint8_t* packed, size_t count, uint8_t* codes)
{
  for (size_t i = 0; i < count; ++i)
    codes[i] = (i % 2 == 0) ? (packed[i / 2] & 0x0F) : (packed[i / 2] >> 4);
}

constexpr DTypeInfo Std(const char* name, size_t bytes)
{
  return {name, DTypeKind::Standard, 1, bytes, bytes, name, nullptr, nullptr};
}

constexpr DTypeInfo Nibble(const char* name)
{
  return {name, DTypeKind::Packed, 2, 1, 1, "U8", PackNibbles, UnpackNibbles};
}

constexpr DTypeInfo kRegistry[] = {
    Std("F64", 8),  Std("F32", 4),  Std("F16", 2),  Std("BF16", 2), Std("I64", 8), Std("I32", 4),
    Std("I16", 2),  Std("I8", 1),   Std("U64", 8),  Std("U32", 4),  Std("U16", 2), Std("U8", 1),
    Std("BOOL", 1), Nibble("FP4_E2M1"), Nibble("POSIT4_0"), Nibble("POSIT4_1"), Nibble("POSIT4_2"),
};
} // namespace

const DTypeInfo* FindDType(const std::string& name) noexcept
{
  for (const auto& row : kRegistry)
    if (name == row.name)
      return &row;
  return nullptr;
}

size_t StorageBytes(const DTypeInfo& info, size_t elements) noexcept
{
  return (elements + info.elementsPerBlock - 1) / info.elementsPerBlock * info.bytesPerBlock;
}

void ConvertEndianness(void* data, size_t elemBytes, size_t count, bool hostIsBigEndian) noexcept
{
  if (!hostIsBigEndian || elemBytes < 2)
    return;
  auto* p = static_cast<uint8_t*>(data);
  for (size_t i = 0; i < count; ++i, p += elemBytes)
    std::reverse(p, p + elemBytes);
}
} // namespace Deep::io
