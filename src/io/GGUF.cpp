#include "deepity/io/GGUF.h"
#include "GGUFTables.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>

namespace Deep::io
{
namespace
{
constexpr uint64_t kMaxCount = 1u << 24;       // tensors or key-value pairs
constexpr uint64_t kMaxString = 64u << 20;     // bytes in one string
constexpr size_t kInlineArray = 256;
constexpr uint32_t kMaxDims = 8;

[[noreturn]] void Fail(const std::string& path, const std::string& why)
{
  throw GGUFError("GGUF: " + path + ": " + why);
}

float HalfToFloat(uint16_t h) noexcept
{
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
  if (exp == 0)
  {
    if (mant == 0) bits = sign;
    else
    {
      exp = 1;
      while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
      mant &= 0x3FF;
      bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
  }
  else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
  else bits = sign | ((exp + 112) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

float Bf16ToFloat(uint16_t b) noexcept
{
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

struct TypeInfo
{
  const char* name;
  size_t blockElems;
  size_t blockBytes;
  bool integer;
  bool decodable;
};

// Block sizes of ggml types; unknown ids get blockElems 0.
TypeInfo InfoOf(uint32_t id) noexcept
{
  switch (id)
  {
  case 0: return {"F32", 1, 4, false, true};
  case 1: return {"F16", 1, 2, false, true};
  case 2: return {"Q4_0", 32, 18, false, true};
  case 3: return {"Q4_1", 32, 20, false, true};
  case 6: return {"Q5_0", 32, 22, false, true};
  case 7: return {"Q5_1", 32, 24, false, true};
  case 8: return {"Q8_0", 32, 34, false, true};
  case 9: return {"Q8_1", 32, 40, false, false};
  case 10: return {"Q2_K", 256, 84, false, true};
  case 11: return {"Q3_K", 256, 110, false, true};
  case 12: return {"Q4_K", 256, 144, false, true};
  case 13: return {"Q5_K", 256, 176, false, true};
  case 14: return {"Q6_K", 256, 210, false, true};
  case 15: return {"Q8_K", 256, 292, false, true};
  case 16: return {"IQ2_XXS", 256, 66, false, true};
  case 17: return {"IQ2_XS", 256, 74, false, true};
  case 18: return {"IQ3_XXS", 256, 98, false, true};
  case 19: return {"IQ1_S", 256, 50, false, true};
  case 20: return {"IQ4_NL", 32, 18, false, true};
  case 21: return {"IQ3_S", 256, 110, false, true};
  case 22: return {"IQ2_S", 256, 82, false, true};
  case 23: return {"IQ4_XS", 256, 136, false, true};
  case 29: return {"IQ1_M", 256, 56, false, true};
  case 24: return {"I8", 1, 1, true, true};
  case 25: return {"I16", 1, 2, true, true};
  case 26: return {"I32", 1, 4, true, true};
  case 27: return {"I64", 1, 8, true, true};
  case 28: return {"F64", 1, 8, false, true};
  case 30: return {"BF16", 1, 2, false, true};
  default: return {nullptr, 0, 0, false, false};
  }
}

/// Bounds-checked cursor over the mapped file.
class Cursor
{
public:
  Cursor(const unsigned char* data, size_t size, const std::string& path) : data(data), size(size), path(path) {}
  size_t Pos() const noexcept { return pos; }
  void Need(uint64_t n) const
  {
    if (n > size - pos) Fail(path, "the file ends inside the header");
  }
  template <typename T> T Read()
  {
    Need(sizeof(T));
    T v;
    std::memcpy(&v, data + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  std::string String()
  {
    const uint64_t n = Read<uint64_t>();
    if (n > kMaxString) Fail(path, "a string is " + std::to_string(n) + " bytes long");
    Need(n);
    std::string s(reinterpret_cast<const char*>(data + pos), static_cast<size_t>(n));
    pos += static_cast<size_t>(n);
    return s;
  }
  void Skip(uint64_t n)
  {
    Need(n);
    pos += static_cast<size_t>(n);
  }

private:
  const unsigned char* data;
  size_t size;
  std::string path;
  size_t pos = 0;
};

size_t ScalarSize(uint32_t type) noexcept
{
  switch (type)
  {
  case 0: case 1: case 7: return 1;
  case 2: case 3: return 2;
  case 4: case 5: case 6: return 4;
  case 10: case 11: case 12: return 8;
  default: return 0; // 8 = string, 9 = array
  }
}

nlohmann::json ReadScalar(Cursor& c, uint32_t type)
{
  switch (type)
  {
  case 0: return c.Read<uint8_t>();
  case 1: return c.Read<int8_t>();
  case 2: return c.Read<uint16_t>();
  case 3: return c.Read<int16_t>();
  case 4: return c.Read<uint32_t>();
  case 5: return c.Read<int32_t>();
  case 6: return c.Read<float>();
  case 7: return c.Read<uint8_t>() != 0;
  case 8: return c.String();
  case 10: return c.Read<uint64_t>();
  case 11: return c.Read<int64_t>();
  case 12: return c.Read<double>();
  default: throw GGUFError("GGUF: unknown metadata value type " + std::to_string(type));
  }
}

nlohmann::json ReadValue(Cursor& c, uint32_t type, const std::string& path, int depth = 0)
{
  if (type != 9) return ReadScalar(c, type);
  if (depth > 0) Fail(path, "nested metadata arrays are not supported");
  const uint32_t inner = c.Read<uint32_t>();
  const uint64_t n = c.Read<uint64_t>();
  const size_t fixed = ScalarSize(inner);
  if (fixed != 0 && n > std::numeric_limits<uint64_t>::max() / fixed) Fail(path, "an array is impossibly long");
  if (fixed != 0) c.Need(n * fixed); // a count the file cannot hold is rejected before any allocation
  else if (inner != 8) Fail(path, "unsupported metadata array element type " + std::to_string(inner));
  else c.Need(n * 8);                // every string is at least its 8-byte length
  if (n > kInlineArray)
  {
    for (uint64_t i = 0; i < n; ++i) ReadScalar(c, inner);
    return {{"array_length", n}};
  }
  nlohmann::json out = nlohmann::json::array();
  for (uint64_t i = 0; i < n; ++i) out.push_back(ReadScalar(c, inner));
  return out;
}

/// Scale and minimum `j` (0..7) packed in the 12 scale bytes of Q4_K and Q5_K.
void ScaleMinK4(int j, const unsigned char* q, uint8_t* d, uint8_t* m) noexcept
{
  if (j < 4)
  {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  }
  else
  {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

float HalfAt(const unsigned char* q) noexcept
{
  uint16_t h;
  std::memcpy(&h, q, 2);
  return HalfToFloat(h);
}

/// Decodes one 256-element K-quant block, following ggml's reference dequantizers.
void DequantizeKBlock(uint32_t type, const unsigned char* p, float* y) noexcept
{
  switch (type)
  {
  case 10: // Q2_K: scales[16], qs[64], d, dmin
  {
    const float d = HalfAt(p + 80), mn = HalfAt(p + 82);
    const unsigned char* q = p + 16;
    int is = 0;
    for (int n = 0; n < 256; n += 128, q += 32)
    {
      int shift = 0;
      for (int j = 0; j < 4; ++j, shift += 2)
        for (int half = 0; half < 2; ++half)
        {
          const unsigned char sc = p[is++];
          const float dl = d * (sc & 0x0F), ml = mn * (sc >> 4);
          for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16 * half] >> shift) & 3) - ml;
        }
    }
    break;
  }
  case 11: // Q3_K: hmask[32], qs[64], scales[12], d
  {
    const float dAll = HalfAt(p + 108);
    uint32_t aux[4];
    std::memcpy(aux, p + 96, 12);
    aux[3] = 0;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
    int8_t scales[16];
    std::memcpy(scales, aux, 16);
    const unsigned char* hm = p;
    const unsigned char* q = p + 32;
    uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < 256; n += 128, q += 32)
    {
      int shift = 0;
      for (int j = 0; j < 4; ++j, shift += 2, m <<= 1)
        for (int half = 0; half < 2; ++half)
        {
          const float dl = dAll * (scales[is++] - 32);
          for (int l = 0; l < 16; ++l)
          {
            const int lo = (q[l + 16 * half] >> shift) & 3;
            *y++ = dl * (lo - ((hm[l + 16 * half] & m) ? 0 : 4));
          }
        }
    }
    break;
  }
  case 12: // Q4_K: d, dmin, scales[12], qs[128]
  {
    const float d = HalfAt(p), mn = HalfAt(p + 2);
    const unsigned char* sc = p + 4;
    const unsigned char* q = p + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64, q += 32, is += 2)
    {
      uint8_t s1, m1, s2, m2;
      ScaleMinK4(is, sc, &s1, &m1);
      ScaleMinK4(is + 1, sc, &s2, &m2);
      const float d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - mm1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - mm2;
    }
    break;
  }
  case 13: // Q5_K: d, dmin, scales[12], qh[32], qs[128]
  {
    const float d = HalfAt(p), mn = HalfAt(p + 2);
    const unsigned char* sc = p + 4;
    const unsigned char* qh = p + 16;
    const unsigned char* ql = p + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2)
    {
      uint8_t s1, m1, s2, m2;
      ScaleMinK4(is, sc, &s1, &m1);
      ScaleMinK4(is + 1, sc, &s2, &m2);
      const float d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
      for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - mm1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - mm2;
    }
    break;
  }
  case 14: // Q6_K: ql[128], qh[64], scales[16] (int8), d
  {
    const float d = HalfAt(p + 208);
    const unsigned char* ql = p;
    const unsigned char* qh = p + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
    for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8)
      for (int l = 0; l < 32; ++l)
      {
        const int is = l / 16;
        const int q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l] = d * sc[is] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
    break;
  }
  default: // Q8_K: float d, qs[256], bsums[16]
  {
    float d;
    std::memcpy(&d, p, 4);
    for (int j = 0; j < 256; ++j) y[j] = d * static_cast<int8_t>(p[4 + j]);
  }
  }
}

constexpr int8_t kValuesIQ4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

uint16_t U16At(const unsigned char* q) noexcept
{
  uint16_t v;
  std::memcpy(&v, q, 2);
  return v;
}

uint32_t U32At(const unsigned char* q) noexcept
{
  uint32_t v;
  std::memcpy(&v, q, 4);
  return v;
}

/// Writes `sign * scale * grid[0..7]`, where bit j of `signBits` set means negative.
void EmitSigned(float scale, const uint8_t* grid, unsigned signBits, float* y) noexcept
{
  for (int j = 0; j < 8; ++j) y[j] = scale * grid[j] * (((signBits >> j) & 1) ? -1.f : 1.f);
}

/// Decodes one block of an IQ (importance) format. Layouts and arithmetic follow the reference
/// numpy decoders in the `gguf` package, which this is tested against.
void DequantizeIQBlock(uint32_t type, const unsigned char* p, float* y) noexcept
{
  using namespace iq;
  switch (type)
  {
  case 16: // IQ2_XXS: d, 8 groups of (4 grid bytes, scale+signs word)
  {
    const float d = HalfAt(p);
    for (int g = 0; g < 8; ++g)
    {
      const unsigned char* q = p + 2 + 8 * g;
      const uint32_t aux = U32At(q + 4);
      const float db = d * (0.5f + (aux >> 28)) * 0.25f;
      for (int k = 0; k < 4; ++k, y += 8) EmitSigned(db, kGridIQ2XXS + 8 * q[k], kSignsIQ2[(aux >> (7 * k)) & 127], y);
    }
    break;
  }
  case 17: // IQ2_XS: d, qs[32] (uint16), scales[8]
  {
    const float d = HalfAt(p);
    const unsigned char* sc = p + 2 + 64;
    for (int e = 0; e < 32; ++e, y += 8)
    {
      const uint16_t q = U16At(p + 2 + 2 * e);
      const int s = (sc[e / 4] >> (4 * ((e / 2) & 1))) & 0x0F;
      EmitSigned(d * (0.5f + s) * 0.25f, kGridIQ2XS + 8 * (q & 511), kSignsIQ2[q >> 9], y);
    }
    break;
  }
  case 22: // IQ2_S: d, qs[32], signs[32], qh[8], scales[8]
  {
    const float d = HalfAt(p);
    const unsigned char *qs = p + 2, *signs = p + 34, *qh = p + 66, *sc = p + 74;
    for (int e = 0; e < 32; ++e, y += 8)
    {
      const int s = (sc[e / 4] >> (4 * ((e / 2) & 1))) & 0x0F;
      const int idx = qs[e] | (((qh[e / 4] >> (2 * (e % 4))) & 3) << 8);
      EmitSigned(d * (0.5f + s) * 0.25f, kGridIQ2S + 8 * idx, signs[e], y);
    }
    break;
  }
  case 18: // IQ3_XXS: d, qs[64], scale+signs words[8]
  {
    const float d = HalfAt(p);
    const unsigned char* qs = p + 2;
    for (int g = 0; g < 8; ++g)
    {
      const uint32_t aux = U32At(p + 2 + 64 + 4 * g);
      const float db = d * (0.5f + (aux >> 28)) * 0.5f;
      for (int k = 0; k < 4; ++k, y += 8)
      {
        const unsigned signBits = kSignsIQ2[(aux >> (7 * k)) & 127];
        for (int h = 0; h < 2; ++h)
          for (int j = 0; j < 4; ++j)
            y[4 * h + j] = db * kGridIQ3XXS[4 * qs[8 * g + 2 * k + h] + j] * (((signBits >> (4 * h + j)) & 1) ? -1.f : 1.f);
      }
    }
    break;
  }
  case 21: // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
  {
    const float d = HalfAt(p);
    const unsigned char *qs = p + 2, *qh = p + 66, *signs = p + 74, *sc = p + 106;
    for (int g = 0; g < 8; ++g)
    {
      const float db = d * (1 + 2 * ((sc[g / 2] >> (4 * (g % 2))) & 0x0F));
      for (int k = 0; k < 4; ++k, y += 8)
      {
        const unsigned signBits = signs[4 * g + k];
        for (int h = 0; h < 2; ++h)
        {
          const int e = 8 * g + 2 * k + h;
          const int idx = qs[e] | (((qh[e / 8] >> (e % 8)) & 1) << 8);
          for (int j = 0; j < 4; ++j)
            y[4 * h + j] = db * kGridIQ3S[4 * idx + j] * (((signBits >> (4 * h + j)) & 1) ? -1.f : 1.f);
        }
      }
    }
    break;
  }
  case 19: // IQ1_S: d, qs[32], qh[8] (uint16)
  {
    const float d = HalfAt(p);
    for (int g = 0; g < 8; ++g)
    {
      const uint16_t qh = U16At(p + 34 + 2 * g);
      const float dl = d * (2 * ((qh >> 12) & 7) + 1);
      const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
      for (int k = 0; k < 4; ++k, y += 8)
      {
        const int idx = p[2 + 4 * g + k] | (((qh >> (3 * k)) & 7) << 8);
        for (int j = 0; j < 8; ++j) y[j] = dl * (kGridIQ1S[8 * idx + j] + delta);
      }
    }
    break;
  }
  case 29: // IQ1_M: qs[32], qh[16], scales[4] (uint16); d is the scales' top nibbles
  {
    const unsigned char *qs = p, *qh = p + 32;
    uint16_t sc[4];
    for (int i = 0; i < 4; ++i) sc[i] = U16At(p + 48 + 2 * i);
    const uint16_t dBits = (sc[0] >> 12) | ((sc[1] >> 12) << 4) | ((sc[2] >> 12) << 8) | ((sc[3] >> 12) << 12);
    const float d = HalfToFloat(dBits);
    for (int e = 0; e < 32; ++e, y += 8)
    {
      const int scale = (sc[(e / 2) / 4] >> (3 * ((e / 2) % 4))) & 7;
      const int nib = (qh[e / 2] >> (4 * (e % 2))) & 0x0F;
      const float dl = d * (2 * scale + 1);
      const float delta = (nib & 8) ? -0.125f : 0.125f;
      const int idx = qs[e] | ((nib & 7) << 8);
      for (int j = 0; j < 8; ++j) y[j] = dl * (kGridIQ1S[8 * idx + j] + delta);
    }
    break;
  }
  case 20: // IQ4_NL: d, qs[16]
  {
    const float d = HalfAt(p);
    for (int j = 0; j < 16; ++j)
    {
      y[j] = d * kValuesIQ4[p[2 + j] & 0x0F];
      y[j + 16] = d * kValuesIQ4[p[2 + j] >> 4];
    }
    break;
  }
  default: // IQ4_XS: d, scales_h (uint16), scales_l[4], qs[128]
  {
    const float d = HalfAt(p);
    const uint16_t sh = U16At(p + 2);
    const unsigned char *sl = p + 4, *qs = p + 8;
    for (int g = 0; g < 8; ++g, y += 32)
    {
      const int ls = ((sl[g / 2] >> (4 * (g % 2))) & 0x0F) | (((sh >> (2 * g)) & 3) << 4);
      const float dl = d * (ls - 32);
      for (int j = 0; j < 16; ++j)
      {
        y[j] = dl * kValuesIQ4[qs[16 * g + j] & 0x0F];
        y[j + 16] = dl * kValuesIQ4[qs[16 * g + j] >> 4];
      }
    }
  }
  }
}

bool MulOverflows(uint64_t a, uint64_t b, uint64_t* out) noexcept
{
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return true;
  *out = a * b;
  return false;
}
} // namespace

size_t GGUFTensor::Elements() const noexcept
{
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return n;
}

GGUFReader::GGUFReader(const std::string& filePath) : path(filePath)
{
  // Header fields and block data are read in host order, and GGUF is little endian.
  if constexpr (std::endian::native != std::endian::little)
    Fail(filePath, "GGUF files can only be read on a little-endian host");
  file = MappedFile::Open(path);
  if (file.Data() == nullptr) Fail(path, "cannot open or map the file (missing, empty or unreadable)");
  Cursor c(file.Data(), file.Size(), path);
  char magic[4];
  c.Need(4);
  std::memcpy(magic, file.Data(), 4);
  c.Skip(4);
  if (std::memcmp(magic, "GGUF", 4) != 0) Fail(path, "not a GGUF file (bad magic)");
  version = c.Read<uint32_t>();
  if (version < 2 || version > 3) Fail(path, "unsupported GGUF version " + std::to_string(version) + " (this reader knows 2 and 3)");
  const uint64_t tensorCount = c.Read<uint64_t>(), kvCount = c.Read<uint64_t>();
  if (tensorCount > kMaxCount || kvCount > kMaxCount) Fail(path, "implausible tensor or key-value count");

  nlohmann::json meta = nlohmann::json::object();
  uint64_t alignment = 32;
  for (uint64_t i = 0; i < kvCount; ++i)
  {
    const std::string key = c.String();
    const uint32_t type = c.Read<uint32_t>();
    nlohmann::json value = ReadValue(c, type, path);
    if (key == "general.alignment")
    {
      if (!value.is_number_unsigned() || value.get<uint64_t>() == 0 || value.get<uint64_t>() > 4096 ||
          (value.get<uint64_t>() & (value.get<uint64_t>() - 1)) != 0)
        Fail(path, "general.alignment must be a power of two up to 4096");
      alignment = value.get<uint64_t>();
    }
    meta[key] = std::move(value);
  }
  // Strings in a hostile file may not be UTF-8; replace bad bytes instead of throwing a foreign exception.
  metadata = meta.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);

  struct Raw { GGUFTensor t; uint64_t relOffset; };
  std::vector<Raw> raws;
  raws.reserve(static_cast<size_t>(std::min<uint64_t>(tensorCount, 4096)));
  for (uint64_t i = 0; i < tensorCount; ++i)
  {
    Raw r;
    r.t.name = c.String();
    const uint32_t nDims = c.Read<uint32_t>();
    if (nDims == 0 || nDims > kMaxDims) Fail(path, "tensor '" + r.t.name + "' has " + std::to_string(nDims) + " dimensions");
    std::vector<uint64_t> ne(nDims);
    for (auto& d : ne) d = c.Read<uint64_t>();
    r.t.ggmlType = c.Read<uint32_t>();
    r.relOffset = c.Read<uint64_t>();
    const TypeInfo info = InfoOf(r.t.ggmlType);
    r.t.typeName = info.name != nullptr ? info.name : "type_" + std::to_string(r.t.ggmlType);
    if (info.blockElems == 0) Fail(path, "tensor '" + r.t.name + "' has unknown ggml type " + std::to_string(r.t.ggmlType));
    r.t.integer = info.integer;
    uint64_t elems = 1;
    for (uint64_t d : ne)
      if (d == 0 || MulOverflows(elems, d, &elems)) Fail(path, "tensor '" + r.t.name + "' has an empty or overflowing shape");
    if (ne[0] % info.blockElems != 0)
      Fail(path, "tensor '" + r.t.name + "' row length " + std::to_string(ne[0]) + " is not a multiple of the " + r.t.typeName + " block size");
    uint64_t bytes = 0;
    if (MulOverflows(elems / info.blockElems, info.blockBytes, &bytes)) Fail(path, "tensor '" + r.t.name + "' is impossibly large");
    r.t.byteSize = bytes;
    r.t.shape.assign(ne.rbegin(), ne.rend());
    raws.push_back(std::move(r));
  }

  const uint64_t dataStart = (c.Pos() + alignment - 1) / alignment * alignment;
  if (dataStart > file.Size()) Fail(path, "the file ends before the tensor data");
  std::map<std::string, bool> seen;
  for (auto& r : raws)
  {
    if (!seen.emplace(r.t.name, true).second) Fail(path, "duplicate tensor name '" + r.t.name + "'");
    if (r.relOffset % alignment != 0) Fail(path, "tensor '" + r.t.name + "' offset is not aligned");
    if (r.relOffset > file.Size() - dataStart || r.t.byteSize > file.Size() - dataStart - r.relOffset)
      Fail(path, "tensor '" + r.t.name + "' lies outside the file");
    r.t.offset = dataStart + r.relOffset;
    tensors.push_back(std::move(r.t));
  }
}

bool GGUFReader::CanDecode(const GGUFTensor& t) noexcept
{
  return InfoOf(t.ggmlType).decodable;
}

void GGUFReader::ReadFloat32(const GGUFTensor& t, float* dst) const
{
  const TypeInfo info = InfoOf(t.ggmlType);
  if (info.integer || !info.decodable)
    throw GGUFError("GGUF: tensor '" + t.name + "' has type " + t.typeName + ", which this reader cannot decode to float32");
  const unsigned char* p = file.Data() + t.offset;
  const size_t n = t.Elements();
  auto half = [](const unsigned char* q) { uint16_t h; std::memcpy(&h, q, 2); return HalfToFloat(h); };
  switch (t.ggmlType)
  {
  case 0: std::memcpy(dst, p, n * 4); break;
  case 28:
    for (size_t i = 0; i < n; ++i) { double v; std::memcpy(&v, p + i * 8, 8); dst[i] = static_cast<float>(v); }
    break;
  case 1:
    for (size_t i = 0; i < n; ++i) dst[i] = half(p + i * 2);
    break;
  case 30:
    for (size_t i = 0; i < n; ++i) { uint16_t b; std::memcpy(&b, p + i * 2, 2); dst[i] = Bf16ToFloat(b); }
    break;
  case 10: case 11: case 12: case 13: case 14: case 15:
    for (size_t b = 0; b < n / 256; ++b, p += info.blockBytes, dst += 256) DequantizeKBlock(t.ggmlType, p, dst);
    break;
  case 16: case 17: case 18: case 19: case 21: case 22: case 23: case 29:
    for (size_t b = 0; b < n / 256; ++b, p += info.blockBytes, dst += 256) DequantizeIQBlock(t.ggmlType, p, dst);
    break;
  case 20:
    for (size_t b = 0; b < n / 32; ++b, p += info.blockBytes, dst += 32) DequantizeIQBlock(20, p, dst);
    break;
  default:
    for (size_t b = 0; b < n / 32; ++b, p += info.blockBytes, dst += 32)
    {
      const float d = half(p);
      if (t.ggmlType == 8) // Q8_0
        for (int j = 0; j < 32; ++j) dst[j] = static_cast<int8_t>(p[2 + j]) * d;
      else if (t.ggmlType == 2 || t.ggmlType == 3) // Q4_0, Q4_1
      {
        const float m = t.ggmlType == 3 ? half(p + 2) : 0.f;
        const unsigned char* qs = p + (t.ggmlType == 3 ? 4 : 2);
        for (int j = 0; j < 16; ++j)
        {
          const int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
          dst[j] = t.ggmlType == 2 ? (lo - 8) * d : lo * d + m;
          dst[j + 16] = t.ggmlType == 2 ? (hi - 8) * d : hi * d + m;
        }
      }
      else // Q5_0, Q5_1
      {
        const float m = t.ggmlType == 7 ? half(p + 2) : 0.f;
        const unsigned char* q = p + (t.ggmlType == 7 ? 4 : 2);
        uint32_t qh;
        std::memcpy(&qh, q, 4);
        const unsigned char* qs = q + 4;
        for (int j = 0; j < 16; ++j)
        {
          const int x0 = (qs[j] & 0x0F) | static_cast<int>(((qh >> j) << 4) & 0x10);
          const int x1 = (qs[j] >> 4) | static_cast<int>((qh >> (j + 12)) & 0x10);
          dst[j] = t.ggmlType == 6 ? (x0 - 16) * d : x0 * d + m;
          dst[j + 16] = t.ggmlType == 6 ? (x1 - 16) * d : x1 * d + m;
        }
      }
    }
  }
}

void GGUFReader::ReadInt64(const GGUFTensor& t, int64_t* dst) const
{
  if (!t.integer) throw GGUFError("GGUF: tensor '" + t.name + "' is not an integer tensor");
  const unsigned char* p = file.Data() + t.offset;
  const size_t n = t.Elements();
  for (size_t i = 0; i < n; ++i)
    switch (t.ggmlType)
    {
    case 24: dst[i] = static_cast<int8_t>(p[i]); break;
    case 25: { int16_t v; std::memcpy(&v, p + i * 2, 2); dst[i] = v; break; }
    case 26: { int32_t v; std::memcpy(&v, p + i * 4, 4); dst[i] = v; break; }
    default: { int64_t v; std::memcpy(&v, p + i * 8, 8); dst[i] = v; }
    }
}
} // namespace Deep::io
