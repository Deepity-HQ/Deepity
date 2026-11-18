#include <algorithm>
#include <bit>
#include <cstring>

#define HWY_TARGET_INCLUDE "io/Blake3.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "deepity/io/Blake3.h"

// The message words below are loaded in host byte order, but BLAKE3 defines them little endian. A
// big-endian build would write hashes no other machine can verify, so it must not compile until the
// loads are byte-swapped and tested on real hardware.
static_assert(std::endian::native == std::endian::little,
              "Blake3.cpp assumes a little-endian host; see docs/model-format.md, section Byte order");

HWY_BEFORE_NAMESPACE();
namespace Deep::io
{
namespace HWY_NAMESPACE
{
namespace hn = hwy::HWY_NAMESPACE;
using State = detail::Blake3State;

constexpr uint32_t kIv[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
constexpr uint8_t kSchedule[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};

constexpr uint32_t kChunkStart = 1u;
constexpr uint32_t kChunkEnd = 2u;
constexpr uint32_t kParent = 4u;
constexpr uint32_t kRoot = 8u;
constexpr size_t kMaxLanes = 16;
constexpr size_t kMaxBatches = 16;

template <int N, class D> HWY_INLINE hn::Vec<D> Rotr(D d, hn::Vec<D> x)
{
#if HWY_TARGET <= HWY_AVX3
  (void)d;
  return hn::RotateRight<N>(x);
#else
  if constexpr (N == 16)
  {
    const hn::Repartition<uint8_t, D> d8;
    const auto idx = hn::Dup128VecFromValues(d8, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
    return hn::BitCast(d, hn::TableLookupBytes(hn::BitCast(d8, x), idx));
  }
  else if constexpr (N == 8)
  {
    const hn::Repartition<uint8_t, D> d8;
    const auto idx = hn::Dup128VecFromValues(d8, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
    return hn::BitCast(d, hn::TableLookupBytes(hn::BitCast(d8, x), idx));
  }
  else
    return hn::RotateRight<N>(x);
#endif
}

template <class D>
HWY_INLINE void QRound(D d, hn::Vec<D>* w, int a, int b, int c, int e, hn::Vec<D> m0, hn::Vec<D> m1)
{
  w[a] = hn::Add(hn::Add(w[a], w[b]), m0);
  w[e] = Rotr<16, D>(d, hn::Xor(w[e], w[a]));
  w[c] = hn::Add(w[c], w[e]);
  w[b] = Rotr<12, D>(d, hn::Xor(w[b], w[c]));
  w[a] = hn::Add(hn::Add(w[a], w[b]), m1);
  w[e] = Rotr<8, D>(d, hn::Xor(w[e], w[a]));
  w[c] = hn::Add(w[c], w[e]);
  w[b] = Rotr<7, D>(d, hn::Xor(w[b], w[c]));
}

/// @brief One BLAKE3 compression across all lanes. `cv` goes in as the chaining value and comes
/// out as the new one (the first 8 output words), the layout every caller here needs.
template <class D>
HWY_INLINE void Compress(D d, hn::Vec<D>* cv, const hn::Vec<D>* m, hn::Vec<D> counterLow,
                         hn::Vec<D> counterHigh, hn::Vec<D> blockLen, hn::Vec<D> flags)
{
  hn::Vec<D> w[16];
  for (int i = 0; i < 8; ++i)
    w[i] = cv[i];
  for (int i = 0; i < 4; ++i)
    w[i + 8] = hn::Set(d, kIv[i]);
  w[12] = counterLow;
  w[13] = counterHigh;
  w[14] = blockLen;
  w[15] = flags;

  for (int r = 0; r < 7; ++r)
  {
    const uint8_t* s = kSchedule[r];
    QRound(d, w, 0, 4, 8, 12, m[s[0]], m[s[1]]);
    QRound(d, w, 1, 5, 9, 13, m[s[2]], m[s[3]]);
    QRound(d, w, 2, 6, 10, 14, m[s[4]], m[s[5]]);
    QRound(d, w, 3, 7, 11, 15, m[s[6]], m[s[7]]);
    QRound(d, w, 0, 5, 10, 15, m[s[8]], m[s[9]]);
    QRound(d, w, 1, 6, 11, 12, m[s[10]], m[s[11]]);
    QRound(d, w, 2, 7, 8, 13, m[s[12]], m[s[13]]);
    QRound(d, w, 3, 4, 9, 14, m[s[14]], m[s[15]]);
  }
  for (int i = 0; i < 8; ++i)
    cv[i] = hn::Xor(w[i], w[i + 8]);
}

/// @brief Loads one 64-byte block from each of `lanes` chunks and transposes it so m[j] holds
/// message word j of every chunk. Lane counts 8 and 4 use interleave transposes.
template <class D>
HWY_INLINE void LoadBlock(D d, const uint8_t* const* ptrs, size_t offset, hn::Vec<D>* m)
{
  using V = hn::Vec<D>;
  const hn::Repartition<uint8_t, D> d8;
  const hn::Repartition<uint64_t, D> d64;
  const size_t lanes = hn::Lanes(d);

  if (lanes == 8)
  {
    for (size_t half = 0; half < 2; ++half)
    {
      const size_t o = offset + half * 32;
      const V r0 = hn::BitCast(d, hn::LoadU(d8, ptrs[0] + o));
      const V r1 = hn::BitCast(d, hn::LoadU(d8, ptrs[1] + o));
      const V r2 = hn::BitCast(d, hn::LoadU(d8, ptrs[2] + o));
      const V r3 = hn::BitCast(d, hn::LoadU(d8, ptrs[3] + o));
      const V r4 = hn::BitCast(d, hn::LoadU(d8, ptrs[4] + o));
      const V r5 = hn::BitCast(d, hn::LoadU(d8, ptrs[5] + o));
      const V r6 = hn::BitCast(d, hn::LoadU(d8, ptrs[6] + o));
      const V r7 = hn::BitCast(d, hn::LoadU(d8, ptrs[7] + o));

      const auto t0 = hn::BitCast(d64, hn::InterleaveLower(d, r0, r1));
      const auto t1 = hn::BitCast(d64, hn::InterleaveUpper(d, r0, r1));
      const auto t2 = hn::BitCast(d64, hn::InterleaveLower(d, r2, r3));
      const auto t3 = hn::BitCast(d64, hn::InterleaveUpper(d, r2, r3));
      const auto t4 = hn::BitCast(d64, hn::InterleaveLower(d, r4, r5));
      const auto t5 = hn::BitCast(d64, hn::InterleaveUpper(d, r4, r5));
      const auto t6 = hn::BitCast(d64, hn::InterleaveLower(d, r6, r7));
      const auto t7 = hn::BitCast(d64, hn::InterleaveUpper(d, r6, r7));

      const V u0 = hn::BitCast(d, hn::InterleaveLower(d64, t0, t2));
      const V u1 = hn::BitCast(d, hn::InterleaveUpper(d64, t0, t2));
      const V u2 = hn::BitCast(d, hn::InterleaveLower(d64, t1, t3));
      const V u3 = hn::BitCast(d, hn::InterleaveUpper(d64, t1, t3));
      const V u4 = hn::BitCast(d, hn::InterleaveLower(d64, t4, t6));
      const V u5 = hn::BitCast(d, hn::InterleaveUpper(d64, t4, t6));
      const V u6 = hn::BitCast(d, hn::InterleaveLower(d64, t5, t7));
      const V u7 = hn::BitCast(d, hn::InterleaveUpper(d64, t5, t7));

      V* out = m + half * 8;
      out[0] = hn::ConcatLowerLower(d, u4, u0);
      out[1] = hn::ConcatLowerLower(d, u5, u1);
      out[2] = hn::ConcatLowerLower(d, u6, u2);
      out[3] = hn::ConcatLowerLower(d, u7, u3);
      out[4] = hn::ConcatUpperUpper(d, u4, u0);
      out[5] = hn::ConcatUpperUpper(d, u5, u1);
      out[6] = hn::ConcatUpperUpper(d, u6, u2);
      out[7] = hn::ConcatUpperUpper(d, u7, u3);
    }
    return;
  }

  if (lanes == 4)
  {
    for (size_t quarter = 0; quarter < 4; ++quarter)
    {
      const size_t o = offset + quarter * 16;
      const V r0 = hn::BitCast(d, hn::LoadU(d8, ptrs[0] + o));
      const V r1 = hn::BitCast(d, hn::LoadU(d8, ptrs[1] + o));
      const V r2 = hn::BitCast(d, hn::LoadU(d8, ptrs[2] + o));
      const V r3 = hn::BitCast(d, hn::LoadU(d8, ptrs[3] + o));

      const auto t0 = hn::BitCast(d64, hn::InterleaveLower(d, r0, r1));
      const auto t1 = hn::BitCast(d64, hn::InterleaveUpper(d, r0, r1));
      const auto t2 = hn::BitCast(d64, hn::InterleaveLower(d, r2, r3));
      const auto t3 = hn::BitCast(d64, hn::InterleaveUpper(d, r2, r3));

      V* out = m + quarter * 4;
      out[0] = hn::BitCast(d, hn::InterleaveLower(d64, t0, t2));
      out[1] = hn::BitCast(d, hn::InterleaveUpper(d64, t0, t2));
      out[2] = hn::BitCast(d, hn::InterleaveLower(d64, t1, t3));
      out[3] = hn::BitCast(d, hn::InterleaveUpper(d64, t1, t3));
    }
    return;
  }

  // Any other width (AVX-512's 16 lanes, scalar fallback): gather word by word.
  HWY_ALIGN uint32_t laneWords[kMaxLanes];
  for (int j = 0; j < 16; ++j)
  {
    for (size_t lane = 0; lane < lanes; ++lane)
      std::memcpy(&laneWords[lane], ptrs[lane] + offset + j * 4, sizeof(uint32_t));
    m[j] = hn::Load(d, laneWords);
  }
}

/// @brief Hashes up to `lanes` consecutive 1 KiB chunks at once, one chunk per lane.
/// Lanes beyond `valid` read chunk 0 and are ignored by the caller.
template <class D>
HWY_INLINE void HashChunks(D d, const uint8_t* data, size_t valid, uint64_t counter, hn::Vec<D>* cv)
{
  using V = hn::Vec<D>;
  const size_t lanes = hn::Lanes(d);

  HWY_ALIGN uint32_t counterLo[kMaxLanes];
  HWY_ALIGN uint32_t counterHi[kMaxLanes];
  const uint8_t* ptrs[kMaxLanes];
  for (size_t lane = 0; lane < lanes; ++lane)
  {
    counterLo[lane] = static_cast<uint32_t>(counter + lane);
    counterHi[lane] = static_cast<uint32_t>((counter + lane) >> 32);
    ptrs[lane] = data + (lane < valid ? lane * 1024 : 0);
  }

  const V counterLow = hn::Load(d, counterLo);
  const V counterHigh = hn::Load(d, counterHi);
  const V blockLen = hn::Set(d, 64u);
  for (int j = 0; j < 8; ++j)
    cv[j] = hn::Set(d, kIv[j]);

  V block[16];
  for (size_t offset = 0; offset < 1024; offset += 64)
  {
    LoadBlock(d, ptrs, offset, block);
    const uint32_t flags = (offset == 0 ? kChunkStart : 0u) | (offset == 960 ? kChunkEnd : 0u);
    Compress(d, cv, block, counterLow, counterHigh, blockLen, hn::Set(d, flags));
  }
}

/// @brief Merges neighbouring lanes of `lo` and `hi` into parent chaining values.
template <class D>
HWY_INLINE void MergeParents(D d, const hn::Vec<D>* lo, const hn::Vec<D>* hi, hn::Vec<D>* out)
{
  hn::Vec<D> block[16];
  hn::Vec<D> cv[8];
  for (int j = 0; j < 8; ++j)
  {
    block[j] = hn::ConcatEven(d, hi[j], lo[j]);
    block[j + 8] = hn::ConcatOdd(d, hi[j], lo[j]);
    cv[j] = hn::Set(d, kIv[j]);
  }
  Compress(d, cv, block, hn::Zero(d), hn::Zero(d), hn::Set(d, 64u), hn::Set(d, kParent));
  for (int j = 0; j < 8; ++j)
    out[j] = cv[j];
}

/// @brief Hashes batches * lanes aligned chunks into one subtree root.
template <class D>
HWY_INLINE void HashGroup(D d, const uint8_t* data, size_t batches, uint64_t counter, uint32_t* out)
{
  using V = hn::Vec<D>;
  const size_t lanes = hn::Lanes(d);

  V cvs[kMaxBatches][8];
  for (size_t b = 0; b < batches; ++b)
    HashChunks(d, data + b * lanes * 1024, lanes, counter + b * lanes, cvs[b]);
  for (size_t n = batches; n > 1; n /= 2)
    for (size_t i = 0; i < n / 2; ++i)
      MergeParents(d, cvs[2 * i], cvs[2 * i + 1], cvs[i]);
  for (size_t width = lanes; width > 1; width /= 2)
    MergeParents(d, cvs[0], cvs[0], cvs[0]);
  for (int j = 0; j < 8; ++j)
    out[j] = hn::GetLane(cvs[0][j]);
}

/// @brief One parent compression on scalar chaining values.
inline void MergePair(const uint32_t* left, const uint32_t* right, uint32_t* out)
{
  const hn::ScalableTag<uint32_t> d;
  using V = hn::Vec<decltype(d)>;
  V cv[8];
  V block[16];
  for (int j = 0; j < 8; ++j)
  {
    cv[j] = hn::Set(d, kIv[j]);
    block[j] = hn::Set(d, left[j]);
    block[j + 8] = hn::Set(d, right[j]);
  }
  Compress(d, cv, block, hn::Zero(d), hn::Zero(d), hn::Set(d, 64u), hn::Set(d, kParent));
  for (int j = 0; j < 8; ++j)
    out[j] = hn::GetLane(cv[j]);
}

/// @brief Pushes a finished subtree of 2^log2Chunks chunks onto the stack, merging equal-sized
/// neighbours the way the BLAKE3 tree requires.
inline void PushChainingValue(State& s, const uint32_t cv[8], unsigned log2Chunks)
{
  uint32_t current[8];
  std::memcpy(current, cv, sizeof(current));
  uint64_t total = (s.chunkCounter >> log2Chunks) + 1;
  while ((total & 1u) == 0)
  {
    --s.stackLen;
    uint32_t merged[8];
    MergePair(&s.cvStack[s.stackLen * 8], current, merged);
    std::memcpy(current, merged, sizeof(current));
    total >>= 1;
  }
  std::memcpy(&s.cvStack[s.stackLen * 8], current, sizeof(current));
  ++s.stackLen;
}

/// @brief Hashes the full chunk held in the buffer.
inline void ProcessChunk(State& s)
{
  const hn::ScalableTag<uint32_t> d;
  using V = hn::Vec<decltype(d)>;
  V cv[8];
  V block[16];
  for (int j = 0; j < 8; ++j)
    cv[j] = hn::Set(d, kIv[j]);
  const V counterLow = hn::Set(d, static_cast<uint32_t>(s.chunkCounter));
  const V counterHigh = hn::Set(d, static_cast<uint32_t>(s.chunkCounter >> 32));

  for (int i = 0; i < 1024; i += 64)
  {
    for (int j = 0; j < 16; ++j)
    {
      uint32_t val;
      std::memcpy(&val, s.chunkBuffer + i + j * 4, sizeof(val));
      block[j] = hn::Set(d, val);
    }
    const uint32_t flags = (i == 0 ? kChunkStart : 0u) | (i == 960 ? kChunkEnd : 0u);
    Compress(d, cv, block, counterLow, counterHigh, hn::Set(d, 64u), hn::Set(d, flags));
  }

  uint32_t result[8];
  for (int j = 0; j < 8; ++j)
    result[j] = hn::GetLane(cv[j]);
  PushChainingValue(s, result, 0);
}

/// @brief Hashes `numChunks` whole chunks straight from the caller's memory.
inline void ProcessChunksSimd(State& s, const uint8_t* data, size_t numChunks)
{
  const hn::ScalableTag<uint32_t> d;
  using V = hn::Vec<decltype(d)>;
  const size_t lanes = hn::Lanes(d);

  while (numChunks > 0)
  {
    if (numChunks >= lanes && (s.chunkCounter & (lanes - 1)) == 0)
    {
      size_t batches = 1;
      while (batches < kMaxBatches && numChunks >= 2 * batches * lanes &&
             (s.chunkCounter & (2 * batches * lanes - 1)) == 0)
        batches *= 2;

      uint32_t root[8];
      HashGroup(d, data, batches, s.chunkCounter, root);
      const size_t total = batches * lanes;
      PushChainingValue(s, root, static_cast<unsigned>(std::countr_zero(total)));
      s.chunkCounter += total;
      data += total * 1024;
      numChunks -= total;
      continue;
    }

    // Not lane-aligned yet: hash up to the next alignment point, one chunk at a time on the stack.
    const size_t toBoundary = lanes - (s.chunkCounter & (lanes - 1));
    const size_t count = std::min(numChunks, toBoundary);
    V cv[8];
    HashChunks(d, data, count, s.chunkCounter, cv);

    HWY_ALIGN uint32_t laneWords[kMaxLanes];
    uint32_t laneResults[kMaxLanes][8];
    for (int j = 0; j < 8; ++j)
    {
      hn::Store(cv[j], d, laneWords);
      for (size_t lane = 0; lane < count; ++lane)
        laneResults[lane][j] = laneWords[lane];
    }
    for (size_t lane = 0; lane < count; ++lane)
    {
      PushChainingValue(s, laneResults[lane], 0);
      ++s.chunkCounter;
    }
    data += count * 1024;
    numChunks -= count;
  }
}

void UpdateHwy(State& s, const uint8_t* ptr, size_t len)
{
  while (len > 0)
  {
    if (s.bufferLen == 1024)
    {
      ProcessChunk(s);
      ++s.chunkCounter;
      s.bufferLen = 0;
    }
    // The last chunk is never hashed early: it may need the ROOT flag when finalizing.
    if (s.bufferLen == 0 && len > 1024)
    {
      const size_t whole = (len - 1) / 1024;
      ProcessChunksSimd(s, ptr, whole);
      ptr += whole * 1024;
      len -= whole * 1024;
      continue;
    }
    const size_t take = std::min(len, 1024 - s.bufferLen);
    std::memcpy(s.chunkBuffer + s.bufferLen, ptr, take);
    s.bufferLen += take;
    ptr += take;
    len -= take;
  }
}

/// @brief Finishes a copy of the state and writes the 32-byte digest.
void FinalizeHwy(const State& in, uint8_t* out)
{
  const State s = in;
  const hn::ScalableTag<uint32_t> d;
  using V = hn::Vec<decltype(d)>;
  V cv[8];
  V block[16];
  for (int j = 0; j < 8; ++j)
    cv[j] = hn::Set(d, kIv[j]);

  const V counterLow = hn::Set(d, static_cast<uint32_t>(s.chunkCounter));
  const V counterHigh = hn::Set(d, static_cast<uint32_t>(s.chunkCounter >> 32));
  const size_t numBlocks = s.bufferLen == 0 ? 1 : (s.bufferLen + 63) / 64;

  for (size_t i = 0; i < numBlocks; ++i)
  {
    const size_t offset = i * 64;
    const size_t remaining = s.bufferLen > offset ? s.bufferLen - offset : 0;
    const size_t blockLen = std::min<size_t>(remaining, 64);

    uint8_t bytes[64] = {};
    if (blockLen > 0)
      std::memcpy(bytes, s.chunkBuffer + offset, blockLen);
    for (int j = 0; j < 16; ++j)
    {
      uint32_t val;
      std::memcpy(&val, bytes + j * 4, sizeof(val));
      block[j] = hn::Set(d, val);
    }

    const bool last = i + 1 == numBlocks;
    uint32_t flags = (i == 0 ? kChunkStart : 0u) | (last ? kChunkEnd : 0u);
    if (last && s.stackLen == 0)
      flags |= kRoot;
    Compress(d, cv, block, counterLow, counterHigh, hn::Set(d, static_cast<uint32_t>(blockLen)),
             hn::Set(d, flags));
  }

  for (size_t stack = s.stackLen; stack > 0;)
  {
    --stack;
    const uint32_t* left = &s.cvStack[stack * 8];
    for (int j = 0; j < 8; ++j)
    {
      block[j] = hn::Set(d, left[j]);
      block[j + 8] = cv[j];
      cv[j] = hn::Set(d, kIv[j]);
    }
    const uint32_t flags = kParent | (stack == 0 ? kRoot : 0u);
    Compress(d, cv, block, hn::Zero(d), hn::Zero(d), hn::Set(d, 64u), hn::Set(d, flags));
  }

  for (int j = 0; j < 8; ++j)
  {
    const uint32_t word = hn::GetLane(cv[j]);
    for (int k = 0; k < 4; ++k)
      out[j * 4 + k] = static_cast<uint8_t>(word >> (8 * k));
  }
}
} // namespace HWY_NAMESPACE
} // namespace Deep::io
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace Deep::io
{
HWY_EXPORT(UpdateHwy);
HWY_EXPORT(FinalizeHwy);

void Blake3::Update(const void* data, size_t len) noexcept
{
  HWY_DYNAMIC_DISPATCH(UpdateHwy)(state, static_cast<const uint8_t*>(data), len);
}

std::string Blake3::Hex() const
{
  uint8_t digest[32];
  HWY_DYNAMIC_DISPATCH(FinalizeHwy)(state, digest);
  constexpr char kHex[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (uint8_t byte : digest)
  {
    hex.push_back(kHex[byte >> 4]);
    hex.push_back(kHex[byte & 0x0F]);
  }
  return hex;
}
} // namespace Deep::io
#endif
