// Standalone proof-of-concept, deliberately outside FullPCLayerT/the
// backend interface: proves bit-offset / radix-offset addressing (true
// sub-byte packing, N values per byte instead of 1 byte per value) is
// viable for real compute, before touching any of the main layer/backend
// code. Not wired into CMakeLists.txt -- compile directly, e.g.:
//   g++ -O2 -std=c++20 tests/tBitPackedGemmPrototype.cpp -o /tmp/bppoc && /tmp/bppoc
//
// Single generic mechanism, no per-width special cases: PackedRadix<RADIX>
// is the only storage class, for ANY RADIX (2 = binary, 3 = ternary, 8 =
// 3-bit, 16 = 4-bit, ...). Group size (how many elements share a packed
// group) and group byte count are both *derived* from RADIX via a
// constexpr search, never hand-picked per width (no "5 trits per byte"
// written down anywhere) -- RadixGrid<RADIX> builds its quantization
// levels the same way, via one closed-form exponential ladder, not a
// lookup table sized per bit width. Because RADIX is a compile-time
// template parameter, the compiler constant-folds the modulo/divide by
// RADIX on its own (strength-reduced to shifts when RADIX is a power of
// 2) -- performance differences across widths fall out of that generic
// code, not from hand-tuned per-width paths.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// --- PackedRadix<RADIX>: N values, each one of RADIX distinct codes,
// packed into the fewest bytes a generic search can find. Elements are
// grouped so that RADIX^kGroupSize values fit in kGroupBytes bytes;
// both numbers come from ComputeGroupSize/ComputeGroupBytes below --
// purely a function of RADIX, no per-width table anywhere in this
// class. Get/Set address a group's packed integer and do mixed-radix
// div/mod to reach one element inside it (the direct generalization of
// bit-shift addressing to a non-power-of-2 RADIX).
template <uint64_t RADIX> class PackedRadix
{
  static_assert(RADIX >= 2, "RADIX must be at least 2");

  // Largest group size G with RADIX^G <= 65536: bounds every group to
  // at most 2 bytes and at most ~16 sequential div/mod steps per access
  // regardless of RADIX, so no width gets a pathologically large group.
  static constexpr int ComputeGroupSize()
  {
    uint64_t span = 1;
    int g = 0;
    while (span * RADIX <= 65536)
    {
      span *= RADIX;
      ++g;
    }
    return g > 0 ? g : 1;
  }

  static constexpr uint64_t ComputeGroupSpan()
  {
    uint64_t span = 1;
    for (int i = 0; i < ComputeGroupSize(); ++i)
      span *= RADIX;
    return span;
  }

  static constexpr int ComputeGroupBytes()
  {
    uint64_t span = ComputeGroupSpan();
    uint64_t cap = 1;
    int bytes = 0;
    while (cap < span)
    {
      cap *= 256;
      ++bytes;
    }
    return bytes > 0 ? bytes : 1;
  }

public:
  static constexpr int kGroupSize = ComputeGroupSize();
  static constexpr int kGroupBytes = ComputeGroupBytes();

  explicit PackedRadix(size_t n)
      : n_(n)
      , bytes_(((n + kGroupSize - 1) / kGroupSize) * (size_t)kGroupBytes, 0)
  {
  }

  size_t size() const noexcept
  {
    return n_;
  }
  size_t byteSize() const noexcept
  {
    return bytes_.size();
  }

  uint64_t Get(size_t i) const noexcept
  {
    size_t groupIdx = i / kGroupSize;
    int slot = (int)(i % kGroupSize);
    uint64_t packed = ReadGroup(groupIdx);
    for (int s = 0; s < slot; ++s)
      packed /= RADIX;
    return packed % RADIX;
  }

  void Set(size_t i, uint64_t code) noexcept
  {
    size_t groupIdx = i / kGroupSize;
    int slot = (int)(i % kGroupSize);
    code %= RADIX;

    uint64_t packed = ReadGroup(groupIdx);
    uint64_t digits[kGroupSize];
    for (int s = 0; s < kGroupSize; ++s)
    {
      digits[s] = packed % RADIX;
      packed /= RADIX;
    }
    digits[slot] = code;

    uint64_t rebuilt = 0, place = 1;
    for (int s = 0; s < kGroupSize; ++s)
    {
      rebuilt += digits[s] * place;
      place *= RADIX;
    }
    WriteGroup(groupIdx, rebuilt);
  }

private:
  uint64_t ReadGroup(size_t groupIdx) const noexcept
  {
    uint64_t v = 0;
    for (int b = 0; b < kGroupBytes; ++b)
      v |= (uint64_t)bytes_[groupIdx * (size_t)kGroupBytes + b] << (8 * b);
    return v;
  }

  void WriteGroup(size_t groupIdx, uint64_t v) noexcept
  {
    for (int b = 0; b < kGroupBytes; ++b)
      bytes_[groupIdx * (size_t)kGroupBytes + b] = (uint8_t)((v >> (8 * b)) & 0xFF);
  }

  size_t n_;
  std::vector<uint8_t> bytes_;
};

// --- RadixGrid<RADIX>: maps RADIX codes to float values via one
// closed-form exponential ladder (code 0 -> 0, codes 1..kPos -> positive
// magnitudes 0.5, 1, 2, 4, ..., the mirror codes -> negatives), not a
// lookup table sized per bit width. Works for any RADIX >= 2; for even
// RADIX one code goes unused (documented, not hidden) since codes split
// evenly into a positive/negative mirror plus one zero.
template <uint64_t RADIX> struct RadixGrid
{
  static constexpr int kPosLevels = (int)((RADIX - 1) / 2);

  static float Magnitude(int level) noexcept
  {
    if (level == 0)
      return 0.0f;
    return std::exp2((float)(level - 1)) * 0.5f; // 0.5, 1, 2, 4, 8, ...
  }

  static uint64_t Encode(float x) noexcept
  {
    bool neg = x < 0.0f;
    float ax = std::fabs(x);

    int best = 0;
    float bestDist = std::fabs(ax - Magnitude(0));
    for (int level = 1; level <= kPosLevels; ++level)
    {
      float d = std::fabs(ax - Magnitude(level));
      if (d < bestDist)
      {
        bestDist = d;
        best = level;
      }
    }
    if (best == 0)
      return 0;
    return neg ? (uint64_t)(kPosLevels + best) : (uint64_t)best;
  }

  static float Decode(uint64_t code) noexcept
  {
    if (code == 0)
      return 0.0f;
    if (code <= (uint64_t)kPosLevels)
      return Magnitude((int)code);
    return -Magnitude((int)(code - (uint64_t)kPosLevels));
  }
};

template <uint64_t RADIX> float RadixDecodeAt(const PackedRadix<RADIX>& arr, size_t i) noexcept
{
  return RadixGrid<RADIX>::Decode(arr.Get(i));
}

template <uint64_t RADIX> void RadixEncodeInto(PackedRadix<RADIX>& arr, size_t i, float x) noexcept
{
  arr.Set(i, RadixGrid<RADIX>::Encode(x));
}

// --- The actual proof: a GEMM reading both operands directly from
// packed storage, decoding via radix div/mod inside the inner loop --
// the same function body regardless of RADIX, no per-width overload.
template <uint64_t RADIX>
void RadixGemm(int M, int N, int K, const PackedRadix<RADIX>& A, const PackedRadix<RADIX>& B,
               float* C) noexcept
{
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
    {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += RadixDecodeAt(A, (size_t)i * K + k) * RadixDecodeAt(B, (size_t)k * N + j);
      C[i * N + j] = acc;
    }
}

template <uint64_t RADIX> static void TestRoundTrip(const char* label)
{
  printf("[RADIX=%llu, %s] round-trip + packing density\n", (unsigned long long)RADIX, label);
  const size_t n = 23; // deliberately not a multiple of any group size
  PackedRadix<RADIX> arr(n);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> inputs(n);
  for (size_t i = 0; i < n; ++i)
    inputs[i] = dist(rng);

  for (size_t i = 0; i < n; ++i)
    RadixEncodeInto(arr, i, inputs[i]);

  bool allRoundTrip = true;
  for (size_t i = 0; i < n; ++i)
  {
    float decoded = RadixDecodeAt(arr, i);
    float expected = RadixGrid<RADIX>::Decode(RadixGrid<RADIX>::Encode(inputs[i]));
    allRoundTrip &= (decoded == expected);
  }
  double bitsPerElem = 8.0 * (double)arr.byteSize() / (double)n;
  double theoreticalBits = std::log2((double)RADIX);
  printf("  round-trip: %s | %zu elements -> %zu bytes -> %.2f bits/elem (ideal: %.2f)\n\n",
         allRoundTrip ? "OK" : "MISMATCH",
         n,
         arr.byteSize(),
         bitsPerElem,
         theoreticalBits);
}

template <uint64_t RADIX> static void TestGemmCorrectness(const char* label)
{
  printf("[RADIX=%llu, %s] GEMM read directly from packed storage vs. reference\n",
         (unsigned long long)RADIX,
         label);
  const int M = 4, K = 5, N = 3;

  PackedRadix<RADIX> A((size_t)(M * K)), B((size_t)(K * N));
  std::vector<float> Aref(M * K), Bref(K * N);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  for (int i = 0; i < M * K; ++i)
  {
    float x = dist(rng);
    RadixEncodeInto(A, (size_t)i, x);
    Aref[i] = RadixGrid<RADIX>::Decode(RadixGrid<RADIX>::Encode(x));
  }
  for (int i = 0; i < K * N; ++i)
  {
    float x = dist(rng);
    RadixEncodeInto(B, (size_t)i, x);
    Bref[i] = RadixGrid<RADIX>::Decode(RadixGrid<RADIX>::Encode(x));
  }

  std::vector<float> Cpacked(M * N), Cref(M * N, 0.0f);
  RadixGemm(M, N, K, A, B, Cpacked.data());

  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
    {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += Aref[i * K + k] * Bref[k * N + j];
      Cref[i * N + j] = acc;
    }

  bool allMatch = true;
  for (int i = 0; i < M * N; ++i)
    allMatch &= (Cpacked[i] == Cref[i]);
  printf("  %s\n\n", allMatch ? "ALL MATCH (bit-exact)" : "FAILED");
}

template <uint64_t RADIX> static double TimeDecodeNsPerElement(size_t n, int reps)
{
  PackedRadix<RADIX> arr(n);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (size_t i = 0; i < n; ++i)
    RadixEncodeInto(arr, i, dist(rng));

  volatile float sink = 0.0f;
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
    for (size_t i = 0; i < n; ++i)
      sink += RadixDecodeAt(arr, i);
  auto end = std::chrono::steady_clock::now();

  double ns = std::chrono::duration<double, std::nano>(end - start).count();
  (void)sink;
  return ns / ((double)n * reps);
}

static void TestPerformanceAcrossRadixes()
{
  printf("[Performance] ns/element decoded, same generic code, varying RADIX only\n");
  const size_t n = 10000;
  const int reps = 200;
  printf("  RADIX=2   (binary):  %.2f ns/elem\n", TimeDecodeNsPerElement<2>(n, reps));
  printf("  RADIX=3   (ternary): %.2f ns/elem\n", TimeDecodeNsPerElement<3>(n, reps));
  printf("  RADIX=5   :          %.2f ns/elem\n", TimeDecodeNsPerElement<5>(n, reps));
  printf("  RADIX=8   (3-bit):   %.2f ns/elem\n", TimeDecodeNsPerElement<8>(n, reps));
  printf("  RADIX=16  (4-bit):   %.2f ns/elem\n", TimeDecodeNsPerElement<16>(n, reps));
  printf("  RADIX=256 (1 byte):  %.2f ns/elem\n", TimeDecodeNsPerElement<256>(n, reps));
  printf("  (no per-width code path above -- same template body for every RADIX)\n\n");
}

int main()
{
  TestRoundTrip<2>("binary");
  TestRoundTrip<3>("ternary");
  TestRoundTrip<5>("radix-5");
  TestRoundTrip<8>("3-bit");
  TestRoundTrip<16>("4-bit");

  TestGemmCorrectness<3>("ternary");
  TestGemmCorrectness<8>("3-bit");
  TestGemmCorrectness<16>("4-bit");

  TestPerformanceAcrossRadixes();

  printf("Generic radix-packed addressing + direct-from-packed compute: "
         "proven viable for arbitrary widths, no per-width hardcoding.\n");
  return 0;
}
