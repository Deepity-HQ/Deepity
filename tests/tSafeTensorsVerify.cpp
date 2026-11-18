/**
 * @file tSafeTensorsVerify.cpp
 * @brief Tests for the standalone safetensors reader/writer (include/deepity/io/SafeTensors.h).
 *
 * Covers: BLAKE3 vectors, round trips, alignment, atomic writes, hash verification, rejection of
 * every kind of malformed file, a checked-in golden file (v1) and a file written by 0.1.0a8.
 * Run with `--write-golden <path>` or `--write-golden-dtypes <path>` to regenerate a golden file.
 */
#include "deepity/io/Blake3.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deepity/io/SafeTensors.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace Deep::io;
namespace fs = std::filesystem;

#ifndef DEEPITY_TEST_DATA_DIR
#define DEEPITY_TEST_DATA_DIR "tests/data"
#endif

namespace
{
int failures = 0;

void Check(bool ok, const std::string& what)
{
  printf("  [%s] %s\n", ok ? "PASSED" : "FAILED", what.c_str());
  failures += ok ? 0 : 1;
}

template <typename F> bool Throws(F&& f)
{
  try
  {
    f();
  }
  catch (const SafeTensorsError&)
  {
    return true;
  }
  return false;
}

std::string Tmp(const std::string& name)
{
  return (fs::temp_directory_path() / ("deepity_st_" + name)).string();
}

std::vector<char> Slurp(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void Spit(const std::string& path, const std::vector<char>& bytes)
{
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/// Builds a file from a raw header string and data bytes, bypassing the writer's checks.
void WriteRaw(const std::string& path, const std::string& header, size_t dataBytes)
{
  std::vector<char> f(8 + header.size() + dataBytes, 1);
  const uint64_t n = header.size();
  std::memcpy(f.data(), &n, 8);
  std::memcpy(f.data() + 8, header.data(), header.size());
  Spit(path, f);
}

struct Sample
{
  std::vector<float> a = std::vector<float>(15);
  std::vector<uint16_t> b = std::vector<uint16_t>(7);
  std::vector<uint8_t> c = std::vector<uint8_t>(11);
  std::vector<double> d = std::vector<double>(4);
  Sample()
  {
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = 0.5f * static_cast<float>(i) - 3.0f;
    for (size_t i = 0; i < b.size(); ++i)
      b[i] = static_cast<uint16_t>(100 + i);
    for (size_t i = 0; i < c.size(); ++i)
      c[i] = static_cast<uint8_t>(i * 7);
    for (size_t i = 0; i < d.size(); ++i)
      d[i] = 1.0 / static_cast<double>(i + 1);
  }
  void Fill(SafeTensorsWriter& w) const
  {
    // Deliberately small-to-large, so the writer's reordering is exercised.
    w.AddTensor("bytes", "U8", {11}, c.data(), c.size());
    w.AddTensor("half", "F16", {7}, b.data(), b.size() * 2);
    w.AddTensor("empty", "F32", {0, 4}, nullptr, 0);
    w.AddTensor("mat", "F32", {3, 5}, a.data(), a.size() * 4);
    w.AddTensor("wide", "F64", {2, 2}, d.data(), d.size() * 8);
    w.SetMetadata("deepity.format_version", "1");
    w.SetMetadata("deepity.kind", "inference");
  }
};

// One 5-element tensor for every dtype name the format defines. The names are listed here on purpose,
// not read from the registry, so removing or renaming a dtype breaks this test and the golden file.
const char* const kGoldenDTypes[] = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U64", "U32", "U16",
                                     "U8",  "BOOL", "FP4_E2M1", "POSIT4_0", "POSIT4_1", "POSIT4_2"};

std::vector<uint8_t> GoldenBytes(const DTypeInfo& info, size_t index)
{
  std::vector<uint8_t> bytes(StorageBytes(info, 5));
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<uint8_t>(i * 37 + 11 + index);
  if (info.kind == DTypeKind::Packed) bytes.back() &= 0x0F; // padding nibble of the odd last block is zero
  return bytes;
}

void WriteGoldenDTypes(const std::string& path)
{
  SafeTensorsWriter w;
  std::vector<std::vector<uint8_t>> keep; // the writer holds pointers until Write()
  keep.reserve(std::size(kGoldenDTypes));
  for (const char* name : kGoldenDTypes)
  {
    keep.push_back(GoldenBytes(*FindDType(name), keep.size()));
    w.AddTensor(std::string("t_") + name, name, {5}, keep.back().data(), keep.back().size());
  }
  w.SetMetadata("deepity.format_version", "1");
  w.SetMetadata("deepity.kind", "inference");
  w.Write(path);
}

struct SimulatedBigEndianHost
{
  SimulatedBigEndianHost() { SimulateBigEndianHost(true); }
  ~SimulatedBigEndianHost() { SimulateBigEndianHost(false); }
};

/// @brief The little-endian bytes of a golden tensor as a big-endian host holds them in memory.
std::vector<uint8_t> AsBigEndianMemory(const DTypeInfo& info, std::vector<uint8_t> bytes)
{
  if (info.kind == DTypeKind::Standard && info.memBytes > 1)
    for (size_t e = 0; e < 5; ++e)
      std::reverse(bytes.begin() + e * info.memBytes, bytes.begin() + (e + 1) * info.memBytes);
  return bytes;
}

/// @brief No big-endian hardware exists to run on, so SimulateBigEndianHost() drives the real reader
/// and writer as if the host were one: files must come out byte-identical to the little-endian
/// host's, and reads must hand back big-endian memory.
void TestBigEndianHost()
{
  printf("--- simulated big-endian host ---\n");
  std::vector<std::vector<uint8_t>> little, big;
  for (const char* name : kGoldenDTypes)
  {
    const DTypeInfo& info = *FindDType(name);
    little.push_back(GoldenBytes(info, little.size()));
    big.push_back(AsBigEndianMemory(info, little.back()));
  }
  auto fill = [&](SafeTensorsWriter& w, const std::vector<std::vector<uint8_t>>& memory)
  {
    size_t i = 0;
    for (const char* name : kGoldenDTypes)
    {
      w.AddTensor(std::string("t_") + name, name, {5}, memory[i].data(), memory[i].size());
      ++i;
    }
  };

  const std::string plain = Tmp("le_host.safetensors"), simulated = Tmp("be_host.safetensors");
  {
    SafeTensorsWriter w;
    fill(w, little);
    w.Write(plain);
  }
  {
    SimulatedBigEndianHost host;
    SafeTensorsWriter w;
    fill(w, big);
    w.Write(simulated);
  }
  const auto plainBytes = Slurp(plain);
  Check(plainBytes == Slurp(simulated), "a big-endian host writes the same file, hash included");

  uint64_t headerSize = 0;
  for (int i = 7; i >= 0; --i)
    headerSize = (headerSize << 8) | static_cast<unsigned char>(plainBytes[i]);
  Check(8 + headerSize < plainBytes.size() && plainBytes[8] == '{',
        "the header length field is little endian");

  auto readsBack = [&](bool mmap, bool simulate, const std::vector<std::vector<uint8_t>>& want)
  {
    SimulatedBigEndianHost* host = simulate ? new SimulatedBigEndianHost : nullptr;
    SafeTensorsReader r(plain, true, mmap);
    bool ok = true;
    size_t i = 0;
    for (const char* name : kGoldenDTypes)
    {
      const TensorEntry* t = r.Find(std::string("t_") + name);
      std::vector<uint8_t> got(want[i].size());
      r.Read(*t, got.data(), got.size());
      const DTypeInfo& info = *FindDType(name);
      const bool swapped = simulate && !t->packed && info.memBytes > 1;
      // Zero-copy views are raw file bytes, so a host that must swap gets none for multi-byte types.
      const bool viewExpected = mmap && !swapped;
      ok &= got == want[i] && (r.Bytes(*t) != nullptr) == viewExpected;
      ++i;
    }
    delete host;
    return ok;
  };
  Check(readsBack(true, false, little), "little-endian host, mapped: contents and zero-copy views");
  Check(readsBack(false, false, little), "little-endian host, streamed: contents");
  Check(readsBack(true, true, big), "big-endian host, mapped: contents are swapped, no view for multi-byte types");
  Check(readsBack(false, true, big), "big-endian host, streamed: contents are swapped");

  const std::string dirA = Tmp("shards_le"), dirB = Tmp("shards_be");
  for (const auto& d : {dirA, dirB})
  {
    fs::remove_all(d);
    fs::create_directories(d);
  }
  {
    SafeTensorsWriter w;
    fill(w, little);
    w.WriteSharded(dirA + "/m.safetensors.index.json", 64);
  }
  {
    SimulatedBigEndianHost host;
    SafeTensorsWriter w;
    fill(w, big);
    w.WriteSharded(dirB + "/m.safetensors.index.json", 64);
  }
  std::vector<std::string> names;
  for (const auto& e : fs::directory_iterator(dirA))
    names.push_back(e.path().filename().string());
  bool sameShards = names.size() > 1;
  for (const auto& n : names)
    sameShards &= fs::exists(dirB + "/" + n) && Slurp(dirA + "/" + n) == Slurp(dirB + "/" + n);
  Check(sameShards, "a big-endian host writes the same shard set (" + std::to_string(names.size()) + " files)");
  {
    SimulatedBigEndianHost host;
    SafeTensorsReader r(dirA + "/m.safetensors.index.json");
    bool ok = true;
    size_t i = 0;
    for (const char* name : kGoldenDTypes)
    {
      std::vector<uint8_t> got(big[i].size());
      r.Read(*r.Find(std::string("t_") + name), got.data(), got.size());
      ok &= got == big[i++];
    }
    Check(ok, "big-endian host reads a shard set swapped");
  }
  for (const auto& d : {dirA, dirB})
    fs::remove_all(d);
  fs::remove(plain);
  fs::remove(simulated);
}

void TestBlake3()
{
  printf("--- blake3 ---\n");
  // Vectors from the reference implementation (Python blake3 1.0.11) over bytes i % 251.
  const struct { size_t n; const char* hex; } vectors[] = {
      {0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
      {1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
      {63, "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"},
      {64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"},
      {65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"},
      {1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"},
      {1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
      {1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"},
      {2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"},
      {3073, "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"},
      {8192, "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63"},
      {100000, "d93c23eedaf165a7e0be908ba86f1a7a520d568d2d13cde787c8580c5c72cc54"},
      {1048577, "2f053cd7472cf0cd2f9adaf45c1180255b91b9a865404a63671a0ee5f792ed33"},
  };
  for (const auto& v : vectors)
  {
    std::vector<uint8_t> data(v.n);
    for (size_t i = 0; i < v.n; ++i)
      data[i] = static_cast<uint8_t>(i % 251);
    Blake3 whole;
    whole.Update(data.data(), v.n);
    Check(whole.Hex() == v.hex, "BLAKE3 of " + std::to_string(v.n) + " bytes, one Update");
    Blake3 split; // odd, growing piece sizes exercise the buffer and alignment paths
    size_t pos = 0, step = 1;
    while (pos < v.n)
    {
      const size_t take = std::min(step, v.n - pos);
      split.Update(data.data() + pos, take);
      pos += take;
      step = step * 3 + 7;
    }
    Check(split.Hex() == v.hex, "BLAKE3 of " + std::to_string(v.n) + " bytes, split Updates");
    Check(whole.Hex() == whole.Hex(), "Hex() does not change the state");
  }
}

void TestRoundTrip()
{
  printf("--- round trip, layout, atomicity ---\n");
  const std::string path = Tmp("rt.safetensors");
  Sample s;
  SafeTensorsWriter w;
  s.Fill(w);
  w.Write(path);
  Check(!fs::exists(path + ".tmp"), "no temporary file left behind");

  SafeTensorsReader r(path);
  Check(r.Tensors().size() == 5, "five tensors read back");
  Check(r.DataOffset() % 64 == 0, "data region starts on a 64-byte boundary");
  Check(r.Metadata().at("deepity.kind") == "inference" && r.Metadata().count(kHashKey) == 1,
        "metadata present, hash written by default");

  bool aligned = true, sizesOk = true;
  for (const auto& t : r.Tensors())
  {
    aligned &= (t.begin % DTypeSize(t.dtype)) == 0;
    sizesOk &= t.ByteSize() == DTypeSize(t.dtype) * [&] { size_t n = 1; for (size_t d : t.shape) n *= d; return n; }();
  }
  Check(aligned, "every tensor is aligned to its element size");
  Check(sizesOk, "byte sizes match shape and dtype");

  std::vector<float> a(15);
  r.Read(*r.Find("mat"), a.data(), a.size() * 4);
  std::vector<uint16_t> b(7);
  r.Read(*r.Find("half"), b.data(), b.size() * 2);
  std::vector<uint8_t> c(11);
  r.Read(*r.Find("bytes"), c.data(), c.size());
  std::vector<double> d(4);
  r.Read(*r.Find("wide"), d.data(), d.size() * 8);
  Check(a == s.a && b == s.b && c == s.c && d == s.d, "tensor contents survive");
  Check(Throws([&] { std::vector<float> x(3); r.Read(*r.Find("mat"), x.data(), 12); }),
        "Read() with the wrong destination size is rejected");

  // Overwriting an existing file works and stays atomic.
  w.Write(path);
  Check(SafeTensorsReader(path).Tensors().size() == 5, "overwrite an existing file");
  fs::remove(path);
}

void TestWriterRejects()
{
  printf("--- writer input checks ---\n");
  SafeTensorsWriter w;
  float x[4] = {};
  Check(Throws([&] { w.AddTensor("t", "F31", {4}, x, 16); }), "unknown dtype");
  Check(Throws([&] { w.AddTensor("t", "F32", {4}, x, 12); }), "byte size does not match shape");
  w.AddTensor("t", "F32", {4}, x, 16);
  Check(Throws([&] { w.AddTensor("t", "F32", {4}, x, 16); }), "duplicate name");
  Check(Throws([&] { w.AddTensor("__metadata__", "F32", {4}, x, 16); }), "reserved name");
  Check(Throws([&] { w.SetMetadata(kHashKey, "x"); }), "reserved metadata key");
  Check(Throws([&] { w.AddTensor("n", "F32", {4}, nullptr, 16); }), "null data with non-zero size");
}

void TestHash()
{
  printf("--- hash ---\n");
  const std::string path = Tmp("hash.safetensors");
  Sample s;
  SafeTensorsWriter w;
  s.Fill(w);
  w.Write(path);
  auto bytes = Slurp(path);
  bytes[bytes.size() - 3] ^= 0x40; // flip a bit in the last tensor
  Spit(path, bytes);
  Check(Throws([&] { SafeTensorsReader r(path); }), "flipped data bit is caught by the hash");
  Check(!Throws([&] { SafeTensorsReader r(path, /*verifyHash=*/false); }), "verification can be skipped");

  SafeTensorsWriter nohash;
  s.Fill(nohash);
  nohash.SetHash(false);
  nohash.Write(path);
  Check(SafeTensorsReader(path).Metadata().count(kHashKey) == 0, "hash can be turned off");
  fs::remove(path);
}

void TestMalformed()
{
  printf("--- malformed files ---\n");
  const std::string p = Tmp("bad.safetensors");
  auto rejects = [&](const char* what, const std::string& header, size_t data)
  {
    WriteRaw(p, header, data);
    Check(Throws([&] { SafeTensorsReader r(p); }), what);
  };
  const std::string ok = R"({"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
  WriteRaw(p, ok, 8);
  Check(!Throws([&] { SafeTensorsReader r(p); }), "hand-built valid file is accepted");

  rejects("invalid JSON", "{not json", 0);
  rejects("header is not an object", "[1,2]", 0);
  rejects("unknown dtype", R"({"t":{"dtype":"F31","shape":[2],"data_offsets":[0,8]}})", 8);
  rejects("negative dimension", R"({"t":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})", 8);
  rejects("fractional dimension", R"({"t":{"dtype":"F32","shape":[2.5],"data_offsets":[0,8]}})", 8);
  rejects("byte size does not match shape", R"({"t":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", 8);
  rejects("offsets past the data region", R"({"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", 4);
  rejects("begin after end", R"({"t":{"dtype":"F32","shape":[0],"data_offsets":[8,0]}})", 8);
  rejects("overlapping tensors",
          R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})", 12);
  rejects("gap between tensors",
          R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})", 12);
  rejects("trailing bytes not covered by any tensor", ok, 16);
  rejects("shape product overflows",
          R"({"t":{"dtype":"F32","shape":[4294967296,4294967296,4],"data_offsets":[0,0]}})", 0);
  rejects("non-string metadata", R"({"__metadata__":{"k":1}})", 0);
  rejects("missing dtype", R"({"t":{"shape":[2],"data_offsets":[0,8]}})", 8);

  WriteRaw(p, ok, 8);
  auto bytes = Slurp(p);
  auto truncated = bytes;
  truncated.resize(bytes.size() - 3);
  Spit(p, truncated);
  Check(Throws([&] { SafeTensorsReader r(p); }), "truncated file");
  truncated.resize(5);
  Spit(p, truncated);
  Check(Throws([&] { SafeTensorsReader r(p); }), "file shorter than the size field");
  auto huge = bytes;
  const uint64_t big = 1ull << 40;
  std::memcpy(huge.data(), &big, 8);
  Spit(p, huge);
  Check(Throws([&] { SafeTensorsReader r(p); }), "huge header size");
  auto past = bytes;
  const uint64_t beyond = bytes.size();
  std::memcpy(past.data(), &beyond, 8);
  Spit(p, past);
  Check(Throws([&] { SafeTensorsReader r(p); }), "header size runs past the file end");
  fs::remove(p);
  Check(Throws([&] { SafeTensorsReader r(Tmp("does_not_exist")); }), "missing file");
}

void TestDTypes()
{
  printf("--- dtype registry, packed types, byte order ---\n");
  Check(FindDType("F32") != nullptr && FindDType("F32")->bytesPerBlock == 4, "F32 is a 4-byte standard type");
  Check(FindDType("FP4_E2M1")->kind == DTypeKind::Packed && StorageBytes(*FindDType("FP4_E2M1"), 7) == 4,
        "7 four-bit elements take 4 bytes (last block padded)");
  Check(FindDType("F31") == nullptr, "unknown dtype is not in the registry");

  // Every 4-bit code survives pack/unpack, for even and odd counts.
  for (size_t n : {size_t(0), size_t(1), size_t(2), size_t(7), size_t(16), size_t(33)})
  {
    std::vector<uint8_t> codes(n), packed(StorageBytes(*FindDType("POSIT4_1"), n)), back(n);
    for (size_t i = 0; i < n; ++i)
      codes[i] = static_cast<uint8_t>((i * 7 + 3) & 0x0F);
    FindDType("POSIT4_1")->pack(codes.data(), n, packed.data());
    FindDType("POSIT4_1")->unpack(packed.data(), n, back.data());
    Check(codes == back, "4-bit pack/unpack round trip, " + std::to_string(n) + " elements");
  }

  // Byte order: the swap acts only when the host is big endian, and is its own inverse.
  uint32_t v = 0x01020304;
  ConvertEndianness(&v, 4, 1, false);
  Check(v == 0x01020304, "no swap on a little-endian host");
  ConvertEndianness(&v, 4, 1, true);
  Check(v == 0x04030201, "32-bit swap on a big-endian host");
  ConvertEndianness(&v, 4, 1, true);
  Check(v == 0x01020304, "swap twice restores the value");
  uint16_t h[2] = {0x0102, 0x0304};
  ConvertEndianness(h, 2, 2, true);
  Check(h[0] == 0x0201 && h[1] == 0x0403, "16-bit swap covers every element");
  uint8_t bytes[3] = {1, 2, 3};
  ConvertEndianness(bytes, 1, 3, true);
  Check(bytes[0] == 1 && bytes[2] == 3, "single bytes are never swapped");

  // A packed tensor is stored as 1-D U8 plus deepity.dtypes, and reads back with its logical type.
  const std::string path = Tmp("packed.safetensors");
  const std::vector<uint8_t> packed = {0x21, 0x43, 0x65, 0x07}; // 7 elements 1..7
  const float f[2] = {1.5f, -2.5f};
  {
    SafeTensorsWriter w;
    w.AddTensor("q", "FP4_E2M1", {7}, packed.data(), packed.size());
    w.AddTensor("f", "F32", {2}, f, sizeof(f));
    w.Write(path);
  }
  SafeTensorsReader r(path);
  const TensorEntry* q = r.Find("q");
  Check(q != nullptr && q->dtype == "FP4_E2M1" && q->shape == std::vector<size_t>{7} && q->packed && q->ByteSize() == 4,
        "packed tensor reports its logical dtype and shape");
  std::vector<uint8_t> got(4);
  r.Read(*q, got.data(), got.size());
  Check(got == packed, "packed bytes read back unchanged");
  Check(r.Metadata().count(kDTypesKey) == 1, "deepity.dtypes is written");
  {
    const auto raw = Slurp(path);
    uint64_t n = 0;
    std::memcpy(&n, raw.data(), 8);
    const auto header = nlohmann::json::parse(std::string(raw.data() + 8, n));
    Check(header["q"]["dtype"] == "U8" && header["q"]["shape"] == nlohmann::json::array({4}),
          "header stores the packed tensor as 1-D U8, so stock readers open it");
  }
  Check(Throws([&] { SafeTensorsWriter w; w.SetMetadata(kDTypesKey, "{}"); }), "deepity.dtypes is reserved");
  Check(Throws([&] { SafeTensorsWriter w; w.AddTensor("q", "FP4_E2M1", {7}, packed.data(), 7); }),
        "packed tensor with an unpacked byte count is rejected");

  // The reader checks deepity.dtypes against the header instead of trusting it.
  const std::string bad = Tmp("bad_dtypes.safetensors");
  auto rejects = [&](const char* what, const std::string& overlay, const std::string& dtype, const std::string& shape)
  {
    const std::string header = std::string(R"({"q":{"dtype":")") + dtype + R"(","shape":)" + shape +
                               R"(,"data_offsets":[0,4]},"__metadata__":{"deepity.dtypes":)" +
                               nlohmann::json(overlay).dump() + "}}";
    WriteRaw(bad, header, 4);
    Check(Throws([&] { SafeTensorsReader rr(bad); }), what);
  };
  rejects("overlay with the wrong element count", R"({"q":{"dtype":"FP4_E2M1","shape":[20]}})", "U8", "[4]");
  rejects("overlay naming a missing tensor", R"({"nope":{"dtype":"FP4_E2M1","shape":[7]}})", "U8", "[4]");
  rejects("overlay on a tensor that is not U8", R"({"q":{"dtype":"FP4_E2M1","shape":[1]}})", "I8", "[4]");
  rejects("overlay on a tensor that is not 1-D", R"({"q":{"dtype":"FP4_E2M1","shape":[7]}})", "U8", "[2,2]");
  rejects("overlay with a standard logical dtype", R"({"q":{"dtype":"F32","shape":[1]}})", "U8", "[4]");
  rejects("overlay with an unknown logical dtype", R"({"q":{"dtype":"FP3","shape":[1]}})", "U8", "[4]");
  rejects("overlay that is not an object", "[1]", "U8", "[4]");
  rejects("overlay with a negative dimension", R"({"q":{"dtype":"FP4_E2M1","shape":[-7]}})", "U8", "[4]");
}

void TestMmap()
{
  printf("--- mmap and buffered reads ---\n");
  const std::string path = Tmp("mmap.safetensors");
  std::vector<float> a(1000), b(33);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.5f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = -static_cast<float>(i);
  {
    SafeTensorsWriter w;
    w.AddTensor("a", "F32", {1000}, a.data(), a.size() * 4);
    w.AddTensor("b", "F32", {33}, b.data(), b.size() * 4);
    w.Write(path);
  }
  SafeTensorsReader mapped(path, true, true), buffered(path, true, false);
  Check(mapped.IsMapped() && !buffered.IsMapped(), "mmap on by default, off when asked");
  std::vector<float> m(1000), f(1000);
  mapped.Read(*mapped.Find("a"), m.data(), 4000);
  buffered.Read(*buffered.Find("a"), f.data(), 4000);
  Check(m == a && f == a, "mapped and buffered reads return identical bytes");
  const void* p = mapped.Bytes(*mapped.Find("b"));
  Check(p != nullptr && std::memcmp(p, b.data(), b.size() * 4) == 0, "Bytes() is a zero-copy view of the tensor");
  Check(buffered.Bytes(*buffered.Find("a")) == nullptr, "Bytes() is null without a mapping");

  // The hash is checked over the mapping too.
  auto raw = Slurp(path);
  raw[raw.size() - 1] ^= 0x1;
  const std::string bad = Tmp("mmap_bad.safetensors");
  { std::ofstream o(bad, std::ios::binary); o.write(raw.data(), static_cast<std::streamsize>(raw.size())); }
  Check(Throws([&] { SafeTensorsReader r(bad, true, true); }), "mapped reader rejects a corrupted file");
  Check(Throws([&] { SafeTensorsReader r(bad, true, false); }), "buffered reader rejects a corrupted file");
  Check(!Throws([&] { SafeTensorsReader r(bad, false, true); }), "verifyHash=false skips the check");

  // A view keeps the data alive after the reader is gone.
  std::shared_ptr<const MappedFile> keep;
  (void)keep;
  const void* viewPtr = nullptr;
  {
    SafeTensorsReader r(path);
    keep = r.Mapping(*r.Find("a"));
    viewPtr = r.Bytes(*r.Find("a"));
  }
  Check(viewPtr != nullptr && std::memcmp(viewPtr, a.data(), 4000) == 0, "mapping outlives the reader");
}

void TestGolden()
{
  printf("--- golden and legacy files ---\n");
  const std::string dir = DEEPITY_TEST_DATA_DIR;
  SafeTensorsReader g(dir + "/golden_v1.safetensors");
  Sample s;
  std::vector<float> a(15);
  g.Read(*g.Find("mat"), a.data(), 60);
  Check(g.Tensors().size() == 5 && a == s.a && g.DataOffset() % 64 == 0 &&
            g.Metadata().at("deepity.format_version") == "1",
        "golden v1 file loads with the expected contents");

  SafeTensorsReader l(dir + "/legacy_a8_dkp.safetensors");
  Check(l.Tensors().size() == 9 && l.Find("layer_0.W") != nullptr && l.Find("layer_1.Psi") != nullptr &&
            l.Metadata().at("format") == "deepity_safetensors",
        "file written by 0.1.0a8 loads (unaligned, legacy names)");
  Check(l.DataOffset() % 64 != 0, "legacy file really is unaligned, so the reader does not require alignment");

  SafeTensorsReader d(dir + "/golden_v1_dtypes.safetensors");
  size_t index = 0, good = 0;
  for (const char* name : kGoldenDTypes)
  {
    const DTypeInfo* info = FindDType(name);
    const TensorEntry* t = d.Find(std::string("t_") + name);
    const auto want = GoldenBytes(*info, index++);
    good += t != nullptr && t->dtype == name && t->shape == std::vector<size_t>{5} &&
            t->packed == (info->kind == DTypeKind::Packed) && t->ByteSize() == want.size() &&
            std::memcmp(d.Bytes(*t), want.data(), want.size()) == 0;
  }
  Check(good == std::size(kGoldenDTypes) && d.Tensors().size() == good,
        "golden dtype file: all " + std::to_string(good) + " dtypes read back with exact bytes");
}
} // namespace

int main(int argc, char** argv)
{
  if (argc == 3 && std::string(argv[1]) == "--write-golden")
  {
    Sample s;
    SafeTensorsWriter w;
    s.Fill(w);
    w.Write(argv[2]);
    printf("wrote %s\n", argv[2]);
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--write-golden-dtypes")
  {
    WriteGoldenDTypes(argv[2]);
    printf("wrote %s\n", argv[2]);
    return 0;
  }
  TestBlake3();
  TestRoundTrip();
  TestWriterRejects();
  TestHash();
  TestMalformed();
  TestDTypes();
  TestMmap();
  TestGolden();
  TestBigEndianHost();
  printf(failures == 0 ? "PASSED: SafeTensors reader/writer\n" : "FAILED: %d check(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
