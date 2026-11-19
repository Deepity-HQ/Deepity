/**
 * @file tFormatFuzz.cpp
 * @brief Mutation fuzzing of the two file parsers (safetensors and shard index).
 *
 * Starts from valid files and applies thousands of seeded random mutations (byte flips, truncation,
 * overwritten numbers, spliced chunks, appended junk). Each mutant must either parse or throw the
 * format's own error type; any other exception, crash, hang or out-of-bounds access (visible under
 * a sanitizer build) fails the run. The seed is fixed so a failure reproduces.
 */
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deepity/io/SafeTensors.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace Deep::io;
namespace fs = std::filesystem;

namespace
{
std::vector<char> Slurp(const std::string& p)
{
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void Spit(const std::string& p, const std::vector<char>& b)
{
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(b.data(), static_cast<std::streamsize>(b.size()));
}

std::vector<char> Mutate(const std::vector<char>& src, std::mt19937_64& rng)
{
  std::vector<char> b = src;
  const int kind = static_cast<int>(rng() % 6);
  auto at = [&](size_t n) { return n == 0 ? size_t(0) : static_cast<size_t>(rng() % n); };
  switch (kind)
  {
  case 0: // flip a few bytes, favouring the header
    for (int i = 0, n = 1 + static_cast<int>(rng() % 4); i < n && !b.empty(); ++i)
      b[at(std::min<size_t>(b.size(), 256))] ^= static_cast<char>(1u << (rng() % 8));
    break;
  case 1: // truncate
    b.resize(at(b.size() + 1));
    break;
  case 2: // overwrite 8 bytes with a hostile number
  {
    static const uint64_t magic[] = {0, 1, ~0ull, 1ull << 63, 1ull << 40, 0xFFFFFFFFull, 7};
    if (b.size() >= 8)
    {
      const uint64_t v = magic[rng() % 7];
      std::memcpy(&b[at(std::min<size_t>(b.size() - 7, 300))], &v, 8);
    }
    break;
  }
  case 3: // random bytes anywhere
    for (int i = 0, n = 1 + static_cast<int>(rng() % 8); i < n && !b.empty(); ++i)
      b[at(b.size())] = static_cast<char>(rng());
    break;
  case 4: // append junk
    for (int i = 0, n = static_cast<int>(rng() % 64); i < n; ++i) b.push_back(static_cast<char>(rng()));
    break;
  default: // duplicate a chunk over another place
    if (b.size() > 16)
    {
      const size_t len = 1 + at(16), from = at(b.size() - len), to = at(b.size() - len);
      std::copy_n(src.begin() + from, len, b.begin() + to);
    }
  }
  return b;
}

int failures = 0;

template <typename Parse> void Fuzz(const char* name, const std::vector<char>& seed, const std::string& scratch, size_t rounds, Parse&& parse)
{
  std::mt19937_64 rng(0xDEE917);
  size_t parsed = 0, rejected = 0;
  for (size_t i = 0; i < rounds; ++i)
  {
    Spit(scratch, Mutate(seed, rng));
    try
    {
      parse(scratch);
      ++parsed;
    }
    catch (const SafeTensorsError&) { ++rejected; }
    catch (const std::exception& e)
    {
      printf("  [FAILED] %s: mutant %zu threw an unexpected exception: %s\n", name, i, e.what());
      ++failures;
      return;
    }
  }
  printf("  [PASSED] %s: %zu mutants, %zu rejected cleanly, %zu still valid\n", name, rounds, rejected, parsed);
}
} // namespace

int main()
{
  // One directory per process: the plain and sanitized runs execute in parallel under ctest -j.
  const fs::path dir = fs::temp_directory_path() / ("deepity_fuzz_" + std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(dir);
  const std::string scratch = (dir / "m.bin").string();

  // Seed 1: a safetensors file with a standard tensor, a packed one and metadata.
  const float f[6] = {1, 2, 3, 4, 5, 6};
  const uint8_t q[4] = {0x21, 0x43, 0x65, 0x07};
  SafeTensorsWriter w;
  w.SetMetadata("deepity.kind", "inference");
  w.AddTensor("a", "F32", {2, 3}, f, sizeof(f));
  w.AddTensor("q", "FP4_E2M1", {7}, q, sizeof(q));
  const std::string st = (dir / "seed.safetensors").string();
  w.Write(st);
  Fuzz("safetensors", Slurp(st), (dir / "fuzz.safetensors").string(), 6000,
       [](const std::string& p)
       {
         for (bool mmap : {true, false})
         {
           SafeTensorsReader r(p, true, mmap);
           for (const auto& t : r.Tensors())
           {
             std::vector<char> buf(t.ByteSize());
             r.Read(t, buf.data(), buf.size());
           }
         }
       });

  // Seed 2: a shard index plus its shards (only the index is mutated).
  const std::string index = (dir / "m.safetensors.index.json").string();
  w.WriteSharded(index, 20);
  Fuzz("shard index", Slurp(index), (dir / "fuzz.safetensors.index.json").string(), 4000,
       [&](const std::string& p) { SafeTensorsReader r(p); });

  fs::remove_all(dir);
  printf(failures == 0 ? "PASSED: format fuzzing\n" : "FAILED: format fuzzing\n");
  return failures == 0 ? 0 : 1;
}
