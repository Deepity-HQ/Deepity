/**
 * @file FuzzFormats.cpp
 * @brief libFuzzer target for the safetensors and shard index parsers (clang only).
 *
 * The first input byte picks the parser, the rest is the file. Anything but a clean parse or the
 * format's own error type aborts, and AddressSanitizer/UBSan report memory errors.
 *   cmake -DDEEPITY_BUILD_FUZZ=ON ...; then run FuzzFormats corpus/ -max_total_time=600
 */
#include <cstdint>
#include <cstdio>
#include <deepity/io/SafeTensors.h>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>

using namespace Deep::io;
namespace fs = std::filesystem;

namespace
{
fs::path Dir()
{
  static const fs::path dir = fs::temp_directory_path() / ("deepity_libfuzz_" + std::to_string(::getpid()));
  return dir;
}

void Put(const fs::path& p, const uint8_t* data, size_t size)
{
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}
} // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***)
{
  fs::create_directories(Dir());
  // Valid shards for the index inputs to point at.
  const float f[4] = {1, 2, 3, 4};
  SafeTensorsWriter w;
  w.AddTensor("a", "F32", {4}, f, sizeof(f));
  w.AddTensor("b", "F32", {4}, f, sizeof(f));
  w.WriteSharded((Dir() / "m.safetensors.index.json").string(), 16);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
  if (size == 0) return 0;
  const uint8_t which = data[0] % 2;
  ++data;
  --size;
  try
  {
    if (which == 0)
    {
      const fs::path p = Dir() / "f.safetensors";
      Put(p, data, size);
      for (bool mmap : {true, false})
      {
        SafeTensorsReader r(p.string(), true, mmap);
        for (const auto& t : r.Tensors())
        {
          if (t.ByteSize() > (1u << 24)) continue;
          std::vector<char> buf(t.ByteSize());
          r.Read(t, buf.data(), buf.size());
        }
      }
    }
    else if (which == 1)
    {
      const fs::path p = Dir() / "x.safetensors.index.json";
      Put(p, data, size);
      SafeTensorsReader r(p.string());
    }
  }
  catch (const SafeTensorsError&) {}
  return 0;
}
