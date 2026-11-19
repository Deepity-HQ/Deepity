#pragma once
#include <cstddef>
#include <string>

namespace Deep::io
{
/// @brief A read-only memory mapping of a whole file. Move-only; unmaps on destruction.
///
/// Open() returns an empty mapping (`Data() == nullptr`) when the file cannot be mapped, such as
/// an empty file or a filesystem that refuses mmap, so callers fall back to ordinary reads.
class MappedFile
{
public:
  MappedFile() = default;
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  static MappedFile Open(const std::string& path) noexcept;

  const unsigned char* Data() const noexcept
  {
    return data;
  }
  size_t Size() const noexcept
  {
    return size;
  }

private:
  void Reset() noexcept;
  const unsigned char* data = nullptr;
  size_t size = 0;
#ifdef _WIN32
  void* mapping = nullptr; ///< HANDLE of the file-mapping object.
#endif
};
} // namespace Deep::io
