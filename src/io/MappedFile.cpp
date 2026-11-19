#include "deepity/io/MappedFile.h"
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Deep::io
{
MappedFile::~MappedFile()
{
  Reset();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
{
  *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept
{
  if (this == &other) return *this;
  Reset();
  data = std::exchange(other.data, nullptr);
  size = std::exchange(other.size, 0);
#ifdef _WIN32
  mapping = std::exchange(other.mapping, nullptr);
#endif
  return *this;
}

#ifdef _WIN32
void MappedFile::Reset() noexcept
{
  if (data != nullptr) UnmapViewOfFile(data);
  if (mapping != nullptr) CloseHandle(mapping);
  data = nullptr;
  mapping = nullptr;
  size = 0;
}

MappedFile MappedFile::Open(const std::string& path) noexcept
{
  MappedFile m;
  const int wideLen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  if (wideLen <= 0) return m;
  std::wstring wide(static_cast<size_t>(wideLen), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), wideLen);
  HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return m;
  LARGE_INTEGER fileSize;
  if (GetFileSizeEx(file, &fileSize) && fileSize.QuadPart > 0)
  {
    m.mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m.mapping != nullptr)
    {
      m.data = static_cast<const unsigned char*>(MapViewOfFile(m.mapping, FILE_MAP_READ, 0, 0, 0));
      if (m.data != nullptr) m.size = static_cast<size_t>(fileSize.QuadPart);
      else
      {
        CloseHandle(m.mapping);
        m.mapping = nullptr;
      }
    }
  }
  CloseHandle(file);
  return m;
}
#else
void MappedFile::Reset() noexcept
{
  if (data != nullptr) munmap(const_cast<unsigned char*>(data), size);
  data = nullptr;
  size = 0;
}

MappedFile MappedFile::Open(const std::string& path) noexcept
{
  MappedFile m;
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return m;
  struct stat st;
  if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0)
  {
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (p != MAP_FAILED)
    {
      m.data = static_cast<const unsigned char*>(p);
      m.size = static_cast<size_t>(st.st_size);
    }
  }
  close(fd);
  return m;
}
#endif
} // namespace Deep::io
