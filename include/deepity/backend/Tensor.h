#pragma once
#include <cstddef>
#include <deepity/backend/DeviceType.h>
#include <deepity/backend/IComputeBackend.h>
#include <vector>

/**
 * @file Tensor.h
 * @brief A device-aware, RAII-managed, move-only buffer. Device is
 * fixed at construction; a Tensor cannot be moved between CPU and GPU
 * afterward. To load host-side data onto a GPU Tensor, construct it
 * directly from a host buffer or `std::vector` (see the constructors
 * below).
 */

namespace Deep
{
/// @brief A device-aware, RAII-managed, move-only buffer.
class Tensor
{
public:
  /// @brief Allocates numFloats on the given backend/device,
  /// zero-initialized.
  /// @param backend Non-owning pointer to the backend that will
  /// perform the allocation; must outlive this Tensor.
  /// @param device Which device this Tensor lives on, must match
  /// what `backend` itself actually allocates on (a CPUBackend
  /// paired with DEVICE_GPU, or vice versa, is a caller error).
  /// @param numFloats Number of floats to allocate.
  Tensor(IComputeBackend* backend, DeviceType device, size_t numFloats)
      : backend(backend)
      , device(device)
      , size(numFloats)
  {
    data = backend->Allocate(numFloats);
    backend->Zero(data, numFloats);
  }

  /// @brief Allocates numFloats on the given backend/device, then
  /// copies hostData in.
  Tensor(IComputeBackend* backend, DeviceType device, const float* hostData, size_t numFloats)
      : backend(backend)
      , device(device)
      , size(numFloats)
  {
    data = backend->Allocate(numFloats);
    backend->CopyFromHost(data, hostData, numFloats);
  }

  /// @brief Same as above, taking a std::vector directly.
  Tensor(IComputeBackend* backend, DeviceType device, const std::vector<float>& hostData)
      : Tensor(backend, device, hostData.data(), hostData.size())
  {
  }

  /// @brief Frees the underlying device allocation, if any.
  ~Tensor()
  {
    if (data != nullptr)
    {
      backend->Free(data);
    }
  }

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  /// @brief Move constructor: takes over `other`'s allocation,
  /// leaving it empty (Free()-safe, zero size).
  Tensor(Tensor&& other) noexcept
      : backend(other.backend)
      , device(other.device)
      , data(other.data)
      , size(other.size)
  {
    other.data = nullptr;
    other.size = 0;
  }

  /// @brief Move assignment: frees this Tensor's current allocation
  /// (if any), then takes over `other`'s, leaving it empty.
  Tensor& operator=(Tensor&& other) noexcept
  {
    if (this != &other)
    {
      if (data != nullptr)
      {
        backend->Free(data);
      }
      backend = other.backend;
      device = other.device;
      data = other.data;
      size = other.size;
      other.data = nullptr;
      other.size = 0;
    }
    return *this;
  }

  /// @brief Returns the raw device buffer. Only dereference directly
  /// from code running on this Tensor's own device -- never dereference
  /// a DEVICE_GPU Tensor's Data() from host code.
  float* Data() noexcept
  {
    return data;
  }
  /// @brief const overload of Data() above.
  const float* Data() const noexcept
  {
    return data;
  }

  /// @brief Number of floats this Tensor holds.
  size_t Size() const noexcept
  {
    return size;
  }
  /// @brief Which device this Tensor lives on.
  DeviceType Device() const noexcept
  {
    return device;
  }
  /// @brief The backend this Tensor was allocated through.
  IComputeBackend* Backend() const noexcept
  {
    return backend;
  }

  /// @brief Copies this Tensor's data out to a host buffer, e.g.
  /// for Save()/inspection. A no-op memcpy if this Tensor is
  /// already DEVICE_CPU.
  /// @param hostDst Must point to at least Size() floats.
  void CopyToHost(float* hostDst) const noexcept
  {
    backend->CopyToHost(hostDst, data, size);
  }

  /// @brief Same as above, resizing hostDst to fit if needed.
  void CopyToHost(std::vector<float>& hostDst) const
  {
    hostDst.resize(size);
    CopyToHost(hostDst.data());
  }

private:
  IComputeBackend* backend = nullptr; // non-owning; must outlive this Tensor
  DeviceType device;
  float* data = nullptr;
  size_t size = 0;
};
} // namespace Deep