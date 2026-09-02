#pragma once
#include <cstddef>
#include <vector>
#include <deepity/backend/IComputeBackend.h>

/**
 * @file Tensor.h
 * @brief A device-aware, RAII-managed buffer. Device is fixed at
 * construction -- there is no support for moving a live Tensor between
 * CPU and GPU after creation. Loading weights saved on CPU and running
 * them on GPU doesn't need that: read the saved file into a plain
 * std::vector on the host, then construct a DEVICE_GPU Tensor directly
 * from that vector (see the host-data constructors below) -- the
 * one-time host-to-device copy happens inside the constructor, and the
 * temporary host vector falls out of scope immediately after. The
 * Tensor itself is a GPU object from the moment it's born.
 *
 * Move-only, matching MemoryArena's own established RAII convention in
 * this codebase (a Tensor owns a single device allocation; copying would
 * risk a double-free).
 */

namespace Deep
{
    enum class DeviceType
    {
        DEVICE_CPU,
        DEVICE_GPU
    };

    class Tensor
    {
    public:
        /// @brief Allocates numFloats on the given backend/device,
        /// zero-initialized.
        /// @param backend Non-owning pointer to the backend that will
        /// perform the allocation; must outlive this Tensor.
        /// @param device Which device this Tensor lives on -- must match
        /// what `backend` itself actually allocates on (a CPUBackend
        /// paired with DEVICE_GPU, or vice versa, is a caller error).
        Tensor(IComputeBackend *backend, DeviceType device, size_t numFloats)
            : backend(backend), device(device), size(numFloats)
        {
            data = backend->Allocate(numFloats);
            backend->Zero(data, numFloats);
        }

        /// @brief Allocates numFloats on the given backend/device, then
        /// copies hostData in via backend->CopyFromHost(). This is the
        /// "load weights saved on CPU, run on GPU" pattern -- works
        /// identically for device=DEVICE_CPU too, where CopyFromHost()
        /// is just a memcpy.
        Tensor(IComputeBackend *backend, DeviceType device, const float *hostData, size_t numFloats)
            : backend(backend), device(device), size(numFloats)
        {
            data = backend->Allocate(numFloats);
            backend->CopyFromHost(data, hostData, numFloats);
        }

        /// @brief Same as above, taking a std::vector directly -- the
        /// expected common case (read a saved model's weights into a
        /// std::vector, then hand it straight to this constructor).
        Tensor(IComputeBackend *backend, DeviceType device, const std::vector<float> &hostData)
            : Tensor(backend, device, hostData.data(), hostData.size())
        {
        }

        /// @brief Frees the underlying device allocation, if any (not
        /// called on a moved-from Tensor -- see the move constructor).
        ~Tensor()
        {
            if (data != nullptr)
            {
                backend->Free(data);
            }
        }

        // Move-only -- see file-level note.
        Tensor(const Tensor &) = delete;
        Tensor &operator=(const Tensor &) = delete;

        Tensor(Tensor &&other) noexcept
            : backend(other.backend), device(other.device), data(other.data), size(other.size)
        {
            other.data = nullptr;
            other.size = 0;
        }

        Tensor &operator=(Tensor &&other) noexcept
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

        /// @brief Returns the raw device buffer. Only safe to dereference
        /// directly from code that already knows it's running on the
        /// same device this Tensor lives on (e.g. CPUBackend's own
        /// methods for a DEVICE_CPU Tensor) -- never dereference a
        /// DEVICE_GPU Tensor's Data() from host code directly.
        float *Data() noexcept { return data; }
        const float *Data() const noexcept { return data; }

        size_t Size() const noexcept { return size; }
        DeviceType Device() const noexcept { return device; }
        IComputeBackend *Backend() const noexcept { return backend; }

        /// @brief Copies this Tensor's data out to a host buffer, e.g.
        /// for Save()/inspection. A no-op memcpy if this Tensor is
        /// already DEVICE_CPU.
        /// @param hostDst Must point to at least Size() floats.
        void CopyToHost(float *hostDst) const noexcept
        {
            backend->CopyToHost(hostDst, data, size);
        }

        /// @brief Same as above, resizing hostDst to fit if needed.
        void CopyToHost(std::vector<float> &hostDst) const
        {
            hostDst.resize(size);
            CopyToHost(hostDst.data());
        }

    private:
        IComputeBackend *backend = nullptr; // non-owning; must outlive this Tensor
        DeviceType device;
        float *data = nullptr;
        size_t size = 0;
    };
}