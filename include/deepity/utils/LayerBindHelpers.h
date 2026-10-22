#pragma once
#include <cstddef>
#include <memory>
#include <type_traits>

#include <deepity/backend/IComputeBackend.h>
#include <deepity/utils/MemoryArena.h>

/**
 * @file LayerBindHelpers.h
 * @brief Shared BindMemory() boilerplate every PC layer repeated once per
 * weight-like tensor: Adam/AdamW state allocation, the device-resident
 * timestep/learning-rate mirrors, and the local-arena release at the end.
 */
namespace Deep
{
/// @brief Allocates and zeros a parameter's Adam/AdamW optimizer state
/// (gradient, first moment, second moment) in the given arena.
template <typename ArenaT>
inline void BindAdamState(ArenaT& arena, IComputeBackend* backend, size_t paramSize,
                           float*& grad, float*& m, float*& v)
{
  grad = arena.AllocateFloats(paramSize);
  m = arena.AllocateFloats(paramSize);
  v = arena.AllocateFloats(paramSize);
  backend->Zero(m, paramSize);
  backend->Zero(v, paramSize);
  backend->Zero(grad, paramSize);
}

/// @brief Allocates a layer's device-resident Adam timestep/learning-rate
/// mirrors and initializes them from the host, so graph-captured Adam
/// steps on CUDA can read them each replay without a host round-trip.
template <typename ArenaT>
inline void BindLearningRateState(ArenaT& arena, IComputeBackend* backend, int*& timestepDevice,
                                   float*& lrDevice, float lr)
{
  timestepDevice = reinterpret_cast<int*>(arena.AllocateFloats(1));
  lrDevice = arena.AllocateFloats(1);
  int zero = 0;
  backend->CopyFromHost(reinterpret_cast<float*>(timestepDevice),
                        reinterpret_cast<float*>(&zero), 1);
  backend->CopyFromHost(lrDevice, &lr, 1);
}

/// @brief Releases a layer's private fallback arena once BindMemory() has
/// run: dropped if it was bound into a network-wide MemoryArena other
/// than its own, and always dropped for a device arena (a layer never
/// keeps its own DeviceMemoryArena).
template <typename ArenaT>
inline void ReleaseLocalArenaIfForeign(std::unique_ptr<MemoryArena>& localArena, ArenaT& arena)
{
  if constexpr (std::is_same_v<ArenaT, MemoryArena>)
  {
    if (localArena && localArena.get() != &arena)
      localArena.reset();
  }
  else
  {
    localArena.reset();
  }
}
} // namespace Deep
