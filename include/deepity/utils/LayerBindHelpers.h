#pragma once
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

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

/// @brief Templated counterpart to BindAdamState(), for FullPCLayer<T,
/// AccumT>: `grad`/`m`/`v` are AccumT-typed (the optimizer's own wider
/// accumulator, see IComputeBackendT<T, AccumT>'s own doc comment), not
/// T-typed like the layer's forward-path buffers -- so, unlike those,
/// they're never allocated from a layer's T-typed arena (a
/// MemoryArenaT<T>'s Allocate() returns T*, which only happens to equal
/// AccumT* when T==AccumT; for a genuinely different pair, e.g.
/// Fp4E2M1Value's storage vs. its float accumulator, it's a flat type
/// mismatch). Each one is instead a `std::vector<AccumT>` the layer
/// owns directly. Valid for a host-resident backend only (every
/// templated backend today); a future CUDABackendT<T, AccumT> would
/// need its own device-resident accumulator storage instead of a host
/// std::vector.
template <typename AccumT>
inline void BindAdamStateT(size_t paramSize, std::vector<AccumT>& grad, std::vector<AccumT>& m,
                           std::vector<AccumT>& v)
{
  grad.assign(paramSize, AccumT(0));
  m.assign(paramSize, AccumT(0));
  v.assign(paramSize, AccumT(0));
}

/// @brief Templated counterpart to BindLearningRateState(), for
/// FullPCLayer<T, AccumT>. Same host-only-AccumT-scratch reasoning as
/// BindAdamStateT() above -- `lr`/`timestep` are plain scalar members,
/// not arena-allocated pointers, with their address taken at each
/// backend call site that needs a `const AccumT*`/`const int*`.
template <typename AccumT>
inline void BindLearningRateStateT(int& timestep, AccumT& lrState, AccumT lr)
{
  timestep = 0;
  lrState = lr;
}

/// @brief Templated counterpart to ReleaseLocalArenaIfForeign(), for
/// FullPCLayer<T, AccumT>'s own MemoryArenaT<T> local arena.
template <typename T, typename ArenaT>
inline void ReleaseLocalArenaIfForeignT(std::unique_ptr<MemoryArenaT<T>>& localArena,
                                        ArenaT& arena)
{
  if constexpr (std::is_same_v<ArenaT, MemoryArenaT<T>>)
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
