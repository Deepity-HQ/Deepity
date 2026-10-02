/**
 * @file CPUBackendHighway.cpp
 * @brief Google Highway (github.com/google/highway) runtime-dispatched
 * implementations for several CPUBackend elementwise kernels that were
 * previously plain scalar loops: AddBiasPerChannel, Fill, MultiplyInto,
 * and (for a subset of ActivationType, see below)
 * FusedActivationDerivativeMultiply, FusedStateUpdate and
 * FusedStateUpdateMomentum. See CPUBackendHighway.h for why this is its
 * own CMake target, compiled without the project's -march pin.
 *
 * dGELU is never vectorized here, in any of these kernels: its derivative
 * needs Sleef_tanhf_u10, a transcendental function, and wiring SLEEF's own
 * vectorized tanh into a Highway lane loop is a real, separate piece of
 * work, not something to rush into this pass. dSIGMOID and dTANH are
 * vectorized in FusedActivationDerivativeMultiply (its "from an already-
 * activated value" closed form needs no transcendental: sig*(1-sig),
 * 1-tanh^2) but NOT in FusedStateUpdate/FusedStateUpdateMomentum (their
 * dType is evaluated from the raw pre-activation z, which for dSIGMOID
 * and dTANH needs std::exp / Sleef_tanhf_u10 respectively -- see
 * ActivationDerivativeScalar in Activations.h). All three transcendental
 * cases stay on CPUBackend.cpp's existing scalar path for whichever
 * kernel needs them.
 */
#include <cmath>
#include <cstddef>

#define HWY_TARGET_INCLUDE "backend/CPUBackendHighway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "CPUBackendHighway.h"

HWY_BEFORE_NAMESPACE();
namespace Deep
{
namespace HWY_NAMESPACE
{
namespace hn = hwy::HWY_NAMESPACE;

void AddBiasPerChannelHwy(float* buf, const float* bias, size_t channels, size_t spatialSize)
{
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);

  for (size_t c = 0; c < channels; ++c)
  {
    const auto biasVec = hn::Set(d, bias[c]);
    float* row = buf + c * spatialSize;

    size_t s = 0;
    for (; s + lanes <= spatialSize; s += lanes)
    {
      auto v = hn::LoadU(d, row + s);
      hn::StoreU(hn::Add(v, biasVec), d, row + s);
    }
    for (; s < spatialSize; ++s) // scalar tail, never an out-of-bounds read
      row[s] += bias[c];
  }
}

// Mirrors ActivationDerivativeFromActivatedScalar (Activations.h) exactly
// for every case reachable here (dGELU is routed to the scalar path by
// the caller, see CPUBackendHighway.h).
void FusedActivationDerivativeMultiplyHwy(float* dst, const float* a, float* activatedInOut,
                                          ActivationType dType, size_t n)
{
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto one = hn::Set(d, 1.0f);
  const auto two = hn::Set(d, 2.0f);
  const auto zero = hn::Zero(d);

  size_t i = 0;
  for (; i + lanes <= n; i += lanes)
  {
    auto av = hn::LoadU(d, a + i);
    auto act = hn::LoadU(d, activatedInOut + i);
    decltype(av) deriv;

    switch (dType)
    {
    case ActivationType::dRELU:
      deriv = hn::IfThenElseZero(hn::Gt(act, zero), one);
      break;
    case ActivationType::dSIGMOID:
      deriv = hn::Mul(act, hn::Sub(one, act));
      break;
    case ActivationType::d_eSIGMOID:
      deriv = hn::Mul(two, hn::Mul(act, hn::Sub(one, act)));
      break;
    case ActivationType::dTANH:
      deriv = hn::Sub(one, hn::Mul(act, act));
      break;
    default: // dLINEAR, NONE (dGELU never reaches this function)
      deriv = one;
      break;
    }

    hn::StoreU(hn::Mul(av, deriv), d, dst + i);
    hn::StoreU(deriv, d, activatedInOut + i);
  }

  for (; i < n; ++i) // scalar tail, must match the vector formulas exactly
  {
    float act = activatedInOut[i];
    float deriv;
    switch (dType)
    {
    case ActivationType::dRELU:
      deriv = (act > 0.0f) ? 1.0f : 0.0f;
      break;
    case ActivationType::dSIGMOID:
      deriv = act * (1.0f - act);
      break;
    case ActivationType::d_eSIGMOID:
      deriv = 2.0f * act * (1.0f - act);
      break;
    case ActivationType::dTANH:
      deriv = 1.0f - act * act;
      break;
    default:
      deriv = 1.0f;
      break;
    }
    dst[i] = a[i] * deriv;
    activatedInOut[i] = deriv;
  }
}

void FillHwy(float* buf, size_t n, float value)
{
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto v = hn::Set(d, value);

  size_t i = 0;
  for (; i + lanes <= n; i += lanes)
    hn::StoreU(v, d, buf + i);
  for (; i < n; ++i) // scalar tail, never an out-of-bounds write
    buf[i] = value;
}

void MultiplyIntoHwy(float* dst, const float* a, const float* b, size_t n)
{
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);

  size_t i = 0;
  for (; i + lanes <= n; i += lanes)
  {
    auto va = hn::LoadU(d, a + i);
    auto vb = hn::LoadU(d, b + i);
    hn::StoreU(hn::Mul(va, vb), d, dst + i);
  }
  for (; i < n; ++i) // scalar tail
    dst[i] = a[i] * b[i];
}

// Derivative-from-raw-z formula for the three transcendental-free cases
// this file handles (dRELU, d_eSIGMOID, dLINEAR/NONE); mirrors
// ActivationDerivativeScalar (Activations.h) exactly for those cases.
// dGELU/dSIGMOID/dTANH are never passed in here -- see
// CPUBackendHighway.h's note on the caller-side routing.
template <class D, class V>
V DerivFromRawZHwy(D d, V z, ActivationType dType)
{
  const auto one = hn::Set(d, 1.0f);
  switch (dType)
  {
  case ActivationType::dRELU:
    return hn::IfThenElseZero(hn::Gt(z, hn::Zero(d)), one);
  case ActivationType::d_eSIGMOID:
  {
    auto a = hn::Add(one, hn::Abs(z));
    return hn::Div(hn::Set(d, 0.5f), hn::Mul(a, a));
  }
  default: // dLINEAR, NONE
    return one;
  }
}

void FusedStateUpdateHwy(float* z, const float* feedback, ActivationType dType, const float* e,
                         size_t n, float ir)
{
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto irVec = hn::Set(d, ir);

  size_t i = 0;
  for (; i + lanes <= n; i += lanes)
  {
    auto zv = hn::LoadU(d, z + i);
    auto deriv = DerivFromRawZHwy(d, zv, dType);
    auto fv = hn::LoadU(d, feedback + i);
    auto ev = hn::LoadU(d, e + i);
    auto update = hn::Sub(hn::Mul(fv, deriv), ev);
    hn::StoreU(hn::MulAdd(irVec, update, zv), d, z + i);
  }
  for (; i < n; ++i) // scalar tail, must match the vector formulas exactly
  {
    float deriv = (dType == ActivationType::dRELU) ? ((z[i] > 0.0f) ? 1.0f : 0.0f)
                 : (dType == ActivationType::d_eSIGMOID)
                     ? (0.5f / ((1.0f + std::fabs(z[i])) * (1.0f + std::fabs(z[i]))))
                     : 1.0f;
    z[i] += ir * ((feedback[i] * deriv) - e[i]);
  }
}

void FusedStateUpdateMomentumHwy(float* z, float* v, const float* feedback, ActivationType dType,
                                 const float* e, size_t n, float ir, float beta)
{
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto irVec = hn::Set(d, ir);
  const auto betaVec = hn::Set(d, beta);
  const auto oneMinusBetaVec = hn::Set(d, 1.0f - beta);

  size_t i = 0;
  for (; i + lanes <= n; i += lanes)
  {
    auto zv = hn::LoadU(d, z + i);
    auto deriv = DerivFromRawZHwy(d, zv, dType);
    auto fv = hn::LoadU(d, feedback + i);
    auto ev = hn::LoadU(d, e + i);
    auto update = hn::Sub(hn::Mul(fv, deriv), ev);

    auto vv = hn::LoadU(d, v + i);
    vv = hn::MulAdd(betaVec, vv, hn::Mul(oneMinusBetaVec, update));
    hn::StoreU(vv, d, v + i);
    hn::StoreU(hn::MulAdd(irVec, vv, zv), d, z + i);
  }
  for (; i < n; ++i) // scalar tail, must match the vector formulas exactly
  {
    float deriv = (dType == ActivationType::dRELU) ? ((z[i] > 0.0f) ? 1.0f : 0.0f)
                 : (dType == ActivationType::d_eSIGMOID)
                     ? (0.5f / ((1.0f + std::fabs(z[i])) * (1.0f + std::fabs(z[i]))))
                     : 1.0f;
    float update = (feedback[i] * deriv) - e[i];
    v[i] = beta * v[i] + (1.0f - beta) * update;
    z[i] += ir * v[i];
  }
}
} // namespace HWY_NAMESPACE
} // namespace Deep
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace Deep
{
HWY_EXPORT(AddBiasPerChannelHwy);
HWY_EXPORT(FusedActivationDerivativeMultiplyHwy);
HWY_EXPORT(FillHwy);
HWY_EXPORT(MultiplyIntoHwy);
HWY_EXPORT(FusedStateUpdateHwy);
HWY_EXPORT(FusedStateUpdateMomentumHwy);

void AddBiasPerChannelDispatch(float* buf, const float* bias, size_t channels,
                               size_t spatialSize) noexcept
{
  HWY_DYNAMIC_DISPATCH(AddBiasPerChannelHwy)(buf, bias, channels, spatialSize);
}

void FusedActivationDerivativeMultiplyDispatch(float* dst, const float* a, float* activatedInOut,
                                               ActivationType dType, size_t n) noexcept
{
  HWY_DYNAMIC_DISPATCH(FusedActivationDerivativeMultiplyHwy)(dst, a, activatedInOut, dType, n);
}

void FillDispatch(float* buf, size_t n, float value) noexcept
{
  HWY_DYNAMIC_DISPATCH(FillHwy)(buf, n, value);
}

void MultiplyIntoDispatch(float* dst, const float* a, const float* b, size_t n) noexcept
{
  HWY_DYNAMIC_DISPATCH(MultiplyIntoHwy)(dst, a, b, n);
}

void FusedStateUpdateDispatch(float* z, const float* feedback, ActivationType dType, const float* e,
                              size_t n, float ir) noexcept
{
  HWY_DYNAMIC_DISPATCH(FusedStateUpdateHwy)(z, feedback, dType, e, n, ir);
}

void FusedStateUpdateMomentumDispatch(float* z, float* v, const float* feedback, ActivationType dType,
                                      const float* e, size_t n, float ir, float beta) noexcept
{
  HWY_DYNAMIC_DISPATCH(FusedStateUpdateMomentumHwy)(z, v, feedback, dType, e, n, ir, beta);
}
} // namespace Deep
#endif // HWY_ONCE
