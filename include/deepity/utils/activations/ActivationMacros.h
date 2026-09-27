#pragma once

/**
 * @file ActivationMacros.h
 * @brief The `RESTRICT` portability macro shared by every activation
 * header. Split out so each per-activation header is includable on its
 * own, without relying on include order through the Activations.h
 * umbrella.
 */

/// @brief Portability macro for the C++ `restrict` pointer qualifier
/// (MSVC spells it differently from everyone else).
#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
