## Description

<!-- What does this change do, and why? Link the related issue if one exists. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Build/tooling
- [ ] Breaking change (existing C++ or Python API behavior changes)

## Testing

- [ ] `python build.py Release` builds and `DeepityTests` passes
- [ ] `python build.py Debug` builds and `DeepityTests` passes (required for anything correctness-sensitive)
- [ ] `pyright` passes (if you touched `bindings/`, `pydeepity/`, `examples/`, or `experiments/`)

**Platform(s) and compiler(s) tested:**

<!-- e.g. Ubuntu 24.04 + Clang 18, Windows 11 + MSVC -->

## Performance impact

<!--
If this touches anything in the hot path (layer updates, activations, memory
arena, SIMD kernels), include before/after numbers - a quick tests/tBenchmark.cpp
or tests/tProfile.cpp run is enough, it doesn't need to be a formal writeup.
Delete this section if not applicable.
-->

## Checklist

- [ ] I've read [CONTRIBUTING.md](../CONTRIBUTING.md), including the portability rules and numerical gotchas relevant to this change.
- [ ] I've added a [CHANGELOG.md](../CHANGELOG.md) entry under `[Unreleased]` (for any user-facing change).
- [ ] I've updated relevant docs (README, Doxygen comments) if behavior or the public API changed.
- [ ] New/changed code follows the project's `.clang-format` style.

## Notes for reviewers

<!-- Anything you're unsure about, deliberately left out of scope, or want specific feedback on. -->
