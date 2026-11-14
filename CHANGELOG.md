# Changelog

All notable changes to Deepity will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once versioned releases begin (see [Versioning](#versioning) below).

Changes prior to the introduction of this file were not formally tracked;
history effectively begins at `[Unreleased]` below. See the git log for
earlier history.

## [Unreleased]

### Added

- `DKPPCN.configure(psi_decay=...)` (and `set_psi_decay()` on `DirectKPPCNetwork` / `DirectKPPCLayer`): a decay applied to the direct feedback weights Psi alone, separate from `lmbda`, which keeps governing W. The default (`None`) keeps the previous behavior exactly: no decay on Psi under SGD, `lmbda` under AdamW. Only the float DKP family has it; `DirectKPPCNetworkT` / `DirectKPPCLayerT` do not yet.

### Changed

-

### Deprecated

-

### Removed

-

### Fixed

-

### Security

-

<!--
When cutting a release:
  1. Rename [Unreleased] to the new version + date, e.g. [0.2.0] - 2026-09-01
  2. Add a fresh, empty [Unreleased] section above it using the template above
  3. Add a link reference at the bottom (see the pattern below)
  4. Bump the version in pyproject.toml to match
-->

## [0.1.0a7] - 2026-11-09

### Added

- CUDA backend packaged into the published wheels: `pip install pydeepity[cuda]` pulls the matching CUDA 12.6 runtime/cuBLAS libraries, dlopen'd on demand (see `CUDAPluginLoader`). Plain `pip install pydeepity` remains CPU-only and unaffected.

### Fixed

- `tests/*.cpp` benchmark/verification executables didn't receive the same `-march`/ISA compile flags as the `Deepity`/`pydeepity` targets, so Deepity's header-only activation kernels silently compiled against the compiler's default ISA (SSE2) instead of whatever `build.py` selected -- `ActivationBenchmark`/`GflopsBenchmark` were measuring the wrong kernel. `Deepity`/`pydeepity` themselves were unaffected.
- `examples/mnist_cuda.py`'s `TERMINAL_FEED_WEIGHT_DECAY` stabilizer (see `examples/mnist.py` for the full derivation) was never retuned for cuBLAS's GEMM reduction order -- `2e-4` let the terminal-feed weight matrix diverge mid-training (energy spikes, accuracy crashing to the 30-40% range) before partially recovering. Swept and validated over a 100-epoch OSC A100 run; `5e-4` holds a clean >99% plateau with no late crash. Now the default, overridable via a new 4th CLI arg.
- cibuildwheel's `CIBW_ENVIRONMENT_LINUX` (set in CI) was silently overriding, not merging with, `pyproject.toml`'s `[tool.cibuildwheel.linux]` environment table, so the Linux wheel build's CUDA toolchain discovery never saw `CUDACXX`/`CUDAToolkit_ROOT`. Those variables now live directly in `ci.yml`.
- MSVC-only flags (`/openmp:llvm`, `/arch:AVX2`) were leaking onto nvcc's command line during CUDA-language compilation and being misparsed as stray input files; they're no longer applied to `.cu` files on Windows.
- CUTLASS was pinned down to `v3.7.0` (from `v4.8.0`) to sidestep a current, unfixed upstream MSVC incompatibility in its Blackwell (SM100/SM103) kernel headers (`constexpr` functions returning `dim3`, a non-literal type) -- this project only targets `compute_70`-`compute_90` and has no use for those kernels.

<!--
## [0.1.0] - YYYY-MM-DD

### Added

- Initial tracked release.
-->

[Unreleased]: https://github.com/Deepity-HQ/Deepity/compare/v0.1.0a7...HEAD
[0.1.0a7]: https://github.com/Deepity-HQ/Deepity/compare/v0.1.0a6...v0.1.0a7

## Versioning

Deepity follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`)
once tagged releases begin:

- **MAJOR** — incompatible C++ or Python API changes (e.g. changing
  `AddLayer`'s signature, renaming/removing a public header or `deepity`
  function).
- **MINOR** — backwards-compatible functionality (new layer types, new
  optimizers, new Python bindings, new build targets).
- **PATCH** — backwards-compatible bug fixes, performance improvements that
  don't change behavior, and documentation-only changes.

Until the first `1.0.0` release, the project may make breaking changes in
`0.x` releases as the API stabilizes; these will still be called out under
**Changed**/**Removed** in this file.

The authoritative version number lives in [`pyproject.toml`](pyproject.toml)
and should be bumped as part of the same PR/commit that promotes
`[Unreleased]` to a versioned entry in this file.
