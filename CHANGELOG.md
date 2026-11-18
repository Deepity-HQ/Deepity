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

- Self-describing model files for `DKPPCN` and `FullPCN`: `net.save(path)` stores the architecture, hyperparameters and default settling steps next to the weights, and `deepity.load(path)` rebuilds the network from the file alone. `predict()` now defaults to the model's `default_inference_steps` (1 for a new model; a loaded file restores its saved value). New `deepity.dy.read_model_metadata(path)` reads a file's metadata without loading weights, and every native network's `save()` accepts an optional `metadata` map.

- Training checkpoints for `DKPPCN` and `FullPCN` (CPU): `net.save_checkpoint(path, training_state)` stores the weights, Adam/AdamW moments and step counters plus any JSON dict you pass (epoch, step, RNG state), and `deepity.resume(path)` returns `(net, training_state)` ready to continue training bit-for-bit. In C++ it also works for `DirectKPPCNetworkT`, `FullConvPCNetwork`, `GaussSeidelPCNetwork(T)`, `DiscriminativePCNetwork` and `ConvPCNetwork` (no Python wrappers for those yet). In C++: `ModelIO::SaveCheckpoint` / `LoadCheckpoint`, backed by new `Layer::GetOptimizerStateDict()`, `GetOptimizerCounters()` and `SupportsCheckpoint()`. Other networks and GPU networks throw a clear error instead of writing a checkpoint that would not resume.

- `net.load_weights(path, strict=True, rename=None)` (C++: `ModelIO::LoadWeights`) loads weights from any safetensors file, such as a PyTorch export, mapping names with `rename` and reporting or rejecting names that do not line up.
- `deepity.inference` now records `mode` (`"iterative"`), and `deepity.load` rejects a mode it does not know.

### Changed

- `ModelIO::Save()` now writes a validated, aligned safetensors file atomically (temp file plus rename) with a BLAKE3 content hash (`deepity.content_blake3`), `deepity.format_version` and `deepity.kind` metadata, and tensor names `layers.{i}.weight|bias|psi`. `ModelIO::Load()` verifies the file (header limits, offsets, dtypes, hash) before reading, throws `SafeTensorsError` on a corrupt file, and still reads 0.1.0a8 and earlier files (`layer_{i}.W|b|Psi`).

- `ModelIO::Save/Load` (and the network `Save/Load` wrappers) now report failure by exception with the reason instead of returning `false`; they return `true` on success. `FullConvPCNetwork::Save/Load` and `DiscriminativePCNetwork::Save/Load` are no longer `noexcept`, because a corrupt file would otherwise terminate the process.

- `ModelIO::Save()` now throws instead of writing an empty file when a network exposes no weights (the Simple networks, `ConvPCNetworkT` and `FullConvPCNetworkT` have no state dict yet). Before, it silently produced a model with no tensors.

- 4-bit weight types (`fp4_e2m1`, `posit4_0/1/2`) now save packed, two codes per byte, as a 1-D `U8` tensor plus a `deepity.dtypes` entry, so those model files are about half the size and still open in stock safetensors readers. Files from 0.1.0a8 that stored them one byte per element still load. Model files are little endian on every host; big-endian hosts byte-swap on read and write.

### Deprecated

-

### Removed

-

### Fixed

- `ModelIO::Save()` truncated every `Psi` tensor (DKP and FullPC feedback weights) whenever a layer's size differed from the next layer's: the state dict labeled Psi `[nextSize, terminal]` while the buffer is `[size, terminal]`, so rows were dropped and a resumed run loaded partly wrong feedback weights. Affects `DirectKPPCLayer`, `DirectKPPCLayerT`, `FullPCLayer` and `FullPCLayerT` (the last also staged too few Psi rows on the host). Files written by 0.1.0a8 and earlier still load; their missing rows keep the network's current values.
- `ModelIO::Load()` now throws on a tensor shape mismatch instead of reading mismatched bytes.

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
