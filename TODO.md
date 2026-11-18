# BetterModelIO: plan

Branch: `BetterModelIO`. Goal: make Deepity models portable. Save, pause, resume, ship, download,
and move between formats, with IO fast enough that nobody writes a Python loop over weights.

## Decisions so far

- [x] Custom dtypes: stored as `U8` plus `deepity.dtypes` metadata.
- [x] Sharding: planned in the spec, built later (Phase 4).
- [x] Content hash: optional, written by default.

- [x] **Native format is real safetensors** plus a versioned Deepity metadata block. Chosen for
      cross-platform support, zero-copy mmap and adoption.
- [x] **Settling settings are part of the artifact** (`inference_steps`, inference rate, per-layer
      activations) and are the defaults after loading. Callers can override them.
- [x] **Both import and export.** Models should move in and out (ONNX, GGUF, PyTorch state_dict),
      and training must be able to pause and resume.
- [x] **Audience: Python users building products, not only researchers.** Priorities are reliable
      loading, fast cold start, small files (quantization) and clear errors.
- [x] **Speed rule:** all bulk tensor work happens in C++. Python passes paths and buffers, never
      loops over weights.

## What exists today (`include/deepity/utils/ModelIO.h`)

- Weights only, in a safetensors-shaped file. Keys are `layer_{index}.{name}`.
- No architecture or inference settings in the file. `Load()` needs a network already built with
  the same layers, and matches tensors by position.
- No version field, checksum, optimizer state or RNG state.
- `Save`/`Load` return `bool` with no reason. Writes are not atomic.
- Python API is inconsistent (`SequentialPCN.save` takes a directory, others take a file).
- Custom dtype names (`FP4_E2M1`, `POSIT4_*`) live only in `DTypeTag.h`.
- Verified 2026-11-14 on a small DKPPCN: stock `safetensors` (`safe_open`) reads our files fine.
- But the header is not padded to 8 bytes, so tensor data starts at an unaligned offset (670 in the
  test file). That breaks aligned mmap access.
- **Bug:** `DirectKPPCLayer::GetStateDict()` labels Psi as `[nextSize, terminalSize]`, but the
  buffer is `[size, terminalSize]`. Save writes only the first `nextSize` rows. Round trip on
  `Linear(8,6),ReLU,Linear(6,4)`: layer 1 Psi loses rows 4 and 5, layer 0 loses rows 6 and 7, the
  terminal layer's Psi is not saved. Resumed DKP training gets partly wrong feedback weights.
  Fixing it changes the stored shape, so do it together with the versioned format (Phase 1).
- Zero-size tensors are written for the terminal layer (`layer_2.W` has shape `[0, 4]`).
- The `__metadata__.dtype` field holds the last tensor's dtype, which is meaningless.

## Phase 0: spec and test fixtures (before any code)

- [~] Write `docs/model-format.md`: file layout, metadata schema, dtype names, tensor naming,
      layout conventions (conv `[outC, inC*kH*kW]`, Psi padding), versioning rules.
      Draft v0 written; open questions are listed at the bottom of the file.
- [x] Check that stock `safetensors` reads our current files. Gaps recorded above.
- [x] Add a DKP save/load round-trip regression test (`tests/tModelIOVerify.cpp`). Failed on the
      old code at the predicted rows, passes now. Legacy (a8) files verified to still load.
- [x] Round-trip test covers `DirectKPPCNetwork`, `FullPCNetwork`, `FullPCNetworkT<double>`,
      `FullConvPCNetwork`, dtype mismatch and corruption rejection (`tests/tModelIOVerify.cpp`).
- [~] Golden files: `tests/data/golden_v1.safetensors` and `legacy_a8_dkp.safetensors` added and
      loaded by `SafeTensorsVerify`. Still to add: one per extra dtype once those are written. CI now runs
      ctest and `tests/pyModelFilesVerify.py` on Linux (gcc, clang, icpx); Windows still builds only.
      Note `.gitignore` excludes `*.safetensors` and `*_meta.json`, so fixtures need an exception.
- [ ] Decide naming: stable tensor names (for example `layers.3.weight`) instead of positional keys.

## Phase 1: core format cleanup

- [x] Standalone reader/writer (`include/deepity/io/SafeTensors.h`, `src/io/SafeTensors.cpp`):
      header padding, 64-byte data alignment, string metadata, contiguous offsets, atomic write,
      BLAKE3 hash, strict validation. Output opens in stock `safetensors`. Not yet wired into
      `ModelIO::Save/Load`.
- [ ] `format_version` plus unknown-field-tolerant reads. Reader for current a7/a8 files.
- [x] Validate untrusted input (in the new reader): header size cap, offsets against the real file
      size, no overlaps or gaps, `prod(shape) * dtype size == bytes`, overflow-safe, dtype allow-list.
- [x] `ModelIO::Save/Load` use the validating reader/writer, new `layers.{i}.weight|bias|psi` names,
      legacy `layer_{i}.*` names still load.
- [x] Atomic writes (temp file plus rename) and content hash, in the new writer.
- [x] Failures throw with the reason (`ModelIO` and network `Save/Load`); `true` on success is kept
      for source compatibility.
- [x] Aligned data region (64 bytes), tensors aligned to element size.
- [x] Hash: BLAKE3 in Highway with runtime dispatch (`src/io/Blake3.cpp`, adapted from
      Universal-Blake3), checked against the reference implementation. Dispatch costs nothing
      versus the static upstream header (same speed, same compiler). Absolute speed here varies
      with the CPU governor (1.9 to 3.5 GB/s per thread on one laptop).

## Phase 2: dtype system

- [ ] Registry that records logical type, storage layout, block size and bytes per block.
      Needed for sub-byte types (fp4, posit4) and GGUF-style block quant (Q4_K and similar).
- [ ] Endianness policy (little endian on disk, byte-swap on big-endian hosts).

## Phase 3: self-describing models

- [~] Architecture and inference settings serialized as JSON in the metadata block. Done for
      `DKPPCN` and `FullPCN` (`save()` writes them). Other networks still save weights only.
- [x] `deepity.load(path)` rebuilds the network from the file (`DKPPCN`, `FullPCN`).
      Test: `tests/pyModelFilesVerify.py` (run in the Linux CI job).
- [ ] `net.load_weights(path, strict=False, rename=...)` for partial loads.
- [~] Two artifact kinds: inference artifact and training checkpoint. Checkpoint done for
      `DirectKPPCNetwork`, `FullPCNetwork` and `FullPCNetworkT` on CPU (Adam/AdamW moments, step
      counters, caller training state). Still to do: conv family, `DirectKPPCNetworkT`, GPU
      networks (moments live on the device), and an inference artifact that omits Psi.
- [ ] Make settling semantics explicit in the file. An exported model says whether it is the full
      iterative model or a forward-only or unrolled approximation.

## Phase 4: sources and speed

- [ ] Small `Source`/`Sink` interface: file, memory buffer, mmap, then `https://` and `s3://`,
      with a local cache directory.
- [ ] mmap fast path, lazy tensor loading, and direct-to-device copies for CUDA.
- [ ] Bulk Python bindings: numpy and buffer protocol, no per-tensor Python loops.
- [ ] Benchmark: cold-load time and GB/s against raw `safetensors` and `np.load`.
- [ ] Sharding plus an index file for large models.

## Phase 5: converters (separate from the core)

- [ ] Import: safetensors and PyTorch `state_dict` with name mapping.
- [ ] Export: ONNX for the forward path (`predict` with 0 steps), and optionally an unrolled
      settling graph with a fixed step count.
- [ ] GGUF: read first (quantized weights), write later.
- [ ] Round-trip tests: Deepity to ONNX to onnxruntime, outputs within tolerance.

## Phase 6: ecosystem polish

- [ ] Model card metadata (license, training data, metrics, Deepity version).
- [ ] CLI: `deepity inspect model.safetensors` and `deepity convert in out`.
- [ ] Fuzz tests for the header parser. Security note in `SECURITY.md` about untrusted models.
- [ ] CHANGELOG entries and docs.

## Open questions

- Tensor naming scheme and whether to keep a compatibility alias for `layer_N.name`.
- Single file with metadata, or `config.json` sidecar, or both?
- Which PC networks get self-describing save first (suggest `DKPPCN` and `FullPCN`).
- How to handle the two layer hierarchies (float classes and `*LayerT` templates) in one schema.
- GGUF scope: which quant types matter for Deepity users first?

## Housekeeping

- [x] Root cleanup committed.
- [ ] `pyrightconfig.json` is tracked but also listed in `.gitignore`. Pick one.
