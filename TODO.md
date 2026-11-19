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

- [x] Write `docs/model-format.md`: file layout, metadata schema, dtype names, tensor naming,
      layout conventions (conv `[outC, inC*kH*kW]`, Psi padding), versioning rules. Now v1.
- [x] Check that stock `safetensors` reads our current files. Gaps recorded above.
- [x] Add a DKP save/load round-trip regression test (`tests/tModelIOVerify.cpp`). Failed on the
      old code at the predicted rows, passes now. Legacy (a8) files verified to still load.
- [x] Round-trip test covers `DirectKPPCNetwork`, `FullPCNetwork`, `FullPCNetworkT<double>`,
      `FullConvPCNetwork`, dtype mismatch and corruption rejection (`tests/tModelIOVerify.cpp`).
- [x] Golden files in `tests/data/`: `golden_v1` (container), `golden_v1_dtypes` (all 17 dtypes, exact
      bytes), `legacy_a8_dkp`. Loaded by `SafeTensorsVerify`. CI runs ctest and the Python tests on Linux
      (gcc, clang, icpx). The Windows job now also runs the file-format tests (first run pending).
      `.gitignore` excludes `*.safetensors` with an exception for `tests/data/`.
- [x] Naming decided: `layers.{i}.weight|bias|psi`, legacy `layer_N.*` read-only (see the spec).

## Phase 1: core format cleanup

- [x] Standalone reader/writer (`include/deepity/io/SafeTensors.h`, `src/io/SafeTensors.cpp`):
      header padding, 64-byte data alignment, string metadata, contiguous offsets, atomic write,
      BLAKE3 hash, strict validation. Output opens in stock `safetensors`. Not yet wired into
      `ModelIO::Save/Load`.
- [x] `format_version` checked on every load path (integer, newer is refused naming both versions, missing
      is fine); unknown metadata keys, tensors and JSON fields are ignored. a7 and a8 wrote the same
      format (identical `ModelIO.h`), covered by `legacy_a8_dkp`.
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

- [x] Registry that records logical type, storage layout, block size and bytes per block
      (`include/deepity/io/DType.h`). fp4 and posit4 now store packed, half the bytes, as `U8` plus
      `deepity.dtypes`; stock `safetensors` opens them. GGUF-style block quant (Q4_K and similar)
      is one more registry row plus codec each, added in Phase 5.
- [x] Endianness policy: little endian on disk. The header length field is now explicitly little endian,
      and a simulated big-endian host (`SimulateBigEndianHost()`) drives the real swap code and writes
      byte-identical files. Still open: BLAKE3 and GGUF reading assume a little-endian host (the build
      refuses a big-endian target for BLAKE3), and nothing has run on real big-endian hardware.

## Phase 3: self-describing models

- [~] Architecture and inference settings serialized as JSON in the metadata block. Done for
      every Python network class (`save()` writes them).
- [x] `deepity.load(path)` rebuilds the network from the file (every Python network class).
      Test: `tests/pyModelFilesVerify.py` (run in the Linux CI job).
- [x] `net.load_weights(path, strict=True, rename=...)` for partial loads and foreign names.
- [~] Two artifact kinds: inference artifact and training checkpoint. Checkpoints (Adam/AdamW
      moments, step counters, caller training state) work on CPU for `DirectKPPCNetwork(T)`,
      `FullPCNetwork(T)`, `FullConvPCNetwork`, `GaussSeidelPCNetwork(T)`, `DiscriminativePCNetwork`
      and `ConvPCNetwork`, each verified to resume bit-identically. GPU networks stage
      through the backend (verified on an A100 at OSC for FullPCN, DKPPCN, SimplePCN, SimpleConvolutionalPCN and ConvolutionalPCN with `tests/pyGpuCheckpointVerify.py`;
      CPU/GPU predictions differ by about 1e-3 relative, GPU-to-GPU resume is exact). `FullConvPCNetwork` (native only), `GaussSeidelPCN` and `SequentialPCN` are untested on a GPU.
      An inference artifact that omits Psi is done (`save(for_inference=True)`, `deepity convert --inference`).
- [x] State dicts for every network class (Simple family, `ConvPCNetworkT`, `FullConvPCNetworkT`);
      checkpoints resume bit-identically for all of them on CPU.
- [x] Make settling semantics explicit in the file (`deepity.inference.mode`; only "iterative" is written). An exported model says whether it is the full
      iterative model or a forward-only or unrolled approximation.

## Phase 4: sources and speed

- [x] Sources (`deepity/_sources.py`): path, `file://`, `https://`, `s3://` (boto3 or public HTTPS), `hf://` and bytes, resolved
      through a download cache with ETag revalidation, size cap, hash check, shard sets and resume of cut-short downloads
      (Range plus If-Range). Tested against local servers (`tests/pySourcesVerify.py`, in CI) and a real one, Hugging Face
      (`tests/pyRealFilesVerify.py`, not in CI). Not built: uploading (Sink), a no-disk C++ memory reader; real S3 is untested.
- [x] mmap fast path (`MappedFile`, zero-copy `Bytes()`). Direct-to-device copies: a float tensor loads straight
      from the mapped file to the device in one copy (was device to host, file to host, host to device). Tested
      on the CPU with `ForceDeviceStaging()` and on an A100 through `tests/pyGpuCheckpointVerify.py`.
- [x] Bulk Python read and write: `deepity.read_tensors` (zero-copy when mapped) and `deepity.write_tensors` (any numpy dtype safetensors has; index path shards).
- [x] Benchmark script (`experiments/benchmark_model_io.py`); host load numbers collected on an Ascend node and recorded in
      the README and `docs/model-format.md`. The script does not time the direct-to-device CUDA load yet.
- [x] Sharding plus an index file (C++ `SaveSharded`, Python `save(path, max_shard_bytes=...)`).

## Phase 5: converters (separate from the core)

- [x] Import: safetensors and PyTorch `state_dict` with name mapping (`deepity.import_weights`). Checked against real
      `torch.save` files (`tests/pyTorchImportVerify.py`, CI): Sequential and Conv2d state dicts, `{"state_dict": ...}`
      checkpoints, fp16 and bf16, matching torch's own forward pass; a whole pickled model and a malicious pickle are refused.
- [x] Export: ONNX for the forward path (dense and convolutional) and unrolled settling for `SimplePCN`, `SequentialPCN`,
      `SimpleConvolutionalPCN` and `ConvolutionalPCN` (`deepity.export_onnx`, relu/sigmoid/tanh/gelu, strided layers
      via `output_padding`); the other dense classes need no settling. `FullConvPCNetwork` is native only.
- [x] GGUF read (`deepity.read_gguf`, `import_weights(.gguf)`): every type but Q8_1, TQ and MXFP4, checked against the
      reference `gguf` package and on real llama.cpp models (`tests/pyRealFilesVerify.py`, not in CI). GGUF write
      (`deepity.write_gguf`, `export_gguf`, `convert ... .gguf`): F32/F16/BF16/F64/ints and Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
      Q2_K to Q6_K (own quantizers, `deepity/_gguf_quant.py`). The IQ quants are read-only.
- [x] Round-trip test: Deepity to ONNX to onnxruntime within 1e-5 (`tests/pyOnnxVerify.py`, CI).

## Phase 6: ecosystem polish

- [x] Model card metadata: `net.card = {...}` is stored as `deepity.card`, restored by `deepity.load`, read by `deepity.read_card`.
- [x] CLI: `deepity inspect` and `deepity convert` (safetensors, shard sets, GGUF both ways).
- [x] Mutation fuzzing of the safetensors, shard index and GGUF parsers (`FormatFuzz`, found one real bug: invalid UTF-8 in a GGUF string).
      Security note in `SECURITY.md`. Also run under ASan/UBSan (`FormatFuzzSanitized`, `-DDEEPITY_BUILD_FUZZ=ON`, in CI) and as a libFuzzer
      target (`FuzzFormats`, clang/icpx): 818k runs in two minutes, no findings.
- [x] CHANGELOG entries and docs.

## Open questions

None open. Decided: positional tensor names, one file with metadata (no sidecar), `FullPCN`/`DKPPCN`
first and now every Python class, `*NetworkT` templates save tensors without a description, GGUF
covers every read quant and writes F32/F16/BF16/F64/ints and the Q and K quants.

## Housekeeping

- [x] Root cleanup committed.
- [x] `pyrightconfig.json` is untracked and ignored.

## Still open

- GPU: `GaussSeidelPCN` (Python `device=` added, test case written, not yet run), `SequentialPCN` (its Python class has no
  `device` and the native base takes no batch size with a device) and `FullConvPCNetwork` (native only).
- `dp.load` builds and randomizes a network, then overwrites it: about 1.6 s of a 2.0 s load of a 512 MiB model is that
  initialization (`configure`), the weight load itself is about 0.4 s. Skipping it needs care for Psi in inference artifacts.
- No upload (Sink), no C++ memory-buffer reader, no big-endian host support (BLAKE3 and GGUF reader are little-endian only),
  no IQ-quant GGUF writing, real S3 untested.
