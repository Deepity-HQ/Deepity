# BetterModelIO: plan

Branch: `BetterModelIO`. Goal: make Deepity models portable. Save, pause, resume, ship, download,
and move between formats, with IO fast enough that nobody writes a Python loop over weights.

## Decisions so far

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
- Not yet verified: whether the stock `safetensors` library can read our current files.

## Phase 0: spec and test fixtures (before any code)

- [ ] Write `docs/model-format.md`: file layout, metadata schema, dtype names, tensor naming,
      layout conventions (conv `[outC, inC*kH*kW]`, Psi padding), versioning rules.
- [ ] Check that stock `safetensors` reads and writes our current files. Record the gaps.
- [ ] Add golden files (a7/a8-era file, plus one per dtype) and a CI test that loads them forever.
      Note `.gitignore` excludes `*.safetensors` and `*_meta.json`, so fixtures need an exception.
- [ ] Decide naming: stable tensor names (for example `layers.3.weight`) instead of positional keys.

## Phase 1: core format cleanup

- [ ] Conform to the safetensors spec (header padding, metadata as string map, contiguous offsets).
- [ ] `format_version` plus unknown-field-tolerant reads. Reader for current a7/a8 files.
- [ ] Validate untrusted input: header size cap, offsets against real file size, no overlaps,
      `prod(shape) * dtype size == bytes`, dtype allow-list.
- [ ] Atomic writes (temp file plus rename) and optional content hash.
- [ ] Replace `bool` returns with exceptions or a result type that carries the reason.
- [ ] Aligned tensor data (64 bytes) so mmap and SIMD paths work.

## Phase 2: dtype system

- [ ] Registry that records logical type, storage layout, block size and bytes per block.
      Needed for sub-byte types (fp4, posit4) and GGUF-style block quant (Q4_K and similar).
- [ ] Endianness policy (little endian on disk, byte-swap on big-endian hosts).

## Phase 3: self-describing models

- [ ] Architecture and inference settings serialized as versioned JSON in the metadata block.
- [ ] `deepity.load(path)` rebuilds the network from the file. No hand-built architecture first.
- [ ] `net.load_weights(path, strict=False, rename=...)` for partial loads.
- [ ] Two artifact kinds: inference artifact (weights plus settings) and training checkpoint
      (adds Psi, optimizer state, RNG, step, learning-rate schedule position).
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

- [ ] Commit the root cleanup (deleted `.whl`, `gpu_interop_verify.*`).
- [ ] `pyrightconfig.json` is tracked but also listed in `.gitignore`. Pick one.
