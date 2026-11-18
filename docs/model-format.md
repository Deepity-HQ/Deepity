# Deepity model format (v1)

Status: format version 1, implemented. The container, hash, tensor names, `deepity.model` and
`deepity.inference` are written and read for every Python network class. Checkpoints, sharding,
mmap and the converters are described below. Open items are in `TODO.md`.

## Goals

- A model is one file that can be moved between machines, downloaded, paused and resumed.
- Loading needs no hand-built architecture: the file describes it.
- Readable by the stock `safetensors` library in any language, with zero-copy mmap.
- Safe to load from an untrusted source.

## Container

The container is the safetensors format, unchanged:

1. 8 bytes: little-endian `u64` header size `N`.
2. `N` bytes: UTF-8 JSON header, padded with trailing spaces so `8 + N` is a multiple of 64.
3. Tensor data, back to back with no gaps (stock safetensors rejects gaps). The header's
   `data_offsets` are relative to the end of the header. The writer orders tensors by decreasing
   element size, so every tensor starts at a multiple of its own element size, and the data
   region starts on a 64-byte boundary. Readers do not require alignment.

Tensor entries in the JSON header use the standard keys: `dtype`, `shape`, `data_offsets`.
`__metadata__` is a string-to-string map, as safetensors requires.

## Metadata keys (`__metadata__`)

All values are strings. Deepity keys are prefixed `deepity.`.

| Key | Value |
|---|---|
| `deepity.format_version` | `"1"`. Bumped only for incompatible changes. |
| `deepity.kind` | `"inference"` or `"checkpoint"`. |
| `deepity.training_tensors` | Optional. `"omitted"` marks an inference artifact written without training-only tensors (Psi). |
| `deepity.model` | JSON string: architecture, see below. |
| `deepity.inference` | JSON string: default settling settings, see below. |
| `deepity.created_by` | e.g. `"deepity 0.1.0a9"`. Informational. |
| `deepity.content_blake3` | Optional hex hash of the data region. |
| `deepity.dtypes` | Optional JSON map from tensor name to logical dtype, for non-standard types. |

Readers ignore unknown `deepity.*` keys, unknown tensors, and unknown fields inside the JSON
values (a hyperparameter this version cannot take is dropped with a warning). Only the version is
checked, on every load path (`load`, `load_weights`, `resume`, metadata reads):

- No `format_version` at all (a foreign or pre-versioned file) loads.
- A value that is not a positive integer is an error ("invalid").
- A version greater than the reader's is an error naming both ("written with model format version 2,
  but this build reads up to version 1"). `deepity inspect` still opens such a file and shows it.

What can still fail is a model description the build cannot construct: an unknown `network`, layer
`kind` or activation, or a layer missing a required field. Each is an error that names it. The
version is bumped for any change an older reader could not safely ignore, such as a new required
tensor, a changed layout or meaning of an existing field, or a new layer kind.

## Model description (`deepity.model`)

Written by every Python network class (weights never pass through Python). The C++ `*NetworkT`
templates save and load the same tensors under the same names but write no `deepity.model`, so a
file made from them needs a hand-built network of matching layers to load into.

```json
{
  "network": "DKPPCN",
  "batch_size": 64,
  "device": "cpu",
  "dtype": "float32",
  "layers": [
    {"kind": "linear", "in": 784, "out": 512, "activation": "relu"},
    {"kind": "linear", "in": 512, "out": 10, "activation": "linear"}
  ],
  "hyperparameters": {"learning_rate": 0.001, "inference_rate": 0.1, "lmbda": 0.01}
}
```

`network` selects the class (`dtype` appears only for networks that have one). `hyperparameters`
are the keyword arguments of the network's `configure()`. A convolution layer is
`{"kind": "convolution", "out_channels", "kernel": [h, w], "stride": [h, w], "pad": [h, w], "activation"}`.
Layer kinds and fields are versioned by `format_version`. Unknown fields are ignored.
`deepity.load(path)` rebuilds the network, then loads the weights over the fresh initialization.

## Inference defaults (`deepity.inference`)

```json
{"mode": "iterative", "inference_steps": 1}
```

These are the defaults a loaded model uses for `predict`. Callers can override them. `mode` states
which prediction the file describes: `"iterative"` is the full settling model (the only mode
written today); `"forward"` is reserved for a forward-only export. A file with a missing `mode` is
`"iterative"`, and a loader rejects a mode it does not know instead of guessing.

## Importing weights from other files

`net.load_weights(path, strict=True, rename=None)` (C++: `ModelIO::LoadWeights`) fills a network
from any safetensors file, such as a PyTorch export. `rename` maps each file tensor name to a
Deepity name (`layers.0.weight`). Dtype and shape must match. With `strict=True` any missing or
unused tensor raises, naming all of them; with `strict=False` the call returns
`(missing, unexpected)` and loads the rest. Optimizer tensors in a checkpoint are not "unexpected".

## Tensor names

`layers.{i}.weight`, `layers.{i}.bias` for every artifact, plus `layers.{i}.p` (precisions) where a network
has them and `layers.{i}.psi` (feedback weights) for the DKP family. `predict()` never reads Psi, so
an **inference artifact** may leave it out: `save(path, for_inference=True)` or
`deepity convert SRC DST --inference` (which also turns a checkpoint into one, dropping the optimizer
tensors and `deepity.training`). Such a file has `deepity.training_tensors = "omitted"`. It loads and predicts identically; a
network loaded from it that trains again starts from a fresh Psi, so use a checkpoint to resume.
A plain `save()` still writes Psi. Without the marker, a file that lacks any tensor the network needs
is an error naming it (a truncated file never loads with random weights). A checkpoint cannot omit
training tensors. A checkpoint adds:

- `layers.{i}.psi` (DKP feedback weights, full `[size, terminal]` shape; also in plain `save()` files)
- `optimizer.layers.{i}.{weight,bias,psi}.{m,v}`: Adam/AdamW moments, same shapes and dtype
  rules as the matching parameter (`F32` or `F64` by the network's accumulation type)
- `optimizer.layers.{i}.step` and `optimizer.layers.{i}.psi_step`: `I32`, shape `[1]`, the
  Adam/AdamW step counts for W and b, and for Psi
- Caller state (epoch, global step, learning-rate schedule position, RNG state) is the string
  metadata key `deepity.training`, a JSON object the caller supplies and gets back unchanged

A checkpoint has `deepity.kind = "checkpoint"`. Loading one as an inference model reads its
weights and ignores the rest. Implemented for `DirectKPPCNetwork`, `DirectKPPCNetworkT`, `FullPCNetwork`,
`FullPCNetworkT`, `FullConvPCNetwork`, `GaussSeidelPCNetwork`, `GaussSeidelPCNetworkT`,
`DiscriminativePCNetwork`, `ConvPCNetwork`, `SimplePCNetwork`, `SimpleConvPCNetwork`,
`ConvPCNetworkT`, `FullConvPCNetworkT` and the `T` forms of the Simple networks: every network class has
a state dict now. After loading weights, layers that cache log(precision) rebuild it
from the loaded p. (The conv layers already keep their
moments in the ordinary state dict as `m_W`, `v_W`, ..., so a conv checkpoint adds only the step
counters.) A network whose layers have no optimizer accessors refuses to write one.

On a GPU backend a layer names the backend that holds its tensors and counters
(`Layer::DeviceStateBackend()`), and `ModelIO` copies them through it. The file is identical to a CPU
one, so a checkpoint moves between devices. Tested on the CPU with `ForceDeviceStaging()`, and on an
A100 for `FullPCN`, `DKPPCN`, `SimplePCN`, `SimpleConvolutionalPCN` and `ConvolutionalPCN`. A GPU
resumes its own checkpoint exactly; CPU and GPU predictions differ by about 1e-3 relative (TF32).
`GaussSeidelPCN`, `SequentialPCN` and `FullConvPCNetwork` have not been run on a GPU.

Layers with no weights (the terminal layer) write no tensors. Zero-size tensors are not written.
Legacy files use `layer_{i}.{W,b,Psi}`; a reader detects them by
`__metadata__.format == "deepity_safetensors"` and maps the names.

Decided: names are positional per layer (`layers.{i}.weight`), not per-module paths. A network is a
flat list of layers, so the index is stable and needs no registry. The legacy `layer_N.*` names are
read for compatibility and never written. Other files' names are mapped on import with `rename`.

## Layout rules

- Row-major, little-endian on disk.
- `shape` is the logical shape. Padded rows (Psi `rowStride`) are never stored.
- Dense `weight`: `[out, in]`. Convolution `weight`: `[outChannels, inChannels * kH * kW]`.

## Dtypes

Every dtype is a row in the registry (`include/deepity/io/DType.h`): a logical name, a block of
`elementsPerBlock` elements that takes `bytesPerBlock` bytes on disk, and a storage dtype.

- **Standard** types (`F64 F32 F16 BF16 I64 I32 I16 I8 U64 U32 U16 U8 BOOL`) store as themselves,
  one element per block, in the header's own dtype.
- **Packed** types store as a 1-D `U8` tensor of whole blocks, with the logical dtype and shape in
  `deepity.dtypes`, a JSON object `{"<tensor>": {"dtype": "FP4_E2M1", "shape": [7]}}`. Stock
  readers see opaque bytes and open the file; only Deepity interprets them. The last block is
  zero padded.
- Defined packed types: `FP4_E2M1`, `POSIT4_0`, `POSIT4_1`, `POSIT4_2`. Two 4-bit codes per byte
  (a 2-element block of 1 byte), the first element in the low nibble. In memory each code takes a
  byte.
- A reader validates the overlay against the header: the tensor exists, is stored as 1-D `U8`, the
  logical dtype is a packed type it knows, and the byte count equals the block arithmetic for the
  logical shape. Anything else is rejected.
- A new block type (for example a GGUF-style quantization) is one registry row plus its codec.
- Files from 0.1.0a8 that wrote these names directly as header dtypes (one byte per element) still
  load.

## Byte order

Files are little endian, always: the 8-byte header length, the tensor data and the hash. The
container code (`SafeTensorsReader`/`Writer`) swaps each multi-byte element of a standard dtype on
write and on read on a big-endian host; packed types and single bytes need no swap, and a mapped
file gives no zero-copy view of a tensor that would need swapping. The content hash covers the bytes
as stored, so it is the same on every host.

**Big-endian hosts are not supported yet.** The swap logic is tested by simulating such a host
(`SimulateBigEndianHost()`, which must write byte-identical files, single and sharded, and read them
back swapped), but two parts read little-endian data in host order and are unfinished: BLAKE3
(`Blake3.cpp` refuses to compile on a big-endian target) and the GGUF reader (throws at open).
Making them correct needs byte-swapped loads and a real big-endian machine or emulator to test on.

## Validation (every reader)

- Header size is capped (suggested 16 MiB) and checked against the file size.
- Every `data_offsets` range lies inside the file, ranges do not overlap, and offsets are in order.
- `prod(shape) * dtype size` equals the byte length of each tensor.
- dtype is on an allow-list; unknown dtypes are an error, not a guess.
- A tensor required by the model description but missing, or with the wrong shape, is an error
  that names the tensor.

## Compatibility

- Format versions are integers. A reader supports every version up to its own.
- Golden files for each version live in `tests/data/` and are loaded by CI on every build:
  `golden_v1.safetensors` (container), `golden_v1_dtypes.safetensors` (one tensor for every dtype,
  exact bytes checked) and `legacy_a8_dkp.safetensors`. Regenerate with
  `SafeTensorsVerify --write-golden <path>` or `--write-golden-dtypes <path>`, only on a deliberate
  format change.
- Files written by 0.1.0a7 and a8 (legacy names, no metadata) stay readable. The two releases wrote
  the same format (`ModelIO.h` is identical between the tags), so one fixture, `legacy_a8_dkp.safetensors`,
  covers both.

## Sharding

A large model may be split across files with an index, as Hugging Face does. `ModelIO::SaveSharded`
(and `SaveCheckpointSharded`) write `<stem>-00001-of-0000N.safetensors` files plus
`<stem>.safetensors.index.json`. From Python, `net.save(path, max_shard_bytes=...)` and
`save_checkpoint` do the same when `path` ends in `.safetensors.index.json` (default 2 GiB per shard).

```json
{"metadata": {"total_size": 123456, "shard_count": 4},
 "weight_map": {"layers.0.weight": "m-00001-of-00004.safetensors", "...": "..."}}
```

- A single file is always a valid model. The index is optional and lives next to the shards.
- Tensor names never encode the shard. Readers resolve names through `weight_map`.
- Every shard is a complete safetensors file with the full `deepity.*` metadata and its own
  `deepity.content_blake3`, so a shard also opens alone in a stock reader.
- Shards fill in tensor order and close when the next tensor would pass the size limit. A tensor
  larger than the limit gets a shard of its own.
- Shards are written first and the index last, each atomically, so an index never names a missing
  shard. Saving again over an old set does not delete shards the new set no longer uses.
- A reader opens a path ending in `.index.json` as one logical file. It rejects a shard name that
  is not a plain file name beside the index, a tensor present in a different shard than the index
  says, an indexed tensor that no shard holds, and an index with no tensors.

## Importing from PyTorch

`deepity.import_weights(net, source, rename=None, strict=False)` accepts a `state_dict`, a `.pt` file
(loaded with `weights_only=True`), a safetensors file or a dict of arrays, and loads it through the
same C++ path as `load_weights`. `rename="sequential"` maps each module's `weight` and `bias`, in
file order, to `layers.0`, `layers.1`, ...; PyTorch `Linear.weight` is `[out, in]`, as here. Feedback
weights (`psi`) are never expected from a foreign file. The terminal layer's own tensors are reported
missing, since a PyTorch model has no counterpart.

## Exporting to ONNX

`deepity.export_onnx(net, path)` writes `act(x W0^T + b0) W1^T + b1 ...` as Gemm and activation nodes
(opset 20, dynamic batch). That is `net.predict(x, inference_steps=0)`, verified against
onnxruntime to 1e-5 for relu, sigmoid, tanh and gelu (Deepity's tanh approximation). It carries no
settling, so it can differ from `predict` with the model's default steps. Convolutional networks
(`SimpleConvolutionalPCN`, `ConvolutionalPCN`) export as Reshape, Conv, activation and Flatten nodes; the input
stays the flattened `[batch, C*H*W]` and the reference is `predict_with_projection(x, 0)`, because their
plain `predict(x, 0)` leaves the output at zero. Mixing dense and convolution layers is not supported.

`export_onnx(net, path, settling_steps=N)` makes the graph match `net.predict(x, N)`. `SimplePCN` and
`SequentialPCN` predict from a cleared state, so their output depends on the step count and the settling is
unrolled into the graph (about 15 nodes per layer per step; verified against `predict` to 1e-7 for relu, tanh,
sigmoid and gelu at 1, 5 and 20 steps, batch size dynamic). `SequentialPCN`'s learned per-feature precisions
(`layers.i.p`) are baked in as constants. `FullPCN`, `DKPPCN` and `GaussSeidelPCN` predict from the forward pass, where every error is
already zero, so settling changes nothing and the plain forward graph is exact for any N. the conv
networks raise `NotImplementedError` for `settling_steps > 0`.

A `GaussSeidelPCN` has no separate terminal layer, so its `predict` returns the last layer's belief, the
previous layer's prediction; its exported graph therefore leaves the last `Linear` out and ends one layer earlier.

## Reading GGUF

`deepity.read_gguf(path)` returns `(tensors, metadata)` for GGUF versions 2 and 3. Decoded types:
F32, F16, BF16, F64, I8..I64 and the block formats Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K and the IQ quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS (all checked against the reference `gguf` package; the IQ lookup grids in `src/io/GGUFTables.cpp` are generated from it). Floats come back as
float32, integers as int64, shapes reversed to row-major. Other types (Q8_1, the ternary TQ formats, MXFP4) raise an
error naming the type. Reading is bounds checked: counts, string and
array lengths and every tensor range must fit the file, names must be unique, and alignment must be
a power of two. GGUF is not a Deepity format; weights still need `import_weights` and a name map.

## Writing GGUF

`deepity.write_gguf(path, tensors, metadata, dtype)` writes numpy arrays as GGUF v3 (32-byte alignment,
atomic write). Floats are stored as F32 or F16 by default, F64 is kept, and `dtype` can force `"f32"`,
`"f16"`, `"bf16"` or `"q8_0"` for every float tensor (Q8_0 leaves 1-D tensors and widths not divisible
by 32 as F32). Integer tensors keep their width. Names are limited to 64 bytes and shapes to 4
dimensions, GGUF's own limits, and empty tensors are rejected. Metadata values may be str, int, float,
bool, homogeneous lists, or dicts (stored as JSON strings); `general.architecture` defaults to
`"deepity"`. `deepity.export_gguf(net, path, dtype)` writes a network's weights with its description under
`deepity.model`, and `deepity convert SRC out.gguf [--dtype q8_0]` does the same from a safetensors file.
Other quant formats are not written. A Deepity GGUF file loads back with `import_weights`; llama.cpp has no
loader for Deepity architectures, so the file is for exchange and inspection, not for running there.

## Model card (`deepity.card`)

A JSON object of plain values the author chooses, at most 64 KiB, for example
`{"license": "MIT", "description": "...", "training_data": "MNIST", "metrics": {"accuracy": 0.97}}`.
Set it with `net.card = {...}` before `save`. `deepity.load` puts it back on the network, so a
re-save keeps it. No field is required and loaders ignore ones they do not know.

## Command line

```
deepity inspect model.safetensors [--json] [--no-verify]
deepity convert model.gguf model.safetensors
deepity convert model.safetensors model.safetensors.index.json --max-shard-size 500MB
```

`inspect` validates the file (hash included unless `--no-verify`) and prints the network, layers,
card, metadata and a tensor table. `convert` reads a safetensors file, shard set or GGUF file.

## Plain tensor files

`deepity.write_tensors` and `deepity.read_tensors` use the same container for arbitrary numpy data
with no model attached (no `deepity.kind`, only the metadata you pass and the content hash).
Supported: float16/32/64, int8..64, uint8..64 and bool (stored as `BOOL`, one byte).

## Sources

Every Python loader (`deepity.load`, `resume`, `net.load_weights`, `net.load_checkpoint`, `read_tensors`,
`read_card`, `import_weights`, `read_gguf`, `deepity inspect` and `convert`) accepts any of these where it
takes a path, and resolves it to a local file first (`deepity/_sources.py`):

| Source | Behaviour |
|---|---|
| a path | used as is, no cost |
| `file://...` | the local path |
| `http://`, `https://` | downloaded to the cache, then loaded through mmap |
| `s3://bucket/key` | through `boto3` when installed (private buckets, your credentials), else the bucket's public HTTPS address |
| `bytes`, `bytearray`, `memoryview` | written once to the cache under their content hash |

The cache is `$DEEPITY_CACHE_DIR`, else `~/.cache/deepity`. Rules:

- A finished download is checked before it is put in place (structure and the content hash when the file has
  one), by an atomic rename, so an interrupted or corrupt download leaves nothing behind.
- A cached file is revalidated with the server's ETag on every load (an unchanged model costs one small
  request, not a download). With no network the cached copy is used with a warning. A server that sends no
  validator is trusted as is; `fetch(url, refresh=True)` forces a download. `DEEPITY_OFFLINE=1` never
  touches the network.
- `deepity.fetch(url, ...)` takes `headers`, `timeout`, `max_bytes` (also `$DEEPITY_MAX_DOWNLOAD_BYTES`, default
  256 GiB, enforced with or without a `Content-Length`) and `sha256` (the expected hash of the file).
- A `.safetensors.index.json` URL downloads the index and then every shard it names from the same
  folder. A shard name that is not a plain file name is refused, as in the local reader.
- Redirects are followed only over http(s), never from https down to http, and credentials are dropped when
  a redirect goes to another host.
- A signed URL's query string is part of its cache key, so a fresh signature downloads again, and a
  sharded model needs an unsigned base URL.

Not built: uploading (no Sink), `hf://`, a C++ memory-buffer reader (bytes are written to the cache, then
mapped), and resuming a partial download.

## Memory mapping

`SafeTensorsReader` maps the file by default and falls back to buffered reads. `Bytes(entry)`
gives a zero-copy pointer and `Mapping(entry)` an owner that keeps it valid, which is how
`deepity.read_tensors` returns read-only numpy views. Truncating a mapped file while it is open
can crash the reader, as with any mmap; write models by replacing the file, as `Save` does.

## Hash

`deepity.content_blake3` is optional and written by default. It is the BLAKE3 hash (32 bytes) of the
data region (every byte after the header), as lowercase hex. Loaders verify it when present, and offer a flag to skip verification for
trusted local files. (Decided 2026-11-14; SHA-256 replaced by BLAKE3 on 2026-11-17 for speed. Files with a
`deepity.content_sha256` key from drafts are not supported.)
