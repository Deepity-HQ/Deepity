# Deepity model format (v1)

Status: format version 1, implemented. The container, hash, tensor names, `deepity.model` and
`deepity.inference` are written and read for every Python network class. Checkpoints, sharding,
mmap and the converters are described below. Open items are in `TODO.md`.

## Goals

- A model is one file that can be moved between machines, paused and resumed.
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
from any safetensors file, such as one exported from another framework. `rename` maps each file tensor name to a
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
- A new block type is one registry row plus its codec.
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
back swapped), but one part reads little-endian data in host order and is unfinished: BLAKE3
(`Blake3.cpp` refuses to compile on a big-endian target).
Making it correct needs byte-swapped loads and a real big-endian machine or emulator to test on.

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

## Exporting to ONNX

`deepity.export_onnx(net, path)` writes `act(x W0^T + b0) W1^T + b1 ...` as Gemm and activation nodes
(opset 20, dynamic batch). That is `net.predict(x, inference_steps=0)`, verified against
onnxruntime to 1e-5 for relu, sigmoid, tanh and gelu (Deepity's tanh approximation). It carries no
settling, so it can differ from `predict` with the model's default steps. Convolutional networks
(`SimpleConvolutionalPCN`, `ConvolutionalPCN`) export as Reshape, Conv, activation and Flatten nodes; the input
stays the flattened `[batch, C*H*W]` and the reference is `predict_with_projection(x, 0)`, because their
plain `predict(x, 0)` leaves the output at zero. Mixing dense and convolution layers is not supported.

`SimplePCN` and `SequentialPCN` predict by settling from a cleared state, so their plain `predict` is not the
forward path and the exported graph is only a forward approximation of them; `FullPCN`, `DKPPCN` and
`GaussSeidelPCN` predict from the forward pass, where every error is already zero, so the graph matches
`predict` at any step count. The iterative settling itself is not exported.

A `GaussSeidelPCN` has no separate terminal layer, so its `predict` returns the last layer's belief, the
previous layer's prediction; its exported graph therefore leaves the last `Linear` out and ends one layer earlier.

## Model card (`deepity.card`)

A JSON object of plain values the author chooses, at most 64 KiB, for example
`{"license": "MIT", "description": "...", "training_data": "MNIST", "metrics": {"accuracy": 0.97}}`.
Set it with `net.card = {...}` before `save`. `deepity.load` puts it back on the network, so a
re-save keeps it. No field is required and loaders ignore ones they do not know.

## Command line

```
deepity inspect model.safetensors [--json] [--no-verify]
deepity convert model.safetensors model.safetensors.index.json --max-shard-size 500MB
```

`inspect` validates the file (hash included unless `--no-verify`) and prints the network, layers,
card, metadata and a tensor table. `convert` converts between a single safetensors file and a shard set, or strips a checkpoint down to an
inference file (`--inference`).

## Plain tensor files

`deepity.write_tensors` and `deepity.read_tensors` use the same container for arbitrary numpy data
with no model attached (no `deepity.kind`, only the metadata you pass and the content hash).
Supported: float16/32/64, int8..64, uint8..64 and bool (stored as `BOOL`, one byte).

## Memory mapping

`SafeTensorsReader` maps the file by default and falls back to buffered reads. `Bytes(entry)`
gives a zero-copy pointer and `Mapping(entry)` an owner that keeps it valid, which is how
`deepity.read_tensors` returns read-only numpy views. Truncating a mapped file while it is open
can crash the reader, as with any mmap; write models by replacing the file, as `Save` does.

On Windows a mapped file cannot be replaced or deleted at all, so saving over a path whose tensors are still
viewed (for example an array from `read_tensors`) fails with an error that says so. Drop the arrays, or pass
`mmap=False` to `read_tensors`, before overwriting the file.

## Hash

`deepity.content_blake3` is optional and written by default. It is the BLAKE3 hash (32 bytes) of the
data region (every byte after the header), as lowercase hex. Loaders verify it when present, and offer a flag to skip verification for
trusted local files. (Decided 2026-11-14; SHA-256 replaced by BLAKE3 on 2026-11-17 for speed. Files with a
`deepity.content_sha256` key from drafts are not supported.)

## Load speed

Measured with `experiments/benchmark_model_io.py 512` on an OSC Ascend node (a0104), 2026-11-18: one
512 MiB file of 8 float32 tensors, a full load that touches every byte, best of 5, warm page cache.
Cold-disk loads will be slower. Reproduce with
`OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run --with safetensors python experiments/benchmark_model_io.py 512`.

| Load | Time | Throughput |
|---|---|---|
| Deepity file, mmap, verify off | 101.6 ms | 5.28 GB/s |
| `deepity.read_tensors`, mmap, verify off | 115.1 ms | 4.66 GB/s |
| `safetensors.numpy.load_file` | 189.2 ms | 2.84 GB/s |
| `deepity.read_tensors`, copy, verify off | 201.2 ms | 2.67 GB/s |
| Deepity file, mmap, verify on (BLAKE3) | 334.7 ms | 1.60 GB/s |
| `np.load` (`.npz`) | 356.1 ms | 1.51 GB/s |

Mapping makes a load about 1.7 times faster than stock safetensors, and the copying path is on par with it.
Checking the BLAKE3 hash adds about 230 ms here (roughly 2.2 GB/s of hashing), so pass the skip-verification
flag for a trusted local file when load time matters. The direct-to-device CUDA load is not in this table
because the script does not time it yet.
