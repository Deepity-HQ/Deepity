# Deepity model format (draft v0)

Status: draft. The container, hash, tensor names and `deepity.model`/`deepity.inference` for
`DKPPCN` and `FullPCN` are implemented; the rest is planned. See `TODO.md`.

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
| `deepity.model` | JSON string: architecture, see below. |
| `deepity.inference` | JSON string: default settling settings, see below. |
| `deepity.created_by` | e.g. `"deepity 0.1.0a9"`. Informational. |
| `deepity.content_blake3` | Optional hex hash of the data region. |
| `deepity.dtypes` | Optional JSON map from tensor name to logical dtype, for non-standard types. |

Readers ignore unknown `deepity.*` keys. Readers must reject a `format_version` greater than the
one they support, with a message naming both versions.

## Model description (`deepity.model`)

Implemented for `DKPPCN` and `FullPCN` (Python layer; weights never pass through Python).

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
{"inference_steps": 1}
```

These are the defaults a loaded model uses for `predict`. Callers can override them. A file
states which prediction it describes: the full iterative model, or a forward-only export.

## Tensor names

`layers.{i}.weight`, `layers.{i}.bias` for every artifact. A checkpoint adds:

- `layers.{i}.psi` (DKP feedback weights, full `[size, terminal]` shape)
- `optimizer.layers.{i}.{weight,bias,psi}.{m,v}`: Adam/AdamW moments, same shapes and dtype
  rules as the matching parameter (`F32` or `F64` by the network's accumulation type)
- `optimizer.layers.{i}.step` and `optimizer.layers.{i}.psi_step`: `I32`, shape `[1]`, the
  Adam/AdamW step counts for W and b, and for Psi
- Caller state (epoch, global step, learning-rate schedule position, RNG state) is the string
  metadata key `deepity.training`, a JSON object the caller supplies and gets back unchanged

A checkpoint has `deepity.kind = "checkpoint"`. Loading one as an inference model reads its
weights and ignores the rest. Implemented for `DirectKPPCNetwork`, `FullPCNetwork`,
`FullPCNetworkT` (any dtype) and `FullConvPCNetwork`, CPU only. (The conv layers already keep their
moments in the ordinary state dict as `m_W`, `v_W`, ..., so a conv checkpoint adds only the step
counters.) Other networks and GPU networks refuse to write one.

Layers with no weights (the terminal layer) write no tensors. Zero-size tensors are not written.
Legacy files use `layer_{i}.{W,b,Psi}`; a reader detects them by
`__metadata__.format == "deepity_safetensors"` and maps the names.

## Layout rules

- Row-major, little-endian on disk.
- `shape` is the logical shape. Padded rows (Psi `rowStride`) are never stored.
- Dense `weight`: `[out, in]`. Convolution `weight`: `[outChannels, inChannels * kH * kW]`.

## Dtypes

Standard safetensors names are used where one exists (`F32`, `F16`, `F64`). Types safetensors
does not know (posit4, packed fp4 variants) are stored as `U8` with the true type recorded in
`deepity.dtypes`, so stock readers still open the file. A stock reader sees opaque bytes, which is
safe; only Deepity interprets them. (Decided 2026-11-14.)

## Validation (every reader)

- Header size is capped (suggested 16 MiB) and checked against the file size.
- Every `data_offsets` range lies inside the file, ranges do not overlap, and offsets are in order.
- `prod(shape) * dtype size` equals the byte length of each tensor.
- dtype is on an allow-list; unknown dtypes are an error, not a guess.
- A tensor required by the model description but missing, or with the wrong shape, is an error
  that names the tensor.

## Compatibility

- Format versions are integers. A reader supports every version up to its own.
- Golden files for each version live in `tests/data/` and are loaded by CI on every build.
- Files written by 0.1.0a7 and a8 (legacy names, no metadata) stay readable.

## Sharding (planned, not built yet)

Large models may later be split across files with a `model.safetensors.index.json` mapping tensor
names to shard files, as Hugging Face does. Rules reserved now so v1 files stay valid:

- A single file is always a valid model. The index is optional and lives next to the shards.
- Tensor names never encode the shard. Readers resolve names through the index when it exists.
- `deepity.format_version` and `deepity.model` appear in every shard's metadata.

## Hash

`deepity.content_blake3` is optional and written by default. It is the BLAKE3 hash (32 bytes) of the
data region (every byte after the header), as lowercase hex. Loaders verify it when present, and offer a flag to skip verification for
trusted local files. (Decided 2026-11-14; SHA-256 replaced by BLAKE3 on 2026-11-17 for speed. Files with a
`deepity.content_sha256` key from drafts are not supported.)

## Open questions

- How much of the layer description is shared between the float and templated layer families?
