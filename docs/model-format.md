# Deepity model format (draft v0)

Status: draft for discussion. Nothing here is implemented yet. See `TODO.md` for the plan.

## Goals

- A model is one file that can be moved between machines, downloaded, paused and resumed.
- Loading needs no hand-built architecture: the file describes it.
- Readable by the stock `safetensors` library in any language, with zero-copy mmap.
- Safe to load from an untrusted source.

## Container

The container is the safetensors format, unchanged:

1. 8 bytes: little-endian `u64` header size `N`.
2. `N` bytes: UTF-8 JSON header, padded with trailing spaces so `8 + N` is a multiple of 64.
3. Tensor data, back to back. Each tensor starts at a 64-byte aligned file offset (insert
   alignment gaps if needed; the header's `data_offsets` are relative to the end of the header).

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
| `deepity.content_sha256` | Optional hex hash of the data region. |
| `deepity.dtypes` | Optional JSON map from tensor name to logical dtype, for non-standard types. |

Readers ignore unknown `deepity.*` keys. Readers must reject a `format_version` greater than the
one they support, with a message naming both versions.

## Model description (`deepity.model`)

```json
{
  "network": "DKPPCN",
  "batch_size": 64,
  "layers": [
    {"kind": "linear", "in": 784, "out": 512, "activation": "relu"},
    {"kind": "linear", "in": 512, "out": 10, "activation": "linear"}
  ]
}
```

`network` selects the class. Layer kinds and fields are defined per network and versioned by
`format_version`. Unknown fields are ignored.

## Inference defaults (`deepity.inference`)

```json
{"inference_steps": 1, "inference_rate": 0.1, "cross_entropy": false}
```

These are the defaults a loaded model uses for `predict`. Callers can override them. A file
states which prediction it describes: the full iterative model, or a forward-only export.

## Tensor names

`layers.{i}.weight`, `layers.{i}.bias` for every artifact. A checkpoint adds:

- `layers.{i}.psi` (DKP feedback weights, full `[size, terminal]` shape)
- `optimizer.layers.{i}.{weight,bias,psi}.{m,v}` and scalar `optimizer.step`
- `rng.state`

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
`deepity.dtypes`, so stock readers still open the file. Open question below.

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

## Open questions

- Custom dtype storage: `U8` plus `deepity.dtypes`, or register names with the safetensors project?
- Single file only, or also a sharded layout with `model.safetensors.index.json` for large models?
- How much of the layer description is shared between the float and templated layer families?
- Is `deepity.content_sha256` mandatory for checkpoints?
