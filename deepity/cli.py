"""Command line tools: `deepity inspect FILE` and `deepity convert SRC DST`."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Optional

import numpy as np

from ._backend import dy
from ._gguf_write import write_gguf
from ._serialization import DEFAULT_SHARD_BYTES, INDEX_SUFFIX, write_tensors
from ._sources import resolve

_UNITS = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12, "ki": 1 << 10, "mi": 1 << 20, "gi": 1 << 30, "ti": 1 << 40}


def parse_size(text: str) -> int:
    """`2GiB`, `500MB`, `1048576` -> bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]i?)?b?\s*", text.lower())
    if not match:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (try 500MB or 2GiB)")
    size = int(float(match.group(1)) * _UNITS[match.group(2) or ""])
    if size <= 0:
        raise argparse.ArgumentTypeError("size must be above zero")
    return size


def _human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def describe(path: str, verify_hash: bool = True) -> dict[str, Any]:
    """Everything `inspect` prints, as a dict (also its --json output)."""
    table = dy.file_table(resolve(path), verify_hash)
    info: dict[str, Any] = {"path": path, "format": table["format"], "version": table["version"]}
    tensors = [{"name": n, "dtype": d, "shape": list(s), "bytes": b} for n, d, s, b in table["tensors"]]
    info["tensors"] = tensors
    info["tensor_count"] = len(tensors)
    info["total_bytes"] = sum(t["bytes"] for t in tensors)
    info["parameters"] = sum(_elements(t["shape"]) for t in tensors)
    if table["format"] == "gguf":
        info["metadata"] = json.loads(table["metadata_json"])
        return info
    meta = dict(table["metadata"])
    info["shards"] = table["shards"]
    info["hash_verified"] = table["hash_verified"]
    info["kind"] = meta.get("deepity.kind")
    info["format_version"] = meta.get("deepity.format_version")
    info["created_by"] = meta.get("deepity.created_by")
    for key, name in (("deepity.model", "model"), ("deepity.inference", "inference"), ("deepity.card", "card"), ("deepity.training", "training")):
        if key in meta:
            info[name] = json.loads(meta[key])
    info["other_metadata"] = {k: v for k, v in meta.items() if not k.startswith("deepity.")}
    return info


def _elements(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _print(info: dict[str, Any], out) -> None:
    w = lambda *a: print(*a, file=out)
    w(f"{info['path']}")
    w(f"  format       {info['format']} v{info['version']}" + (f", {info['shards']} shard(s)" if info.get("shards", 1) > 1 else ""))
    if info["format"] == "safetensors":
        w(f"  kind         {info.get('kind') or 'plain tensors (no deepity.kind)'}")
        if info.get("created_by"):
            w(f"  created by   {info['created_by']}")
        w(f"  content hash {'verified' if info['hash_verified'] else 'not present or not checked'}")
        model = info.get("model")
        if model:
            w(f"  network      {model.get('network')}, batch {model.get('batch_size')}, device {model.get('device')}"
              + (f", dtype {model['dtype']}" if "dtype" in model else ""))
            sizes = " -> ".join(str(l.get("in", l.get("out_channels"))) for l in model.get("layers", [])) 
            w(f"  layers       {len(model.get('layers', []))}: {sizes}")
        if info.get("inference"):
            w(f"  inference    {info['inference']}")
        for key, value in (info.get("card") or {}).items():
            w(f"  card.{key:<8}{value}")
        if info.get("training"):
            w(f"  training     {json.dumps(info['training'])[:200]}")
        for key, value in info.get("other_metadata", {}).items():
            w(f"  meta.{key}  {str(value)[:120]}")
    else:
        for key, value in list(info["metadata"].items())[:40]:
            w(f"  {key}  {json.dumps(value)[:120]}")
    w(f"  {info['tensor_count']} tensors, {info['parameters']:,} elements, {_human(info['total_bytes'])}")
    width = max([len(t["name"]) for t in info["tensors"]] + [4])
    for t in info["tensors"][:200]:
        w(f"    {t['name']:<{width}}  {t['dtype']:<9} {str(tuple(t['shape'])):<18} {_human(t['bytes'])}")
    if info["tensor_count"] > 200:
        w(f"    ... {info['tensor_count'] - 200} more (use --json for the full table)")


def _training_only(name: str) -> bool:
    """True for a tensor name only training needs: Psi and optimizer state (mirrors C++ IsTrainingOnlyTensor)."""
    return name.startswith("optimizer.") or (name.startswith("layers.") and name.endswith(".psi")) or (
        name.startswith("layer_") and name.endswith(".Psi")
    )


def convert(
    src: str, dst: str, max_shard_bytes: Optional[int] = None, dtype: Optional[str] = None, for_inference: bool = False
) -> None:
    """Converts `src` (safetensors, shard index or GGUF) to a safetensors file at `dst`, or to GGUF when
    `dst` ends in `.gguf` (float tensors as `dtype`: f32, f16, bf16 or q8_0; default keeps them).

    A `dst` ending in `.safetensors.index.json` writes a sharded model (2 GiB shards unless
    `max_shard_bytes` is given). A safetensors source keeps every dtype, packed 4-bit ones included.
    A GGUF source is decoded to float32 (int tensors to int64) and its key-value header is kept as the
    `gguf.metadata` entry.

    With `for_inference`, tensors only training uses (Psi, optimizer state) and the saved training state are
    dropped, and the file is marked as an inference artifact. A checkpoint becomes a model file that way.

    `src` may also be an http(s) or s3 URL, which is downloaded to the cache first.
    """
    src = resolve(src)
    if dst.endswith(".gguf"):
        if max_shard_bytes:
            raise ValueError("GGUF output is a single file; --max-shard-size does not apply")
        if src.endswith(".gguf"):
            tensors, metadata = dy.read_gguf(src)
            kept = metadata if isinstance(metadata, dict) else json.loads(metadata)
            metadata = {k: v for k, v in kept.items() if k != "general.alignment"}
        else:
            tensors, metadata = dy.read_tensors(src), {}
            for key, value in dict(dy.read_model_metadata(src)).items():
                if key.startswith("deepity.") and key != "deepity.content_blake3":
                    metadata[key] = value
        arrays = {k: np.array(v) for k, v in tensors.items() if not (for_inference and _training_only(k))}
        write_gguf(dst, {k: v for k, v in arrays.items() if v.size > 0}, metadata, dtype)
        return
    if dtype:
        raise ValueError("--dtype only applies to a .gguf destination")
    shard = 0
    if dst.endswith(INDEX_SUFFIX):
        shard = max_shard_bytes or DEFAULT_SHARD_BYTES
    elif max_shard_bytes:
        raise ValueError(f"--max-shard-size needs a destination ending in {INDEX_SUFFIX}")
    if src.endswith(".gguf"):
        tensors, metadata = dy.read_gguf(src)
        write_tensors(dst, tensors, {"gguf.metadata": metadata, "converted_from": "gguf"}, shard or None)
    else:
        dy.convert_safetensors(src, dst, shard, for_inference)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="deepity", description="Inspect and convert Deepity model files.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_inspect = sub.add_parser("inspect", help="validate a model file and print what is in it")
    p_inspect.add_argument("file", help=".safetensors, .safetensors.index.json or .gguf")
    p_inspect.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_inspect.add_argument("--no-verify", action="store_true", help="skip the content hash check (trusted local files)")
    p_convert = sub.add_parser("convert", help="convert between safetensors, shard sets and GGUF")
    p_convert.add_argument("src")
    p_convert.add_argument("dst")
    p_convert.add_argument("--max-shard-size", type=parse_size, help="shard size for a .safetensors.index.json destination, e.g. 500MB")
    p_convert.add_argument("--inference", action="store_true", help="drop what only training uses (Psi, optimizer state) to make a smaller inference file")
    p_convert.add_argument("--dtype", choices=["f32", "f16", "bf16", "q8_0"], help="float storage type for a .gguf destination")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            info = describe(args.file, not args.no_verify)
            if args.json:
                json.dump(info, sys.stdout, indent=2)
                print()
            else:
                _print(info, sys.stdout)
        else:
            convert(args.src, args.dst, args.max_shard_size, args.dtype, args.inference)
            print(f"wrote {args.dst}")
    except Exception as error:  # noqa: BLE001 - a CLI reports every failure the same way
        print(f"deepity: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
