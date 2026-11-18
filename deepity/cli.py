"""Command line tools: `deepity inspect FILE` and `deepity convert SRC DST`."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Optional

from ._backend import dy
from ._serialization import DEFAULT_SHARD_BYTES, INDEX_SUFFIX, write_tensors

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
    table = dy.file_table(path, verify_hash)
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


def convert(src: str, dst: str, max_shard_bytes: Optional[int] = None) -> None:
    """Converts `src` (safetensors, shard index or GGUF) to a safetensors file at `dst`.

    A `dst` ending in `.safetensors.index.json` writes a sharded model (2 GiB shards unless
    `max_shard_bytes` is given). A safetensors source keeps every dtype, packed 4-bit ones included.
    A GGUF source is decoded to float32 (int tensors to int64) and its key-value header is kept as the
    `gguf.metadata` entry.
    """
    shard = 0
    if dst.endswith(INDEX_SUFFIX):
        shard = max_shard_bytes or DEFAULT_SHARD_BYTES
    elif max_shard_bytes:
        raise ValueError(f"--max-shard-size needs a destination ending in {INDEX_SUFFIX}")
    if src.endswith(".gguf"):
        tensors, metadata = dy.read_gguf(src)
        write_tensors(dst, tensors, {"gguf.metadata": metadata, "converted_from": "gguf"}, shard or None)
    else:
        dy.convert_safetensors(src, dst, shard)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="deepity", description="Inspect and convert Deepity model files.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_inspect = sub.add_parser("inspect", help="validate a model file and print what is in it")
    p_inspect.add_argument("file", help=".safetensors, .safetensors.index.json or .gguf")
    p_inspect.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_inspect.add_argument("--no-verify", action="store_true", help="skip the content hash check (trusted local files)")
    p_convert = sub.add_parser("convert", help="convert between safetensors, shard sets and GGUF (read only)")
    p_convert.add_argument("src")
    p_convert.add_argument("dst")
    p_convert.add_argument("--max-shard-size", type=parse_size, help="shard size for a .safetensors.index.json destination, e.g. 500MB")
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
            convert(args.src, args.dst, args.max_shard_size)
            print(f"wrote {args.dst}")
    except Exception as error:  # noqa: BLE001 - a CLI reports every failure the same way
        print(f"deepity: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
