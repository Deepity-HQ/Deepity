"""Model sources: a local path, `file://`, `http(s)://`, `s3://` or raw bytes, resolved to a local file.

Every loader in Deepity calls `resolve(source)` first and then uses the ordinary mmap path, so a remote
model costs one download and loads as fast as a local one afterwards. Downloads go to a cache
(`$DEEPITY_CACHE_DIR`, else `~/.cache/deepity`), are checked before they are put in place, and are
revalidated with the server's ETag on later loads. Standard library only; `s3://` uses `boto3` when it is
installed (private buckets) and falls back to the bucket's public HTTPS endpoint.

Use `fetch()` directly for options (headers, timeout, size cap, expected SHA-256, forced refresh).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional

from ._backend import dy

INDEX_SUFFIX = ".safetensors.index.json"
_SCHEMES = ("http", "https", "s3", "file")
_CHUNK = 1 << 20
DEFAULT_MAX_BYTES = 256 << 30
_MAX_REDIRECTS = 5


def is_remote(source: Any) -> bool:
    """True for a string with a scheme this module resolves (a one-letter scheme is a Windows drive)."""
    if not isinstance(source, str) or "://" not in source[:16]:
        return False
    return urllib.parse.urlparse(source).scheme in _SCHEMES


def cache_root(override: Optional[str] = None) -> Path:
    """The cache root: `override`, else `$DEEPITY_CACHE_DIR`, else `$XDG_CACHE_HOME/deepity`."""
    chosen = override or os.environ.get("DEEPITY_CACHE_DIR")
    if chosen:
        return Path(chosen)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "deepity"


def _offline() -> bool:
    return os.environ.get("DEEPITY_OFFLINE", "") not in ("", "0")


class _Response:
    """What a download needs from a response, whether it came from urllib or boto3."""

    def __init__(self, stream: Any, headers: dict[str, str]) -> None:
        self.stream = stream
        self.headers = {k.lower(): v for k, v in headers.items()}

    def read(self, n: int) -> bytes:
        return self.stream.read(n)

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if close:
            close()


class _NotModified(Exception):
    pass


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follows only http(s), never from https down to http, and never takes credentials to another host."""

    max_redirections = _MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        old, new = urllib.parse.urlparse(req.full_url), urllib.parse.urlparse(newurl)
        if new.scheme not in ("http", "https"):
            raise urllib.error.HTTPError(req.full_url, code, f"refusing a redirect to {new.scheme}://", headers, fp)
        if old.scheme == "https" and new.scheme == "http":
            raise urllib.error.HTTPError(req.full_url, code, "refusing a redirect from https to http", headers, fp)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and old.netloc != new.netloc:
            for name in ("Authorization", "Cookie", "Proxy-Authorization"):
                redirected.remove_header(name)
                redirected.headers.pop(name.capitalize(), None)
        return redirected


def _s3_parts(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"{url!r} must look like s3://bucket/key")
    return parsed.netloc, urllib.parse.unquote(parsed.path.lstrip("/"))


def _s3_https_url(url: str) -> str:
    """The public HTTPS address of an `s3://` object (works for public objects, no credentials)."""
    bucket, key = _s3_parts(url)
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def _open_http(url: str, headers: dict[str, str], etag: Optional[str], timeout: float) -> _Response:
    request = urllib.request.Request(url, headers={"User-Agent": "deepity", **headers})
    if etag:
        request.add_header("If-None-Match", etag)
    opener = urllib.request.build_opener(_SafeRedirect)
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 304:
            raise _NotModified from error
        hint = ""
        if error.code in (401, 403) and url.startswith("https://") and ".s3.amazonaws.com" in url:
            hint = " (a private S3 object needs boto3 and AWS credentials)"
        raise OSError(f"{url}: HTTP {error.code} {error.reason}{hint}") from error
    return _Response(response, dict(response.headers.items()))


def _open_s3(url: str, headers: dict[str, str], etag: Optional[str], timeout: float) -> _Response:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError:
        return _open_http(_s3_https_url(url), headers, etag, timeout)
    bucket, key = _s3_parts(url)
    args: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if etag:
        args["IfNoneMatch"] = etag
    try:
        result = boto3.client("s3").get_object(**args)
    except Exception as error:  # noqa: BLE001 - botocore's ClientError carries the HTTP status
        code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
        status = getattr(error, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 304 or code in ("304", "NotModified"):
            raise _NotModified from error
        raise OSError(f"{url}: {error}") from error
    return _Response(result["Body"], {"etag": result.get("ETag", ""), "content-length": str(result.get("ContentLength", ""))})


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(str(path) + ".meta").read_text())
    except (OSError, ValueError):
        return {}


def _validate(path: Path, original: str) -> None:
    """Checks a finished download is a well-formed model file (structure and, when present, content hash)."""
    try:
        dy.file_table(str(path), True)
    except Exception as error:  # noqa: BLE001
        raise OSError(f"{original}: the downloaded file is not a valid model file: {error}") from error


def _fetch_one(
    url: str,
    directory: Path,
    name: str,
    *,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
    sha256: Optional[str],
    refresh: bool,
    offline: bool,
) -> Path:
    """Puts one remote file at `directory/name` and returns it. Safe to interrupt: the final file only
    ever appears complete and checked."""
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / name
    have = dest.is_file()
    if have and sha256 and _hash_file(dest) != sha256.lower():
        have = False  # the cached copy is not the one the caller expects
    if offline:
        if not have:
            raise OSError(f"{url}: not in the cache and DEEPITY_OFFLINE is set")
        return dest
    meta = _read_meta(dest) if have else {}
    etag = None if refresh else meta.get("etag")
    if have and not refresh and not etag and not meta.get("last_modified"):
        return dest  # the server gave no validator, so the cached copy cannot be checked; keep it

    opener = _open_s3 if url.startswith("s3://") else _open_http
    try:
        response = opener(url, headers, etag if have else None, timeout)
    except _NotModified:
        return dest
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        if have:
            warnings.warn(f"{url}: cannot reach the server ({error}); using the cached copy.", stacklevel=4)
            return dest
        raise OSError(f"{url}: cannot reach the server: {error}") from error

    part = directory / f".part-{os.getpid()}-{time.monotonic_ns()}-{name}"
    try:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise OSError(f"{url}: {declared} bytes is over the {max_bytes} byte limit (set DEEPITY_MAX_DOWNLOAD_BYTES)")
        received = 0
        digest = hashlib.sha256()
        with open(part, "wb") as out:
            try:
                while chunk := response.read(_CHUNK):
                    received += len(chunk)
                    if received > max_bytes:
                        raise OSError(f"{url}: the download passed the {max_bytes} byte limit (set DEEPITY_MAX_DOWNLOAD_BYTES)")
                    digest.update(chunk)
                    out.write(chunk)
            except http.client.HTTPException as error:  # IncompleteRead and friends
                raise OSError(f"{url}: the download was cut short ({error})") from error
        if declared and declared.isdigit() and received != int(declared):
            raise OSError(f"{url}: the download was cut short ({received} of {declared} bytes)")
        if sha256 and digest.hexdigest() != sha256.lower():
            raise OSError(f"{url}: SHA-256 is {digest.hexdigest()}, expected {sha256.lower()}")
        if not name.endswith(INDEX_SUFFIX):  # an index is checked as JSON; its shards are checked one by one
            _validate(part, url)
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        response.close()
    Path(str(dest) + ".meta").write_text(
        json.dumps({"url": url, "etag": response.headers.get("etag"), "last_modified": response.headers.get("last-modified"),
                    "size": received, "fetched": time.time()})
    )
    return dest


def _safe_name(name: str, index_url: str) -> str:
    if not name or name != os.path.basename(name) or name.startswith(".") or "\\" in name or "/" in name:
        raise ValueError(f"{index_url}: the index names a shard '{name}', which is not a plain file name")
    return name


def fetch(
    url: str,
    *,
    cache_dir: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
    max_bytes: Optional[int] = None,
    sha256: Optional[str] = None,
    refresh: bool = False,
) -> str:
    """Downloads `url` (http, https or s3; a `.safetensors.index.json` also fetches every shard) into the
    cache and returns the local path. A cached copy is revalidated with its ETag, so an unchanged model is
    not downloaded again, and is used with a warning if the server cannot be reached.

    `sha256` is the expected SHA-256 of the file (not of a shard set). `max_bytes` caps each file (default
    `$DEEPITY_MAX_DOWNLOAD_BYTES` or 256 GiB). `refresh=True` downloads again regardless. Setting
    `DEEPITY_OFFLINE=1` uses only what is cached. The finished file is validated before it is kept. A
    signed URL's query string is part of the cache key, so a new signature downloads again; sharded
    models need an unsigned base URL because shards are found next to the index.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        return urllib.request.url2pathname(parsed.path)
    if parsed.scheme not in _SCHEMES:
        raise ValueError(f"cannot fetch {url!r}: only {', '.join(_SCHEMES)} sources are supported")
    limit = max_bytes or int(os.environ.get("DEEPITY_MAX_DOWNLOAD_BYTES", DEFAULT_MAX_BYTES))
    options: dict[str, Any] = dict(headers=headers or {}, timeout=timeout, max_bytes=limit, refresh=refresh, offline=_offline())
    key = hashlib.sha256(url.split("#", 1)[0].encode()).hexdigest()
    directory = cache_root(cache_dir) / key[:2] / key
    name = os.path.basename(urllib.parse.unquote(parsed.path))
    if name in ("", ".", ".."):
        name = "model.safetensors"
    path = _fetch_one(url, directory, name, sha256=sha256, **options)
    if name.endswith(INDEX_SUFFIX):
        weight_map = json.loads(path.read_text()).get("weight_map", {})
        base = url.split("?", 1)[0].rsplit("/", 1)[0]
        for shard in sorted(set(weight_map.values())):
            shard = _safe_name(shard, url)
            _fetch_one(f"{base}/{urllib.parse.quote(shard)}", directory, shard, sha256=None, **options)
    return str(path)


def _spill(data: bytes | bytearray | memoryview) -> str:
    """Writes an in-memory model to the cache (named by its content) and returns the path."""
    raw = bytes(data)
    suffix = ".gguf" if raw[:4] == b"GGUF" else ".safetensors"
    digest = hashlib.sha256(raw).hexdigest()
    directory = cache_root() / "memory"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + suffix)
    if not path.is_file():
        handle, temporary = tempfile.mkstemp(dir=directory, prefix=".part-")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(raw)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return str(path)


def resolve(source: Any) -> str:
    """A local file path for `source`: a path (returned as is), `file://`, an http(s) or s3 URL (downloaded
    to the cache), or bytes (written to the cache). Local paths cost nothing."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _spill(source)
    if isinstance(source, os.PathLike):
        return os.fspath(source)
    if is_remote(source):
        return fetch(source)
    return source


def read_tensors(source: Any, verify_hash: bool = True, mmap: bool = True) -> dict[str, Any]:
    """Reads every tensor of a model file (path, URL or bytes) as numpy arrays. Zero-copy when mapped."""
    return dy.read_tensors(resolve(source), verify_hash, mmap)
