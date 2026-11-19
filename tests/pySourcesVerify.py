"""Model sources: URLs, s3://, bytes and the download cache, against local servers (no internet needed).

Run from the repo root with the built extension importable, e.g.
  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 uv run python tests/pySourcesVerify.py
"""
import contextlib
import io
import os
import sys
import tempfile
import threading
import types
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deepity as dp  # noqa: E402
from deepity import _sources  # noqa: E402
from deepity.cli import main as cli_main  # noqa: E402

failures = 0


def check(ok: bool, what: str) -> None:
    global failures
    print(("  [PASSED] " if ok else "  [FAILED] ") + what)
    failures += not ok


def error_text(fn) -> str:
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}"
    return ""


class Site:
    """A tiny HTTP server over a dict of path -> (body, etag), logging every request."""

    def __init__(self) -> None:
        self.files: dict[str, tuple[bytes, str | None]] = {}
        self.log: list[tuple[str, dict[str, str], int]] = []
        self.routes: dict[str, object] = {}
        site = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # noqa: ANN002
                pass

            def reply(self, code: int, body: bytes = b"", headers: dict[str, str] | None = None, length: int | None = None):
                site.log.append((self.path, {k.lower(): v for k, v in self.headers.items()}, code))
                self.send_response(code)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body) if length is None else length))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                path = self.path.split("?")[0]
                route = site.routes.get(path)
                if callable(route):
                    return route(self)
                if path not in site.files:
                    return self.reply(404, b"no")
                body, etag = site.files[path]
                if etag and self.headers.get("If-None-Match") == etag:
                    return self.reply(304, b"", {"ETag": etag})
                self.reply(200, body, {"ETag": etag} if etag else {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def count(self, path: str, code: int | None = None) -> int:
        return sum(1 for p, _, c in self.log if p.split("?")[0] == path and (code is None or c == code))


x = np.random.default_rng(0).standard_normal((8, 12)).astype(np.float32)


def make(seed: int):
    net = dp.FullPCN(dp.Linear(12, 10), dp.ReLU(), dp.Linear(10, 4), batch_size=8).configure(learning_rate=1e-3)
    net.train_step(x, np.random.default_rng(seed).standard_normal((8, 4)).astype(np.float32), 2)
    return net


with tempfile.TemporaryDirectory() as tmp:
    os.environ["DEEPITY_CACHE_DIR"] = os.path.join(tmp, "cache")
    os.environ.pop("DEEPITY_OFFLINE", None)
    cache = os.path.join(tmp, "cache")

    net_a, net_b = make(1), make(2)
    path_a, path_b = os.path.join(tmp, "a.safetensors"), os.path.join(tmp, "b.safetensors")
    net_a.save(path_a)
    net_b.save(path_b)
    body_a, body_b = open(path_a, "rb").read(), open(path_b, "rb").read()
    want_a, want_b = net_a.predict(x, 2), net_b.predict(x, 2)

    site, other = Site(), Site()
    site.files["/m.safetensors"] = (body_a, '"v1"')

    print("--- download and cache ---")
    url = site.url + "/m.safetensors"
    loaded = dp.load(url)
    check(np.array_equal(loaded.predict(x, 2), want_a), "deepity.load(url) predicts like the original")
    check(site.count("/m.safetensors", 200) == 1, "one download")
    dp.load(url)
    check(site.count("/m.safetensors", 200) == 1 and site.count("/m.safetensors", 304) == 1,
          "a second load revalidates with the ETag (304) and does not download again")
    check(any(h.get("if-none-match") == '"v1"' for p, h, _ in site.log), "the conditional request carries the ETag")
    site.files["/m.safetensors"] = (body_b, '"v2"')
    check(np.array_equal(dp.load(url).predict(x, 2), want_b), "a changed file (new ETag) is downloaded again")
    check(dp.read_card(url) == {}, "read_card accepts a URL")
    check(np.array_equal(dp.read_tensors(url)["layers.0.weight"], dp.read_tensors(path_b)["layers.0.weight"]),
          "read_tensors accepts a URL")
    with contextlib.redirect_stdout(io.StringIO()):
        inspected = cli_main(["inspect", url])
    check(inspected == 0, "deepity inspect accepts a URL")

    site.files["/noetag.safetensors"] = (body_a, None)
    dp.fetch(site.url + "/noetag.safetensors")
    dp.fetch(site.url + "/noetag.safetensors")
    check(site.count("/noetag.safetensors") == 1, "a server with no validator is not asked twice (nothing to revalidate with)")
    dp.fetch(site.url + "/noetag.safetensors", refresh=True)
    check(site.count("/noetag.safetensors") == 2, "refresh=True downloads again")

    print("--- failures leave nothing behind ---")
    def leftovers(u: str) -> list[str]:
        import hashlib

        key = hashlib.sha256(u.encode()).hexdigest()
        folder = os.path.join(cache, key[:2], key)
        return os.listdir(folder) if os.path.isdir(folder) else []

    text = error_text(lambda: dp.load(site.url + "/missing.safetensors"))
    check("404" in text and "missing.safetensors" in text, "a 404 is an error naming the URL and status")

    site.files["/hashed.safetensors"] = (body_a, '"h"')
    import hashlib

    good = hashlib.sha256(body_a).hexdigest()
    text = error_text(lambda: dp.fetch(site.url + "/hashed.safetensors", sha256="0" * 64))
    check("SHA-256" in text and leftovers(site.url + "/hashed.safetensors") == [], "a wrong SHA-256 is refused and nothing is kept")
    check(os.path.isfile(dp.fetch(site.url + "/hashed.safetensors", sha256=good)), "the right SHA-256 is accepted")

    text = error_text(lambda: dp.fetch(site.url + "/m.safetensors", max_bytes=100, refresh=True, cache_dir=os.path.join(tmp, "c2")))
    check("limit" in text, "a Content-Length over the cap is refused before downloading")

    def chunked(handler):
        site.log.append((handler.path, {}, 200))
        handler.send_response(200)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for _ in range(40):
            handler.wfile.write(b"%x\r\n" % 65536 + b"x" * 65536 + b"\r\n")
        handler.wfile.write(b"0\r\n\r\n")

    site.routes["/chunked.safetensors"] = chunked
    text = error_text(lambda: dp.fetch(site.url + "/chunked.safetensors", max_bytes=200000, cache_dir=os.path.join(tmp, "c3")))
    check("limit" in text, "a body with no Content-Length is cut off at the cap")

    def short(handler):
        handler.reply(200, body_a[:100], {"ETag": '"s"'}, length=len(body_a))
        handler.close_connection = True

    site.routes["/short.safetensors"] = short
    text = error_text(lambda: dp.fetch(site.url + "/short.safetensors"))
    kept = leftovers(site.url + "/short.safetensors")
    check("cut short" in text and not any(f == "short.safetensors" for f in kept), "a download cut short is refused and no model file is put in place")
    check(sorted(kept) == [".partial-short.safetensors", ".partial-short.safetensors.meta"], "its 100 bytes are kept to resume later, and nothing else")

    flipped = bytearray(body_a)
    flipped[-3] ^= 0x40
    site.files["/corrupt.safetensors"] = (bytes(flipped), '"c"')
    text = error_text(lambda: dp.load(site.url + "/corrupt.safetensors"))
    check("not a valid model file" in text and leftovers(site.url + "/corrupt.safetensors") == [],
          "a corrupt download fails the content hash and is not cached")

    print("--- shard sets ---")
    shard_dir = os.path.join(tmp, "shards")
    os.makedirs(shard_dir)
    net_a.save(os.path.join(shard_dir, "m.safetensors.index.json"), max_shard_bytes=300)
    for name in os.listdir(shard_dir):
        site.files["/sharded/" + name] = (open(os.path.join(shard_dir, name), "rb").read(), f'"{name}"')
    shard_count = len(os.listdir(shard_dir)) - 1
    check(shard_count > 1, f"the test model has {shard_count} shards")
    check(np.array_equal(dp.load(site.url + "/sharded/m.safetensors.index.json").predict(x, 2), want_a),
          "a sharded model loads from its index URL")
    site.files["/evil/m.safetensors.index.json"] = (
        b'{"metadata": {}, "weight_map": {"layers.0.weight": "../outside.safetensors"}}', '"e"')
    check("plain file name" in error_text(lambda: dp.fetch(site.url + "/evil/m.safetensors.index.json")),
          "an index naming a shard outside its folder is refused")

    print("--- resuming a cut-short download ---")
    state = {"cut": 0, "etag": '"r1"', "body": body_a, "mode": "ranges"}

    def ranged(h):
        body, etag = state["body"], state["etag"]
        want = h.headers.get("Range")
        if state["mode"] == "416" and want:
            state["mode"] = "ranges"
            return h.reply(416, b"", {"Content-Range": f"bytes */{len(body)}"})
        if want and state["mode"] != "ignore" and h.headers.get("If-Range") in (None, etag):
            start = int(want.split("=")[1].rstrip("-"))
            offset = start + 1 if state["mode"] == "wrong" else start
            return h.reply(206, body[start:], {"ETag": etag, "Content-Range": f"bytes {offset}-{len(body) - 1}/{len(body)}"})
        if state["cut"]:
            cut, state["cut"] = state["cut"], 0
            h.reply(200, body[:cut], {"ETag": etag}, length=len(body))
            h.close_connection = True
            return None
        return h.reply(200, body, {"ETag": etag})

    import hashlib

    site.routes["/r.safetensors"] = ranged
    ru = site.url + "/r.safetensors"
    folder = os.path.join(cache, hashlib.sha256(ru.encode()).hexdigest()[:2], hashlib.sha256(ru.encode()).hexdigest())
    half = len(body_a) // 2

    def reset():
        for f in leftovers(ru):
            os.unlink(os.path.join(folder, f))

    def parts():
        return [f for f in leftovers(ru) if f.startswith(".part")]

    state["cut"] = half
    text = error_text(lambda: dp.fetch(ru))
    check("cut short" in text and "resumes" in text, "the first attempt is cut short and says the bytes are kept")
    check(os.path.getsize(os.path.join(folder, ".partial-r.safetensors")) == half, f"the {half} bytes received are kept")
    got = dp.fetch(ru)
    asked = [(c, hd) for p_, hd, c in site.log if p_ == "/r.safetensors" and hd.get("range")]
    check(asked and asked[-1][1]["range"] == f"bytes={half}-" and asked[-1][1]["if-range"] == '"r1"' and asked[-1][0] == 206,
          "the next fetch asks for the rest with Range and If-Range")
    check(open(got, "rb").read() == body_a and not parts(), "the resumed file is complete and byte for byte right, and the partial is gone")
    check(np.array_equal(dp.load(ru).predict(x, 2), want_a), "and it loads")

    reset()
    gguf_path = os.path.join(tmp, "r.gguf")
    dp.write_gguf(gguf_path, {"w": x}, None)
    gbody = open(gguf_path, "rb").read()
    site.routes["/r.gguf"] = lambda h: ranged_for(h, gbody)
    gstate = {"cut": len(gbody) // 2}

    def ranged_for(h, body):
        want = h.headers.get("Range")
        if want:
            start = int(want.split("=")[1].rstrip("-"))
            return h.reply(206, body[start:], {"ETag": '"g"', "Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
        if gstate["cut"]:
            cut, gstate["cut"] = gstate["cut"], 0
            h.reply(200, body[:cut], {"ETag": '"g"'}, length=len(body))
            h.close_connection = True
            return None
        return h.reply(200, body, {"ETag": '"g"'})

    error_text(lambda: dp.fetch(site.url + "/r.gguf"))
    got = dp.fetch(site.url + "/r.gguf")
    check(open(got, "rb").read() == gbody and np.array_equal(dp.read_gguf(got)[0]["w"], x), "a cut-short .gguf resumes and is validated as GGUF")

    reset()
    state.update(cut=half, etag='"r2"', body=body_a, mode="ranges")
    error_text(lambda: dp.fetch(ru))
    state.update(etag='"r3"', body=body_b)  # the file changed while the partial waited
    check(open(dp.fetch(ru), "rb").read() == body_b, "a file that changed between attempts is downloaded whole, not stitched")

    reset()
    state.update(cut=half, etag='"r4"', body=body_a, mode="ranges")
    error_text(lambda: dp.fetch(ru))
    state["mode"] = "ignore"
    check(open(dp.fetch(ru), "rb").read() == body_a, "a server that ignores Range sends the whole file and it is used")

    reset()
    state.update(cut=half, etag='"r5"', body=body_a, mode="ranges")
    error_text(lambda: dp.fetch(ru))
    state["mode"] = "wrong"
    text = error_text(lambda: dp.fetch(ru))
    check("wrong offset" in text and not parts(), "a reply that resumes at the wrong offset is refused and the bytes dropped")

    reset()
    state.update(cut=half, etag='"r6"', body=body_a, mode="ranges")
    error_text(lambda: dp.fetch(ru))
    state["mode"] = "416"
    check(open(dp.fetch(ru), "rb").read() == body_a, "a 416 drops the kept bytes and downloads from the start")

    reset()
    state.update(cut=half, etag='"r7"', body=body_a, mode="ranges")
    error_text(lambda: dp.fetch(ru))
    text = error_text(lambda: dp.fetch(ru, sha256="0" * 64))
    check("SHA-256" in text and not parts(), "a resumed file that fails its hash is refused and nothing is kept")

    import deepity._sources as sources

    check(sources._hf_url("hf://owner/repo/path/to/m.safetensors") == "https://huggingface.co/owner/repo/resolve/main/path/to/m.safetensors", "hf:// names a file on the main branch")
    check(sources._hf_url("hf://owner/repo@v1.0/dir/m.gguf") == "https://huggingface.co/owner/repo/resolve/v1.0/dir/m.gguf", "hf:// takes a revision after the repository")
    check(error_text(lambda: sources._hf_url("hf://owner/repo")) != "", "hf:// without a file is refused")
    os.environ["HF_TOKEN"] = "secret"
    check(sources._hf_headers() == {"Authorization": "Bearer secret"}, "HF_TOKEN is sent as a bearer token")
    os.environ.pop("HF_TOKEN")

    print("--- a redirecting origin that validates by its own etag (Hugging Face style) ---")
    # Like the real CDN, this one reports an etag but does not honour a conditional request.
    site.routes["/cdn.safetensors"] = lambda h: h.reply(200, body_a, {"ETag": '"cdn-etag"'})

    def origin(h):
        if h.headers.get("If-None-Match") == '"linked-etag"':
            return h.reply(304, b"", {"ETag": 'W/"x"'})
        h.reply(302, b"", {"Location": "/cdn.safetensors", "X-Linked-ETag": '"linked-etag"'})

    site.routes["/origin.safetensors"] = origin
    check(np.array_equal(dp.load(site.url + "/origin.safetensors").predict(x, 2), want_a), "a file behind such a redirect loads")
    before = site.count("/cdn.safetensors", 200)
    dp.load(site.url + "/origin.safetensors")
    check(site.count("/cdn.safetensors", 200) == before, "the second load revalidates with the origin's etag and downloads nothing")
    print("--- redirects and credentials ---")
    site.routes["/redirect.safetensors"] = lambda h: h.reply(302, b"", {"Location": "/m.safetensors"})
    check(np.array_equal(dp.load(site.url + "/redirect.safetensors").predict(x, 2), want_b), "a redirect on the same host is followed")
    site.routes["/to_file.safetensors"] = lambda h: h.reply(302, b"", {"Location": "file:///etc/passwd"})
    text = error_text(lambda: dp.fetch(site.url + "/to_file.safetensors"))
    check("file://" in text and "HTTP" in text, "a redirect to file:// is refused")
    other.files["/m.safetensors"] = (body_a, '"o"')
    site.routes["/cross.safetensors"] = lambda h: h.reply(302, b"", {"Location": other.url + "/m.safetensors"})
    dp.fetch(site.url + "/cross.safetensors", headers={"Authorization": "Bearer secret"})
    check(any(h.get("authorization") == "Bearer secret" for p, h, _ in site.log if p == "/cross.safetensors")
          and not any("authorization" in h for p, h, _ in other.log),
          "credentials are sent to the first host and dropped on a redirect to another host")

    print("--- other sources ---")
    check(np.array_equal(dp.load("file://" + path_a).predict(x, 2), want_a), "file:// loads")
    check(np.array_equal(dp.load(body_a).predict(x, 2), want_a), "deepity.load(bytes) loads")
    dp.load(bytes(body_a))
    check(len(os.listdir(os.path.join(cache, "memory"))) == 1, "the same bytes are cached once")
    check(np.array_equal(dp.read_tensors(body_a)["layers.0.bias"], dp.read_tensors(path_a)["layers.0.bias"]), "read_tensors accepts bytes")
    check(_sources.resolve("some/relative/model.safetensors") == "some/relative/model.safetensors"
          and _sources.resolve("C://models/x.safetensors") == "C://models/x.safetensors",
          "plain paths, including a Windows drive, pass through untouched")
    check("only" in error_text(lambda: dp.fetch("ftp://example.com/x.safetensors")), "an unsupported scheme is refused")

    print("--- s3:// ---")
    check(_sources._s3_https_url("s3://my-bucket/a b/m.safetensors") == "https://my-bucket.s3.amazonaws.com/a%20b/m.safetensors",
          "without boto3 an s3:// URL maps to the bucket's HTTPS address")

    class FakeClientError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"HTTP {status}")
            self.response = {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": str(status)}}

    calls: list[dict] = []

    class FakeS3:
        def get_object(self, **args):
            calls.append(args)
            if args.get("IfNoneMatch") == '"s3v1"':
                raise FakeClientError(304)
            return {"Body": io.BytesIO(body_a), "ETag": '"s3v1"', "ContentLength": len(body_a)}

    fake = types.ModuleType("boto3")
    fake.client = lambda name: FakeS3()
    sys.modules["boto3"] = fake
    try:
        check(np.array_equal(dp.load("s3://bucket/dir/m.safetensors").predict(x, 2), want_a), "s3:// loads through boto3")
        dp.load("s3://bucket/dir/m.safetensors")
        check(len(calls) == 2 and calls[1].get("IfNoneMatch") == '"s3v1"', "the second s3:// load revalidates with the ETag")
    finally:
        del sys.modules["boto3"]

    print("--- offline and a dead server ---")
    os.environ["DEEPITY_OFFLINE"] = "1"
    before = len(site.log)
    check(np.array_equal(dp.load(url).predict(x, 2), want_b) and len(site.log) == before, "DEEPITY_OFFLINE uses the cache without touching the network")
    check("not in the cache" in error_text(lambda: dp.fetch(site.url + "/never_seen.safetensors")), "offline and not cached is a clear error")
    del os.environ["DEEPITY_OFFLINE"]
    site.stop()
    other.stop()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        revived = dp.load(url)
    check(np.array_equal(revived.predict(x, 2), want_b) and any("cached copy" in str(w.message) for w in caught),
          "a dead server falls back to the cached copy with a warning")
    check("cannot reach" in error_text(lambda: dp.fetch(site.url + "/other.safetensors")), "a dead server and no cache is an error")

print("PASSED" if not failures else f"FAILED: {failures} check(s)")
sys.exit(1 if failures else 0)
