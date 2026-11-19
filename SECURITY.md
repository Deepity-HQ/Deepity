# Security Policy

## Supported Versions

Deepity is under active development and has not yet reached a stable `1.0`
release. Until then, only the latest commit on `main` (and the latest tagged
pre-1.0 release, if any) is supported with security fixes. There is currently
no backport policy for older versions.

| Version            | Supported          |
| ------------------ | ------------------ |
| `main` (latest)    | :white_check_mark: |
| Older tags/commits | :x:                |

This table will be updated once versioned releases begin (see
[`CHANGELOG.md`](CHANGELOG.md)).

## Loading models from untrusted sources

Deepity model files are safetensors, which holds tensors and string metadata and no executable
content. Loading one cannot run code. What it does do:

- The header size, every tensor offset and size, dtype and shape arithmetic are checked against the
  file before anything is read. The content hash (BLAKE3) is verified by default and catches
  corruption, but it is not a signature: a hash stored inside a file proves nothing about who wrote
  it. Check the download against a hash or signature you obtained separately if provenance matters.
- A shard index may only name plain file names beside itself, never paths.
- Both parsers (safetensors and the shard index) are mutation-fuzzed in CI, including under AddressSanitizer
  and UBSan (`tests/tFormatFuzz.cpp`, `-DDEEPITY_BUILD_FUZZ=ON`), and a libFuzzer target
  (`tests/fuzz/FuzzFormats.cpp`) is available for longer runs.
- Memory-mapped files must not be truncated or rewritten in place while open; doing so can crash the
  reader. Replace the file instead, as `Save` does.
- Model metadata (`deepity.model`, `deepity.card`) is data. A loader only reads the layer kinds,
  sizes and hyperparameters it knows, and a model that asks for a network class this build does not
  have is rejected.

Nothing here stops a hostile model from being large, so cap what you accept (file size, tensor
count) when serving models from users.

## Scope

Deepity is a native C++ library with Python bindings. Security-relevant areas
in particular include:

- **Model / checkpoint deserialization** (`ModelIO.h` and related loading
  code) — loading a network's weights and architecture from a file is a
  classic untrusted-input surface. If you can construct a file that causes
  out-of-bounds reads/writes, memory corruption, or arbitrary allocation
  sizes when loaded via `ModelIO`, that's a security issue, not just a bug.
- **Image loading** (`include/stb_image.h`) — `stb_image` has a public history
  of parser vulnerabilities in other projects that embed it. If you find a
  malformed image that triggers a crash, out-of-bounds access, or memory
  corruption when loaded through Deepity, please report it here rather than
  upstream first, since our usage/integration may differ.
- **Memory safety in the core library** — buffer overflows, use-after-free,
  or misaligned access in `MemoryArena`, the layer implementations, or the
  SIMD micro-kernels, particularly anything reachable from Python-supplied
  array shapes/sizes via the `deepity` bindings.
- **The Python bindings themselves** (`bindings/pybinding.cpp`) — any way that
  Python-level inputs (array shapes, dtypes, batch sizes) can be used to
  corrupt memory or crash the interpreter in a way that wouldn't be a "normal"
  Python exception.

Performance regressions, incorrect training results, and non-security build
failures are **not** security issues — please file those as regular GitHub
issues instead so they get triaged through the normal process.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, email **[contact@deepity.dev](mailto:contact@deepity.dev)**
with:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, ideally a minimal repro (a small `.cpp`/`.py` snippet
  and/or the specific malformed input file, if applicable).
- The platform/compiler/configuration you tested on (this project's behavior
  can genuinely differ by compiler and SIMD dispatch path — see
  [`CONTRIBUTING.md`](CONTRIBUTING.md) — so this detail matters more here
  than in most projects).

You should expect an acknowledgment within **5 business days**. This is a
small, largely solo-maintained project, so please be patient with the
timeline for a fix — but you will be kept informed of progress.

### Disclosure

Please give us a reasonable opportunity to investigate and address a
vulnerability before any public disclosure. We'll credit reporters (by name
or handle, your choice) in the fix's changelog entry unless you'd prefer to
remain anonymous.

There is currently no bug bounty program.

## Preferred Languages

Please report in English if possible.
