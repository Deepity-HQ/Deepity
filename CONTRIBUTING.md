# Contributing to Deepity

This document exists mainly to save the next person (possibly future-you) from
re-discovering a long list of cross-platform build gotchas the hard way. If
you hit a build error that looks like it should be obvious, check here first.

## Build prerequisites

- CMake >= 3.16, Ninja
- A C++20 compiler: Clang (recommended, see below), GCC, or MSVC
- OpenBLAS + pybind11 (via vcpkg on Windows, system packages on Linux/macOS)
- Python 3.8+ with development headers

### Configuring locally

Use `CMakePresets.json` where possible instead of hand-typing flags in
whatever shell you happen to be in — shell-specific line-continuation syntax
(`\` in Bash, `` ` `` in PowerShell, `^` in cmd.exe) is a common source of
confusing, silent failures that have nothing to do with the actual build.

If you must configure manually on Windows with Clang, the known-working
invocation is:

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DOpenMP_CXX_FLAGS="-fopenmp=libomp" ^
  -DOpenMP_CXX_LIB_NAMES="libomp" ^
  -DOpenMP_libomp_LIBRARY="C:/Program Files/LLVM/lib/libomp.lib" ^
  -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
  -DBLA_VENDOR=OpenBLAS
```

Adjust `OpenMP_libomp_LIBRARY` if your LLVM install lives elsewhere. Check
with `dir "C:\Program Files\LLVM\lib\libomp*"`.

## Portability rules

These aren't style preferences — each one corresponds to a real build failure
that took real time to diagnose. Follow them going forward rather than
re-deriving the same fixes.

### 1. Never use `ssize_t`

It's POSIX-only and doesn't exist under MSVC or Clang-targeting-MSVC. Use
`std::ptrdiff_t` (from `<cstddef>`) instead — same width, same signedness,
works identically everywhere.

### 2. Never use `std::aligned_alloc` / `std::free` directly for aligned memory

`std::aligned_alloc`-allocated memory must, per the C11 standard, be freed
with plain `free()`. Windows' native aligned allocator (`_aligned_malloc`)
requires the opposite pairing (`_aligned_free`), so many MinGW toolchains
simply omit `std::aligned_alloc` from `<cstdlib>` rather than violate the
standard's contract. Always go through a portable wrapper — see
`MemoryArena.h`'s `detail::AlignedAllocPortable` /
`detail::AlignedFreePortable` for the pattern. Watch the argument order:
`std::aligned_alloc(alignment, size)` vs `_aligned_malloc(size, alignment)`
are reversed.

### 3. Avoid short, generic identifiers as local variable names

Windows headers define a large number of macros with short, plausible-looking
names via plain `#define` (which do blind textual substitution, unlike C++
namespacing). Known collisions we've hit:

- `min`, `max` — defined in `minwindef.h` (pulled in transitively via many
  headers). Fix globally with `add_compile_definitions(NOMINMAX)` near the
  top of `CMakeLists.txt`, before any targets are defined.
- `small` — defined as `char` in `rpcndr.h`. **`NOMINMAX` does not cover
  this one.** There is no compile-define fix; just don't name a variable
  `small`. Same caution applies to `near`, `far`, `interface`, and other
  short/legacy Win32 SDK macro names.

When in doubt, prefer a more specific name than the shortest possible one for
any local variable, especially in SIMD-heavy code where lots of small,
similarly-scoped variables get declared close together.

### 4. Gate compiler-specific flags behind `CMAKE_CXX_COMPILER_ID`

`-fvectorize` / `-fslp-vectorize` are Clang-only spellings; GCC has no flags
by these names and errors outright. GCC's equivalent is
`-ftree-vectorize` / `-ftree-slp-vectorize`. Never assume "the compiler on
this OS" — Linux and Windows can both run either Clang or GCC depending on
how the environment is set up; branch on `CMAKE_CXX_COMPILER_ID`, not on
`WIN32`/`UNIX`.

### 5. `target_compile_definitions` / `target_compile_options` require the target to already exist

CMake processes a `CMakeLists.txt` strictly top-to-bottom. Calling
`target_compile_definitions(deepity ...)` before the `pybind11_add_module
(deepity ...)` call that creates that target produces a confusing "target is
not built by this project" error. Place any per-target configuration
immediately after the `add_library` / `add_executable` /
`pybind11_add_module` call that defines that target — not grouped together
with unrelated targets' configuration elsewhere in the file, even if that
seems tidier.

If you want a define to apply automatically to every target in a directory,
regardless of order, use `add_compile_definitions()` at directory scope
instead of `target_compile_definitions()` on a specific target — it applies
to everything defined afterward in the same scope, sidestepping the ordering
issue entirely.

### 6. SLEEF linked statically needs `SLEEF_STATIC_LIBS` defined

If `SLEEF_BUILD_SHARED_LIBS` is `OFF`, every translation unit that includes
`sleef.h` needs `SLEEF_STATIC_LIBS` defined, or the header declares its
functions as `dllimport` (expecting a DLL) even though you built a static
archive — producing linker errors like:

```
undefined symbol: __declspec(dllimport) Sleef_tanhf4_u10sse2
NOTE: ... available in sleef.lib but cannot be used because it is not an
import library.
```

Set this as a `PUBLIC` compile definition on the library that links SLEEF, so
it propagates to every downstream consumer:

```cmake
target_compile_definitions(Deepity PUBLIC SLEEF_STATIC_LIBS)
```

This must be set **before** the affected files are compiled — if you add it
after already building, you need a clean rebuild (`rm -rf build` /
`rmdir /s /q build`), since already-compiled `.obj` files have the wrong
`dllimport` decision baked in and won't be invalidated by re-running
`cmake --build` alone.

### 7. Runtime DLL deployment is per-target, not automatic

On Windows, `vcpkg install`ed libraries (OpenBLAS, Python, etc.) live in
vcpkg's own tree, which isn't on the default DLL search path. Each
executable/shared-module target that depends on them needs its own
deployment step, or it'll build and link fine but fail at *runtime* with a
vague, unhelpful error (e.g. `0xc0000279`, "the application was unable to
start correctly") and no further detail.

Use the `deploy_runtime_deps()` helper defined near the top of
`CMakeLists.txt`, called once per runnable target right after that target's
`target_link_libraries()` call:

```cmake
deploy_runtime_deps(deepity)
deploy_runtime_deps(DeepityTests)
```

If you add a new executable or Python module target, remember to call this
for it too — nothing enforces it automatically, and a missing call produces
a build that looks completely successful but a binary that won't start.

**Separately**, any dependency installed *outside* vcpkg (e.g. a manually
installed LLVM's `libomp.dll`) is invisible to `vcpkg z-applocal` entirely,
since it only knows about vcpkg's own tracked install tree. Either add that
tool's `bin` directory to `PATH`, or extend the deployment step to copy it
explicitly.

### 8. `-march=native` is not reproducible across machines

It compiles for whatever CPU the *build machine* has, which can differ
between your local machine, CI runners, and end users. This has been
directly load-bearing for correctness in this codebase — AVX2 vs AVX512
dispatch has changed which code path actually executes for precision-related
logic. For CI or distributed builds, prefer an explicit target
(`-march=x86-64-v2`, `-mavx2`, etc.) over `-march=native`, so behavior is
reproducible rather than depending on which machine happened to build it.

## Numerical / algorithmic gotchas

Separate from build portability, but equally worth knowing before touching
`DiscriminativePCLayer.cpp`:

- **BLAS threading is not controlled by `OMP_NUM_THREADS`.** OpenBLAS ships
  its own internal thread pool (usually pthreads-based), invisible to your
  own `#pragma omp` regions. If you need fully deterministic, reproducible
  runs (e.g. for debugging), set `OPENBLAS_NUM_THREADS=1` as well as
  `OMP_NUM_THREADS=1` — multithreaded BLAS reductions are not
  bit-reproducible run-to-run due to non-associative floating-point
  summation order, and this codebase's dynamics have shown real sensitivity
  to sub-ULP differences compounding over many iterations.

- **`net.Compile()` must be called after all `AddLayer()` calls and before
  `RandomizeWeights()`.** Each layer otherwise runs on its own small,
  independently-sized `localArena`, and `MemoryArena::AllocateFloats()`
  rounds every individual allocation up to a 64-byte boundary — meaning many
  small buffers can collectively need more real memory than the unpadded sum
  of `GetRequiredFloats()` implies, if that sum isn't itself computed with
  the same rounding. Always call `Compile()` once, after the full layer list
  is known.

- **A layer's own `sigma_prime`/`p`/`log_p` buffers should only ever be
  written by that layer's own methods.** Several past bugs in this codebase
  involved one layer reading or writing a *neighboring* layer's buffer under
  a mistaken assumption about ordering or ownership. If you're adding new
  cross-layer logic, be explicit about which layer's `UpdateState()`/
  `CalculateState()` is guaranteed to have already run in the same sweep
  (the network's per-layer loops are front-to-back, meaning `layerBelow`'s
  update always happens before the current layer's in the same call) —
  don't assume the opposite direction without checking.