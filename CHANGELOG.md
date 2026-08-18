# Changelog

All notable changes to Deepity will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once versioned releases begin (see [Versioning](#versioning) below).

Changes prior to the introduction of this file were not formally tracked;
history effectively begins at `[Unreleased]` below. See the git log for
earlier history.

## [Unreleased]

### Added

-

### Changed

-

### Deprecated

-

### Removed

-

### Fixed

-

### Security

-

<!--
When cutting a release:
  1. Rename [Unreleased] to the new version + date, e.g. [0.2.0] - 2026-09-01
  2. Add a fresh, empty [Unreleased] section above it using the template above
  3. Add a link reference at the bottom (see the pattern below)
  4. Bump the version in pyproject.toml / setup.py to match
-->

<!--
## [0.1.0] - YYYY-MM-DD

### Added

- Initial tracked release.
-->

[Unreleased]: https://github.com/Ra4ster/Deepity/compare/main...HEAD

## Versioning

Deepity follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`)
once tagged releases begin:

- **MAJOR** — incompatible C++ or Python API changes (e.g. changing
  `AddLayer`'s signature, renaming/removing a public header or `pydeepity`
  function).
- **MINOR** — backwards-compatible functionality (new layer types, new
  optimizers, new Python bindings, new build targets).
- **PATCH** — backwards-compatible bug fixes, performance improvements that
  don't change behavior, and documentation-only changes.

Until the first `1.0.0` release, the project may make breaking changes in
`0.x` releases as the API stabilizes; these will still be called out under
**Changed**/**Removed** in this file.

The authoritative version number lives in [`pyproject.toml`](pyproject.toml)
/ [`setup.py`](setup.py) and should be bumped as part of the same PR/commit
that promotes `[Unreleased]` to a versioned entry in this file.
