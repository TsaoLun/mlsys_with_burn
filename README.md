# MLSys with Burn

[中文](README_CN.md) | English

*Machine Learning Systems: Design and Implementation with Burn and Rust* is
an open-source textbook for Rust developers. Chapters follow OpenMLSys
systems questions, with Burn → CubeCL → CubeK as the running
implementation—from tensor APIs through kernels, data pipelines, training,
serving, and GPU clusters.

This project is adapted from
[OpenMLSys](https://github.com/openmlsys/openmlsys). It is not an official
OpenMLSys or Tracel project and is not affiliated with either organization.

## Project Status

This is the nine-chapter candidate edition aligned with Burn
`0.22.0-pre.1`. See [`planning/STATUS.md`](planning/STATUS.md) for progress
and remaining limitations.

## Dependency Sources

Builds and CI resolve Burn from its GitHub repository at the exact revision
recorded in [`pins.toml`](pins.toml). Burn's own manifest pins the compatible
CubeCL and CubeK revisions for the `0.22.0-pre.1` writing snapshot. Project
Cargo manifests must not use local path dependencies. The pinned `burn-onnx`
checkout references a different Burn revision and is therefore a source-audit
input, not a dependency of the main workspace.

Optional, read-only source mirrors may be placed in the project root:

```text
mlsys_with_burn/
├── burn/
├── burn-onnx/
├── cubecl/
├── cubek/
├── openmlsys/
├── book/
└── examples/
```

These directories are ignored by Git and exist only to let agents inspect
upstream source quickly. They are not required to build or test the project and
must not affect Cargo dependency resolution.

## Read Online

The candidate edition is published as a static mdBook site on GitHub Pages:

https://tsaolun.github.io/mlsys_with_burn/

Pushes to `main` rebuild the site via
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).
Browser formula rendering still depends on the MathJax CDN configured by
mdBook (see D015 / D016).

## Quick Start

Requirements:

- Rust 1.95
- mdBook 0.4.51
- Python 3.11 or later

```bash
make check
```

`make check` uses `--locked`, runs the CPU smoke suite and the Cargo offline
gate after fetching locked dependencies. The generated book is written to
`book/book/` and is not committed. For local preview:

```bash
mdbook serve book
```

Browser reading of formulas still needs the MathJax assets configured by
mdBook; Cargo offline reproducibility does not imply an offline CDN.

If the optional source mirrors are present, verify that they match the remote
snapshot with:

```bash
make check-local-sources
```

## Project Structure

- `book/`: the Chinese mdBook textbook
- `examples/`: runnable Rust examples associated with chapters
- `planning/`: roadmap, chapter mapping, and live status
- `docs/`: architecture, authoring, and maintenance guidance
- `tools/`: version and content consistency checks
- `pins.toml` / `release.toml`: source revisions and release toolchain
- `.cursor/rules/` and `AGENTS.md`: agent collaboration rules

## License

The textbook is an OpenMLSys derivative licensed under CC BY-NC-SA 4.0.
Original Rust examples and tools are licensed under MIT OR Apache-2.0. See
[`LICENSE.md`](LICENSE.md) and [`NOTICE.md`](NOTICE.md) for details.

