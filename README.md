# MLSys with Burn

[中文](README_CN.md) | English

*Machine Learning Systems: Design and Implementation with Burn and Rust* is
an open-source textbook for Rust developers who want to study machine learning
systems. It uses Burn as a continuous case study and follows the
Burn → CubeCL → CubeK stack through tensors, automatic differentiation,
compilation, kernels, training, and deployment.

This project is adapted from
[OpenMLSys](https://github.com/openmlsys/openmlsys). It is not an official
OpenMLSys or Tracel project and is not affiliated with either organization.

## Project Status

This is the nine-chapter candidate edition for the fixed
`burn-0.22.0-pre.1` source snapshot. It is in release-audit stabilization:
the book has CPU-first runnable evidence, source crosswalks, and explicit
optional platform boundaries. See [`planning/STATUS.md`](planning/STATUS.md)
for verified progress and remaining limitations.

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
`book/book/` and is not committed. Browser reading of formulas still needs
the MathJax assets configured by mdBook; Cargo offline reproducibility does
not imply an offline CDN.

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

