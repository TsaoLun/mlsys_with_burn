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

The project is currently establishing its infrastructure and content outline.
See [`planning/STATUS.md`](planning/STATUS.md) for live progress, next tasks,
and handoff notes.

## Local Layout

The project expects five independent, read-only upstream checkouts in the
project root:

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

The root repository does not track these upstream directories. The source
snapshot used by the textbook is recorded in [`pins.toml`](pins.toml). Agents
and contributors must not modify upstream working trees without explicit
authorization.

## Quick Start

Requirements:

- Rust 1.95
- mdBook 0.4
- Python 3.11 or later

```bash
make check-upstreams
make book
make test
```

The generated book is written to `book/book/`.

## Project Structure

- `book/`: the Chinese mdBook textbook
- `examples/`: runnable Rust examples associated with chapters
- `planning/`: roadmap, chapter mapping, and live status
- `docs/`: architecture, authoring, and maintenance guidance
- `tools/`: version and content consistency checks
- `.cursor/rules/` and `AGENTS.md`: agent collaboration rules

## License

The textbook is an OpenMLSys derivative licensed under CC BY-NC-SA 4.0.
Original Rust examples and tools are licensed under MIT OR Apache-2.0. See
[`LICENSE.md`](LICENSE.md) and [`NOTICE.md`](NOTICE.md) for details.

