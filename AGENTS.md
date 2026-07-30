# MLSys with Burn Agent Guide

## Mission

Build a Chinese, Rust-first machine-learning-systems textbook using Burn,
CubeCL, CubeK, and burn-onnx as the continuous implementation stack.
OpenMLSys supplies the source structure and selected material; this project
must add independent verification, modern context, and runnable Rust examples.

## Read Before Working

1. `planning/STATUS.md` — current objective, blockers, and handoff.
2. `planning/MASTER_PLAN.md` — milestones and definitions of done.
3. `planning/CHAPTER_MATRIX.md` — scope and source mapping.
4. `pins.toml` — the only supported upstream snapshot.
5. The latest relevant entry under `planning/session-logs/`.
6. The relevant file under `docs/`.

## Repository Boundaries

- Editable: `book/`, `examples/`, `tools/`, `docs/`, `planning/`, root project
  files, and project CI.
- Optional local source mirrors: `burn/`, `burn-onnx/`, `cubecl/`, `cubek/`,
  `openmlsys/`. When present, they are read-only and exist only for fast source
  inspection.
- Never edit an upstream checkout to make a textbook example pass.
- Never add a Cargo `path` dependency or `[patch]` pointing at a local source
  mirror. Builds use the GitHub revisions in `pins.toml`.
- Never add upstream repositories, nested `.git` directories, `target/`, or
  generated mdBook output to the root repository.
- `cubek/cubecl/` is an untracked duplicate checkout and is not a source of
  truth.

## Content Rules

- Write in Chinese; introduce an English term at first use when useful.
- Explain the framework-independent principle before the Burn implementation.
- Distinguish verified behavior, design interpretation, and future work.
- Do not claim that Burn supports a feature without checking the pinned source.
- Record OpenMLSys source files and material changes in each chapter.
- Do not mechanically translate Python syntax. Redesign examples around Rust
  ownership, types, traits, errors, and backend abstraction.
- Keep historical or non-Rust snippets when they are essential to explain an
  interface boundary or compare systems.

## Example Rules

- Chapter code lives in `examples/chNN-*` and is included into the book rather
  than copied.
- Default examples must run on CPU without proprietary drivers.
- GPU, distributed, browser, and embedded examples must state prerequisites and
  have a CPU-testable core where practical.
- Avoid `unwrap` in instructional library code; in tiny binaries/tests it must
  have an obvious invariant or a descriptive `expect`.
- Before handoff run `make check`, which does not require local mirrors. When
  mirrors are available, also run `make check-local-sources`.

## Planning and Handoff

- Claim one bounded item from `planning/STATUS.md`.
- Keep only verified work under “已完成”.
- On completion update status, validation evidence, changed assumptions, and
  the next concrete action.
- Record durable scope or architecture decisions in
  `planning/DECISIONS.md`; do not bury them only in chat.
- A chapter is not complete until prose, source attribution, runnable examples,
  exercises, and link/build checks all pass.

