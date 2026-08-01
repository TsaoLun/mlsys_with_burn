.PHONY: book mdbook-test check check-local-sources check-upstreams fmt lint test doctest \
	offline-gate smoke capstone-smoke check-release

CARGO_LOCKED = --locked
CARGO_OFFLINE = --locked --offline
CPU_EXAMPLES = \
	ch01-stack-probe \
	ch02-tensor-basics \
	ch03-cubecl-kernel \
	ch03-tile-loads \
	ch04-fusion-inspector \
	ch05-data-pipeline \
	ch06-training-loop \
	ch07-record-roundtrip \
	ch08-rl-rollout \
	ch09-cluster-simulator

book:
	mdbook build book

mdbook-test:
	mdbook test book

check-upstreams:
	python3 tools/check_upstreams.py

check-local-sources: check-upstreams book
	python3 tools/check_upstreams.py --check-local
	python3 tools/check_release.py --check-local-sources --require-built-book --json

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets $(CARGO_LOCKED) -- -D warnings

test:
	cargo test --workspace --all-targets $(CARGO_LOCKED)

doctest:
	cargo test --workspace --doc $(CARGO_LOCKED)

smoke:
	@set -eu; \
	for package in $(CPU_EXAMPLES); do \
		echo "== smoke: $$package =="; \
		cargo run --quiet --package "$$package" $(CARGO_LOCKED); \
	done

capstone-smoke:
	cargo run --quiet --package ch05-ch07-capstone $(CARGO_LOCKED)

offline-gate:
	cargo fetch $(CARGO_LOCKED)
	cargo metadata $(CARGO_OFFLINE) --format-version 1 --no-deps
	cargo clippy --workspace --all-targets $(CARGO_OFFLINE) -- -D warnings
	cargo test --workspace --all-targets $(CARGO_OFFLINE)
	cargo test --workspace --doc $(CARGO_OFFLINE)

check-release: book
	python3 tools/check_release.py --require-built-book --offline-gate --json

check: check-upstreams book mdbook-test fmt lint test doctest smoke capstone-smoke offline-gate check-release

