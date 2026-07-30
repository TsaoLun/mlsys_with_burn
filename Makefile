.PHONY: book check check-local-sources check-upstreams fmt lint test

book:
	mdbook build book

check-upstreams:
	python3 tools/check_upstreams.py

check-local-sources:
	python3 tools/check_upstreams.py --check-local

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

check: check-upstreams book fmt lint test

