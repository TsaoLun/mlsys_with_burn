.PHONY: book check check-upstreams fmt lint test

book:
	mdbook build book

check-upstreams:
	python3 tools/check_upstreams.py

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

check: check-upstreams book fmt lint test

