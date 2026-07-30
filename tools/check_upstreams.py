#!/usr/bin/env python3
"""Validate the pinned upstream snapshot without modifying any checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_pins() -> dict[str, Any]:
    with (ROOT / "pins.toml").open("rb") as file:
        return tomllib.load(file)


def validate_metadata(config: dict[str, Any]) -> None:
    repositories = config.get("repositories", {})
    expected = {"burn", "burn-onnx", "cubecl", "cubek", "openmlsys"}
    if set(repositories) != expected:
        fail(f"repositories must be exactly {sorted(expected)}")

    for name, repository in repositories.items():
        revision = repository.get("rev", "")
        if not SHA_PATTERN.fullmatch(revision):
            fail(f"{name}.rev is not a full lowercase Git SHA")
        if not repository.get("url", "").startswith("https://github.com/"):
            fail(f"{name}.url must be an HTTPS GitHub URL")
        if not repository.get("path"):
            fail(f"{name}.path is missing")

    root_manifest = (ROOT / "Cargo.toml").read_text()
    burn_revision = repositories["burn"]["rev"]
    if f'rev = "{burn_revision}"' not in root_manifest:
        fail("root Cargo.toml does not use the pinned Burn revision")


def validate_checkouts(config: dict[str, Any]) -> None:
    repositories = config["repositories"]
    for name, repository in repositories.items():
        path = ROOT / repository["path"]
        if not (path / ".git").exists():
            fail(f"{name} checkout is missing at {path}")

        actual_revision = git(path, "rev-parse", "HEAD")
        if actual_revision != repository["rev"]:
            fail(
                f"{name} is at {actual_revision}, expected {repository['rev']}"
            )

        actual_url = git(path, "remote", "get-url", "origin").removesuffix("/")
        expected_url = repository["url"].removesuffix(".git").removesuffix("/")
        if actual_url.removesuffix(".git") != expected_url:
            fail(f"{name} origin is {actual_url}, expected {repository['url']}")

        status = git(path, "status", "--short")
        if status:
            print(f"WARN: {name} checkout is not clean:\n{status}")

    relationships = config["relationships"]
    expected_manifest_revisions = {
        ROOT / "burn/Cargo.toml": (
            relationships["burn_cubecl_rev"],
            relationships["burn_cubek_rev"],
        ),
        ROOT / "burn-onnx/Cargo.toml": (
            relationships["burn_onnx_burn_rev"],
        ),
    }
    for manifest, revisions in expected_manifest_revisions.items():
        content = manifest.read_text()
        for revision in revisions:
            if not SHA_PATTERN.fullmatch(revision) or revision not in content:
                fail(f"{manifest} does not contain expected revision {revision}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate committed metadata without requiring local checkouts",
    )
    args = parser.parse_args()

    config = load_pins()
    validate_metadata(config)
    if not args.metadata_only:
        validate_checkouts(config)
    print("Upstream snapshot is consistent.")


if __name__ == "__main__":
    main()

