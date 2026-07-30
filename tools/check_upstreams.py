#!/usr/bin/env python3
"""Validate remote dependency pins and optional local source mirrors."""

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
        if not repository.get("local_path"):
            fail(f"{name}.local_path is missing")

    relationships = config.get("relationships", {})
    expected_relationships = {
        "burn_cubecl_rev",
        "burn_cubek_rev",
        "burn_onnx_burn_rev",
    }
    if set(relationships) != expected_relationships:
        fail(f"relationships must be exactly {sorted(expected_relationships)}")
    for name, revision in relationships.items():
        if not SHA_PATTERN.fullmatch(revision):
            fail(f"relationships.{name} is not a full lowercase Git SHA")
    if relationships["burn_cubecl_rev"] != repositories["cubecl"]["rev"]:
        fail("Burn and CubeCL snapshot revisions disagree")
    if relationships["burn_cubek_rev"] != repositories["cubek"]["rev"]:
        fail("Burn and CubeK snapshot revisions disagree")

    root_manifest = (ROOT / "Cargo.toml").read_text()
    burn_revision = repositories["burn"]["rev"]
    if f'rev = "{burn_revision}"' not in root_manifest:
        fail("root Cargo.toml does not use the pinned Burn revision")
    if repositories["burn"]["url"] not in root_manifest:
        fail("root Cargo.toml does not use the pinned Burn GitHub repository")

    manifests = [ROOT / "Cargo.toml", *sorted((ROOT / "examples").glob("**/Cargo.toml"))]
    for manifest in manifests:
        if re.search(r"(?m)(?:^|[,{])\s*path\s*=", manifest.read_text()):
            fail(f"{manifest} contains a local path dependency")

    lockfile = (ROOT / "Cargo.lock").read_text()
    remote_dependencies = {
        "burn": repositories["burn"],
        "cubecl": repositories["cubecl"],
        "cubek": repositories["cubek"],
    }
    for name, repository in remote_dependencies.items():
        if repository["url"].removesuffix(".git") not in lockfile:
            fail(f"Cargo.lock does not resolve {name} from its GitHub repository")
        if repository["rev"] not in lockfile:
            fail(f"Cargo.lock does not contain the pinned {name} revision")


def validate_checkouts(config: dict[str, Any]) -> None:
    repositories = config["repositories"]
    for name, repository in repositories.items():
        path = ROOT / repository["local_path"]
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
        "--check-local",
        action="store_true",
        help="also verify optional local source mirrors against the remote pins",
    )
    args = parser.parse_args()

    config = load_pins()
    validate_metadata(config)
    if args.check_local:
        validate_checkouts(config)
    print("Upstream snapshot is consistent.")


if __name__ == "__main__":
    main()

