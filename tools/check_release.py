#!/usr/bin/env python3
"""审计教材发布结构、固定快照和默认可复现边界。

这个检查器只依赖 Python 标准库。它不把本地上游镜像当作普通构建依赖；
传入 ``--check-local-sources`` 时才检查镜像中的源码入口。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tomllib
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BOOK_SRC = ROOT / "book" / "src"
BOOK_OUTPUT = ROOT / "book" / "book"
PINS = ROOT / "pins.toml"
CROSSWALK = ROOT / "planning" / "comparison" / "openmlsys-v1-crosswalk.md"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)\s]+)(?:\s+[^)]*)?\)")
INCLUDE_PATTERN = re.compile(
    r"\{\{#include\s+([^}\s:]+)(?::([^}\s]+))?\s*\}\}"
)
ANCHOR_START = re.compile(r"^\s*//\s*ANCHOR:\s*([A-Za-z0-9_-]+)\s*$")
ANCHOR_END = re.compile(r"^\s*//\s*ANCHOR_END:\s*([A-Za-z0-9_-]+)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")

CHAPTER_ENTRIES = {
    f"ch{number:02d}": f"ch{number:02d}-{suffix}.md"
    for number, suffix in (
        (1, "introduction"),
        (2, "programming-and-graph"),
        (3, "accelerator"),
        (4, "compiler-and-runtime"),
        (5, "data-processing"),
        (6, "training-systems"),
        (7, "model-serving"),
        (8, "rl-systems"),
        (9, "gpu-cluster"),
    )
}
SOURCE_FILES = {f"ch{number:02d}.md" for number in range(1, 10)}
SUMMARY_WHITELIST = {"SUMMARY.md", "README.md", "preface.md", "attribution.md"}
LICENSE_FILES = (
    "LICENSE.md",
    "LICENSE_CN.md",
    "NOTICE.md",
    "book/src/attribution.md",
    "LICENSES/CC-BY-NC-SA-4.0.txt",
    "LICENSES/MIT.txt",
    "LICENSES/Apache-2.0.txt",
)


class ReleaseAudit:
    def __init__(
        self,
        *,
        check_local_sources: bool,
        require_built_book: bool,
        run_offline_gate: bool,
    ) -> None:
        self.check_local_sources = check_local_sources
        self.require_built_book = require_built_book
        self.run_offline_gate = run_offline_gate
        self.checks: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.pins: dict[str, Any] = tomllib.loads(
            PINS.read_text(encoding="utf-8")
        )

    def check(self, name: str, function: Any) -> None:
        try:
            function()
        except AssertionError as error:
            self.errors.append(f"{name}: {error}")
        else:
            self.checks.append(name)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def run(self) -> dict[str, Any]:
        self.check("summary-and-book-files", self.validate_summary)
        self.check("includes-and-anchors", self.validate_includes)
        self.check("sources-and-crosswalk", self.validate_sources)
        self.check("pins-and-cargo", self.validate_pins_and_cargo)
        self.check("licenses-and-attribution", self.validate_licenses)
        self.check("markdown-links", self.validate_links)
        self.check("formulas", self.validate_formulas)
        self.check("code-snippet-annotations", self.validate_code_snippet_annotations)
        self.check("repository-hygiene", self.validate_repository_hygiene)
        self.check("generated-book", self.validate_generated_book)
        if self.run_offline_gate:
            self.check("cargo-offline-gate", self.validate_offline_gate)

        report = {
            "snapshot": self.pins.get("snapshot", {}),
            "pins": {
                name: values.get("rev")
                for name, values in self.pins.get("repositories", {}).items()
            },
            "tool_versions": tool_versions(),
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
            "ok": not self.errors,
        }
        return report

    def validate_summary(self) -> None:
        summary = BOOK_SRC / "SUMMARY.md"
        self.require(summary.is_file(), "book/src/SUMMARY.md 不存在")
        text = summary.read_text(encoding="utf-8")
        targets = []
        for target in LINK_PATTERN.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target:
                targets.append(target)

        missing = []
        for target in targets:
            path = (summary.parent / target).resolve()
            if not path.is_file():
                missing.append(target)
        self.require(not missing, f"SUMMARY 目标不存在: {missing}")

        lines = text.splitlines()
        for chapter, entry in CHAPTER_ENTRIES.items():
            entry_index = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if f"]({entry})" in line and line.startswith("- [")
                ),
                None,
            )
            self.require(entry_index is not None, f"{entry} 未进入 SUMMARY")
            section_targets = []
            for line in lines[entry_index + 1 :]:
                if line.startswith("- ["):
                    break
                match = re.match(r"^\s{2}- \[[^\]]+\]\(([^)]+)\)", line)
                if match:
                    section_targets.append(match.group(1))
            self.require(
                len(section_targets) == 8,
                f"{chapter} 应有 8 个小节，实际为 {len(section_targets)}",
            )
            self.require(
                all(target.startswith(f"{chapter}/") for target in section_targets),
                f"{chapter} 小节必须位于 {chapter}/",
            )

        linked = {target.split("#", 1)[0] for target in targets}
        unlisted = []
        for path in sorted(BOOK_SRC.rglob("*.md")):
            relative = path.relative_to(BOOK_SRC).as_posix()
            if relative in SUMMARY_WHITELIST:
                continue
            if relative not in linked:
                unlisted.append(relative)
        self.require(not unlisted, f"未被 SUMMARY 收录的 Markdown: {unlisted}")

    def validate_includes(self) -> None:
        labels: Counter[tuple[Path, str, str]] = Counter()
        include_count = 0
        for source in BOOK_SRC.rglob("*.md"):
            text = source.read_text(encoding="utf-8")
            for match in INCLUDE_PATTERN.finditer(text):
                include_count += 1
                relative_path, label = match.groups()
                included = (source.parent / relative_path).resolve()
                self.require(
                    included.is_file(),
                    f"{source.relative_to(ROOT)} include 不存在: {relative_path}",
                )
                self.require(
                    ROOT.joinpath("examples") in included.parents,
                    f"{source.relative_to(ROOT)} include 只能来自 examples/: {relative_path}",
                )
                if label:
                    labels[(included, label, "start")] += 1
                    labels[(included, label, "end")] += 1
                    starts = 0
                    ends = 0
                    for line in included.read_text(encoding="utf-8").splitlines():
                        starts += bool(ANCHOR_START.match(line) and ANCHOR_START.match(line).group(1) == label)
                        ends += bool(ANCHOR_END.match(line) and ANCHOR_END.match(line).group(1) == label)
                    self.require(
                        starts == 1 and ends == 1,
                        f"{included.relative_to(ROOT)} 的 anchor {label} 未成对且唯一",
                    )
        self.require(include_count > 0, "没有发现任何 examples mdBook include")

    def validate_sources(self) -> None:
        self.require(CROSSWALK.is_file(), "comparison crosswalk 不存在")
        crosswalk = CROSSWALK.read_text(encoding="utf-8")
        self.require(
            "C/S/R/L/E" in crosswalk,
            "crosswalk 未声明 C/S/R/L/E 验收状态",
        )
        for source_name in SOURCE_FILES:
            path = ROOT / "planning" / "chapter-sources" / source_name
            self.require(path.is_file(), f"章节来源映射缺失: {path}")
        for number in range(1, 10):
            entry = BOOK_SRC / CHAPTER_ENTRIES[f"ch{number:02d}"]
            text = entry.read_text(encoding="utf-8")
            self.require("证据状态" in text, f"{entry.relative_to(ROOT)} 缺少证据状态")

        upstream_root = ROOT / "openmlsys" / "v1" / "zh_chapters"
        if upstream_root.is_dir():
            upstream_files = sorted(upstream_root.rglob("*.md"))
            missing = [
                path.relative_to(upstream_root).as_posix()
                for path in upstream_files
                if path.relative_to(upstream_root).as_posix() not in crosswalk
            ]
            self.require(
                not missing,
                f"crosswalk 未覆盖本地 OpenMLSys Markdown: {missing}",
            )
        else:
            self.warnings.append("没有本地 OpenMLSys 镜像，跳过逐文件数量核对")

        repositories = self.pins.get("repositories", {})
        for name, repository in repositories.items():
            revision = repository.get("rev", "")
            self.require(
                revision in crosswalk,
                f"crosswalk 未记录 {name} revision {revision}",
            )
        self.require(
            "明确排除" in crosswalk and "推荐系统" in crosswalk,
            "crosswalk 未记录 OpenMLSys 扩展篇排除范围",
        )

    def validate_pins_and_cargo(self) -> None:
        self.pins = tomllib.loads(PINS.read_text(encoding="utf-8"))
        release_manifest = ROOT / "release.toml"
        self.require(release_manifest.is_file(), "缺少 release.toml 工具版本清单")
        release = tomllib.loads(release_manifest.read_text(encoding="utf-8"))
        self.require(
            release.get("snapshot", {}).get("pins") == "pins.toml",
            "release.toml 未指向 pins.toml",
        )
        repositories = self.pins.get("repositories", {})
        expected = {"burn", "burn-onnx", "cubecl", "cubek", "openmlsys"}
        self.require(set(repositories) == expected, "pins.toml repositories 集合不完整")
        for name, repository in repositories.items():
            revision = repository.get("rev", "")
            self.require(
                bool(SHA_PATTERN.fullmatch(revision)),
                f"{name}.rev 不是 40 位小写 SHA",
            )
            self.require(
                repository.get("url", "").startswith("https://github.com/"),
                f"{name}.url 不是 GitHub HTTPS 地址",
            )

        manifests = [ROOT / "Cargo.toml", *sorted((ROOT / "examples").glob("**/Cargo.toml"))]
        for manifest in manifests:
            content = manifest.read_text(encoding="utf-8")
            self.require(
                not re.search(r"(?m)^\s*(?:path|branch|tag)\s*=", content),
                f"{manifest.relative_to(ROOT)} 含有 path/branch/tag dependency",
            )
            self.require(
                not re.search(r"(?m)^\s*\[(?:patch|replace)\.", content),
                f"{manifest.relative_to(ROOT)} 含有 [patch]/[replace]",
            )
            self.require(
                "burn-onnx" not in content,
                f"{manifest.relative_to(ROOT)} 不得把 burn-onnx 旧 revision 接入主线",
            )
            for git_url in re.findall(r'git\s*=\s*"([^"]+)"', content):
                self.require(
                    "rev =" in content,
                    f"{manifest.relative_to(ROOT)} 的 Git dependency 缺少 rev",
                )
                self.require(
                    git_url.startswith("https://github.com/"),
                    f"{manifest.relative_to(ROOT)} 的 Git dependency 不是 GitHub",
                )

        lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
        lock_sources = [
            package.get("source", "")
            for package in lock.get("package", [])
            if package.get("source")
        ]
        for name in ("burn", "cubecl", "cubek"):
            repository = repositories[name]
            expected_source = repository["url"].removesuffix(".git")
            revision = repository["rev"]
            self.require(
                any(expected_source in source and revision in source for source in lock_sources),
                f"Cargo.lock 未锁定 {name} 的 pins.toml revision",
            )

    def validate_licenses(self) -> None:
        for relative in LICENSE_FILES:
            self.require((ROOT / relative).is_file(), f"缺少许可证文件: {relative}")
        attribution = (ROOT / "book/src/attribution.md").read_text(encoding="utf-8")
        notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
        self.require("CC BY-NC-SA 4.0" in attribution, "书内归属缺少 CC BY-NC-SA 4.0")
        self.require("MIT OR Apache-2.0" in notice or "MIT" in notice, "NOTICE 缺少代码许可证说明")
        self.require("OpenMLSys" in notice, "NOTICE 缺少 OpenMLSys 归属")
        self.require(
            "不是\nOpenMLSys" in attribution or "不是 OpenMLSys" in attribution,
            "书内归属缺少独立性声明",
        )

    def validate_links(self) -> None:
        broken: list[str] = []
        for source in BOOK_SRC.rglob("*.md"):
            text = source.read_text(encoding="utf-8")
            in_fence = False
            for line in text.splitlines():
                if FENCE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                for target in LINK_PATTERN.findall(line):
                    if target.startswith(("http://", "https://", "mailto:")):
                        continue
                    path_part, _, fragment = target.partition("#")
                    resolved = (source.parent / path_part).resolve()
                    if not resolved.is_file():
                        broken.append(f"{source.relative_to(ROOT)} -> {target}")
                    elif fragment and fragment not in heading_ids(resolved):
                        broken.append(f"{source.relative_to(ROOT)} -> {target}")
        self.require(not broken, f"书内链接损坏: {broken}")

    def validate_formulas(self) -> None:
        display_count = 0
        inline_count = 0
        formula_errors: list[str] = []
        for source in BOOK_SRC.rglob("*.md"):
            in_fence = False
            display_open = False
            display_lines: list[str] = []
            inline_dollars = 0
            inline_open_line: int | None = None
            for line_number, line in enumerate(
                source.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if FENCE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                if "$$" in line:
                    if inline_open_line is not None:
                        formula_errors.append(
                            f"{source.relative_to(ROOT)}:{inline_open_line} 行内 $ 与 $$ 交错"
                        )
                        inline_open_line = None
                    pieces = line.split("$$")
                    for index in range(0, len(pieces) - 1):
                        if not display_open:
                            display_open = True
                            display_lines = [pieces[index]]
                        else:
                            display_lines.append(pieces[index])
                            display_count += 1
                            formula_errors.extend(
                                formula_issues(source, line_number, "\n".join(display_lines))
                            )
                            display_open = False
                            display_lines = []
                    if display_open and pieces[-1]:
                        display_lines.append(pieces[-1])
                    continue
                if display_open:
                    display_lines.append(line)
                    continue
                # Strip escaped dollars and inline code before counting $ pairs.
                stripped = re.sub(r"`[^`]*`", "", line)
                stripped = re.sub(r"\\\$", "", stripped).replace("$$", "")
                dollar_count = stripped.count("$")
                inline_dollars += dollar_count
                if dollar_count % 2 == 1:
                    if inline_open_line is None:
                        inline_open_line = line_number
                    else:
                        formula_errors.append(
                            f"{source.relative_to(ROOT)}:{inline_open_line}-{line_number} "
                            "行内公式跨行；请合并到一行或改用 $$...$$"
                        )
                        inline_open_line = None
            self.require(
                not display_open,
                f"{source.relative_to(ROOT)} 的 $$ display math 未闭合",
            )
            self.require(
                inline_open_line is None,
                f"{source.relative_to(ROOT)}:{inline_open_line} 行内 $ math 未闭合",
            )
            self.require(
                inline_dollars % 2 == 0,
                f"{source.relative_to(ROOT)} 的行内 $ math 未成对",
            )
            inline_count += inline_dollars // 2
        self.require(not formula_errors, f"公式下标/结构错误: {formula_errors}")
        self.require(display_count > 0, "没有发现 display math")
        self.require(inline_count > 0, "没有发现 inline math")

    def validate_code_snippet_annotations(self) -> None:
        invalid: list[str] = []
        for source in BOOK_SRC.rglob("*.md"):
            lines = source.read_text(encoding="utf-8").splitlines()
            fence_info: str | None = None
            fence_has_include = False
            for line in lines:
                if fence_info is None:
                    match = re.match(r"^\s*```([^\s]*)\s*$", line)
                    if match:
                        fence_info = match.group(1)
                        fence_has_include = False
                    continue
                if line.strip() == "```":
                    if (
                        fence_info.startswith("rust")
                        and fence_has_include
                        and "ignore" not in fence_info
                    ):
                        invalid.append(str(source.relative_to(ROOT)))
                    fence_info = None
                    fence_has_include = False
                    continue
                fence_has_include |= "{{#include" in line
        self.require(
            not invalid,
            f"依赖上下文的 include Rust 片段必须显式 ignore: {sorted(set(invalid))}",
        )

    def validate_generated_book(self) -> None:
        if not BOOK_OUTPUT.is_dir():
            if self.require_built_book:
                raise AssertionError("book/book 不存在，请先运行 mdbook build book")
            self.warnings.append("book/book 不存在，跳过生成 HTML 公式复查")
            return
        book_toml = (ROOT / "book/book.toml").read_text(encoding="utf-8")
        self.require("mathjax-support = true" in book_toml, "book.toml 未开启 MathJax")
        self.require('theme = "theme"' in book_toml, "book.toml 未启用自定义 theme")
        head_hbs = ROOT / "book/theme/head.hbs"
        self.require(head_hbs.is_file(), "缺少 book/theme/head.hbs（MathJax $ 分隔符配置）")
        head_text = head_hbs.read_text(encoding="utf-8")
        self.require(
            'inlineMath: [["$", "$"]' in head_text or "inlineMath: [['$', '$']" in head_text,
            "theme/head.hbs 未配置 MathJax inlineMath 美元分隔符",
        )
        html_files = sorted(BOOK_OUTPUT.rglob("*.html"))
        self.require(bool(html_files), "book/book 没有 HTML 输出")
        polluted: list[str] = []
        missing_mathjax_config: list[str] = []
        pages_with_math = 0
        for path in html_files:
            content = html.unescape(path.read_text(encoding="utf-8"))
            for match in re.finditer(
                r'<span[^>]+class="[^"]*math[^"]*"[^>]*>(.*?)</span>',
                content,
                flags=re.DOTALL,
            ):
                body = match.group(1)
                if re.search(r"<(?:em|ul|ol)\b", body):
                    polluted.append(str(path.relative_to(ROOT)))
            has_dollar_math = bool(
                re.search(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", content, flags=re.S)
            ) or "$$" in content
            has_paren_math = "\\(" in content or "\\[" in content
            if has_dollar_math or has_paren_math:
                pages_with_math += 1
                if "MathJax" not in content:
                    polluted.append(f"{path.relative_to(ROOT)} (缺少 MathJax)")
                # Custom head must land before MathJax.js and enable $.
                if "inlineMath" not in content or '"$"' not in content:
                    missing_mathjax_config.append(str(path.relative_to(ROOT)))
        self.require(not polluted, f"生成 HTML 数学结构污染: {sorted(set(polluted))}")
        self.require(
            not missing_mathjax_config,
            "生成 HTML 缺少 MathJax $ 分隔符配置: "
            f"{sorted(set(missing_mathjax_config))[:8]}",
        )
        self.require(pages_with_math > 0, "生成 HTML 中没有发现公式页面")

    def validate_repository_hygiene(self) -> None:
        forbidden = ("target/", "book/book/", ".mdbook/", "burn/", "burn-onnx/", "cubecl/", "cubek/", "openmlsys/")
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = []
        for line in result.stdout.splitlines():
            value = line[3:] if len(line) >= 3 else line
            if " -> " in value:
                value = value.split(" -> ", 1)[1]
            paths.append(value)
        bad = [path for path in paths if any(path.startswith(prefix) for prefix in forbidden)]
        self.require(not bad, f"Git 状态包含禁止提交路径: {bad}")

    def validate_offline_gate(self) -> None:
        result = subprocess.run(
            [
                "cargo",
                "metadata",
                "--locked",
                "--offline",
                "--format-version",
                "1",
                "--no-deps",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.require(
            result.returncode == 0,
            f"cargo metadata --locked --offline 失败: {result.stderr.strip()}",
        )


def formula_issues(source: Path, line_number: int, formula: str) -> list[str]:
    issues = []
    if re.search(r"(?<!\\)_", formula):
        issues.append(f"{source.relative_to(ROOT)}:{line_number} 含未转义下标")
    if re.search(r"<(?:em|ul|ol)\b", formula):
        issues.append(f"{source.relative_to(ROOT)}:{line_number} 含 Markdown 结构标签")
    return issues


def heading_ids(path: Path) -> set[str]:
    identifiers = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"`([^`]*)`", r"\1", match.group(1))
        heading = unicodedata.normalize("NFKC", heading).lower()
        heading = re.sub(r"[^\w\u4e00-\u9fff -]", "", heading)
        heading = re.sub(r"\s+", "-", heading.strip())
        identifiers.add(heading)
    return identifiers


def tool_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for command, argument in (("rustc", "--version"), ("mdbook", "--version")):
        try:
            result = subprocess.run(
                [command, argument],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            versions[command] = "unavailable"
        else:
            versions[command] = result.stdout.strip()
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-local-sources",
        action="store_true",
        help="检查 crosswalk 中的 Burn/CubeCL/CubeK/OpenMLSys 本地路径",
    )
    parser.add_argument(
        "--require-built-book",
        action="store_true",
        help="要求 book/book 已由 mdBook 构建并复查生成 HTML",
    )
    parser.add_argument(
        "--offline-gate",
        action="store_true",
        help="额外运行 cargo metadata --locked --offline",
    )
    parser.add_argument("--json", action="store_true", help="只输出机器可读 JSON")
    args = parser.parse_args()

    audit = ReleaseAudit(
        check_local_sources=args.check_local_sources,
        require_built_book=args.require_built_book,
        run_offline_gate=args.offline_gate,
    )
    # local source checks are deliberately separate from ordinary release checks.
    if args.check_local_sources:
        audit.check("local-source-paths", lambda: validate_local_source_paths(audit))
    report = audit.run()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        for check in report["checks"]:
            print(f"PASS {check}")
        for warning in report["warnings"]:
            print(f"WARN {warning}", file=sys.stderr)
        for error in report["errors"]:
            print(f"ERROR {error}", file=sys.stderr)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["ok"] else 1


def validate_local_source_paths(audit: ReleaseAudit) -> None:
    repositories = audit.pins.get("repositories", {})
    crosswalk = CROSSWALK.read_text(encoding="utf-8")
    paths = set(
        re.findall(r"`((?:burn-onnx|burn|cubecl|cubek)/[^`]+)`", crosswalk)
    )
    missing = [path for path in sorted(paths) if not (ROOT / path.rstrip("/")).exists()]
    for name, repository in repositories.items():
        mirror = ROOT / repository.get("local_path", name)
        if not (mirror / ".git").exists():
            missing.append(f"{name}: {mirror.relative_to(ROOT)}/.git")
            continue
        result = subprocess.run(
            ["git", "-C", str(mirror), "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        actual = result.stdout.strip()
        if actual != repository.get("rev"):
            missing.append(
                f"{name}: HEAD {actual} != {repository.get('rev')}"
            )
    audit.require(not missing, f"本地源码入口/revision 审计失败: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
