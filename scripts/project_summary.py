#!/usr/bin/env python3
"""
CooledAI Simulator — Project Summary

Scans the entire repository and prints a structured inventory:
file counts, line counts, entry points, endpoints, test coverage,
and architecture overview.

Usage:
    python3 scripts/project_summary.py
    python3 scripts/project_summary.py --json   # machine-readable output
"""

from __future__ import annotations

import json as _json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    "node_modules", ".next", "__pycache__", ".git", "venv", ".venv",
    "env", ".cursor", ".DS_Store",
}


def _walk(root: Path):
    for entry in sorted(root.iterdir()):
        if entry.name in SKIP_DIRS:
            continue
        if entry.is_dir():
            yield from _walk(entry)
        else:
            yield entry


def count_lines(path: Path) -> int:
    try:
        return len(path.read_text(errors="replace").splitlines())
    except Exception:
        return 0


def scan_files(root: Path):
    by_ext: Dict[str, List[Path]] = defaultdict(list)
    by_dir: Dict[str, int] = defaultdict(int)
    total_lines_by_ext: Dict[str, int] = defaultdict(int)

    for p in _walk(root):
        ext = p.suffix.lower() or p.name
        rel_dir = str(p.parent.relative_to(root)).split("/")[0]
        if rel_dir == ".":
            rel_dir = "(root)"
        by_ext[ext].append(p)
        by_dir[rel_dir] += 1
        if ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".sh", ".yaml", ".yml"):
            total_lines_by_ext[ext] += count_lines(p)

    return by_ext, by_dir, total_lines_by_ext


def find_endpoints(api_main: Path) -> List[str]:
    endpoints = []
    try:
        text = api_main.read_text()
        for m in re.finditer(
            r'@app\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
            text,
        ):
            method = m.group(1).upper()
            path = m.group(2)
            endpoints.append(f"{method:6s} {path}")
    except Exception:
        pass
    return endpoints


def find_pydantic_models(api_main: Path) -> List[str]:
    models = []
    try:
        text = api_main.read_text()
        for m in re.finditer(r'class\s+(\w+)\(BaseModel\)', text):
            models.append(m.group(1))
    except Exception:
        pass
    return models


def find_scripts(scripts_dir: Path) -> List[Dict[str, Any]]:
    scripts = []
    if not scripts_dir.is_dir():
        return scripts
    for p in sorted(scripts_dir.iterdir()):
        if p.name.startswith("."):
            continue
        desc = ""
        try:
            head = p.read_text(errors="replace")[:500]
            for line in head.splitlines():
                stripped = line.strip().strip("#").strip('"').strip("'").strip()
                if stripped and not stripped.startswith("!") and len(stripped) > 10:
                    desc = stripped[:80]
                    break
        except Exception:
            pass
        scripts.append({
            "name": p.name,
            "lines": count_lines(p),
            "description": desc,
        })
    return scripts


def find_tests(tests_dir: Path) -> List[Dict[str, str]]:
    tests = []
    if not tests_dir.is_dir():
        return tests
    for p in sorted(tests_dir.glob("test_*.py")):
        tests.append({"name": p.name, "lines": count_lines(p)})
    return tests


def find_docs(root: Path) -> List[str]:
    docs = []
    for p in sorted(root.glob("*.md")):
        docs.append(p.name)
    return docs


def find_frontend_pages(frontend: Path) -> List[str]:
    pages = []
    app_dir = frontend / "app"
    if not app_dir.is_dir():
        return pages
    for p in sorted(app_dir.rglob("page.tsx")):
        route = "/" + str(p.parent.relative_to(app_dir))
        route = route.replace("/.", "/").rstrip("/") or "/"
        pages.append(route)
    return pages


def build_summary() -> Dict[str, Any]:
    by_ext, by_dir, lines_by_ext = scan_files(ROOT)

    py_files = by_ext.get(".py", [])
    ts_files = by_ext.get(".ts", []) + by_ext.get(".tsx", [])
    total_py_lines = sum(count_lines(p) for p in py_files)
    total_ts_lines = sum(count_lines(p) for p in ts_files)

    api_main = ROOT / "api" / "main.py"
    endpoints = find_endpoints(api_main)
    models = find_pydantic_models(api_main)
    scripts = find_scripts(ROOT / "scripts")
    tests = find_tests(ROOT / "tests")
    docs = find_docs(ROOT)
    pages = find_frontend_pages(ROOT / "frontend")

    # Core modules
    core_modules = []
    core_dir = ROOT / "core"
    if core_dir.is_dir():
        for d in sorted(core_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                py_count = len(list(d.glob("*.py")))
                if py_count:
                    core_modules.append({"name": d.name, "files": py_count})

    return {
        "project": "CooledAI Simulator",
        "root": str(ROOT),
        "file_counts": {
            "python": len(py_files),
            "typescript_tsx": len(ts_files),
            "yaml": len(by_ext.get(".yaml", []) + by_ext.get(".yml", [])),
            "markdown": len(by_ext.get(".md", [])),
            "shell": len(by_ext.get(".sh", [])),
            "total": sum(len(v) for v in by_ext.values()),
        },
        "line_counts": {
            "python": total_py_lines,
            "typescript_tsx": total_ts_lines,
            "total_code": total_py_lines + total_ts_lines,
        },
        "files_by_directory": dict(sorted(by_dir.items(), key=lambda x: -x[1])),
        "api_endpoints": endpoints,
        "pydantic_models": models,
        "core_modules": core_modules,
        "scripts": scripts,
        "tests": tests,
        "docs": docs,
        "frontend_pages": pages,
    }


def print_summary(s: Dict[str, Any]) -> None:
    W = 68

    print()
    print("=" * W)
    print(f"  {s['project']}  —  Full Project Summary")
    print("=" * W)

    # File counts
    fc = s["file_counts"]
    lc = s["line_counts"]
    print(f"""
  Files
  ─────────────────────────────
    Python (.py)        {fc['python']:>5}    ({lc['python']:,} lines)
    TypeScript/TSX      {fc['typescript_tsx']:>5}    ({lc['typescript_tsx']:,} lines)
    YAML configs        {fc['yaml']:>5}
    Markdown docs       {fc['markdown']:>5}
    Shell scripts       {fc['shell']:>5}
    Total files         {fc['total']:>5}    ({lc['total_code']:,} lines of code)
""")

    # By directory
    print("  Files by Directory")
    print("  ─────────────────────────────")
    for d, count in s["files_by_directory"].items():
        print(f"    {d:<22} {count:>4} files")

    # Core modules
    if s["core_modules"]:
        print(f"\n  Core Modules (core/)")
        print("  ─────────────────────────────")
        for m in s["core_modules"]:
            print(f"    {m['name']:<22} {m['files']:>2} files")

    # API endpoints
    if s["api_endpoints"]:
        print(f"\n  API Endpoints ({len(s['api_endpoints'])})")
        print("  ─────────────────────────────")
        for ep in s["api_endpoints"]:
            print(f"    {ep}")

    # Pydantic models
    if s["pydantic_models"]:
        print(f"\n  Pydantic Models ({len(s['pydantic_models'])})")
        print("  ─────────────────────────────")
        for m in s["pydantic_models"]:
            print(f"    {m}")

    # Scripts
    if s["scripts"]:
        print(f"\n  Scripts ({len(s['scripts'])})")
        print("  ─────────────────────────────")
        for sc in s["scripts"]:
            desc = f"  {sc['description']}" if sc["description"] else ""
            print(f"    {sc['name']:<32} {sc['lines']:>4} lines{desc}")

    # Tests
    if s["tests"]:
        print(f"\n  Tests ({len(s['tests'])})")
        print("  ─────────────────────────────")
        for t in s["tests"]:
            print(f"    {t['name']:<36} {t['lines']:>4} lines")

    # Frontend pages
    if s["frontend_pages"]:
        print(f"\n  Frontend Routes ({len(s['frontend_pages'])})")
        print("  ─────────────────────────────")
        for p in s["frontend_pages"]:
            print(f"    {p}")

    # Docs
    if s["docs"]:
        print(f"\n  Documentation ({len(s['docs'])} files at root)")
        print("  ─────────────────────────────")
        for d in s["docs"]:
            print(f"    {d}")

    print()
    print("=" * W)
    print()


def main() -> None:
    summary = build_summary()
    if "--json" in sys.argv:
        print(_json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
