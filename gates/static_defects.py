"""Static defect gate for the production packages (routebalance, bench, analysis, tools).

Each check targets a defect class that produced complete, error-free, wrong cells in v3:
  falsy_or            `a or b` used as a value: a real 0 / "" / [] silently becomes the fallback
  fabricated_default  `.get(k, d)`, `.pop(k, d)`, `.setdefault(k, d)`, `getattr(o, n, d)` with d not None:
                      a missing value is invented (v3: quality 0.5, price 0.01, TPOT 0.05)
  env_param           reads of os.environ / os.getenv, however imported: invisible to config validation
  gpu_map             hardware-type -> GPU knowledge anywhere but the single map in routebalance/hw.py
  model_ladder        hard-coded model lists or model-keyed dicts: a second family is silently ignored

`a or b` inside a boolean test (if / while / assert / conditional expression / comprehension filter / not)
is not a value and is allowed. Anything else needs a same-line waiver that states its reason:
  counts.get(k, 0)  # default-ok: counters start at zero
  os.environ["RB_MODELS"]  # env-ok: artifact root path, not an experiment parameter

Usage: python -m gates.static_defects <dir> [<dir> ...]   (exit 1 on findings, 2 on usage errors)
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WAIVER = {
    "default": re.compile(r"#\s*default-ok:\s*\S.{2,}"),
    "env": re.compile(r"#\s*env-ok:\s*\S.{2,}"),
}
HW_MAP_HOME = "routebalance/hw.py"
# CloudLab hardware types (the ones this project uses plus common others) - never GPU names like a100.
KNOWN_HW = {"d7525", "c4130", "d8545", "c240g5", "r7525", "m510", "xl170", "c6525-25g", "c6525-100g", "c6420",
            "c220g5", "c220g2", "d430", "d710", "d820", "c8220"}
MODEL_NAME = re.compile(r"(?i)(qwen|gemma|llama|mistral|phi|deepseek).*\d+(\.\d+)?b")
HTTP_MODULES = {"requests", "httpx"}   # module-level get(url, params) - not a mapping lookup
ENV_ATTRS = {"environ", "getenv"}


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.kind:18s} {self.path}:{self.line}  {self.message}"


def _boolean_test_nodes(tree: ast.AST) -> set[int]:
    tests: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            tests.add(id(node.test))
        elif isinstance(node, ast.comprehension):
            tests.update(id(c) for c in node.ifs)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            tests.add(id(node.operand))
        elif isinstance(node, ast.match_case) and node.guard is not None:
            tests.add(id(node.guard))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bool":
            tests.update(id(a) for a in node.args)
    # operands of a boolean operator that is itself a test are tests too
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp) and id(node) in tests:
                for v in node.values:
                    if id(v) not in tests:
                        tests.add(id(v))
                        changed = True
    return tests


def _model_names_in(node: ast.AST) -> int:
    return sum(1 for n in ast.walk(node)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and MODEL_NAME.search(n.value))


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_value_default(node: ast.AST) -> bool:
    """A literal value default (`x: float = 0.01`, `= -1`, `= "a30"`); Field(...) calls are checked separately."""
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, (ast.Constant, ast.List, ast.Dict, ast.Tuple, ast.Set)) and not _is_none(node)


def scan_file(path: Path, rel: str, hw_maps: list[tuple[str, int]]) -> list[Finding]:
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src, filename=rel)
    tests = _boolean_test_nodes(tree)
    out: list[Finding] = []

    def waived(node: ast.AST, kind: str) -> bool:
        ln = getattr(node, "lineno", 0)
        return 0 < ln <= len(lines) and bool(WAIVER[kind].search(lines[ln - 1]))

    def add(kind: str, node: ast.AST, msg: str, waiver: str | None = "default") -> None:
        if waiver is None or not waived(node, waiver):
            out.append(Finding(kind, rel, getattr(node, "lineno", 0), msg))

    os_modules: set[str] = set()
    env_names: set[str] = set()
    http_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "os" or (a.asname is None and a.name.startswith("os.")):
                    os_modules.add(a.asname or "os")        # `import os.path` binds `os`
                if a.name in HTTP_MODULES:
                    http_modules.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for a in node.names:
                if a.name in ENV_ATTRS or a.name == "*":
                    env_names.add(a.asname or a.name)
                    add("env_param", node, f"imports os.{a.name}; experiment parameters belong in config", "env")
    rebound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            rebound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            rebound.add(node.id)
    http_modules -= rebound            # a parameter or assignment named like the module shadows it

    ladder_seen: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and id(node) not in tests:
            add("falsy_or", node, "`a or b` used as a value hides a falsy result")
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if any(k.arg == "fill_value" and not _is_none(k.value) for k in node.keywords):
                add("fabricated_default", node, f"{fname}(fill_value=...) invents values for missing cells")
            if fname == "defaultdict" and node.args and not _is_none(node.args[0]) and not (
                    isinstance(node.args[0], ast.Lambda) and _is_none(node.args[0].body)):
                add("fabricated_default", node, "defaultdict(factory) invents values for missing keys")
            if fname in ("nan_to_num", "ffill", "bfill", "interpolate"):
                add("fabricated_default", node, f"{fname}() invents values for missing data")
            if isinstance(f, ast.Attribute):
                default = node.args[1] if len(node.args) >= 2 else next(
                    (k.value for k in node.keywords if k.arg == "default"), None)
                if f.attr in ("get", "pop", "setdefault") and default is not None and not _is_none(default):
                    http = f.attr == "get" and isinstance(f.value, ast.Name) and f.value.id in http_modules
                    if not http:
                        add("fabricated_default", node, f".{f.attr}() with a non-None default")
                elif f.attr == "fillna":
                    add("fabricated_default", node, ".fillna() invents values for missing data")
            elif isinstance(f, ast.Name):
                if f.id == "getattr" and len(node.args) == 3 and not _is_none(node.args[2]):
                    add("fabricated_default", node, "getattr() with a non-None default")
                elif f.id == "next" and len(node.args) == 2 and not _is_none(node.args[1]):
                    add("fabricated_default", node, "next() with a non-None default")
                elif f.id in ("max", "min") and any(k.arg == "default" and not _is_none(k.value) for k in node.keywords):
                    add("fabricated_default", node, f"{f.id}(default=...) invents a value for empty input")
                elif f.id == "dict" and any(k.arg in KNOWN_HW for k in node.keywords):
                    hw_maps.append((rel, node.lineno))
                    if rel != HW_MAP_HOME:
                        add("gpu_map", node, f"hardware map outside {HW_MAP_HOME}", None)
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None and _is_value_default(stmt.value):
                    add("fabricated_default", stmt, "class field with a non-None default; require it in config")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field"
                and any(k.arg == "default" and not _is_none(k.value) for k in node.keywords)):
            add("fabricated_default", node, "Field(default=...) with a non-None default")
        if (isinstance(node, ast.Attribute) and node.attr in ENV_ATTRS
                and isinstance(node.value, ast.Name) and node.value.id in os_modules):
            add("env_param", node, f"reads os.{node.attr}; experiment parameters belong in config", "env")
        elif isinstance(node, ast.Name) and node.id in env_names and isinstance(node.ctx, ast.Load):
            add("env_param", node, f"reads {node.id} (imported from os)", "env")
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if any(k in KNOWN_HW for k in keys):
                hw_maps.append((rel, node.lineno))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)) and id(node) not in ladder_seen:
            if _model_names_in(node) >= 2:
                add("model_ladder", node, "hard-coded model names; models come from the fleet config", None)
                ladder_seen.update(id(n) for n in ast.walk(node))   # report the outermost literal once
        if (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in KNOWN_HW
                and rel != HW_MAP_HOME):
            add("gpu_map", node, f"hardware type {node.value!r} outside {HW_MAP_HOME}", None)
    return out


def scan_paths(dirs: list[Path], base: Path) -> list[Finding]:
    findings: list[Finding] = []
    hw_maps: list[tuple[str, int]] = []
    files = sorted({p.resolve() for d in dirs for p in Path(d).rglob("*.py") if "__pycache__" not in p.parts})
    for path in files:  # a set, so overlapping directory arguments never scan a file twice
        findings.extend(scan_file(path, path.relative_to(Path(base).resolve()).as_posix(), hw_maps))
    homes = [m for m in hw_maps if m[0] == HW_MAP_HOME]
    if len(homes) > 1:
        findings += [Finding("gpu_map", r, ln, f"more than one hardware map in {HW_MAP_HOME}") for r, ln in homes]
    return findings


def scan_package(pkg: Path) -> list[Finding]:
    """Scan one package directory; paths are reported relative to its parent (e.g. routebalance/hw.py)."""
    pkg = Path(pkg)
    return scan_paths([pkg], pkg.parent)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m gates.static_defects <dir> [<dir> ...]", file=sys.stderr)
        return 2
    dirs = [Path(a).resolve() for a in argv]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        print(f"static_defects: not a directory: {missing}", file=sys.stderr)
        return 2
    top = subprocess.run(["git", "-C", str(dirs[0]), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode == 0:
        base = Path(top.stdout.strip()).resolve()
    else:
        base = Path(os.path.commonpath([str(d.parent) for d in dirs])).resolve()
    findings = scan_paths(dirs, base)
    for f in findings:
        print(f)
    print(f"static_defects: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
