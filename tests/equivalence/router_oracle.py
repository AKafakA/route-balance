"""The v3 scheduler tree and its router artifacts under the oracle root, for the baseline replays.

Layout: Block/block/global_scheduler/<pkg>/ (the v3 scheduler: routers/, dispatch/, the serve module),
Block/models/<pkg>/ (best_route_4way_qwen/, avengers_4model/), AvengersPro/ (the upstream checkout)
and rb-exp/v3-exp/cells/s2_p1_l6/ (the banked lambda-6 cells). <pkg> is discovered, not named.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

BANKED_N = 3534


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique(paths: list[Path], what: str) -> Path:
    if len(paths) != 1:
        pytest.fail(f"expected exactly one {what}, found {paths}")
    return paths[0]


@dataclass(frozen=True)
class V3Tree:
    scheduler: Path       # the v3 scheduler package directory
    artifacts: Path       # the v3 model artifact root
    upstream: Path        # upstream AvengersPro checkout
    cells: Path           # banked lambda-6 S2 cells

    @classmethod
    def under(cls, root: Path) -> "V3Tree":
        root = Path(root).resolve()
        routers = root.glob("Block/block/global_scheduler/*/routers/best_route_4way.py")
        scheduler = _unique([p.parent.parent for p in routers], "v3 scheduler package")
        artifacts = _unique([p.parent.parent for p in root.glob("Block/models/*/best_route_4way_qwen/config.json")],
                            "v3 artifact root")
        upstream, cells = root / "AvengersPro", root / "rb-exp/v3-exp/cells/s2_p1_l6"
        for p in (upstream / "simple_cluster_router.py", cells):
            if not p.exists():
                pytest.fail(f"oracle is missing {p}")
        sys.dont_write_bytecode = True   # the mirror is read-only: importing v3 modules must not write .pyc files
        return cls(scheduler, artifacts, upstream, cells)

    def v3_package(self, sub: str):
        """Import v3 plug-in package `sub` (routers or dispatch) under an alias, without its __init__,
        which imports every plug-in. Relative imports inside it resolve through __path__."""
        alias = f"v3_{sub}"
        if alias not in sys.modules:
            pkg = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(alias, None, is_package=True))
            pkg.__path__ = [str(self.scheduler / sub)]
            sys.modules[alias] = pkg
        return lambda name: importlib.import_module(f"{alias}.{name}")

    def v3_unwrap(self):
        """v3's _estimator_prompt, taken verbatim from the serve module's source (importing that module
        would start a server's worth of globals)."""
        serve = self.scheduler / f"{self.scheduler.name}_serve.py"
        tree = ast.parse(serve.read_text())
        keep = [n for n in tree.body
                if (isinstance(n, ast.FunctionDef) and n.name == "_estimator_prompt")
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_QWEN_USER_RE", "_GEMMA_USER_RE")
                                                      for t in n.targets))]
        if len(keep) != 3:
            pytest.fail(f"could not find _estimator_prompt and its two regexes in {serve}")
        ns = {"re": re}
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(serve), "exec"), ns)
        return ns["_estimator_prompt"]

    def banked(self, cell: str) -> list[dict]:
        rows = json.loads((self.cells / f"{cell}_rr_l6_n{BANKED_N}.json").read_text())["response_details"]
        assert len(rows) == BANKED_N, f"{cell}: {len(rows)} rows"
        errors = [r["error"] for r in rows if r["error"]]
        assert not errors, f"{cell}: banked errors {errors[:3]}"
        return rows
