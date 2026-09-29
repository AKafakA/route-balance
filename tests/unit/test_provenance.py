"""D134 (owner 09-29: "manual set the default value ... should never never never allowed"): every configured value in
the scope and the presets, and every numeric constant of the serving code, has its source in configs/provenance.json
(the owner's decision, the upstream system's stock default, or v2/v3's accepted value) and still has the recorded value.
A new, removed or changed value fails here until the table records it with its source."""
import ast
import json
import math
import operator
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = json.loads((ROOT / "configs" / "provenance.json").read_text())
CONSTANT = r"_?[A-Z][A-Z0-9_]*"
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}


def _value(node, env=None):
    """A constant's value, read from the source (no import): numbers, arithmetic, tuples, math.inf, int()/float() of
    one, an earlier constant of the same scope by name, and an index into one (BACKLOG_CAP = ADAPTIVE_LIMITS[-1][1]).
    Anything else (a string, a call, a name from elsewhere) is not a numeric constant and returns None."""
    env = {} if env is None else env
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Subscript):
        base, idx = _value(node.value, env), _value(node.slice, env)
        return base[idx] if isinstance(base, tuple) and type(idx) is int and -len(base) <= idx < len(base) else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("int", "float") \
            and len(node.args) == 1 and not node.keywords:
        v = _value(node.args[0], env)
        return None if v is None or isinstance(v, tuple) else {"int": int, "float": float}[node.func.id](v)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _value(node.operand, env)
        return None if v is None or isinstance(v, tuple) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _value(node.left, env), _value(node.right, env)
        if a is None or b is None or isinstance(a, tuple) or isinstance(b, tuple):
            return None
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and (node.value.id, node.attr) == ("math", "inf"):
        return math.inf
    if isinstance(node, ast.Tuple):
        vals = [_value(e, env) for e in node.elts]
        return None if any(v is None for v in vals) else tuple(vals)
    return None


def serving_constants(root: Path = ROOT, entries=None) -> dict[str, str]:
    """module.NAME -> repr(value) for every module- and class-level numeric constant of the serving modules."""
    out = {}
    for entry in TABLE["modules"] if entries is None else entries:
        base = root / entry
        files = sorted(base.rglob("*.py")) if base.is_dir() else [base]
        for f in files:
            mod = ".".join(f.relative_to(root).with_suffix("").parts)
            tree = ast.parse(f.read_text())
            scopes = [(mod, tree.body)] + [(f"{mod}.{c.name}", c.body) for c in tree.body if isinstance(c, ast.ClassDef)]
            for where, body in scopes:                  # module-level and class-level constants alike
                env: dict = {}
                for node in body:
                    if isinstance(node, ast.Assign):
                        targets = node.targets
                    elif isinstance(node, ast.AnnAssign):
                        targets = [node.target]
                    else:
                        continue
                    for t in targets:
                        if isinstance(t, ast.Name) and re.fullmatch(CONSTANT, t.id) and node.value is not None:
                            v = _value(node.value, env)
                            if v is not None:
                                env[t.id] = v
                                out[f"{where}.{t.id}"] = repr(v)
    return out


def _leaves(o, path):
    """Every leaf: numbers, flags, nulls, strings (a latency source, a dispatcher type, a pinned artifact) and lists."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, f"{path}.{k}")
    else:
        yield path, o


def configured(scope_path: Path = ROOT / "configs" / "scope.json",
               presets_path: Path = ROOT / "configs" / "presets.json") -> dict[str, object]:
    scope = json.loads(scope_path.read_text())
    presets = json.loads(presets_path.read_text())
    out = dict(p for sec in ("server", "bench", "warmup", "drain") for p in _leaves(scope[sec], f"scope.{sec}"))
    return out | dict(_leaves(presets["parts"], "presets.parts"))


def value_problems(now: dict, recorded: dict) -> list[str]:
    return ([f"no recorded source: {k}" for k in sorted(set(now) - set(recorded))]
            + [f"recorded, gone: {k}" for k in sorted(set(recorded) - set(now))]
            + [f"differs from the table: {k}" for k in sorted(set(now) & set(recorded)) if now[k] != recorded[k]])


def test_a_changed_value_or_an_unlisted_constant_is_caught_through_the_extractors(tmp_path):
    """Codex reviews 09-29: mutations go through the real extractors: a stock-sourced timeout set to 1 s, the pending
    decode set to another mode, a new bound, and a derived constant whose index changes."""
    recorded = {k: v for e in TABLE["values"] for k, v in e["values"].items()}
    scope = json.loads((ROOT / "configs" / "scope.json").read_text())
    presets = json.loads((ROOT / "configs" / "presets.json").read_text())
    scope["server"]["request_timeout_s"] = 1.0
    presets["parts"]["rb.pending_decode"] = "lpt_key"          # a dotted key in the presets
    (tmp_path / "scope.json").write_text(json.dumps(scope))
    (tmp_path / "presets.json").write_text(json.dumps(presets))
    assert value_problems(configured(tmp_path / "scope.json", tmp_path / "presets.json"), recorded) == [
        "differs from the table: presets.parts.rb.pending_decode", "differs from the table: scope.server.request_timeout_s"]
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "rounds.py").write_text("LIMITS = ((0.3, 4, 50.0), (1.0, 32, 200.0))\nCAP = LIMITS[-1][1]\n")
    before = serving_constants(tmp_path, ["pkg"])
    assert before == {"pkg.rounds.LIMITS": "((0.3, 4, 50.0), (1.0, 32, 200.0))", "pkg.rounds.CAP": "32"}
    (pkg / "rounds.py").write_text("LIMITS = ((0.3, 4, 50.0), (1.0, 32, 200.0))\nCAP = LIMITS[0][1]\nNEW = 2 ** 10\n")
    assert value_problems(serving_constants(tmp_path, ["pkg"]), before) == [
        "no recorded source: pkg.rounds.NEW", "differs from the table: pkg.rounds.CAP"]


def test_every_entry_names_an_allowed_source_and_a_reference():
    for e in TABLE["values"] + TABLE["constants"]:
        assert e["source"] in TABLE["sources"] and e["ref"].strip(), e


def test_every_configured_value_has_a_source_and_its_recorded_value():
    recorded = {k: v for e in TABLE["values"] for k, v in e["values"].items()}
    assert value_problems(configured(), recorded) == []


def test_every_numeric_serving_constant_has_a_source_and_its_recorded_value():
    recorded = {f"{c['module']}.{c['name']}": c["value"] for c in TABLE["constants"]}
    assert value_problems(serving_constants(), recorded) == []
