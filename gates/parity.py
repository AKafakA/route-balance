"""Parity gate: the cells of one comparison group executed identically, except in the differences the group
declares.

Usage: python -m gates.parity --cells DIR [DIR ...]   (exit 0 on pass, 1 on fail; one reason per line)

The input is each cell's execution manifest (GET /v1/manifest, routebalance/server.py), read before and after
the cell, and its group.json:
  {"group": "<scope group>", "reference": null | "<config id>", "intended_differences": ["<pattern>", ...]}
The driver writes group.json from the scope. A pattern is a manifest path: either an exact leaf, such as
"system.lpt" or "telemetry.dead_reckoning", or a subtree, such as "system.rounds.*".

  - Every cell's manifest_before.json equals its manifest_after.json.
  - The code on disk never changed while the scheduler ran: code.* equals code.*_now.
  - All cells name the same group and the same reference, and the group is their record's labels.group.
  - A main comparison group (reference null): every cell declares the same patterns, usually ["system.*"].
    Two cells may differ only in keys those patterns match. A main group that declares no difference at all
    (one configuration by design, e.g. the W1b latency sweep at one seed) may be a single cell; its own checks
    (manifest unchanged, code unchanged, labels) still apply. Any other group needs at least two cells.
  - An ablation group (reference = a config id): the reference cells (that config id, declaring []) are
    present and identical to one another. Every other cell may differ from them only in keys its own patterns
    match, and must differ in every one of its patterns. A declared difference that did not happen is an
    ablation that did not run.
  - Keys that follow from the others:
      config_digest   is not compared; it is a digest of config sections compared key by key
      loaded.*        (the component classes) may differ only between cells whose system.* differs
      artifacts.*     a file that one cell loaded and the other did not may appear only when system.*
                      differs (or a pattern declares it)
      threadpools.*   which native pools are listed depends on the libraries the scheduler process has
                      loaded so far (one process serves many cells; libraries load lazily), so presence is free
      Across the whole group, whatever is declared, an artifact path has one md5 and a native pool one size in
      every cell that loaded it: a different md5 under one path is a file that changed during the campaign.
  - The bench side of each record is equal across the group: sampling, max_connections, request_timeout_s,
    dataset_sha256, tokenizer_id and prompt_order.
  - The engines (D48, the first rule on execution): each cell's engines_before.json equals its engines_after.json
    (no engine restarted or changed during the cell), and every cell of the group ran on the same engines with
    the same launches: the deploy.sh command, the KV cache each engine built, its model and context length.
    Only the start time may differ between cells (an engine restarted between them with the same launch).
Every offending key is reported with both values and the cells that hold them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gates.cell import Cell, GroupSpec, Problems, changed_keys, engine_changes, flatten, open_cell, run

NOT_COMPARED = ("config_digest",)
FOLLOW_SYSTEM = ("loaded",)                       # may differ exactly where system.* differs
LOADED_WHERE_USED = ("artifacts", "threadpools")  # one value per key across the group (_one_content_per_path)
HISTORY = ("threadpools",)                        # presence follows the process's history, not the system
BENCH_KEYS = ("sampling", "max_connections", "request_timeout_s", "dataset_sha256", "tokenizer_id", "prompt_order")
_MISSING = "<absent>"


def matches(pattern: str, key: str) -> bool:
    if pattern.endswith(".*"):
        root = pattern[:-2]
        return key == root or key.startswith(root + ".") or key.startswith(root + "[")
    return key == pattern


def _show(value: Any) -> str:
    return value if value == _MISSING else json.dumps(value, sort_keys=True)


def offending(a: dict[str, Any], b: dict[str, Any], patterns: list[str]) -> list[tuple[str, Any, Any]]:
    """(key, value in a, value in b) for every difference between two flattened manifests that the patterns
    and the derived-key rules do not allow."""
    system_differs = any(a[k] != b[k] if k in a and k in b else True
                         for k in set(a) | set(b) if k.startswith("system."))
    out = []
    for key in sorted(set(a) | set(b)):
        top = key.split(".")[0].split("[")[0]
        va, vb = (a[key] if key in a else _MISSING), (b[key] if key in b else _MISSING)
        if va == vb or top in NOT_COMPARED or any(matches(p, key) for p in patterns):
            continue
        if top in FOLLOW_SYSTEM and system_differs:
            continue
        if top in LOADED_WHERE_USED and (_MISSING not in (va, vb) or system_differs or top in HISTORY):
            continue                    # contents: _one_content_per_path; presence: system.* (artifacts) or free
        out.append((key, va, vb))
    return out


def parity_problems(cells: list[Cell]) -> tuple[list[str], str]:
    p = Problems()
    ids = [c.cell_id for c in cells]
    alone = len(cells) == 1 and cells[0].group_spec.reference is None and cells[0].group_spec.intended_differences == []
    if len(cells) < 2 and not alone:
        p.add("parity needs at least two cells of one group (one only in a group that declares no difference)",
              f"got {ids}")
    if len(set(ids)) != len(ids):
        p.add("a cell is listed twice", ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    specs: dict[str, GroupSpec] = {c.cell_id: c.group_spec for c in cells}
    flats, bench, config_of, launches = {}, {}, {}, {}
    for c in cells:
        rec, spec = c.record, specs[c.cell_id]
        label = rec.labels["group"] if "group" in rec.labels else _MISSING
        if spec.group != label:
            p.add("group.json names another group than the cell record", f"{c.cell_id}: {spec.group} vs {label}")
        if "config_id" not in rec.labels:
            p.add("a cell record has no config_id label", c.cell_id)
        config_of[c.cell_id] = rec.labels["config_id"] if "config_id" in rec.labels else _MISSING
        before, after = c.manifest("before"), c.manifest("after")
        if before != after:
            p.add("the execution manifest changed during a cell", f"{c.cell_id}: {changed_keys(before, after)}")
        for change in before.code.changed():
            p.add("the scheduler's code changed on disk after it started (restart it)", f"{c.cell_id}: {change}")
        flats[c.cell_id] = flatten(before.model_dump(mode="json"))
        engines_before = c.engines("before")
        changed = engine_changes(engines_before, c.engines("after"))
        if changed:
            p.add("an engine restarted or changed during a cell", f"{c.cell_id}: {changed}")
        launches[c.cell_id] = {i: e.launch() for i, e in engines_before.instances.items()}
        bench[c.cell_id] = {k: getattr(rec, k) for k in BENCH_KEYS} | {"sampling": rec.sampling.model_dump(mode="json")}
    groups = {s.group for s in specs.values()}
    references = {s.reference for s in specs.values()}
    if len(groups) != 1:
        p.add("the cells belong to more than one comparison group", ", ".join(sorted(groups)))
    if len(references) != 1:
        p.add("the cells name different reference configurations", ", ".join(sorted(map(str, references))))
    reference = next(iter(references)) if len(references) == 1 else None
    if len(references) == 1 and reference is None:
        _main_group(p, specs, flats)
    elif len(references) == 1:
        _ablation_group(p, specs, flats, config_of, reference)
    _one_content_per_path(p, flats)
    _bench(p, bench)
    _same_launches(p, launches)
    group = next(iter(groups)) if len(groups) == 1 else "?"
    kind = "main group" if reference is None else f"ablation group against {reference}"
    return p.lines(), f"{len(cells)} cells of {kind} {group} differ only as declared"


def _same_launches(p: Problems, launches: dict[str, dict[str, dict]]) -> None:
    ids = list(launches)
    for cid in ids[1:]:
        a, b = launches[ids[0]], launches[cid]
        if set(a) != set(b):
            p.add("the cells ran on different engines", f"{ids[0]}: {sorted(a)} vs {cid}: {sorted(b)}")
            continue
        for inst in sorted(a):
            for key in sorted(set(a[inst]) | set(b[inst])):
                va = a[inst][key] if key in a[inst] else _MISSING
                vb = b[inst][key] if key in b[inst] else _MISSING
                if va != vb:
                    p.add(f"the engines were launched differently: {inst}.{key}",
                          f"{ids[0]}: {_show(va)} vs {cid}: {_show(vb)}")


def _main_group(p: Problems, specs: dict, flats: dict) -> None:
    declared = {json.dumps(sorted(s.intended_differences)) for s in specs.values()}
    if len(declared) != 1:
        p.add("the cells of a main group declare different intended differences", "; ".join(sorted(declared)))
        return
    patterns = next(iter(specs.values())).intended_differences
    # every pair of distinct manifests (the derived-key rules depend on the pair); cells with one manifest
    # (the same system at other loads and seeds) are compared once, through their first cell
    first: dict[str, str] = {}
    for cid, flat in flats.items():
        first.setdefault(json.dumps(flat, sort_keys=True), cid)   # default-ok: grouping
    ids = list(first.values())
    seen: set[tuple[str, str, str]] = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for key, va, vb in offending(flats[a], flats[b], patterns):
                sig = (key, _show(va), _show(vb))
                if sig not in seen:
                    seen.add(sig)
                    p.add(f"an undeclared execution difference: {key}", f"{a}: {_show(va)} vs {b}: {_show(vb)}")


def _ablation_group(p: Problems, specs: dict, flats: dict, config_of: dict, reference: str) -> None:
    refs = [cid for cid in flats if config_of[cid] == reference]
    if not refs:
        p.add(f"no reference cell (config {reference}) among the cells")
        return
    for cid in refs:
        if specs[cid].intended_differences:
            p.add("a reference cell declares intended differences", f"{cid}: {specs[cid].intended_differences}")
    base = refs[0]
    for cid in refs[1:]:
        for key, va, vb in offending(flats[base], flats[cid], []):
            p.add(f"the reference cells differ: {key}", f"{base}: {_show(va)} vs {cid}: {_show(vb)}")
    for cid in flats:
        if cid in refs:
            continue
        patterns = specs[cid].intended_differences
        if not patterns:
            p.add("a treatment of an ablation group declares no difference (its override was lost?)", cid)
        for key, va, vb in offending(flats[base], flats[cid], patterns):
            p.add(f"an undeclared execution difference: {key}", f"{base} (reference): {_show(va)} vs {cid}: {_show(vb)}")
        a, b = flats[base], flats[cid]
        differing = [k for k in set(a) | set(b) if (a[k] if k in a else _MISSING) != (b[k] if k in b else _MISSING)]
        for pattern in patterns:
            if not any(matches(pattern, k) for k in differing):
                p.add("a declared difference did not happen (the ablation did not run)", f"{cid}: {pattern}")


def _one_content_per_path(p: Problems, flats: dict) -> None:
    """Across the whole group, whatever is declared: an artifact path has one md5 and a native pool one size in
    every cell that loaded it (a different md5 under one path is a file that changed during the campaign). The
    pairwise rules compare an ablation's treatments with the reference only; this also compares the treatments."""
    for section in LOADED_WHERE_USED:
        seen: dict[str, dict[str, list[str]]] = {}
        for cid, flat in flats.items():
            for key, value in flat.items():
                if key.startswith(section + "."):
                    seen.setdefault(key, {}).setdefault(_show(value), []).append(cid)   # default-ok: grouping
        for key, by_value in sorted(seen.items()):
            if len(by_value) > 1:
                p.add(f"{key} differs between the cells that loaded it",
                      "; ".join(f"{v} in {', '.join(sorted(cs)[:3])}" for v, cs in sorted(by_value.items())))


def _bench(p: Problems, bench: dict) -> None:
    flat = {cid: flatten(b) for cid, b in bench.items()}
    keys = sorted(set().union(*(f.keys() for f in flat.values()))) if flat else []
    for key in keys:
        holders: dict[str, list[str]] = {}
        for cid, f in flat.items():
            holders.setdefault(_show(f[key] if key in f else _MISSING), []).append(cid)   # default-ok: grouping
        if len(holders) > 1:
            p.add(f"the bench ran differently: {key}",
                  "; ".join(f"{v} in {', '.join(sorted(cs)[:3])}" for v, cs in sorted(holders.items())))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Diff the execution manifests of one comparison group's cells.")
    ap.add_argument("--cells", type=Path, nargs="+", required=True, help="the group's cell directories")
    a = ap.parse_args(argv)
    return run("parity", lambda: parity_problems([open_cell(d) for d in a.cells]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
