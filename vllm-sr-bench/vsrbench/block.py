"""A family's vLLM-SR block: the given seeds at the main grid's rates in one block (seed 1 only),
seed 1 first, each run by `vsrbench.run_cell`; and the block's coverage.

    python -m vsrbench.block run --family F --render R --traces l12,l30,... --seeds 1,2,3 --reference-cells DIR
                                 --cells-root DIR --mirror DEST [--max-invalid 3] -- <run_cell's other arguments>
    python -m vsrbench.block status --family F --render R --traces ... --seeds ... --cells-root DIR --reference-cells DIR

Each run's template is the round-robin dispatcher's main-grid cell at its rate, or, if the render has none, another
cell of the family's main group at that rate; the run sends the template's trace at its own seed (run_cell --seed): the
arrivals and the prompt order follow from the trace and the seed, as every system of that rate and seed is sent (the
main grid renders seeds 2-3 only at the knee points). Its engines are compared with a published, valid run of that
group (the template if it is one). Its inputs are checked against the render in every verdict, and also against a
published, valid main-grid run at its rate and seed whenever there is one (rechecked here, so a pair published after
the run still counts). Every run of the block records the same versions (a version change is flagged):
`run` stops before a run whose versions differ from the block's first published run, and `status` fails a mixed block.
A run already published is not run again: a valid one counts, an invalid one stays for the main session to retire and
rerun (as the main grid's). `run` stops, as scripts/run_scope.sh does, after more than --max-invalid invalid runs in
this invocation, at once on a run that could not run (run_cell's exit 2; its attempt stays for debugging) and on a
failed mirror; a run already published is mirrored again (verified identical). It exits 0 only when every run of the
block is valid.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from analysis.manifest import cell_problems
from bench.records import cell_dir
from gates.cell import GateError, open_cell, reset_problems, traffic_problems
from bench.scope import expected_cells
from fmrlb.config import ConfigError
from tools import cell_driver
from vsrbench.run_cell import CLIENT_ENV, _sha, input_problems, inputs_reference, shared_tree_check, versions, vsr_cell
from vsrbench.stack import Docker

MODULE = Path(__file__).resolve().parents[1]


def cell_id(family: str, trace: str, seed: int) -> str:
    return f"{family}.vsr.mom-balance__{trace}__s{seed}"


def template_for(scope, family: str, trace: str) -> str:
    """The round-robin dispatcher's main-grid cell at this rate (its lowest seed), else the family's main group's first
    cell at this rate: the trace, dataset and prompt-order rule the run sends at its own seed."""
    cells = sorted((c for c in expected_cells(scope) if c.group == family and c.trace_id == trace),
                   key=lambda c: (c.config_id != f"{family}.disp.rr", c.seed, c.cell_id))
    if not cells:
        raise ConfigError(f"the render has no {family} cell at {trace}: no prompts to send")
    return cells[0].cell_id


def expected(scope, family: str, template: str, seed: int, config_sha: str):
    return vsr_cell(dataclasses.replace(cell_driver.expected(scope, template), seed=seed), family, config_sha, test=False)


PATH_FLAGS = ("--fleet", "--hosts", "--engine-args", "--engines-record", "--training-rows", "--data-root", "--vsr",
              "--templates", "--models-cache", "--work")
BLOCK_FLAGS = ("--profile", "--render", "--template-cell", "--seed", "--reference-cell", "--family",
               "--reference-cells", "--cells-root", "--mirror")    # set by the block for each run; never in the rest


def module_digests() -> dict[str, str]:
    """This module's code, as run_cell's harness record names it (vsr/harness.json "module")."""
    return {str(p.relative_to(MODULE)): _sha(p) for p in sorted(MODULE.glob("vsrbench/*.py"))}


def versions_of(root: Path, cid: str):
    """What a published run executed with (vsr/harness.json): its versions and its code (this module's and the shared
    files' digests), or None if it recorded none."""
    try:
        h = json.loads((cell_dir(root, cid) / "vsr" / "harness.json").read_text())
    except (OSError, ValueError):
        return None
    return {"versions": h.get("versions"), "module": h.get("module"), "shared": h.get("shared")}


def mixed_versions(root: Path, order) -> list[str]:
    """The published runs whose versions or code differ from the first published run's (in the block's order)."""
    got = [(cid, versions_of(root, cid)) for cid, _, _ in order if cell_dir(root, cid).is_dir()]
    return [cid for cid, v in got[1:] if v != got[0][1]] if got else []


def verdict(root: Path, cid: str, scope=None, family: str = "", template: str = "", seed: int = 0,
            reference_cells=None) -> str:
    """missing | valid | invalid: valid only with a boolean true verdict, a readable record of this cell and the
    isolation evidence (the proven cache reset, the traffic check) rechecked now, never trusted from the saved verdict;
    given the render, the run's template and seed, also the main grid's record checks against the current render (id,
    labels, seed, trace, order, dataset, sampling, bench parameters, digests: analysis.manifest.cell_problems), with the
    configuration digest the sha256 of the configuration the run published; given the main grid's cells root, the inputs
    against a published, valid main-grid run at its rate and seed, if there is one now."""
    d = cell_dir(root, cid)
    if not d.is_dir():
        return "missing"
    try:
        valid = json.loads((d / "gates" / "verdict.json").read_text())["valid"]
        cell = open_cell(d)
        rec = cell.record
        config_sha = hashlib.sha256((d / "vsr" / "config.yaml").read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError, GateError):
        return "invalid"
    if valid is not True or rec.cell_id != cid or reset_problems(cell) or traffic_problems(cell):
        return "invalid"
    if scope is not None:
        exp = expected(scope, family, template, seed, config_sha)
        if exp.cell_id != cid or cell_problems(rec, exp, scope.bench):
            return "invalid"
        pair = None if reference_cells is None else inputs_reference(scope, reference_cells, template, seed)
        try:
            if pair is not None and input_problems(rec, open_cell(pair).record):
                return "invalid"
        except GateError:
            return "invalid"
    return "valid"


def reference_for(scope, root: Path, family: str, template: str) -> str:
    """The template if it is published and valid in the main grid's cells root, else the newest valid cell of the
    family's main group there (every cell of a group runs on the same engine launches: gates/parity.py)."""
    exp = {c.cell_id: c for c in expected_cells(scope) if c.group == family}
    if template in exp and cell_driver.status(scope, exp[template], root) == "done":
        return template
    valid = [c for c, e in exp.items() if cell_dir(root, c).is_dir() and cell_driver.status(scope, e, root) == "done"]
    if not valid:
        raise ConfigError(f"no valid {family} main-grid run in {root} to compare the engines with")
    return max(valid, key=lambda c: (cell_dir(root, c) / "gates" / "verdict.json").stat().st_mtime)


def mirror(root: Path, dest: str, cid: str) -> int:
    """scripts/mirror.sh's mirror_cell: never over another run of the cell, then identical (checksums)."""
    return subprocess.run(["bash", "-c", f'source "{MODULE.parent}/scripts/mirror.sh" && mirror_cell "$0" "$1" "$2"',
                           str(root), dest, cid]).returncode


def _arg(rest: list[str], flag: str) -> str:
    if flag not in rest:
        raise ConfigError(f"run needs {flag} among run_cell's arguments")
    return rest[rest.index(flag) + 1]


def plan(scope, family: str, traces: list[str], seeds: list[int]) -> list[tuple[str, str, int]]:
    templates = {t: template_for(scope, family, t) for t in traces}
    return [(cell_id(family, t, s), templates[t], s) for s in seeds for t in traces]


def main(argv: list[str]) -> int:
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    # --flag=value as --flag value, so every check below (and run_cell) reads one form
    rest = [part for tok in rest for part in (tok.split("=", 1) if tok.startswith("--") and "=" in tok else [tok])]
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("command", choices=("run", "status"))
    ap.add_argument("--family", required=True)
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--reference-cells", type=Path, required=True,
                    help="the main grid's cells root: the engines' comparison and the inputs' pairing")
    ap.add_argument("--mirror", help="the second location (rsync syntax)")
    ap.add_argument("--max-invalid", type=int, default=3)     # scripts/run_scope.sh's --max-invalid as the runs use it
    a = ap.parse_args(argv)
    a.render, a.cells_root, a.reference_cells = a.render.resolve(), a.cells_root.resolve(), a.reference_cells.resolve()
    if a.mirror is not None and ":" not in a.mirror:            # a local mirror: the same place for run_cell and here
        a.mirror = str(Path(a.mirror).resolve())
    relative = [f"{rest[i]} {rest[i + 1]}" for i in range(len(rest) - 1)
                if rest[i] in PATH_FLAGS and not Path(rest[i + 1]).is_absolute()]
    if relative:                # run_cell runs in the module's directory: a relative path would land there
        raise ConfigError(f"give absolute paths after --: {relative}")
    shared_tree_check()           # the block's own checks (verdict, pairing) are this tree's too
    given = [tok for tok in rest if tok.startswith("--")]
    repeated = sorted({g for g in given if given.count(g) > 1})
    if repeated:                  # the block reads the first, run_cell's parser the last: never two
        raise ConfigError(f"{repeated} given more than once after --")
    clash = sorted({g for g in given for f in BLOCK_FLAGS if f.startswith(g) and len(g) > 2})   # and any abbreviation
    if clash:
        raise ConfigError(f"{clash} are set by the block for each run; they cannot be given after --")
    scope, _ = cell_driver.load(a.render)
    order = plan(scope, a.family, a.traces.split(","), [int(s) for s in a.seeds.split(",")])
    def states() -> dict[str, str]:
        return {cid: verdict(a.cells_root, cid, scope, a.family, t, s, a.reference_cells) for cid, t, s in order}
    if a.command == "status":
        now = states()
        for cid, t, s in order:
            pair = inputs_reference(scope, a.reference_cells, t, s)
            print(f"{now[cid]:8s} {cid}  inputs paired with: "
                  + ("none published (checked against the render)" if pair is None
                     else f"{pair.name} ({_sha(next(pair.glob('*.cell.json')))[:16]})"))
        count = {k: sum(v == k for v in now.values()) for k in ("valid", "invalid", "missing")}
        mixed = mixed_versions(a.cells_root, order)
        print(f"[vsr block] {count['valid']}/{len(now)} valid, {count['invalid']} invalid, {count['missing']} missing"
              + (f"; versions differ from the first run's in {mixed}" if mixed else ""))
        return 0 if all(s == "valid" for s in now.values()) and not mixed else 1
    if a.mirror is None:
        raise ConfigError("run needs --mirror (the second location, as run_cell's)")
    extra = sorted(set(os.environ) - set(CLIENT_ENV))
    if extra:       # the block's own version check reads what run_cell reads: the client's environment only
        raise ConfigError(f"start the block as run_cell starts (env -i with {CLIENT_ENV}); found also {extra}")
    invalid = 0
    for cid, template, seed in order:
        state = verdict(a.cells_root, cid, scope, a.family, template, seed, a.reference_cells)
        if state != "missing":
            print(f"[vsr block] {cid}: {state}, published; not run again")
            if mirror(a.cells_root, a.mirror, cid) != 0:
                print(f"[vsr block] the mirror of {cid} failed: stopping")
                return 3
            continue
        first = next((c for c, _, _ in order if cell_dir(a.cells_root, c).is_dir()), None)
        if first is not None:       # a version change within the block is flagged; the block stops
            vsr, sg = _arg(rest, "--vsr"), "--docker-sg" in rest
            then = versions_of(a.cells_root, first) or {}
            now = {"versions": versions(Path(vsr), Docker(sg)), "module": module_digests()}
            differ = sorted(k for k in now if now[k] != then.get(k))
            if differ:
                print(f"[vsr block] stopping: the {differ} differ from the block's first run ({first})")
                return 2
        ref = reference_for(scope, a.reference_cells, a.family, template)
        cmd = [sys.executable, "-m", "vsrbench.run_cell", "--profile", "lease", "--render", str(a.render),
               "--template-cell", template, "--seed", str(seed), "--reference-cell", ref, "--family", a.family,
               "--reference-cells", str(a.reference_cells), "--cells-root", str(a.cells_root), "--mirror", a.mirror,
               *rest]
        print(f"[vsr block] {cid}: template {template} at seed {seed}, engines compared with {ref}", flush=True)
        env = {k: os.environ[k] for k in CLIENT_ENV if k in os.environ}   # as scripts/run_cell.sh starts the client
        rc = subprocess.run(cmd, cwd=MODULE, env=env).returncode
        if rc == 2:
            print(f"[vsr block] {cid} could not run (exit 2): stopping; its attempt stays under _attempts")
            return 2
        if rc == 3:
            print(f"[vsr block] {cid}: the mirror failed: stopping")
            return 3
        state = verdict(a.cells_root, cid, scope, a.family, template, seed, a.reference_cells)
        if rc not in (0, 1) or state == "missing":
            print(f"[vsr block] {cid}: run_cell exited {rc} and published nothing: stopping")
            return 2
        invalid += state != "valid"
        if invalid > a.max_invalid:
            print(f"[vsr block] stopping: more than {a.max_invalid} invalid runs")
            return 1
    left = sorted(c for c, st in states().items() if st != "valid")
    mixed = mixed_versions(a.cells_root, order)
    print(f"[vsr block] {a.family}: {invalid} runs invalid in this invocation; "
          + ("every run of the block valid" if not left else f"not valid yet: {left}")
          + (f"; versions differ from the first run's in {mixed}" if mixed else ""))
    return 0 if not left and not mixed else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except ConfigError as e:
        print(f"[vsr block] refused: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:      # a crash is "could not run" (2), never "some runs invalid" (1)
        print(f"[vsr block] failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
