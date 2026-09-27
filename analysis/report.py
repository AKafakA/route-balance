"""The paper's numbers, from the lease's cells (plan Part B; paper rule R12: a number reaches the paper only
through this tool's output, which the paper \\input{}s).

    python -m analysis.report --render DIR --cells ROOT --data-root DIR --fleet NAME=FLEET.json [...]
        --hosts HOSTS.json --slo-base FAMILY=BASE.json [...] [--store STORE.jsonl ...]
        --rules configs/prereg_rules.json --treatment SYSTEM [...] --out DIR
    exit 0 only when every expected cell is ok, none is left out and every group's parity passed over its current
    cells; 1 otherwise (the report is written either way)

Inputs, each checked:
  the render   plan.json (group -> fleet -> family; the scored corpora with their sha256) and scope.json (the
               expected cells); a fleet file must have the plan's fleet_sha256 for its name;
  the cells    analysis/manifest.py check: only ok cells enter; a group whose cells ran on two fleets is left out
               whole (its fleet problem is reported); a cell enters only with a passing gate verdict
               (gates/verdict.json), and no cell of a group whose latest parity record failed enters; a group
               whose parity has not run over its current cells is reported ("parity");
  per family   the SLO bases (tools/slo_base.py) and the scored corpus under --data-root, pinned by the plan;
  the stores   the served-text score stores (tools/judge_served.py), for the budget cells.
Every ok cell is summarised against its group's fleet (analysis/metrics.py summarize_cell) with the preregistered
SLO multipliers on its family's bases. A cell whose summary has problems is reported and kept out of every
comparison, never averaged in. A valid cell's per-request values (the request bootstrap's input) must reproduce
its summary.

Output, a new directory:
  report.json  per cell: the summary, the model mix and the overhead stages; per configuration: the seed means
               (analysis/frontier.py); the frontiers per metric; for each --treatment system, the equal-quality rows
               (analysis/equal_quality.py, each seed's value, the range over the seeds and the request-bootstrap
               interval, D80, D81) by both readings (D86): "equal_quality" (one_config: one treatment
               configuration for every metric) and "equal_quality_per_metric" (per_metric: each metric on its own
               frontier, as v2/v3), each with its "_appendix" rows; and every baseline configuration it could not
               compare, with the
               reason (`not_compared`: an ablation group, whose variants are read against its reference in its
               systems table; no treatment configuration at that group and trace; or no treatment preset with
               cells for every seed of the baseline); goodput at the rules' attainment levels of the tight and
               loose SLO per configuration over the stationary traces, with the rates it was read from
               (analysis/slo.py)
  tables/systems.<group>.<trace>.tex         every configuration of the group at the trace
  tables/equal.<treatment>.<group>.<trace>.tex   (the main text) every compared baseline configuration except the
               appendix-only ones (APPENDIX_SYSTEMS, below): the
               relative change of mean and P99 E2E and cost, and the change of SLO attainment in percentage
               points, each with its range over the seeds (-- where there is no equal-quality point)
  tables/equal_ci.<treatment>.<group>.<trace>.tex   (the appendix) the same changes with their request-bootstrap
               intervals
  tables/equal_seeds.<treatment>.<group>.<trace>.tex   (the appendix) each seed's changes
  tables/equal_appendix.<treatment>.<group>.<trace>.tex   (the appendix) the appendix-only baselines
               (APPENDIX_SYSTEMS: the dispatch-only least-outstanding and power of two counting in-flight requests,
               D84), with their request-bootstrap intervals; they run in the main group so that they pair
               with the main grid's runs, and never enter the equal-quality tables of the main text or
               macros.tex (the systems tables list every configuration of a group, these included)
  macros.tex   per treatment, group and stationary trace: the range over the baseline configurations of the
               relative change of mean E2E, P99 E2E and cost at equal quality (...Min/...Max), and including the
               comparisons read at higher quality (...MinAll/...MaxAll, D87)
A value read at RouteBalance's frontier low end, above the baseline's quality beyond the tolerance (D87), carries
a dagger in the tables. Every equal* table and macro is written for both readings: the one_config names as above,
the per_metric ones with the prefix equal_pm (tables) and eqpm (macros). The paper's framing picks one after the
runs (D86).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from bench.dataset import file_sha256
from bench.schema import loads_strict
from analysis.equal_quality import READINGS, SampleKey, equal_quality
from analysis.frontier import METRICS, CellPoint, ConfigPoint, by_config, cell_point, config_points, frontiers
from analysis.manifest import Coverage, check
from analysis.metrics import CellSummary, RequestValues, request_values, summarize_cell
from analysis.quality import ScoredCorpus, ScoreStore
from analysis.rules import Rules, load_rules
from analysis.slo import CellSlo, SloBase, goodputs
from bench.records import CellRecord, cell_dir, cell_path, read_cell
from bench.scope import ExpectedCell, GroupSpec, Scope, expected_cells, load_scope
from routebalance.config import ConfigError, load_fleet
from routebalance.types import Instance
from tools.campaign import load_plan

SLO_LEVELS = ("tight", "loose")
MACRO_METRICS = {"e2e_mean_ms": "Mean", "e2e_p99_ms": "Tail", "cost_decode_per_1k": "Cost"}
TABLE_METRICS = {**MACRO_METRICS, "slo_miss_tight": "SLO tight (pp)", "slo_miss_loose": "SLO loose (pp)"}
ATTAINMENT = ("slo_miss_tight", "slo_miss_loose")        # shown as the change of attainment, in points
APPENDIX_SYSTEMS = frozenset({"dispatcher-inflight"})  # D84: compared in the appendix only (equal_appendix.*)
# reading -> (report key, table prefix, macro prefix); both readings are always written (D86)
READING_OUT = {"one_config": ("equal_quality", "equal", "eq"),
               "per_metric": ("equal_quality_per_metric", "equal_pm", "eqpm")}
assert set(READING_OUT) == set(READINGS)
_DIGITS = dict(zip("0123456789",
                   ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"), strict=True))


def _pairs(items: Sequence[str], what: str) -> dict[str, Path]:
    out = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path or name in out:
            raise ConfigError(f"--{what} takes NAME=PATH once per name, got {item!r}")
        out[name] = Path(path)
    return out


def stationary_rates(scope: Scope) -> dict[str, float]:
    """trace id -> rate, for the traces with one Poisson segment (the goodput curves use only these)."""
    out = {}
    for tid, t in scope.traces.items():
        if len(t.segments) == 1 and t.burstiness == 1.0:
            out[tid] = float(t.segments[0].rate)
    return out


def model_mix(cell: CellRecord) -> dict[str, float]:
    served = Counter(r.reply.model for r in cell.requests if r.reply is not None)
    n = sum(served.values())
    return {m: c / n for m, c in sorted(served.items())} if n else {}


class Inputs:
    """The render and every input, checked; one fleet, SLO and quality source per group."""

    def __init__(self, render: Path, data_root: Path, fleets: Mapping[str, Path], hosts: Path,
                 slo_bases: Mapping[str, Path], stores: Sequence[Path], rules: Rules):
        self.plan = load_plan(render)
        self.scope = load_scope(Path(render) / "scope.json")
        self.rules = rules
        judge_key = rules.judge_key
        self.fleets: dict[str, list[Instance]] = {}
        for name, entry in self.plan["fleets"].items():
            if name not in fleets:
                continue
            if file_sha256(fleets[name]) != entry["fleet_sha256"]:
                raise ConfigError(f"{fleets[name]} is not the fleet {name} the plan pins")
            self.fleets[name] = load_fleet(fleets[name], hosts)
        self.corpora: dict[str, ScoredCorpus] = {}
        for family, entry in self.plan["scored_corpora"].items():
            path = Path(data_root) / entry["path"]
            if not path.is_file():
                continue
            if file_sha256(path) != entry["sha256"]:
                raise ConfigError(f"{path} is not the scored corpus the plan pins for {family}")
            self.corpora[family] = ScoredCorpus.load([path], judge_key)
        self.store = ScoreStore.load(list(stores), judge_key) if stores else None
        self.slo_bases = {family: SloBase.load(p) for family, p in slo_bases.items()}

    def group_fleet(self, group: str) -> tuple[str, str]:
        name = self.plan["groups"][group]["fleet"]
        return name, self.plan["fleets"][name]["family"]

    def _sources(self, exp: ExpectedCell, cell: CellRecord) -> tuple[ScoredCorpus | ScoreStore, list[Instance], CellSlo]:
        name, family = self.group_fleet(exp.group)
        for have, what, key in ((self.fleets, "--fleet", name), (self.slo_bases, "--slo-base", family)):
            if key not in have:
                raise ConfigError(f"cell {exp.cell_id} (group {exp.group}) needs {what} {key}=...")
        fleet = self.fleets[name]
        slo = CellSlo.of(self.rules.slo.k(), self.slo_bases[family], fleet)
        if cell.budgets_sha256 is not None:
            if self.store is None:
                raise ConfigError(f"budget cell {exp.cell_id} needs the served-text score stores (--store)")
            return self.store, fleet, slo
        if family not in self.corpora:
            raise ConfigError(f"cell {exp.cell_id} needs the {family} scored corpus under --data-root")
        return self.corpora[family], fleet, slo

    def summarize(self, exp: ExpectedCell, cell: CellRecord) -> CellSummary:
        return summarize_cell(cell, *self._sources(exp, cell))

    def request_values(self, exp: ExpectedCell, cell: CellRecord) -> RequestValues:
        return request_values(cell, *self._sources(exp, cell))


def check_values(cell_id: str, s: CellSummary, v: RequestValues) -> None:
    """The per-request values reproduce the cell's summary: the bootstrap resamples what the comparison uses."""
    got = {k: float(x[0]) for k, x in v.at(np.arange(len(v.index))[None, :]).items()}
    want = {"quality": s.quality, "e2e_mean_ms": s.e2e_ms.mean, "e2e_p99_ms": s.e2e_ms.p99,
            "cost_decode_per_1k": s.cost_decode_per_1k, "slo_miss_tight": 1.0 - s.slo_attainment_tight,
            "slo_miss_loose": 1.0 - s.slo_attainment_loose}
    bad = {k: (got[k], w) for k, w in want.items() if not math.isclose(got[k], w, rel_tol=1e-9, abs_tol=1e-12)}
    if bad:
        raise ConfigError(f"cell {cell_id}: the per-request values do not reproduce the summary: {bad}")


def comparable(points: Sequence[CellPoint], treatment: str,
               groups: Mapping[str, GroupSpec]) -> tuple[list[CellPoint], dict[str, str]]:
    """The points equal_quality() compares for `treatment`, and each baseline configuration it cannot compare
    ("group|trace|config" -> why; see the module docstring)."""
    table = by_config(points)
    system_of = {key: next(iter(per_seed.values())).system for key, per_seed in table.items()}
    keep, skipped = set(), {}
    for (g, t, c), per_seed in sorted(table.items()):
        if system_of[(g, t, c)] == treatment:
            continue
        presets = [k for k, s in system_of.items() if k[:2] == (g, t) and s == treatment]
        why = None
        if groups[g].reference is not None:
            why = f"ablation group: read against its reference {groups[g].reference}"
        elif not presets:
            why = f"no {treatment} configuration at {g}/{t}"
        elif not any(set(per_seed) <= set(table[k]) for k in presets):
            why = f"no {treatment} preset has cells for every seed {sorted(per_seed)}"
        if why is None:
            keep.update([(g, t, c), *presets])
        else:
            skipped[f"{g}|{t}|{c}"] = why
    return [p for p in points if (p.group, p.trace_id, p.config_id) in keep], skipped


def gate_problem(cells_root: Path, cell_id: str) -> Optional[str]:
    """Why a cell's per-cell gates do not pass (scripts/run_cell.sh writes gates/verdict.json), or None."""
    v = cell_dir(cells_root, cell_id) / "gates" / "verdict.json"
    if not v.is_file():
        return "no gate verdict (gates/verdict.json)"
    verdict = loads_strict(v.read_text())
    if not isinstance(verdict, dict) or verdict.get("valid") is not True:   # default-ok: a missing key is not valid
        return "its gates did not pass (gates/verdict.json)"
    return None


def parity_state(cells_root: Path, group: str, runs: Mapping[str, str]) -> str:
    """'passed' when the group's latest parity record (tools/cell_driver.py) passed and covers exactly these runs
    ({cell directory name: record sha256}; names, since the record holds the node's paths and the analysis may read
    the mirror), 'failed' when it did not pass, else 'not run' (the group is not complete, or a cell ran again)."""
    d = cells_root / "_groups" / group
    records = sorted(d.glob("parity.*.json")) if d.is_dir() else []
    if not records:
        return "not run"
    last = loads_strict(records[-1].read_text())
    if last["rc"] != 0:
        return "failed"
    covered = {Path(c).name: sha for c, sha in last["cells"]}
    return "passed" if all(covered.get(c) == sha for c, sha in runs.items()) else "not run"


def build(inputs: Inputs, cells_root: Path, treatments: Sequence[str]) -> dict:
    cov: Coverage = check(inputs.scope, cells_root)
    expected = {e.cell_id: e for e in expected_cells(inputs.scope)}
    cells, points, left_out = {}, [], {}
    samples: dict[SampleKey, RequestValues] = {}
    runs: dict[str, dict[str, str]] = {}
    for cid in cov.ok:
        path = cell_path(cell_dir(cells_root, cid), cid)
        runs.setdefault(expected[cid].group, {})[cell_dir(cells_root, cid).name] = file_sha256(path)   # default-ok: grouping
    parity = {g: parity_state(cells_root, g, r) for g, r in sorted(runs.items())}
    for cid in cov.ok:
        exp = expected[cid]
        if exp.group in cov.fleet_problems:
            left_out[cid] = cov.fleet_problems[exp.group]
            continue
        gate = gate_problem(cells_root, cid)
        if gate is not None:
            left_out[cid] = gate
            continue
        if parity[exp.group] == "failed":
            left_out[cid] = f"parity failed for group {exp.group} (_groups/{exp.group})"
            continue
        rec = read_cell(cell_path(cell_dir(cells_root, cid), cid))
        s = inputs.summarize(exp, rec)
        cells[cid] = {"group": exp.group, "config_id": exp.config_id, "trace_id": exp.trace_id, "seed": exp.seed,
                      "summary": asdict(s), "model_mix": model_mix(rec)}
        if s.valid:
            points.append(cell_point(exp, s, cov))
            v = inputs.request_values(exp, rec)
            check_values(cid, s, v)
            samples[(exp.group, exp.trace_id, exp.config_id, exp.seed)] = v
        else:
            left_out[cid] = "; ".join(s.problems)
    configs = config_points(points)
    fronts = {m: {"|".join(k): [p.config_id for p in v] for k, v in frontiers(configs, m).items()} for m in METRICS}
    readings: dict[str, dict[str, list[dict]]] = {k: {} for key, _, _ in READING_OUT.values()
                                                   for k in (key, f"{key}_appendix")}
    not_compared = {}
    for t in treatments:
        these, not_compared[t] = comparable(points, t, inputs.scope.groups)
        for reading, (key, _, _) in READING_OUT.items():
            rows = [asdict(r) for r in equal_quality(these, t, inputs.rules.equal_quality, samples, reading)]
            readings[key][t] = [r for r in rows if r["system"] not in APPENDIX_SYSTEMS]
            readings[f"{key}_appendix"][t] = [r for r in rows if r["system"] in APPENDIX_SYSTEMS]
    rates = stationary_rates(inputs.scope)
    read_from: dict[str, list[float]] = {}
    for p in configs:
        if p.trace_id in rates:
            read_from.setdefault(f"{p.group}|{p.config_id}", []).append(rates[p.trace_id])  # default-ok: grouping
    good = {f"{lvl}@{att}": {f"{g}|{c}": {**asdict(v), "rates": sorted(read_from[f"{g}|{c}"])}
                             for (g, c), v in goodputs(configs, rates, att, lvl).items()}
            for lvl in SLO_LEVELS for att in inputs.rules.slo.attainment_levels}
    return {"format": "rb-report/1", "coverage": {"expected": cov.expected, "ok": len(cov.ok),
                                                   "missing": list(cov.missing), "invalid": dict(cov.invalid),
                                                   "fleet_problems": dict(cov.fleet_problems),
                                                   "pending": dict(cov.pending), "unexpected": list(cov.unexpected)},
            "left_out": left_out, "parity": parity, "cells": cells, "configs": [asdict(p) for p in configs],
            "frontiers": fronts,
            **readings, "not_compared": not_compared, "goodput": good, "rates": rates}


# ---------------------------------------------------------------- LaTeX

def _tex(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _pct(x: Optional[float]) -> str:
    return "--" if x is None else f"{100 * x:.1f}\\%"


def systems_table(configs: Sequence[ConfigPoint]) -> str:
    rows = [r"\begin{tabular}{llllrrrrrrr}", r"\toprule",
            r"System & Knob & Dispatch & Config & Quality & Mean (ms) & P99 (ms) & \$/1k & SLO tight & SLO loose & "
            r"req/s \\", r"\midrule"]
    for p in sorted(configs, key=lambda p: (p.system, p.knob, p.dispatch, p.config_id)):
        rows.append(f"{_tex(p.system)} & {_tex(p.knob)} & {_tex(p.dispatch)} & {_tex(p.config_id)} & "
                    f"{p.quality:.4f} & {p.e2e_mean_ms:.0f} & {p.e2e_p99_ms:.0f} & {p.cost_decode_per_1k:.4f} & "
                    f"{_pct(1 - p.slo_miss_tight)} & {_pct(1 - p.slo_miss_loose)} & {p.throughput_rps:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


HIGHER = r"$^\dagger$"     # read at RouteBalance's frontier low end, above the baseline's quality (D87)


def _change(metric: str, relative: Optional[float], delta: Optional[float],
            span: Optional[Sequence[float]] = None, span_delta: Optional[Sequence[float]] = None,
            higher: bool = False) -> str:
    """A metric's change as the tables show it: relative (%) for latency and cost; for an SLO miss share, the
    change of attainment in percentage points (minus the miss share's delta), with its interval flipped. A value
    read at higher quality than the baseline's (D87) carries HIGHER."""
    mark = HIGHER if higher else ""
    if metric in ATTAINMENT:
        if delta is None:
            return "--"
        tail = "" if span_delta is None else f" [{-100 * span_delta[1]:+.1f}, {-100 * span_delta[0]:+.1f}]"
        return f"{-100 * delta:+.1f}{mark}{tail}"
    if relative is None:
        return "--"
    tail = "" if span is None else f" [{100 * span[0]:+.1f}, {100 * span[1]:+.1f}]"
    return f"{100 * relative:+.1f}\\%{mark}{tail}"


def _table(header: str, lines: Sequence[str], n_metrics: int, n_lead: int) -> str:
    out = [r"\begin{tabular}{l" + "r" * (n_lead - 1 + n_metrics) + "}", r"\toprule", header + r" \\", r"\midrule",
           *[ln + r" \\" for ln in lines], r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def _any_higher(d: dict, interval: str, shown: Optional[Sequence[float]]) -> bool:
    """A cell is marked when a value it shows was read at higher quality (D87): the value at the means, or, when
    the cell prints its range or interval (`shown`), a seed inside the range or a resample inside the interval."""
    if shown is None:
        within = False
    elif interval == "range":
        within = any(v["at_higher_quality"] for v in d["per_seed"])
    else:
        within = d["higher_resamples"] > 0
    return any((d["at_higher_quality"], within))


def equal_table(rows: Sequence[dict], interval: str = "range") -> str:
    """One line per baseline configuration: each metric's change at equal quality, with its range over the
    seeds (interval "range", the main text) or its request-bootstrap interval ("ci", the appendix)."""
    rel_key, abs_key = {"range": ("range_relative", "range"), "ci": ("ci_relative", "ci")}[interval]
    names = list(TABLE_METRICS)
    lines = []
    for r in sorted(rows, key=lambda r: (r["system"], r["config_id"])):
        by = {d["metric"]: d for d in r["deltas"]}
        cells = [_change(m, by[m]["relative"], by[m]["delta"], by[m][rel_key], by[m][abs_key],
                         _any_higher(by[m], interval, by[m][abs_key] if m in ATTAINMENT else by[m][rel_key]))
                 if m in by else "--" for m in names]
        lines.append(f"{_tex(r['config_id'])} & {r['quality']:.4f} & {len(r['seeds'])} & " + " & ".join(cells))
    return _table("Baseline & Quality & Seeds & " + " & ".join(TABLE_METRICS[m] for m in names), lines, len(names), 3)


def seeds_table(rows: Sequence[dict]) -> str:
    """One line per baseline configuration and seed: each metric's change on that seed alone."""
    names = list(TABLE_METRICS)
    lines = []
    for r in sorted(rows, key=lambda r: (r["system"], r["config_id"])):
        by = {d["metric"]: {v["seed"]: v for v in d["per_seed"]} for d in r["deltas"]}
        for seed in r["seeds"]:
            cells = [_change(m, by[m][seed]["relative"], by[m][seed]["delta"], higher=by[m][seed]["at_higher_quality"])
                     if m in by else "--" for m in names]
            lines.append(f"{_tex(r['config_id'])} & {seed} & " + " & ".join(cells))
    return _table("Baseline & Seed & " + " & ".join(TABLE_METRICS[m] for m in names), lines, len(names), 2)


def macro_name(*parts: str, prefix: str = "eq") -> str:
    """A LaTeX command name from free parts: letters kept, digits spelled, every part capitalised."""
    out = []
    for part in parts:
        word = "".join(_DIGITS[ch] if ch in _DIGITS else ch for ch in part if ch.isalnum())
        out.append(word[:1].upper() + word[1:])
    return prefix + "".join(out)


def macros(report: dict) -> str:
    lines = ["% Written by analysis/report.py. Ranges over the baseline configurations with an equal-quality point;",
             "% eq* read one RouteBalance configuration for every metric, eqpm* each metric on its own frontier (D86).",
             "% ...Min/...Max: at equal quality only; ...MinAll/...MaxAll: also the comparisons read at RouteBalance's",
             "% frontier low end, at higher quality than the baseline's (D87; the tables mark them)."]
    for key, _, prefix in READING_OUT.values():
        lines += _macros(report, key, prefix)
    return "\n".join(lines) + "\n"


def _macros(report: dict, key: str, prefix: str) -> list[str]:
    lines = []
    for t, rows in report[key].items():
        by: dict[tuple[str, str, str], list[tuple[float, bool]]] = {}
        for r in rows:
            if r["trace_id"] not in report["rates"]:
                continue
            for d in r["deltas"]:
                if d["metric"] in MACRO_METRICS and d["relative"] is not None:
                    by.setdefault((r["group"], r["trace_id"], d["metric"]), []).append(  # default-ok: grouping
                        (d["relative"], d["at_higher_quality"]))     # a macro shows the value at the means only
        for (group, trace, metric), vals in sorted(by.items()):
            name = macro_name(t, group, trace, MACRO_METRICS[metric], prefix=prefix)
            strict = [v for v, higher in vals if not higher]
            for suffix, these in (("", strict), ("All", [v for v, _ in vals])):
                if these:
                    lines.append(f"\\newcommand{{\\{name}Min{suffix}}}{{{100 * min(these):.1f}\\%}}")
                    lines.append(f"\\newcommand{{\\{name}Max{suffix}}}{{{100 * max(these):.1f}\\%}}")
    return lines


def write(report: dict, out: Path) -> None:
    if out.exists():
        raise ConfigError(f"{out} exists; write a new directory")
    (out / "tables").mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=list) + "\n")
    configs = [ConfigPoint(**{**c, "seeds": tuple(c["seeds"])}) for c in report["configs"]]
    for key in sorted({(c.group, c.trace_id) for c in configs}):
        subset = [c for c in configs if (c.group, c.trace_id) == key]
        (out / "tables" / f"systems.{key[0]}.{key[1]}.tex").write_text(systems_table(subset))
    for key, table, _ in READING_OUT.values():
        for t, rows in report[key].items():
            for g, trace in sorted({(r["group"], r["trace_id"]) for r in rows}):
                subset = [r for r in rows if (r["group"], r["trace_id"]) == (g, trace)]
                (out / "tables" / f"{table}.{t}.{g}.{trace}.tex").write_text(equal_table(subset))
                (out / "tables" / f"{table}_ci.{t}.{g}.{trace}.tex").write_text(equal_table(subset, "ci"))
                (out / "tables" / f"{table}_seeds.{t}.{g}.{trace}.tex").write_text(seeds_table(subset))
        for t, rows in report[f"{key}_appendix"].items():
            for g, trace in sorted({(r["group"], r["trace_id"]) for r in rows}):
                subset = [r for r in rows if (r["group"], r["trace_id"]) == (g, trace)]
                (out / "tables" / f"{table}_appendix.{t}.{g}.{trace}.tex").write_text(equal_table(subset, "ci"))
    (out / "macros.tex").write_text(macros(report))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--fleet", nargs="+", required=True, help="NAME=FLEET.json for each fleet whose cells enter")
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--slo-base", nargs="+", required=True, help="FAMILY=BASE.json (tools/slo_base.py)")
    ap.add_argument("--store", type=Path, nargs="*", default=[], help="served-text score stores (budget cells)")
    ap.add_argument("--rules", type=Path, required=True)
    ap.add_argument("--treatment", nargs="+", required=True, help="the system(s) compared at equal quality")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    inputs = Inputs(a.render, a.data_root, _pairs(a.fleet, "fleet"), a.hosts, _pairs(a.slo_base, "slo-base"),
                    a.store, load_rules(a.rules))
    known = {e.config.system for e in expected_cells(inputs.scope)}
    unknown = sorted(set(a.treatment) - known)
    if unknown:
        print(f"--treatment {unknown}: no configuration of the render has that system ({sorted(known)})", file=sys.stderr)
        return 2
    report = build(inputs, a.cells, a.treatment)
    write(report, a.out)
    cov = report["coverage"]
    print(f"{a.out}: {cov['ok']}/{cov['expected']} cells ok, {len(report['configs'])} configurations, "
          f"{len(report['left_out'])} cells left out of the comparisons")
    for cid, why in sorted(report["left_out"].items()):
        print(f"  LEFT OUT {cid}: {why}")
    for g, state in report["parity"].items():
        if state != "passed":
            print(f"  PARITY {g}: {state}")
    for t, skipped in report["not_compared"].items():
        for key, why in skipped.items():
            print(f"  NOT COMPARED with {t}: {key}: {why}")
    for reading, (key, _, _) in READING_OUT.items():
        for t, rows in report[key].items():
            for r in rows + report[f"{key}_appendix"][t]:
                for d in r["deltas"]:
                    if d["outside_range"]:
                        print(f"  NOTE ({reading}) {t} vs {r['group']}|{r['trace_id']}|{r['config_id']} {d['metric']}: "
                              "the value at the seed means lies outside the range over the seeds (D81)")
    complete = (cov["ok"] == cov["expected"] and not cov["invalid"] and not cov["pending"] and not cov["unexpected"]
                and not report["left_out"] and all(s == "passed" for s in report["parity"].values()))
    return 0 if complete else 1         # the report is written either way; 1: something is missing or left out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
