"""The paper's numbers, from the lease's cells (plan Part B; paper rule R12: a number reaches the paper only
through this tool's output, which the paper \\input{}s).

    python -m analysis.report --render DIR --cells ROOT --data-root DIR --fleet NAME=FLEET.json [...]
        --hosts HOSTS.json --slo-base FAMILY=BASE.json [...] [--store STORE.jsonl ...]
        --rules configs/prereg_rules.json --treatment SYSTEM [...] --out DIR

Inputs, each checked:
  the render   plan.json (group -> fleet -> family; the scored corpora with their sha256) and scope.json (the
               expected cells); a fleet file must have the plan's fleet_sha256 for its name;
  the cells    analysis/manifest.py check: only ok cells enter; a group whose cells ran on two fleets is left out
               whole (its fleet problem is reported);
  per family   the SLO bases (tools/slo_base.py) and the scored corpus under --data-root, pinned by the plan;
  the stores   the served-text score stores (tools/judge_served.py), for the budget cells.
Every ok cell is summarised against its group's fleet (analysis/metrics.py summarize_cell) with the preregistered
SLO multipliers on its family's bases. A cell whose summary has problems is reported and kept out of every
comparison, never averaged in.

Output, a new directory:
  report.json  per cell: the summary, the model mix and the overhead stages; per configuration: the seed means
               (analysis/frontier.py); the frontiers per metric; for each --treatment system, the equal-quality rows
               (analysis/equal_quality.py) and every baseline configuration it could not compare, with the
               reason (`not_compared`: an ablation group, whose variants are read against its reference in its
               systems table; no treatment configuration at that group and trace; or no treatment preset with
               cells for every seed of the baseline); goodput at the rules' attainment levels of the tight and
               loose SLO per configuration over the stationary traces, with the rates it was read from
               (analysis/slo.py)
  tables/systems.<group>.<trace>.tex         every configuration of the group at the trace
  tables/equal.<treatment>.<group>.<trace>.tex   every compared baseline configuration: the relative change and
               its seed-bootstrap CI (-- where there is no equal-quality point)
  macros.tex   per treatment, group and stationary trace: the range over the baseline configurations of the
               relative change of mean E2E, P99 E2E and cost at equal quality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

from analysis.equal_quality import equal_quality
from analysis.frontier import METRICS, CellPoint, ConfigPoint, by_config, cell_point, config_points, frontiers
from analysis.manifest import Coverage, check
from analysis.metrics import CellSummary, summarize_cell
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            if _sha256(fleets[name]) != entry["fleet_sha256"]:
                raise ConfigError(f"{fleets[name]} is not the fleet {name} the plan pins")
            self.fleets[name] = load_fleet(fleets[name], hosts)
        self.corpora: dict[str, ScoredCorpus] = {}
        for family, entry in self.plan["scored_corpora"].items():
            path = Path(data_root) / entry["path"]
            if not path.is_file():
                continue
            if _sha256(path) != entry["sha256"]:
                raise ConfigError(f"{path} is not the scored corpus the plan pins for {family}")
            self.corpora[family] = ScoredCorpus.load([path], judge_key)
        self.store = ScoreStore.load(list(stores), judge_key) if stores else None
        self.slo_bases = {family: SloBase.load(p) for family, p in slo_bases.items()}

    def group_fleet(self, group: str) -> tuple[str, str]:
        name = self.plan["groups"][group]["fleet"]
        return name, self.plan["fleets"][name]["family"]

    def summarize(self, exp: ExpectedCell, cell: CellRecord) -> CellSummary:
        name, family = self.group_fleet(exp.group)
        for have, what, key in ((self.fleets, "--fleet", name), (self.slo_bases, "--slo-base", family)):
            if key not in have:
                raise ConfigError(f"cell {exp.cell_id} (group {exp.group}) needs {what} {key}=...")
        fleet = self.fleets[name]
        slo = CellSlo.of(self.rules.slo.k(), self.slo_bases[family], fleet)
        if cell.budgets_sha256 is not None:
            if self.store is None:
                raise ConfigError(f"budget cell {exp.cell_id} needs the served-text score stores (--store)")
            quality = self.store
        else:
            if family not in self.corpora:
                raise ConfigError(f"cell {exp.cell_id} needs the {family} scored corpus under --data-root")
            quality = self.corpora[family]
        return summarize_cell(cell, quality, fleet, slo)


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


def build(inputs: Inputs, cells_root: Path, treatments: Sequence[str]) -> dict:
    cov: Coverage = check(inputs.scope, cells_root)
    expected = {e.cell_id: e for e in expected_cells(inputs.scope)}
    cells, points, left_out = {}, [], {}
    for cid in cov.ok:
        exp = expected[cid]
        if exp.group in cov.fleet_problems:
            left_out[cid] = cov.fleet_problems[exp.group]
            continue
        rec = read_cell(cell_path(cell_dir(cells_root, cid), cid))
        s = inputs.summarize(exp, rec)
        cells[cid] = {"group": exp.group, "config_id": exp.config_id, "trace_id": exp.trace_id, "seed": exp.seed,
                      "summary": asdict(s), "model_mix": model_mix(rec)}
        if s.valid:
            points.append(cell_point(exp, s, cov))
        else:
            left_out[cid] = "; ".join(s.problems)
    configs = config_points(points)
    fronts = {m: {"|".join(k): [p.config_id for p in v] for k, v in frontiers(configs, m).items()} for m in METRICS}
    eq, not_compared = {}, {}
    for t in treatments:
        these, not_compared[t] = comparable(points, t, inputs.scope.groups)
        eq[t] = [asdict(r) for r in equal_quality(these, t, inputs.rules.equal_quality)]
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
                                                   "fleet_problems": dict(cov.fleet_problems)},
            "left_out": left_out, "cells": cells, "configs": [asdict(p) for p in configs], "frontiers": fronts,
            "equal_quality": eq, "not_compared": not_compared, "goodput": good, "rates": rates}


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


def _delta(d: dict) -> str:
    if d["relative"] is None:
        return "--"
    ci = d["ci_relative"]
    tail = "" if ci is None else f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
    return f"{100 * d['relative']:+.1f}\\%{tail}"


def equal_table(rows: Sequence[dict]) -> str:
    names = [m for m in MACRO_METRICS]
    out = [r"\begin{tabular}{lrr" + "r" * len(names) + "}", r"\toprule",
           "Baseline & Quality & Seeds & " + " & ".join(MACRO_METRICS[m] for m in names) + r" \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: (r["system"], r["config_id"])):
        by = {d["metric"]: d for d in r["deltas"]}
        out.append(f"{_tex(r['config_id'])} & {r['quality']:.4f} & {len(r['seeds'])} & " +
                   " & ".join(_delta(by[m]) if m in by else "--" for m in names) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def macro_name(*parts: str) -> str:
    """A LaTeX command name from free parts: letters kept, digits spelled, every part capitalised."""
    out = []
    for part in parts:
        word = "".join(_DIGITS[ch] if ch in _DIGITS else ch for ch in part if ch.isalnum())
        out.append(word[:1].upper() + word[1:])
    return "eq" + "".join(out)


def macros(report: dict) -> str:
    lines = ["% Written by analysis/report.py. Ranges over the baseline configurations with an equal-quality point."]
    for t, rows in report["equal_quality"].items():
        by: dict[tuple[str, str, str], list[float]] = {}
        for r in rows:
            if r["trace_id"] not in report["rates"]:
                continue
            for d in r["deltas"]:
                if d["metric"] in MACRO_METRICS and d["relative"] is not None:
                    by.setdefault((r["group"], r["trace_id"], d["metric"]), []).append(d["relative"])  # default-ok: grouping
        for (group, trace, metric), vals in sorted(by.items()):
            name = macro_name(t, group, trace, MACRO_METRICS[metric])
            lo, hi = min(vals), max(vals)
            lines.append(f"\\newcommand{{\\{name}Min}}{{{100 * lo:.1f}\\%}}")
            lines.append(f"\\newcommand{{\\{name}Max}}{{{100 * hi:.1f}\\%}}")
    return "\n".join(lines) + "\n"


def write(report: dict, out: Path) -> None:
    if out.exists():
        raise ConfigError(f"{out} exists; write a new directory")
    (out / "tables").mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=list) + "\n")
    configs = [ConfigPoint(**{**c, "seeds": tuple(c["seeds"])}) for c in report["configs"]]
    for key in sorted({(c.group, c.trace_id) for c in configs}):
        subset = [c for c in configs if (c.group, c.trace_id) == key]
        (out / "tables" / f"systems.{key[0]}.{key[1]}.tex").write_text(systems_table(subset))
    for t, rows in report["equal_quality"].items():
        for key in sorted({(r["group"], r["trace_id"]) for r in rows}):
            subset = [r for r in rows if (r["group"], r["trace_id"]) == key]
            (out / "tables" / f"equal.{t}.{key[0]}.{key[1]}.tex").write_text(equal_table(subset))
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
    report = build(inputs, a.cells, a.treatment)
    write(report, a.out)
    cov = report["coverage"]
    print(f"{a.out}: {cov['ok']}/{cov['expected']} cells ok, {len(report['configs'])} configurations, "
          f"{len(report['left_out'])} cells left out of the comparisons")
    for cid, why in sorted(report["left_out"].items()):
        print(f"  LEFT OUT {cid}: {why}")
    for t, skipped in report["not_compared"].items():
        for key, why in skipped.items():
            print(f"  NOT COMPARED with {t}: {key}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
