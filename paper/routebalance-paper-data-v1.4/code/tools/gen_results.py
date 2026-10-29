"""The paper's result numbers and tables, from the report tool's output: nothing here is typed by hand.

    PYTHONPATH=<route-balance repo> python tools/gen_results.py --data data --out . [--interim]

Inputs (paper/data/, each listed with its sha256 in data/SHA256SUMS, checked before use):
  qwen_report.json    analysis/report.py over the first-seed Qwen runs (render 36)
  gemma_report.json   analysis/report.py over Gemma's first-seed runs at OpenRouter's prices (render 36)
  fleet_qwen.json     the Qwen fleet (tiers, prices, instances)
  tpot_qwen.json      the pinned TPOT table (the constant RouteBalance uses per tier)
  prefill_qwen.json   the pinned prefill line per tier
  round_shares.json   tools/round_share.py over the decision logs of RouteBalance 80/10/10, seed 1, at 12, 30, 42 req/s
  spread_qwen.json    tools/seed_spread.py over a report of Qwen's repeated configurations (seeds 1-3)
  spread_gemma.json   tools/seed_spread.py over a report of Gemma's runs at seeds 1-3
  vsr_failure.json    tools/vsr_failure.py over the vLLM Semantic Router runs' own records (Appendix K's analysis)

Output:
  generated/results.tex     \\resdef{key}{value} for every number the text cites (main.tex's \\res{key} reads them;
                            an unknown key stops the build)
  generated/plot_points.json  the points the figure scripts draw, with each point's 3-D-frontier flag
  tables/gen_*.tex          the generated tables

Definitions (the paper's):
  RouteBalance is shown at its eight presets; a baseline setting is one router setting on one dispatcher, or
  one dispatcher alone, at one rate. A point beats another when it is at least as good on quality, mean E2E latency and
  cost per 1,000 requests, and better on one. The equal-quality reading is per metric
  (v2/v3): each metric on RouteBalance's own quality-metric frontier over the eight presets, read at the
  baseline's quality with analysis.equal_quality.position_at and the report's tolerance; never extrapolated above the
  frontier; below it, the lowest preset (marked as compared at higher quality).
Every value is from one run (seed 1): each configuration's `seeds` must be [1]. The spread across seeds 1-3
comes from the spread files alone, whose seed-1 values must equal the first-seed reports'.
"""
from __future__ import annotations

import argparse
import hashlib
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
import json
import math
import re
import statistics
from pathlib import Path

from analysis.equal_quality import position_at
from analysis.frontier import pareto

EIGHT = ("uniform", "q50l30c20", "q60l20c20", "q80l10c10", "q90l05c05", "q05l25c70", "q22l18c60", "q25l05c70")
RATES = ("l12", "l18", "l24", "l30", "l36", "l42")
THREE = ("l12", "l30", "l42")
ROUTER_DISPATCH = ("lor", "rr", "po2", "lowest-latency")
DISPATCHERS = ("lor", "rr", "po2", "random", "lowest-latency")
AP = ("pw025", "pw039", "pw053", "pw070", "pw080")
BR = ("t00", "t03", "t05", "t07")
ABL_REF = "abl.qwen.rb-static.q80l10c10"
ABLATIONS = (                # dead reckoning off and the LPT key variants are not in the paper
    ("lpt-off", "abl.qwen.rb-static.abl-lpt-off.q80l10c10"),
    ("round32", "abl.qwen.rb-static.abl-round32.q80l10c10"),
    ("wlat0", "abl.qwen.rb-static.abl-wlat0.q80l10c10"),
    ("queueblind", "abl.qwen.abl-static-prior.q80l10c10"),
    ("learned", "abl.qwen.rb-learned-retrained.q80l10c10"),
    ("promptblind", "abl.qwen.rb-static.abl-prompt-blind.q80l10c10"),
    ("labels75", "abl.qwen.rb-static.abl-labels75.q80l10c10"),
    ("labels50", "abl.qwen.rb-static.abl-half-labels.q80l10c10"),
    ("labels25", "abl.qwen.rb-static.abl-labels25.q80l10c10"),
    ("lodogsm", "abl.qwen.rb-static.abl-lodo-gsm8k.q80l10c10"),
    ("lodolmsys", "abl.qwen.rb-static.abl-lodo-lmsys.q80l10c10"),
)
BURSTS = (("ga025", "0.25"), ("ga05", "0.5"), ("ga2", "2"))
SINGLE = "qwen.rb-static.abl-round1."            # RB-single, RouteBalance with one request per round: + preset
GEMMA_SINGLE = "gemma.rb-static.abl-round1."      # its Gemma runs
GEMMA_W = ("q80l10c10", "q50l30c20", "uniform")
BURST_SYSTEMS = (("rb", "qwen.rb-static.q80l10c10"), ("dec", "qwen.rb-static-dec.q80l10c10"),
                 ("ap", "qwen.ap.pw080.lor"), ("single", SINGLE + "q80l10c10"))
BUDGET_MIXES = ("b75", "b45", "b30")
BUDGET_SYSTEMS = (("filter", "qwen.rb-static-budget.uniform"), ("rb", "qwen.rb-static.uniform"),
                  ("br", "qwen.br.t00.lor"))
GEMMA_SETTINGS = (
    ("rb.q80l10c10", "gemma.rb-static.q80l10c10"), ("rb.q50l30c20", "gemma.rb-static.q50l30c20"),
    ("rb.uniform", "gemma.rb-static.uniform"),
    ("dec.q80l10c10", "gemma.rb-static-dec.q80l10c10"), ("dec.q50l30c20", "gemma.rb-static-dec.q50l30c20"),
    ("dec.uniform", "gemma.rb-static-dec.uniform"),
    ("ap.pw080", "gemma.ap.pw080.lor"), ("ap.pw053", "gemma.ap.pw053.lor"),
    ("br.t05", "gemma.br.t05.lor"), ("br.t03", "gemma.br.t03.lor"), ("br.t00", "gemma.br.t00.lor"),
    ("lor", "gemma.disp.lor"),
    ("single.q80l10c10", GEMMA_SINGLE + "q80l10c10"), ("single.q50l30c20", GEMMA_SINGLE + "q50l30c20"),
    ("single.uniform", GEMMA_SINGLE + "uniform"),
)
GEMMA_RATES = ("gl12", "gl30", "gl42")
VSR_RATES = ("l6", "l9")
VSR_STOP_RATES = ("l9", "l12", "l18", "l24", "l30")     # the rates at which Appendix K says its router stopped
VSR_SYSTEMS = (("rb", "qwen.rb-static.q80l10c10"), ("rr", "qwen.disp.rr"), ("vsr", "qwen.vsr.mom-balance"))
STAGES = (("wait", "round_wait_ms"), ("est", "estimator_ms"), ("read", "stats_fetch_ms"), ("sched", "schedule_ms"))
TIER_ORDER = ("qwen2.5-72b_a100", "qwen2.5-14b_v100", "qwen2.5-7b_a30", "qwen2.5-3b_a30")
GPU = {"a100": r"A100$\times$4", "v100": r"V100$\times$4", "a30": "A30"}
INPUTS = ("qwen_report.json", "gemma_report.json", "vsr_summary.json", "fleet_qwen.json", "tpot_qwen.json",
          "prefill_qwen.json", "prereg_rules.json", "parity.json", "corpus_stats.json",
          "fleet_gemma.json", "round_shares.json", "spread_qwen.json", "spread_gemma.json", "budget_judge.json",
          "vsr_failure.json", "round_range.json")
# The parity gate over the first-seed runs (data/parity.json): every group must pass, the main group whole (the two
# round-robin runs at 6 and 9 req/s, set aside in version 1.0 for a thread count their manifests lacked, were rerun).
PARITY_REQUIRED = {"qwen / all": ("qwen_report.json", "qwen"),
                   "qwen-ablation / all": ("qwen_report.json", "qwen-ablation"),
                   "qwen-budget-30 / all": ("qwen_report.json", "qwen-budget-30"),
                   "qwen-budget-45 / all": ("qwen_report.json", "qwen-budget-45"),
                   "qwen-budget-75 / all": ("qwen_report.json", "qwen-budget-75"),
                   "gemma / all": ("gemma_report.json", "gemma")}


class Missing(KeyError):
    pass


def verified(data: Path) -> dict[str, dict]:
    sums = {}
    for line in (data / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    out = {}
    for name in INPUTS:
        raw = (data / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sums.get(name):
            raise SystemExit(f"{name}: sha256 differs from data/SHA256SUMS")
        out[name] = json.loads(raw)
        out[name + ".sha256"] = sums[name]
    return out


class Points:
    """(trace, config) -> (quality, mean E2E s, p99 E2E s, cost per 1k requests) from one report's cells, seed 1."""

    def __init__(self, report: dict, groups: tuple[str, ...]):
        self.p: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        self.cell: dict[tuple[str, str], dict] = {}
        for c in report["cells"].values():
            if c["group"] not in groups:
                continue
            if c["seed"] != 1:
                raise SystemExit(f"{c['config_id']} at {c['trace_id']}: seed {c['seed']}, version 1.0 is seed 1 only")
            s = c["summary"]
            key = (c["trace_id"], c["config_id"])
            if key in self.p:
                raise SystemExit(f"{key}: two cells")
            self.cell[key] = c
            if s.get("quality") is None:
                continue
            self.p[key] = (s["quality"], s["e2e_ms"]["mean"] / 1000.0, s["e2e_ms"]["p99"] / 1000.0,
                           s["cost_total_per_1k"])
        for cfg in report["configs"]:
            if cfg["group"] in groups and cfg["seeds"] != [1]:
                raise SystemExit(f"{cfg['config_id']} at {cfg['trace_id']}: seeds {cfg['seeds']}")

    def get(self, trace: str, config: str):
        if (trace, config) not in self.p:
            raise Missing(f"no run of {config} at {trace}")
        return self.p[(trace, config)]


def beats(a, b) -> bool:
    """a at least as good as b on quality (up), mean E2E and cost (down), and better on one."""
    (qa, ma, _, ca), (qb, mb, _, cb) = a, b
    return qa >= qb and ma <= mb and ca <= cb and (qa > qb or ma < mb or ca < cb)


def main_baselines() -> list[str]:
    out = [f"qwen.ap.{pw}.{d}" for pw in AP for d in ROUTER_DISPATCH]
    out += [f"qwen.br.{t}.{d}" for t in BR for d in ROUTER_DISPATCH]
    out += [f"qwen.disp.{d}" for d in DISPATCHERS]
    return out


def shipped_baselines() -> list[str]:
    return [f"qwen.ap.{pw}.lor-shipped" for pw in AP] + [f"qwen.br.{t}.lor-shipped" for t in BR]


def per_metric(rb: list[tuple[float, float]], quality: float, tolerance: float):
    """(value, at higher quality) of one metric on RouteBalance's quality-metric frontier at `quality`, or None."""
    order = pareto(rb)
    qs = [rb[i][0] for i in order]
    pos = position_at(quality, qs, tolerance)
    if pos is None:
        return None
    i, j, w, higher = pos
    lo, hi = rb[order[i]][1], rb[order[j]][1]
    return lo + (hi - lo) * w, higher


def pct(x: float, digits: int = 0) -> str:
    """A signed percentage; a value that rounds to zero is written 0."""
    v = round(100 * x, digits)
    if v == 0:
        return "0"
    text = f"{abs(v):.{digits}f}" if digits else f"{int(abs(v)):,}".replace(",", "{,}")   # 1{,}226, as the tables
    return ("+" if v > 0 else "$-$") + text


def dq3(x: float) -> str:
    """A signed quality difference to 3 decimals; one that rounds to zero is written 0.000."""
    s = f"{x:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s.replace("-", "$-$")


def thousandths(s: str) -> str:
    """dq3's string in thousandths, as Figure 3 prints it: "+0.005" -> "+5", "0.000" -> "0", "$-$0.012" -> "$-$12"."""
    m = re.fullmatch(r"(\+|\$-\$)?0\.(\d{3})", s)
    if not m:
        raise SystemExit(f"thousandths: unexpected quality string {s!r}")
    n = int(m.group(2))
    return "0" if n == 0 else (m.group(1) or "+") + str(n)


def share(x: float, digits: int = 0) -> str:
    """A share in [0, 1] as a percentage."""
    if not 0 <= x <= 1:
        raise ValueError(f"{x} is not a share")
    v = round(100 * x, digits)
    return f"{v:.{digits}f}" if digits else f"{int(v)}"


def red(x: float, digits: int = 0) -> str:
    """A reduction (x < 0) as its magnitude in percent, for "X% lower" in the text. An increase or no change is written
    as \\resnotlower, which stops the build if the text cites it as a reduction."""
    if not x < 0:
        return r"\resnotlower"
    v = round(-100 * x, digits)
    return f"{v:.{digits}f}" if digits else f"{int(v)}"


def inc(x: float, digits: int = 0) -> str:
    """An increase (x > 0) as its magnitude in percent, for "X% slower"; otherwise \\resnothigher, which stops the
    build if the text cites it."""
    if not x > 0:
        return r"\resnothigher"
    v = round(100 * x, digits)
    return f"{v:,.{digits}f}".replace(",", "{,}") if digits else f"{int(v):,}".replace(",", "{,}")


def ms(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{v:,.0f}".replace(",", "{,}")


def gap(x: float, digits: int = 1) -> str:
    """|x| in percent where the text states the direction itself (e.g. "within X%")."""
    v = abs(round(100 * x, digits))
    return f"{v:.{digits}f}" if digits else f"{int(v)}"


def up(x: float, digits: int) -> float:
    """x rounded up at `digits` decimals, for a value the text states as a bound ("at most", "within")."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_CEILING))  # the shortest
    # decimal of x, ceiled: exact values stay (0.006 -> 0.006), any excess rounds up (0.0060000000001 -> 0.007)


def down(x: float, digits: int) -> float:
    """x rounded down at `digits` decimals, for a value the text states as a lower bound ("or more", "at least")."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_FLOOR))


def gap_up(x: float, digits: int = 1) -> str:
    """|x| in percent, rounded up: for a bound the text states with its direction ("within X%", "by at most X%")."""
    v = up(abs(100 * x), digits)
    return f"{v:.{digits}f}" if digits else f"{int(v)}"


def inc_up(x: float, digits: int = 0) -> str:
    """An increase in percent, rounded up, for "at most X% slower"; otherwise \\resnothigher, as inc()."""
    if not x > 0:
        return r"\resnothigher"
    v = up(100 * x, digits)
    return f"{v:,.{digits}f}".replace(",", "{,}") if digits else f"{int(v):,}".replace(",", "{,}")


def red_up(x: float, digits: int = 0) -> str:
    """red() rounded up, for "by up to X%": the largest reduction, so the bound holds."""
    if not x < 0:
        return r"\resnotlower"
    v = up(-100 * x, digits)
    return f"{v:.{digits}f}" if digits else f"{int(v)}"


def inc_down(x: float, digits: int = 0) -> str:
    """inc() rounded down, for the low end of a range the text states as holding at every rate and seed."""
    if not x > 0:
        return r"\resnothigher"
    v = down(100 * x, digits)
    return f"{v:,.{digits}f}".replace(",", "{,}") if digits else f"{int(v):,}".replace(",", "{,}")


def red_down(x: float, digits: int = 0) -> str:
    """red() rounded down, for the low end of a range the text states as holding at every seed: the smallest
    reduction, so the bound holds."""
    if not x < 0:
        return r"\resnotlower"
    v = down(-100 * x, digits)
    return f"{v:.{digits}f}" if digits else f"{int(v)}"


def bound_up(v: float, digits: int) -> str:
    """An upper bound the text states beside a table that rounds v to nearest at `digits`: the table's own value when it
    already holds (it is not below v), otherwise v rounded up with one digit more, so the bound is true and never
    prints below the table."""
    tab = float(f"{v:.{digits}f}")
    return f"{tab:.{digits}f}" if tab >= v else f"{up(v, digits + 1):.{digits + 1}f}"


def bound_down(v: float, digits: int) -> float:
    """bound_up's lower end: the table's value when it already holds (it is not above v), otherwise v rounded down
    with one digit more. Returned as a number, with its digits as the second item."""
    tab = float(f"{v:.{digits}f}")
    return (tab, digits) if tab <= v else (down(v, digits + 1), digits + 1)


def ms_up(v: float) -> str:
    """ms() rounded up, for "at most X ms"."""
    return f"{up(v, 1):.1f}" if up(v, 1) < 10 else f"{int(up(v, 0)):,}".replace(",", "{,}")


def median(xs) -> float:
    return statistics.median(xs)


def rate(t: str) -> str:
    return re.sub(r"^g?l", "", t)


class Out:
    def __init__(self):
        self.res: dict[str, str] = {}
        self.tables: dict[str, str] = {}

    def __setitem__(self, key: str, value: str):
        if not re.fullmatch(r"[a-z0-9.\-]+", key):
            raise ValueError(f"bad key {key!r}")
        if key in self.res:
            raise ValueError(f"{key} written twice")
        self.res[key] = value


def dominance(q: Points, out: Out, plot: dict) -> None:
    base = main_baselines()
    beaten_all = total_all = 0
    ap_all = br_nonzero_all = True
    for t in RATES:
        rb = {w: q.get(t, f"qwen.rb-static.{w}") for w in EIGHT}
        beaten, open_ = 0, []
        for b in base:
            p = q.get(t, b)
            if any(beats(v, p) for v in rb.values()):
                beaten += 1
            else:
                open_.append(b)
                if ".ap." in b:
                    ap_all = False
                if ".br." in b and ".t00." not in b:
                    br_nonzero_all = False
        out[f"dom.{rate(t)}.n"] = str(beaten)
        out[f"dom.{rate(t)}.of"] = str(len(base))
        out[f"dom.{rate(t)}.open"] = str(len(open_))
        plot.setdefault("open", {})[t] = open_
        beaten_all += beaten
        total_all += len(base)
    named = ({(t, f"qwen.br.t00.{d}") for t in ("l12", "l18", "l24") for d in ROUTER_DISPATCH}
             | {("l30", "qwen.br.t00.lowest-latency"), ("l24", "qwen.disp.lowest-latency"), ("l30", "qwen.disp.lowest-latency")})
    found = {(t, b) for t in RATES for b in plot["open"][t]}
    if found != named:      # the text lists them: BEST-Route 0 on every dispatcher at 12-24 and on LL at 30, LL alone
        raise SystemExit(f"the settings no preset beats are not the text's list: {sorted(found ^ named)}")
    closed = [rate(t) for t in RATES if out.res[f"dom.{rate(t)}.open"] == "0"]
    if closed != ["36", "42"]:
        raise SystemExit(f"the text names 36 and 42 req/s as the rates where every baseline setting is beaten: {closed}")
    out["dom.all.n"] = str(beaten_all)
    out["dom.all.of"] = str(total_all)
    out["dom.all.open"] = str(total_all - beaten_all)
    out["dom.settings"] = str(len(base))
    if not ap_all or not br_nonzero_all:
        raise SystemExit("an AvengersPro setting or a BEST-Route threshold above 0 is not beaten at some rate")
    out["dom.ap.allbeaten"] = "yes" if ap_all else "no"
    out["dom.br.nonzero.allbeaten"] = "yes" if br_nonzero_all else "no"
    # the as-shipped routers (appendix)
    sb = so = 0
    for t in THREE:
        rb = [q.get(t, f"qwen.rb-static.{w}") for w in EIGHT]
        for b in shipped_baselines():
            so += 1
            sb += any(beats(v, q.get(t, b)) for v in rb)
    out["dom.shipped.n"] = str(sb)
    out["dom.shipped.of"] = str(so)


OPEN_OUT: dict[str, str] = {}


def open_table(q: Points, plot: dict) -> str:
    """Every main-grid baseline setting no preset beats, with the nearest preset by quality (appendix)."""
    rows = []
    for t in RATES:
        for b in plot["open"][t]:
            bq, bm, _, bc = q.get(t, b)
            best = min(EIGHT, key=lambda w: (abs(q.get(t, f"qwen.rb-static.{w}")[0] - bq), w))
            rq, rm, _, rc = q.get(t, f"qwen.rb-static.{best}")
            slug = f"open.{rate(t)}." + b.replace("qwen.", "").replace(".", "-")
            plot.setdefault("open_rows", []).append((t, b, best))
            for k, v in (("q", f"{bq:.3f}"), ("lat", f"{bm:.2f}"), ("cost", f"{1000 * bc:.1f}"), ("rb", wlabel(best)),
                         ("rb.q", f"{rq:.3f}"), ("rb.lat", f"{rm:.2f}"), ("rb.cost", f"{1000 * rc:.1f}")):
                OPEN_OUT[f"{slug}.{k}"] = v
            rows.append(f"{rate(t)} & {label(b)} & {bq:.4f} & {bm:.2f} & {1000 * bc:.1f} & {wlabel(best)} & "
                        f"{rq:.4f} & {rm:.2f} & {1000 * rc:.1f} \\\\")
    return table(
        "tab:open", r"Baseline settings that no \sys preset beats on quality, latency and cost together, beside the "
        r"\sys preset nearest in quality. Latency is the mean E2E in seconds and cost is USD per million requests.",
        "rlrrrlrrr", r"$\lambda$ & Baseline & Q & E2E & Cost & \sys & Q & E2E & Cost",   # one column per header
        rows, wide=True)


def label(cfg: str) -> str:
    m = re.fullmatch(r"(?:qwen|gemma)\.(ap|br)\.(pw|t)(\d+)\.([a-z0-9\-]+?)(-shipped)?", cfg)
    disp = {"lor": "LOR", "rr": "RR", "po2": "Po2", "random": "random", "lowest-latency": "LL"}
    if m:
        kind, _, v, d, shipped = m.groups()
        val = f"0.{v[1:]}" if kind == "ap" else ("0" if v == "00" else f"0.{v[1]}")
        name = "AvengersPro" if kind == "ap" else "BEST-Route"
        return f"{name} {val} / {disp[d]}" + (" (serial)" if shipped else "")
    m = re.fullmatch(r"(?:qwen|gemma)\.disp\.([a-z0-9\-]+)", cfg)
    if m:
        return disp[m.group(1)] + " alone"
    raise ValueError(cfg)


def wlabel(w: str) -> str:
    if w == "uniform":
        return "uniform"
    m = re.fullmatch(r"q(\d\d)l(\d\d)c(\d\d)", w)
    return "/".join(str(int(x)) for x in m.groups())


def equal_quality(q: Points, out: Out, tolerance: float) -> None:
    lat_all, cost_all = [], []
    higher_all = 0
    for t in RATES:
        presets = [q.get(t, f"qwen.rb-static.{w}") for w in EIGHT]
        lat, cost = [], []
        n = higher = 0
        for b in main_baselines():
            bq, bm, _, bc = q.get(t, b)
            l = per_metric([(p[0], p[1]) for p in presets], bq, tolerance)
            c = per_metric([(p[0], p[3]) for p in presets], bq, tolerance)
            if l is None or c is None:
                continue
            n += 1
            higher += l[1] or c[1]
            lat.append(l[0] / bm - 1)
            cost.append(c[0] / bc - 1)
        out[f"eq.{rate(t)}.n"] = str(n)
        out[f"eq.{rate(t)}.higher"] = str(higher)
        out[f"eq.{rate(t)}.lat.hi"], out[f"eq.{rate(t)}.cost.hi"] = red(min(lat)), red(min(cost))
        out[f"eq.{rate(t)}.lat.median"] = red(median(lat))
        out[f"eq.{rate(t)}.cost.median"] = red(median(cost))
        out[f"eq.{rate(t)}.lat.lower"] = str(sum(x < 0 for x in lat))
        out[f"eq.{rate(t)}.cost.lower"] = str(sum(x < 0 for x in cost))
        lat_all += lat
        cost_all += cost
        higher_all += higher
    out["eq.n"] = str(len(lat_all))
    out["eq.higher"] = str(higher_all)
    out["eq.lat.lower"] = str(sum(x < 0 for x in lat_all))
    out["eq.cost.lower"] = str(sum(x < 0 for x in cost_all))
    out["eq.lat.median"] = red(median(lat_all))
    out["eq.cost.median"] = red(median(cost_all))
    out["eq.lat.best"] = red(min(lat_all))
    out["eq.cost.best"] = red(min(cost_all))


def main_table(q: Points, out: Out) -> str:
    rows = []
    for t in RATES:
        r = rate(t)
        rows.append(f"{r} & {out.res[f'dom.{r}.n']} of {out.res[f'dom.{r}.of']} & "
                    f"{out.res[f'eq.{r}.lat.median']}\\% & {out.res[f'eq.{r}.lat.hi']}\\% & "
                    f"{out.res[f'eq.{r}.cost.median']}\\% & {out.res[f'eq.{r}.cost.hi']}\\% \\\\")
    return table(
        "tab:main", rf"\sys at its eight presets against the {out.res['dom.settings']} baseline settings at each rate: the settings it beats "
        r"on quality, mean E2E latency and cost together, and its median and largest reductions of mean E2E latency "
        r"and cost at equal quality.",
        "rrrrrr", r"$\lambda$ & Beaten & \multicolumn{2}{c}{E2E reduction} & "
        r"\multicolumn{2}{c}{Cost reduction} \\ & & median & largest & median & largest", rows)


VARIANTS = (("RB-decoupled", "qwen.rb-static-dec."), ("RB-single", SINGLE))


def variant_cells(q: Points, prefix: str, w: str, t: str) -> tuple[tuple[float, float, float], tuple[str, str, str]]:
    """RouteBalance against a variant at one preset and rate: the change of mean E2E and of cost and the quality
    difference, and their printed forms (pct, pct, dq3), the same for the appendix table and the figure."""
    a, v = q.get(t, f"qwen.rb-static.{w}"), q.get(t, prefix + w)
    x = (a[1] / v[1] - 1, a[3] / v[3] - 1, a[0] - v[0])
    return x, (pct(x[0]), pct(x[1]), dq3(x[2]))


def decoupled(q: Points, out: Out, plot: dict) -> str:
    """The appendix table of the variants (the main text's Figure 3 draws it): RouteBalance against
    RB-decoupled and RB-single at every preset and rate. The figure's cells (plot["variants"]) are this table's values
    and printed strings; their counts must equal the text's."""
    rows, fig = [], {}
    for name, prefix in VARIANTS:
        fig[name] = {"lat": [], "cost": [], "q": []}
        for w in EIGHT:
            vals = [variant_cells(q, prefix, w, t) for t in RATES]
            for k, key in enumerate(("lat", "cost", "q")):
                fig[name][key].append({"val": [x[k] for x, _ in vals], "txt": [s_[k] for _, s_ in vals]})
    for name, _ in VARIANTS:                     # Figure 3's layout: a short table, not 48 rows
        rows.append(rf"\multicolumn{{{1 + 3 * len(RATES)}}}{{l}}{{\emph{{{name}}}}} \\")
        for i, w in enumerate(EIGHT):
            cells = [s_ if k != "q" else thousandths(s_) for j in range(len(RATES)) for k in ("lat", "cost", "q")
                     for s_ in (fig[name][k][i]["txt"][j],)]
            rows.append(f"{wlabel(w)} & " + " & ".join(cells) + r" \\")
    plot["variants"] = {"presets": [wlabel(w) for w in EIGHT], "rates": [rate(t) for t in RATES], "panels": fig}
    lat, cost, dq = [], [], []
    for w in EIGHT:
        for t in RATES:
            a, d = q.get(t, f"qwen.rb-static.{w}"), q.get(t, f"qwen.rb-static-dec.{w}")
            lat.append(a[1] / d[1] - 1)
            cost.append(a[3] / d[3] - 1)
            dq.append(a[0] - d[0])
    n = len(lat)
    out["dec.n"] = str(n)
    out["dec.lat.lower"] = str(sum(x < 0 for x in lat))
    out["dec.cost.lower"] = str(sum(x < 0 for x in cost))
    out["dec.lat.best"] = red_up(min(lat))                                   # "by up to": rounded up
    out["dec.cost.best"] = red_up(min(cost))
    lo, nd = bound_down(min(dq), 3)                                         # "between A and B": outward, and
    out["dec.q.lowest"] = f"{lo:+.{nd}f}".replace("-", "$-$")               # beside Figure 3
    out["dec.q.maxloss"] = f"{-min(dq):.3f}" if min(dq) < 0 else r"\resnotlower"
    q80 = [q.get(t, "qwen.rb-static.q80l10c10")[0] - q.get(t, "qwen.rb-static-dec.q80l10c10")[0] for t in RATES]
    out["dec.q80.qgap"] = bound_up(max(abs(x) for x in q80), 3)          # "within": a bound beside Figure 3
    for k, i in (("lat", 1), ("cost", 3)):
        v = [q.get(t, "qwen.rb-static.q05l25c70")[i] / q.get(t, "qwen.rb-static-dec.q05l25c70")[i] - 1 for t in RATES]
        out[f"dec.q05l25c70.{k}.maxabs"] = bound_up(100 * max(abs(x) for x in v), 0)   # "within": beside Figure 3
    if any(q.cell[(t, "qwen.rb-static.q05l25c70")]["model_mix"].get("Qwen/Qwen2.5-72B", 0.0) > 0 for t in RATES):
        raise SystemExit("5/25/70 sends requests to the 72B model; the text says it sends none")
    hq = bound_up(max(dq), 3)                                               # beside Figure 3
    out["dec.q.highest"] = ("+" + hq) if not hq.startswith("-") else "$-$" + hq[1:]   # the minus as dq3 writes it
    out["dec.q.lower"] = str(sum(x < 0 for x in dq))
    for w in EIGHT:
        for t in THREE:
            a, d = q.get(t, f"qwen.rb-static.{w}"), q.get(t, f"qwen.rb-static-dec.{w}")
            out[f"dec.{w}.{rate(t)}.lat"] = red(a[1] / d[1] - 1)
            out[f"dec.{w}.{rate(t)}.cost"] = red(a[3] / d[3] - 1)
    for name, _ in VARIANTS:                     # the figure's counts are the text's (dec.* here, single.* in single())
        key = "dec" if name == "RB-decoupled" else "single"
        for metric, mk in (("lat", "lat"), ("cost", "cost")):
            below = sum(x < 0 for row in fig[name][metric] for x in row["val"])
            if str(below) != out.res[f"{key}.{mk}.lower"]:
                raise SystemExit(f"the {name} figure's {metric} cells below 0 ({below}) differ from the text's count")
    head = "Preset & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{rate(t)}\,req/s}}" for t in RATES) + r" \\ "
    head += "".join(rf"\cmidrule(lr){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(len(RATES)))
    head += " & " + " & ".join("E2E & Cost & Q" for _ in RATES)
    return table(
        "tab:decoupled", r"\sys against RB-decoupled and RB-single at the same weights, at every preset and rate: "
        r"\sys's change of mean E2E latency and cost (\%) and its difference in quality, in thousandths, as in the "
        r"main text's Figure~3.",
        "l" + "rrr" * len(RATES), head, rows, wide=True, colsep="3.5pt")


def seeds_pt(v: dict):
    """A spread file's seed values as a point (quality, mean E2E, -, cost) for beats()."""
    return v["quality"], v["e2e_mean_s"], None, v["cost_per_1k"]


def single(q: Points, g: Points, out: Out) -> list[str]:
    """RB-single (RouteBalance with one request per round) against RouteBalance at the same preset and rate, pair
    by pair as RB-decoupled (not in the main count). Seed 1 from the first-seed report, seeds 2 and 3 from
    the spread files' set-aside runs (their seed 1 is checked against the report in spread()). Guards: RB-single beats
    RouteBalance in no pair at any seed; the pairs where RouteBalance is not cheaper are 5/25/70's alone, at every seed,
    and 5/25/70 sends no request to the 72B model in either; RB-single keeps one 72B share at 80/10/10 at every rate,
    RB-decoupled's; the queue rule: the lowest rate at which RB-single's mean wait for a decision exceeds 1 s at every
    preset and seed, with every higher rate above it too. Returns its rows at three rates (seed 1), which no table uses
    since decoupled() draws the appendix table's rows for both variants at every rate."""
    big = "Qwen/Qwen2.5-72B"
    lat, cost = {}, {}
    rows = []
    for w in EIGHT:
        cells = []
        for t in RATES:
            a, v = q.get(t, f"qwen.rb-static.{w}"), q.get(t, SINGLE + w)
            if beats(v, a):
                raise SystemExit(f"RB-single beats RouteBalance at {w}, {t}")
            lat[(w, t)], cost[(w, t)] = a[1] / v[1] - 1, a[3] / v[3] - 1
            if t in THREE:
                cells += [pct(lat[(w, t)]), pct(cost[(w, t)]), dq3(a[0] - v[0])]
        rows.append(f"{wlabel(w)} & " + " & ".join(cells) + r" \\")
    out["single.n"] = str(len(lat))
    out["single.lat.lower"] = str(sum(x < 0 for x in lat.values()))
    out["single.cost.lower"] = str(sum(x < 0 for x in cost.values()))
    out["single.lat.best"] = red_up(min(lat.values()))                       # "by up to": rounded up
    out["single.cost.best"] = red_up(min(cost.values()))
    five = "q05l25c70"
    if {k for k, x in cost.items() if x >= 0} != {(five, t) for t in RATES}:
        raise SystemExit("the pairs where RouteBalance is not cheaper than RB-single are not 5/25/70's alone")
    for t in RATES:
        for c in (f"qwen.rb-static.{five}", SINGLE + five):
            if q.cell[(t, c)]["model_mix"].get(big, 0.0) > 0:
                raise SystemExit(f"{c} at {t} sends requests to the 72B model; the text says 5/25/70 sends none")
    sav = [q.get(t, SINGLE + five)[3] / q.get(t, f"qwen.rb-static.{five}")[3] - 1 for t in RATES]
    loss = [q.get(t, f"qwen.rb-static.{five}")[0] - q.get(t, SINGLE + five)[0] for t in RATES]
    if max(sav) >= 0 or min(loss) <= 0:
        raise SystemExit("at 5/25/70 RB-single is not cheaper at lower quality at every rate")
    out["single.five.cost.lo"], out["single.five.cost.hi"] = red(max(sav), 1), red_up(min(sav), 1)   # "up to"
    out["single.five.q.lo"], out["single.five.q.hi"] = f"{min(loss):.3f}", bound_up(max(loss), 3)   # "up to", beside
    # Figure 3
    for t in THREE:
        out[f"single.q80.{rate(t)}.lat"] = red(lat[("q80l10c10", t)])
        out[f"single.q80.{rate(t)}.cost"] = red(cost[("q80l10c10", t)])
    q80cost = [cost[("q80l10c10", t)] for t in THREE]                 # the range's ends, in no assumed order
    out["single.q80.cost.lo"], out["single.q80.cost.hi"] = red(max(q80cost)), red(min(q80cost))
    shares = {share(q.cell[(t, SINGLE + "q80l10c10")]["model_mix"][big]) for t in RATES}
    dec = {share(q.cell[(t, "qwen.rb-static-dec.q80l10c10")]["model_mix"][big]) for t in RATES}
    if len(shares) != 1 or shares != dec:
        raise SystemExit("the text says RB-single keeps the 72B share at 80/10/10 the same at every rate, as RB-decoupled")
    out["single.mix72"] = shares.pop()
    # seeds 1 to 3: the counts, the guards, the waits for a decision
    runs, aside = SPREAD_RUNS["qwen"], SPREAD_ASIDE["qwen"]
    waits: dict[str, list[float]] = {t: [] for t in RATES}
    rbwait, net = [], []
    for i, x in enumerate(("1", "2", "3")):
        faster = cheaper = 0
        for w in EIGHT:
            for t in RATES:
                a, v = runs[(t, f"qwen.rb-static.{w}")][i], aside[(t, SINGLE + w)][i]
                if beats(seeds_pt(v), seeds_pt(a)):
                    raise SystemExit(f"RB-single beats RouteBalance at {w}, {t}, seed {x}")
                faster += a["e2e_mean_s"] < v["e2e_mean_s"]
                cheaper += a["cost_per_1k"] < v["cost_per_1k"]
                if (a["cost_per_1k"] < v["cost_per_1k"]) == (w == five):   # cheaper exactly outside 5/25/70
                    raise SystemExit(f"at {w}, {t}, seed {x}: the pairs where RouteBalance is not cheaper than RB-single "
                                     "are not 5/25/70's alone")
                if v["round_wait_mean_s"] is None or a["round_wait_mean_s"] is None:
                    raise SystemExit(f"no round wait recorded for {w} at {t}, seed {x}")
                waits[t].append(v["round_wait_mean_s"])
                rbwait.append(a["round_wait_mean_s"])
                if w == "q80l10c10":
                    net.append((v["e2e_mean_s"] - v["round_wait_mean_s"]) / a["e2e_mean_s"] - 1)
        out[f"single.lat.lower.s{x}"], out[f"single.cost.lower.s{x}"] = str(faster), str(cheaper)
        dfast = sum(runs[(t, f"qwen.rb-static.{w}")][i]["e2e_mean_s"] < aside[(t, f"qwen.rb-static-dec.{w}")][i]["e2e_mean_s"]
                    for w in EIGHT for t in RATES)
        dcheap = sum(runs[(t, f"qwen.rb-static.{w}")][i]["cost_per_1k"] < aside[(t, f"qwen.rb-static-dec.{w}")][i]["cost_per_1k"]
                     for w in EIGHT for t in RATES)
        out[f"dec.lat.lower.s{x}"], out[f"dec.cost.lower.s{x}"] = str(dfast), str(dcheap)        # RB-decoupled per seed
    above = [t for t in RATES if min(waits[t]) > 1.0]
    if not above or above != list(RATES[RATES.index(above[0]):]):
        raise SystemExit(f"RB-single's waits do not exceed 1 s from one rate on at every preset and seed: {above}")
    out["single.queue.rate"] = rate(above[0])
    for t in RATES:
        lo, hi = min(waits[t]), max(waits[t])
        out[f"single.wait.{rate(t)}.lo"] = f"{lo:.2f}" if lo < 1 else f"{lo:.1f}" if lo < 10 else f"{lo:.0f}"
        out[f"single.wait.{rate(t)}.hi"] = (f"{up(hi, 2):.2f}" if hi < 1 else f"{up(hi, 1):.1f}" if hi < 10
                                            else f"{up(hi, 0):.0f}")                # "up to": rounded up
    out["single.wait.queued.min"] = f"{down(min(min(waits[t]) for t in above), 1):.1f}"   # "or more": rounded down
    out["rb.wait.max"] = f"{up(max(rbwait), 2):.2f}"                   # "at most": rounded up
    if min(net) <= 0:
        raise SystemExit("net of its wait, RB-single at 80/10/10 is not slower than RouteBalance at every rate and seed")
    out["single.q80.net.lo"], out["single.q80.net.hi"] = inc_down(min(net)), inc_up(max(net))   # "X--Y% slower ... at
    # every rate and seed": a bound, rounded outward
    # Gemma: the sentence beside RB-decoupled's, seed 1; seeds 2 and 3 faster and cheaper in every pair
    glat, gcost, gdq = [], [], []
    for w in GEMMA_W:
        for t in GEMMA_RATES:
            a, v = g.get(t, f"gemma.rb-static.{w}"), g.get(t, GEMMA_SINGLE + w)
            glat.append(a[1] / v[1] - 1)
            gcost.append(a[3] / v[3] - 1)
            gdq.append(a[0] / v[0] - 1)
    if max(glat) >= 0 or max(gcost) >= 0:
        raise SystemExit("Gemma: RouteBalance is not faster and cheaper than RB-single everywhere")
    out["gemma.single.lat.lo"], out["gemma.single.lat.hi"] = red(max(glat)), red(min(glat))
    out["gemma.single.cost.lo"], out["gemma.single.cost.hi"] = red(max(gcost)), red(min(gcost))
    out["gemma.single.q.lo"], out["gemma.single.q.hi"] = red(max(gdq), 1), red(min(gdq), 1)
    gruns, gaside = SPREAD_RUNS["gemma"], SPREAD_ASIDE["gemma"]
    for i, x in enumerate(("1", "2", "3")):
        both = sum(gruns[(t, f"gemma.rb-static.{w}")][i]["e2e_mean_s"] < gaside[(t, GEMMA_SINGLE + w)][i]["e2e_mean_s"]
                   and gruns[(t, f"gemma.rb-static.{w}")][i]["cost_per_1k"] < gaside[(t, GEMMA_SINGLE + w)][i]["cost_per_1k"]
                   for w in GEMMA_W for t in GEMMA_RATES)
        if both != len(GEMMA_W) * len(GEMMA_RATES):
            raise SystemExit(f"Gemma seed {x}: RouteBalance is not faster and cheaper than RB-single in every pair")
    out["gemma.single.pairs"] = str(len(GEMMA_W) * len(GEMMA_RATES))
    return rows


def mix(q: Points, out: Out, plot: dict) -> None:
    big = "Qwen/Qwen2.5-72B"
    shifts = []
    for w in EIGHT:
        for sysname, prefix in (("rb", "qwen.rb-static."), ("dec", "qwen.rb-static-dec."),
                                ("reqnorm", "qwen.rb-static-reqnorm.")):
            for t in RATES:
                frac = q.cell[(t, prefix + w)]["model_mix"].get(big, 0.0)
                plot.setdefault("mix", {}).setdefault(f"{sysname}.{w}", {})[t] = q.cell[(t, prefix + w)]["model_mix"]
                if t in THREE:
                    out[f"mix.{sysname}.{w}.{rate(t)}"] = share(frac)
        a12 = q.cell[("l12", "qwen.rb-static." + w)]["model_mix"].get(big, 0.0)
        a42 = q.cell[("l42", "qwen.rb-static." + w)]["model_mix"].get(big, 0.0)
        shifts.append(a12 - a42)
    out["mix.shift.max"] = share(max(shifts))
    # the text: per-request normalization and RB-decoupled keep 80/10/10's 72B share the same at every rate (all six;
    # the key itself is required, so a renamed model cannot pass as a share of 0)
    for prefix in ("qwen.rb-static-reqnorm.", "qwen.rb-static-dec."):
        if len({share(q.cell[(t, prefix + "q80l10c10")]["model_mix"][big]) for t in RATES}) != 1:
            raise SystemExit(f"the text says {prefix}q80l10c10 keeps the 72B share the same at every rate")


def round_accounting(rs: dict, rr: dict, fleet: dict, q: Points, out: Out) -> None:
    """How a round's own assignments reach its later decisions (tools/round_share.py over the decision logs of
    RouteBalance 80/10/10 in the main comparison, seed 1, at 12, 30 and 42 req/s): the share of decisions scored after
    an earlier assignment of the same round, and the share that go to an instance the round has already given a request
    to. The logs are the ones Table A8 reads (round_range.json: the same sha256 and round count), so the text's shares
    and the table's mean round size come from the same runs; each log is its run's (log_matches_run)."""
    runs = [f"qwen.rb-static.q80l10c10__{t}__s1" for t in THREE]
    if sorted(rs) != sorted(runs):
        raise SystemExit(f"round_shares.json holds {sorted(rs)}, not the seed-1 runs of 80/10/10 at 12, 30 and 42 req/s")
    for t, run in zip(THREE, runs):
        s = rs[run]
        log_matches_run(run, s, q.cell[(t, "qwen.rb-static.q80l10c10")], fleet)
        if s["sha256"] != rr[run]["sha256"] or s["rounds"] != rr[run]["rounds"]:
            raise SystemExit(f"{run}: round_shares.json and round_range.json read different decision logs")
        out[f"round.{rate(t)}.after"] = share(s["after_round_assignment"] / s["decisions"])
        out[f"round.{rate(t)}.same"] = share(s["to_instance_already_given"] / s["decisions"])


def log_matches_run(run: str, s: dict, cell: dict, fleet: dict) -> None:
    """A decision log is its run's: its request count and its decisions per model equal the run's model mix."""
    if s["decisions"] != cell["summary"]["n_requests"]:
        raise SystemExit(f"{run}: {s['decisions']} decisions in its log, not its run's request count")
    per_model = {}
    for tier, k in s["per_tier"].items():
        per_model[fleet["tiers"][tier]["model"]] = per_model.get(fleet["tiers"][tier]["model"], 0) + k
    mix = cell["model_mix"]
    if set(per_model) != {m for m, f in mix.items() if f > 0} or any(
            abs(per_model[m] - mix[m] * s["decisions"]) > 1e-6 for m in per_model):
        raise SystemExit(f"{run}: its log's decisions per model {per_model} are not the run's model mix {mix}")


def round_range(rr: dict, fleet: dict, q: Points, out: Out) -> str:
    """The round's range as the lever of the load (tools/round_range.py over the decision logs of RouteBalance 80/10/10
    in the main comparison, seed 1; appendix): per rate, the mean round size, the mean quality range of the round, the
    72B model's lead over the best other model on the requests it is estimated to answer best (on the judge's scale,
    the same at every rate since the estimates do not depend on the round, and as a share of the range), the share of
    those requests it received, and its share of all requests. The text's "from A to B" pairs take the 12 and 42 req/s
    values, which must be the two ends over the six rates."""
    rows, raws, bests = [], set(), set()
    source = {k: [] for k in ("size", "range", "lead", "kept")}        # the unrounded values, for the endpoint check
    for t in RATES:
        run = f"qwen.rb-static.q80l10c10__{t}__s1"
        s = rr[run]
        log_matches_run(run, s, q.cell[(t, "qwen.rb-static.q80l10c10")], fleet)
        if s["big"] != "Qwen-2.5-72B":
            raise SystemExit(f"{run}: the lead was computed for {s['big']}, not the 72B model")
        if s["big_tie_share"] != 0:           # "estimated to answer best" leaves no request tied with another model
            raise SystemExit(f"{run}: the 72B model ties with another model on {s['big_tie_share']:.4%} of requests")
        r = rate(t)
        out[f"rr.{r}.size"] = f"{s['mean_round_size']:.1f}"
        out[f"rr.{r}.range"] = f"{s['range_mean_rounds']:.2f}"          # the mean over rounds, as the text says
        out[f"rr.{r}.lead"] = f"{s['big_lead_norm_mean_when_best']:.2f}"   # each request against its own round's range
        out[f"rr.{r}.kept"] = share(s["big_kept_share_when_best"])
        out[f"rr.{r}.mix72"] = share(s["big_chosen_share"])
        for key, value in (("size", s["mean_round_size"]), ("range", s["range_mean_rounds"]),
                           ("lead", s["big_lead_norm_mean_when_best"]), ("kept", s["big_kept_share_when_best"])):
            source[key].append(value)
        raws.add(f"{s['big_lead_raw_mean_when_best']:.3f}")
        bests.add(share(s["big_best_share"]))
        # the same prompts and estimates at every rate: the lead on the judge's scale and the best share are equal up
        # to the summation order, not merely at the printed precision
        if abs(s["big_lead_raw_mean_when_best"] - rr["qwen.rb-static.q80l10c10__l12__s1"]["big_lead_raw_mean_when_best"]) > 1e-6 \
                or s["big_best_share"] != rr["qwen.rb-static.q80l10c10__l12__s1"]["big_best_share"]:
            raise SystemExit(f"{run}: the 72B model's lead on the judge's scale or its best share differs from 12 req/s")
        rows.append(f"{r} & {out.res[f'rr.{r}.size']} & {out.res[f'rr.{r}.range']} & "
                    f"{s['big_lead_raw_mean_when_best']:.3f} & {out.res[f'rr.{r}.lead']} & "
                    f"{out.res[f'rr.{r}.kept']} & {out.res[f'rr.{r}.mix72']} \\\\")
    if len(raws) != 1 or len(bests) != 1:
        raise SystemExit("the text says the 72B model's lead on the judge's scale, and the share of requests it is "
                         f"estimated to answer best, are the same at every rate: {sorted(raws)}, {sorted(bests)}")
    out["rr.lead.raw"], out["rr.best"] = raws.pop(), bests.pop()
    for key, low_first in (("size", True), ("range", True), ("lead", False), ("kept", False)):
        vals = source[key]                    # the measured values, not their printed roundings
        ends = (min(vals), max(vals)) if low_first else (max(vals), min(vals))
        if (vals[0], vals[-1]) != ends:
            raise SystemExit(f"rr.{key}: the text gives the change from 12 to 42 req/s, and a rate in between lies "
                             f"outside it: {vals}")
    return table(
        "tab:roundrange", r"The round's range at 80/10/10 (seed 1), from the decision logs: the mean round size, the mean "
        r"quality range of the round, the 72B model's lead over the next model on the " + out.res["rr.best"] +
        r"\% of requests it is estimated to answer best, on the judge's scale and as a share of the range of each request's "
        r"own round, both averaged over those requests, the share of those requests it received (Kept), and its share of "
        r"all requests.",
        "rrrrrrr", r"$\lambda$ & Round & Range & \multicolumn{2}{c}{72B lead} & Kept & 72B \\ "
        r"\cmidrule(lr){4-5} & size & & judge & of range & \% & \%", rows)


SCHED: list[float] = []
BRW: list[tuple[float, str]] = []


def overhead(q: Points, out: Out) -> str:
    rows = []
    for t in RATES:
        o = q.cell[(t, "qwen.rb-static.q80l10c10")]["summary"]["overhead_ms"]
        vals = {k: o[src]["mean"] for k, src in STAGES}
        total = sum(vals.values())
        for k, v in vals.items():
            out[f"ovh.{rate(t)}.{k}"] = ms(v)
        out[f"ovh.{rate(t)}.total"] = ms(total)
        SCHED.append(vals["sched"])
        ap = q.cell[(t, "qwen.ap.pw080.lor")]["summary"]["overhead_ms"]["router_ms"]["mean"]
        br = q.cell[(t, "qwen.br.t05.lor")]["summary"]["overhead_ms"]["router_ms"]["mean"]
        out[f"ovh.{rate(t)}.ap"] = ms(ap)
        out[f"ovh.{rate(t)}.br"] = ms(br)
        BRW.append((br, t))
        rows.append(f"{rate(t)} & " + " & ".join(out.res[f"ovh.{rate(t)}.{k}"] for k, _ in STAGES)
                    + f" & {out.res[f'ovh.{rate(t)}.total']} & {out.res[f'ovh.{rate(t)}.ap']} & "
                      f"{out.res[f'ovh.{rate(t)}.br']} \\\\")
    out["ovh.sched.max"] = ms_up(max(SCHED))                                       # "at most"
    peak, at = max(BRW)
    out["ovh.br.peak"], out["ovh.br.peak.rate"] = ms(peak), rate(at)
    return table(
        "tab:overhead", r"Mean time a request spends in the scheduler before dispatch (ms): \sys at 80/10/10 by "
        r"stage (the round's wait, the batched estimate, the load read and the scheduling), and the router wait of "
        r"AvengersPro 0.80 (AP) and BEST-Route 0.5 (BR) on LOR.",
        "rrrrrrrr", r"$\lambda$ & Wait & Estimate & Load & Schedule & Total & AP & BR",
        rows)


ABL_REL: dict[str, list[tuple[float, float]]] = {}


def ablations(q: Points, out: Out) -> str:
    rows = []
    names = {"lpt-off": "LPT off (arrival order)", "round32": r"Fixed round (32 requests or 200\,ms)",
             "wlat0": r"No latency term ($w_l{=}0$)", "queueblind": "No queue terms (ties to the shorter queue)",
             "learned": "Learned per-tier regressors", "promptblind": "Prompt-blind estimates",
             "labels75": "75\\% of the labels", "labels50": "50\\% of the labels", "labels25": "25\\% of the labels",
             "lodogsm": "Without GSM8K's prompts", "lodolmsys": "Without LMSYS's prompts"}
    for key, cfg in ABLATIONS:
        cells = []
        for t in THREE:
            ref = q.get(t, ABL_REF)
            v = q.get(t, cfg)
            cells += [dq3(v[0] - ref[0]), pct(v[1] / ref[1] - 1), pct(v[3] / ref[3] - 1)]
            out[f"abl.{key}.{rate(t)}.q"] = dq3(v[0] - ref[0])
            out[f"abl.{key}.{rate(t)}.lat"] = pct(v[1] / ref[1] - 1)
            out[f"abl.{key}.{rate(t)}.cost"] = pct(v[3] / ref[3] - 1)
            out[f"abl.{key}.{rate(t)}.latinc"] = inc(v[1] / ref[1] - 1)
            out[f"abl.{key}.{rate(t)}.latx"] = f"{v[1] / ref[1]:.1f}"
            out[f"abl.{key}.{rate(t)}.costinc"] = inc(v[3] / ref[3] - 1)
            ABL_REL.setdefault(key, []).append((v[1] / ref[1] - 1, v[0] - ref[0]))
        rows.append(f"{names[key]} & " + " & ".join(cells) + r" \\")
        if key == "round32":
            # per-request normalization ran in the main grid's group: against the main grid's own 80/10/10 run
            cells = []
            for t in THREE:
                ref, v = q.get(t, "qwen.rb-static.q80l10c10"), q.get(t, "qwen.rb-static-reqnorm.q80l10c10")
                cells += [dq3(v[0] - ref[0]), pct(v[1] / ref[1] - 1), pct(v[3] / ref[3] - 1)]
                out[f"abl.reqnorm.{rate(t)}.latinc"] = inc(v[1] / ref[1] - 1)
            rows.append(r"Scores normalized per request$^{*}$ & " + " & ".join(cells) + r" \\")
    for k, vals in ABL_REL.items():
        out[f"abl.{k}.lat.absmax"] = bound_up(100 * max(abs(x) for x, _ in vals), 0)    # "by at most": a bound
        top = max(x for x, _ in vals)                                                     # beside Table A6
        out[f"abl.{k}.lat.max"] = bound_up(100 * top, 0) if top > 0 else r"\resnothigher"
    out["abl.latsrc.max"] = gap_up(max(abs(x) for k in ("queueblind", "learned") for x, _ in ABL_REL[k]), 0)
    out["abl.lodo.qmax"] = f"{up(max(abs(dq) for k in ('lodogsm', 'lodolmsys') for _, dq in ABL_REL[k]), 3):.3f}"
    out["abl.lodo.latmax"] = gap_up(max(abs(x) for k in ("lodogsm", "lodolmsys") for x, _ in ABL_REL[k]), 0)
    for t in THREE:
        ref = q.get(t, ABL_REF)
        out[f"abl.ref.{rate(t)}.q"] = f"{ref[0]:.3f}"
        out[f"abl.ref.{rate(t)}.lat"] = f"{ref[1]:.2f}"
    return table(
        "tab:ablation", r"Variants of \sys at 80/10/10 against a reference run of \sys at 80/10/10: the difference in "
        r"quality, and the change (\%) of mean E2E latency and of cost ($^{*}$against \sys at 80/10/10 in the main "
        r"comparison).",
        "l" + "rrr" * 3, r"Variant & \multicolumn{3}{c}{12\,req/s} & \multicolumn{3}{c}{30\,req/s} & "
        r"\multicolumn{3}{c}{42\,req/s} \\ & Q & E2E & Cost & Q & E2E & Cost & Q & E2E & Cost", rows, wide=True)


def bursts(q: Points, out: Out) -> str:
    rows = []
    for g, alpha in (BURSTS[0], BURSTS[1], ("", "1 (Poisson)"), BURSTS[2]):
        cells = []
        for t in THREE:
            trace = f"{g}-{t}" if g else t
            vals = {k: q.get(trace, c) for k, c in BURST_SYSTEMS}
            cells += [f"{vals[k][1]:.2f}" for k, _ in BURST_SYSTEMS]
            for k, _ in BURST_SYSTEMS:
                if k != "rb":
                    out[f"burst.{g or 'ga1'}.{rate(t)}.{k}"] = red(vals["rb"][1] / vals[k][1] - 1)
        rows.append(f"{alpha} & " + " & ".join(cells) + r" \\")
    short = {"rb": r"\sys", "dec": "Dec.", "ap": "AP", "single": "Single"}
    k = len(BURST_SYSTEMS)
    return table(
        "tab:burst", r"Mean E2E latency (s) under gamma arrivals of shape $\alpha$ at seed 1: \sys, RB-decoupled "
        r"(Dec.) and RB-single (Single) at 80/10/10, and AvengersPro 0.80 on LOR (AP).",
        "l" + "r" * k * 3, r"$\alpha$ & " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{r}\,req/s}}" for r in ("12", "30", "42"))
        + r" \\ & " + " & ".join(short[s_] for _ in THREE for s_, _c in BURST_SYSTEMS), rows, wide=True)


def budget(q: Points, out: Out) -> str:
    rows = []
    names = {"b75": "75\\%", "b45": "45\\%", "b30": "30\\%"}
    for mixname in BUDGET_MIXES:
        cells = []
        for k, cfg in BUDGET_SYSTEMS:
            c = q.cell[("l16", f"{mixname}.{cfg}")]["summary"]
            ex = c["budget_exhausted"]
            share = None if ex is None else ex / c["n_ok"] if isinstance(ex, int) else ex
            cells += [f"{c['quality']:.3f}", "--" if share is None else f"{100 * share:.1f}"]
            out[f"budget.{mixname}.{k}.q"] = f"{c['quality']:.3f}"
            if share is not None:
                out[f"budget.{mixname}.{k}.ex"] = f"{100 * share:.1f}"
        rows.append(f"{names[mixname]} & " + " & ".join(cells) + r" \\")
        p = lambda k: float(out.res[f"budget.{mixname}.{k}"])        # the printed quality and exhaustion rate
        if not (p("filter.q") > p("br.q") and p("filter.ex") < p("br.ex")):
            raise SystemExit(f"{mixname}: RouteBalance with its filter is not better than BEST-Route 0 on both")
    return table(
        "tab:budget", r"Budgets at 16\,req/s (Appendix~\ref{app:budget}): served quality (Q) and share of "
        r"requests exhausting their budget (Exh., \%).",
        "l" + "rr" * 3, r"Budgeted & \multicolumn{2}{c}{\sys + filter} & \multicolumn{2}{c}{\sys} & "
        r"\multicolumn{2}{c}{BEST-Route 0} \\ & Q & Exh. & Q & Exh. & Q & Exh.", rows)


GEMMA_NUM: dict[str, list[float]] = {}


def gemma(g: Points, out: Out, plot: dict) -> str:
    rows = []
    names = {"rb.q80l10c10": r"\sys 80/10/10", "rb.q50l30c20": r"\sys 50/30/20", "rb.uniform": r"\sys uniform",
             "dec.q80l10c10": "RB-decoupled 80/10/10", "dec.q50l30c20": "RB-decoupled 50/30/20",
             "dec.uniform": "RB-decoupled uniform", "ap.pw080": "AvengersPro 0.80", "ap.pw053": "AvengersPro 0.53",
             "br.t05": "BEST-Route 0.5", "br.t03": "BEST-Route 0.3", "br.t00": "BEST-Route 0", "lor": "LOR alone",
             "single.q80l10c10": "RB-single 80/10/10", "single.q50l30c20": "RB-single 50/30/20",
             "single.uniform": "RB-single uniform"}
    for key, cfg in GEMMA_SETTINGS:
        cells = []
        for t in GEMMA_RATES:
            v = g.get(t, cfg)
            cells += [f"{v[0]:.3f}", f"{v[1]:.2f}", f"{1000 * v[3]:.1f}"]
        rows.append(f"{names[key]} & " + " & ".join(cells) + r" \\")
    rbk = [k for k, _ in GEMMA_SETTINGS if k.startswith("rb.")]
    basek = [k for k, _ in GEMMA_SETTINGS if k.startswith(("ap.", "br.")) or k == "lor"]
    beaten_rb = 0
    for t in GEMMA_RATES:
        pts = {k: g.get(t, c) for k, c in GEMMA_SETTINGS}
        beaten_rb += sum(any(beats(pts[b], pts[r]) for b in basek) for r in rbk)
        rb, ap = pts["rb.q80l10c10"], pts["ap.pw080"]
        out[f"gemma.{rate(t)}.rbap.lat"] = red(rb[1] / ap[1] - 1)
        out[f"gemma.{rate(t)}.rbap.q"] = gap(rb[0] / ap[0] - 1, 1)
        b5 = pts["br.t05"]
        out[f"gemma.{rate(t)}.br05.latx"] = f"{b5[1] / rb[1]:.1f}"
        out[f"gemma.{rate(t)}.br05.costx"] = f"{b5[3] / rb[3]:.1f}"
        out[f"gemma.{rate(t)}.br05.q"] = inc(b5[0] / rb[0] - 1, 1)
        GEMMA_NUM.setdefault("br05.latx", []).append(b5[1] / rb[1])
        GEMMA_NUM.setdefault("br05.costx", []).append(b5[3] / rb[3])
        GEMMA_NUM.setdefault("br05.q", []).append(b5[0] / rb[0] - 1)
        GEMMA_NUM.setdefault("rbap.q", []).append(abs(rb[0] / ap[0] - 1))
    out["gemma.rb.beaten"] = str(beaten_rb)
    plot["gemma"] = {t: {k: g.get(t, c) for k, c in GEMMA_SETTINGS} for t in GEMMA_RATES}
    return table(
        "tab:gemma", r"Gemma~3 with every setting fixed on Qwen: quality, mean E2E latency (s) and cost (USD per "
        r"million requests) at OpenRouter's token prices (\url{https://openrouter.ai/models}). OpenRouter does not list the 1B, which we "
        r"price at \res{price.gemma.1b} per million input/output tokens, the lowest input price it lists for a Gemma 3 "
        r"model.",
        "l" + "rrr" * 3,
        r"Setting & \multicolumn{3}{c}{12\,req/s} & \multicolumn{3}{c}{30\,req/s} & \multicolumn{3}{c}{42\,req/s} "
        r"\\ & Q & E2E & Cost & Q & E2E & Cost & Q & E2E & Cost", rows, wide=True)


def vsr(summary: dict, q: Points, out: Out) -> str:
    """The vLLM-SR runs from tools/vsr_summary.py (the report's own summary function); ours from the report. The
    summary's RouteBalance and round-robin values at 6 req/s must equal the report's (a check of the two paths)."""
    for cfg in ("qwen.rb-static.q80l10c10", "qwen.disp.rr"):
        a, b = summary[f"{cfg}__l6__s1"]["summary"], q.cell[("l6", cfg)]["summary"]
        if not (math.isclose(a["quality"], b["quality"]) and math.isclose(a["e2e_ms"]["mean"], b["e2e_ms"]["mean"])
                and math.isclose(a["cost_total_per_1k"], b["cost_total_per_1k"])):
            raise SystemExit(f"{cfg} at l6: tools/vsr_summary.py and the report disagree")
    rows = []
    names = {"rb": r"\sys 80/10/10", "rr": "RR alone", "vsr": "vLLM-SR"}
    for k, cfg in VSR_SYSTEMS:
        cells = []
        for t in VSR_RATES:
            if k == "vsr":
                s = summary[f"{cfg}__{t}__s1"]["summary"]
            else:
                s = q.cell[(t, cfg)]["summary"]
            cells += [f"{s['n_ok']:,}/{s['n_requests']:,}".replace(",", "{,}"), f"{s['quality']:.3f}",
                      f"{s['e2e_ms']['mean'] / 1000:.2f}", f"{1000 * s['cost_total_per_1k']:.1f}"]
            out[f"vsr.{k}.{rate(t)}.done"] = f"{s['n_ok']:,}".replace(",", "{,}")
            out[f"vsr.{k}.{rate(t)}.of"] = f"{s['n_requests']:,}".replace(",", "{,}")
            out[f"vsr.{k}.{rate(t)}.q"] = f"{s['quality']:.3f}"
            lat = s['e2e_ms']['mean'] / 1000
            out[f"vsr.{k}.{rate(t)}.lat"] = f"{lat:.2f}" if lat < 100 else f"{lat:.0f}"
        rows.append(f"{names[k]} & " + " & ".join(cells) + r" \\")
    return table(
        "tab:vsr", r"The vLLM Semantic Router, with its shipped policy changed only in its quality scores, beside \sys and round robin on the same cluster: requests "
        r"completed, quality of the completed requests, mean E2E latency (s) and cost (USD per million requests).",
        "l" + "rrrr" * 2, r"System & \multicolumn{4}{c}{6\,req/s} & \multicolumn{4}{c}{9\,req/s} \\ & Done & Q & E2E "
        r"& Cost & Done & Q & E2E & Cost", rows, wide=True)


def vsr_failure(diag: dict, summary: dict, out: Out) -> None:
    """Appendix K's analysis, from tools/vsr_failure.py over the vLLM-SR runs' own records: where a request's time goes
    at 6 req/s, and the router's stops at 9 to 30 req/s, counted in each stop's goroutine dump. The
    text says: at 6 req/s it routed and completed every request and did not stop; at each of 9 to 30 req/s it stopped
    at one thread limit, holding requests (all in the signal phase), with threads in native calls and no goroutine in
    the model choice, the step that reads the quality scores; it completed requests at 9 and 12 req/s only, every one
    before it stopped. The stop time is the send of the last request it held or completed, and the next send must
    follow within a second."""
    names = {f"qwen.vsr.mom-balance__{t}__s1": t for t in ("l6",) + VSR_STOP_RATES}
    if sorted(diag) != sorted(names) or any(f"l{diag[c]['rate']:g}" != t for c, t in names.items()):
        raise SystemExit("vsr_failure.json: expected the six vLLM-SR runs, at 6, 9, 12, 18, 24 and 30 req/s")
    cells = {t: diag[c] for c, t in names.items()}
    six, tm = cells["l6"], cells["l6"]["timings"]
    if six["stop"] is not None or tm["n"] != six["requests"] or six["completed"] != six["requests"]:
        raise SystemExit("vLLM-SR at 6 req/s: the text says it routed and completed every request and did not stop")
    for t in VSR_RATES:                                     # the table's counts, from the report's summary function
        if cells[t]["completed"] != summary[f"qwen.vsr.mom-balance__{t}__s1"]["summary"]["n_ok"]:
            raise SystemExit(f"vLLM-SR at {t}: the completed requests differ from tools/vsr_summary.py's")
    stops = [cells[t]["stop"] for t in VSR_STOP_RATES]
    if any(s is None for s in stops):
        raise SystemExit("vLLM-SR: the text says its router stopped at every rate from 9 to 30 req/s")
    limits = {s["thread_limit"] for s in stops}
    if len(limits) != 1:
        raise SystemExit(f"vLLM-SR: the stops name different thread limits {sorted(limits)}")
    limit = limits.pop()
    for t, s in zip(VSR_STOP_RATES, stops):
        if s["model_choice_goroutines"] != 0:
            raise SystemExit(f"vLLM-SR at {t}: a goroutine was in the model choice when the router stopped")
        if s["held_in_signal_phase"] != s["held"] or s["held"] == 0:
            raise SystemExit(f"vLLM-SR at {t}: {s['held']} requests held, {s['held_in_signal_phase']} in the signal phase")
        if not 0 < s["native_threads"] <= s["threads_attached"] <= limit + 1:
            raise SystemExit(f"vLLM-SR at {t}: {s['native_threads']} native-call threads of {s['threads_attached']}")
        if not (s["stop_s"] < s["next_send_s"] <= s["stop_s"] + 1.0 and s["last_log_s"] < s["next_send_s"]):
            raise SystemExit(f"vLLM-SR at {t}: the stop time is not pinned between two sends a second apart")
        if s["completed_before"] != cells[t]["completed"]:
            raise SystemExit(f"vLLM-SR at {t}: a request completed after the router stopped")
    done = {t: cells[t]["completed"] for t in VSR_STOP_RATES}
    if not (done["l9"] > 0 and done["l12"] > 0 and all(done[t] == 0 for t in ("l18", "l24", "l30"))):
        raise SystemExit(f"the text says it completed requests at 9 and 12 req/s and none at 18 to 30: {done}")
    big = lambda v: f"{v:,}".replace(",", "{,}")
    out["vsr.diag.6.envoy"] = f"{tm['envoy_s_mean']:.2f}"
    out["vsr.diag.6.signals"] = f"{tm['signals_s_mean']:.2f}"
    out["vsr.diag.6.share"] = f"{100 * tm['signals_share']:.1f}"
    out["vsr.diag.6.after"] = f"{tm['after_signals_ms_mean']:.1f}"
    out["vsr.diag.6.aftermax"] = f"{tm['after_signals_ms_max']:.0f}"
    out["vsr.stop.limit"] = big(limit)
    for key, vals in (("s", [round(s["stop_s"]) for s in stops]), ("held", [s["held"] for s in stops]),
                      ("native", [s["native_threads"] for s in stops])):
        out[f"vsr.stop.{key}.lo"], out[f"vsr.stop.{key}.hi"] = big(min(vals)), big(max(vals))
    out["vsr.stop.done.9"], out["vsr.stop.done.12"] = str(done["l9"]), str(done["l12"])


def cluster(fleet: dict, tpot: dict, prefill: dict, out: Out) -> str:
    rows = []
    counts = {}
    for inst in fleet["instances"]:
        counts[inst["tier"]] = counts.get(inst["tier"], 0) + 1
    for tier in TIER_ORDER:
        t = fleet["tiers"][tier]
        model = t["model"].split("/")[-1]
        tp = tpot["constant"][tier]
        pf = prefill["prefill"][tier]
        rows.append(f"{model} & {GPU[t['gpu']]} & {counts[tier]} & {tp:.1f} & {pf['intercept_ms']:.0f} + "
                    f"{1000 * pf['per_token_ms']:.0f} & {1e6 * t['cost_in']:.2f}/{1e6 * t['cost_out']:.2f} \\\\")
        out[f"tpot.{model.lower().replace('qwen2.5-', '')}"] = f"{tp:.1f}"
    out["fleet.instances"] = str(len(fleet["instances"]))
    return table(
        "tab:cluster", r"The Qwen2.5 cluster per tier: instances, the TPOT \sys uses (ms per token), prefill time "
        r"(ms, plus $\mu$s per prompt token) and price (USD per million input/output tokens, from "
        r"\citet{novita}, or \citet{together} for the 3B).",
        "llrrrr", r"Model & GPU & Inst. & TPOT & Prefill & Price", rows)


def table(label_: str, caption: str, cols: str, header: str, rows: list[str], wide: bool = False,
          size: str | None = None, colsep: str | None = None) -> str:
    env = "table*" if wide else "table"
    size = size or (r"\footnotesize" if wide else r"\small")
    if colsep:                                  # a narrower column gap, so a wide table prints at its own size
        size += f"\\setlength{{\\tabcolsep}}{{{colsep}}}"
    width = r"\textwidth" if wide else r"\columnwidth"
    # natural size, shrunk only when wider than the column or page (adjustbox's max width)
    return (f"% Generated by tools/gen_results.py from data/ (R12). Do not edit.\n\\genbegin{{{env}}}\n\\centering{size}\n"
            f"\\caption{{{caption}}}\n\\label{{{label_}}}\n\\begin{{adjustbox}}{{max width={width}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n"
            f"\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            f"\\end{{adjustbox}}\n\\end{{{env}}}\n")


def tail(q: Points, g: Points, tolerance: float, out: Out) -> str:
    """Tail latency: the size of the gain at p99. "Beats" stays defined on mean latency only, and
    the p99 shows the range of the gain.
    Per family and rate, the equal-quality p99 reduction against the baseline settings (equal_quality()'s
    per-metric reading on the presets' quality-p99 frontier): median and largest; and p99 at 80/10/10 for
    RouteBalance, RB-decoupled and AvengersPro 0.80 on LOR."""
    rows, allred = [], []
    fams = (("qwen", q, RATES, [f"qwen.rb-static.{w}" for w in EIGHT], main_baselines(), "qwen"),
            ("gemma", g, GEMMA_RATES, [c for k, c in GEMMA_SETTINGS if k.startswith("rb.")],
             [c for k, c in GEMMA_SETTINGS if k.startswith(("ap.", "br.")) or k == "lor"], "gemma"))
    for fam, pts, rates, rb, base, pre in fams:
        fam_red = []
        rows.append(r"\multicolumn{6}{l}{\emph{" + ("Qwen2.5" if fam == "qwen" else "Gemma 3") + r"}} \\")
        for t in rates:
            presets = [pts.get(t, c) for c in rb]
            reds = []
            for b in base:
                v = pts.get(t, b)
                m = per_metric([(p[0], p[2]) for p in presets], v[0], tolerance)
                if m is not None:
                    reds.append(m[0] / v[2] - 1)
            if not reds:
                raise SystemExit(f"{fam} at {t}: no baseline setting compared at equal quality on p99")
            fam_red += reds
            key = f"tail.{fam}.{rate(t)}"
            out[f"{key}.median"] = red(median(reds))
            out[f"{key}.gap"] = bound_up(100 * abs(median(reds)), 0)      # "within X%": a bound beside Table A10
            rb80, dec80 = pts.get(t, f"{pre}.rb-static.q80l10c10"), pts.get(t, f"{pre}.rb-static-dec.q80l10c10")
            ap80 = pts.get(t, f"{pre}.ap.pw080.lor")
            rows.append(f"{rate(t)} & {pct(median(reds))} & {pct(min(reds))} & {rb80[2]:.1f} & {dec80[2]:.1f} & "
                        f"{ap80[2]:.1f} \\\\")
        out[f"tail.{fam}.median"] = red(median(fam_red))
        out[f"tail.{fam}.n"] = str(len(fam_red))
        out[f"tail.{fam}.lower"] = str(sum(x < 0 for x in fam_red))
    return table(
        "tab:tail", r"p99 E2E latency: \sys's change at equal quality against the baseline settings (the median "
        r"change and the largest reduction, \%, negative where \sys is lower) and p99 (s) at 80/10/10 for \sys, "
        r"RB-decoupled (Dec.) and AvengersPro 0.80 on LOR (AP).",
        "lrrrrr", r"$\lambda$ & Median & Largest & \sys & Dec. & AP", rows)


def spread(files: dict, q: Points, g: Points, rules_sha: str, tolerance: float, interim: bool, out: Out) -> str:
    """The spread across seeds 1-3 of the repeated settings, per family, for RouteBalance at its presets and for
    the baselines (RB-decoupled and the per-request variant are RouteBalance's own variants and in neither). Quality's
    range is absolute; mean E2E's and cost's are a share of their mean over the three seeds. Every seed-1 value must be
    the first-seed report's value for that run. Per seed, on the repeated settings alone: the baseline settings some
    preset beats at the same rate, and the presets some baseline setting beats, at the rates with every preset
    repeated; at those rates, the equal-quality reductions paired by seed against the repeated baseline settings (as
    equal_quality(), each seed's presets read at that seed's baseline quality): the median per seed, the settings
    with a latency reduction at all three seeds, and the largest range's reductions. The baselines' largest latency range
    is given with the range of its time off the instances (E2E less the engine's own latency)."""
    fams = (("qwen", files["spread_qwen.json"], q, "qwen", RATES, [f"qwen.rb-static.{w}" for w in EIGHT],
             main_baselines()),
            ("gemma", files["spread_gemma.json"], g, "gemma", GEMMA_RATES,
             [c for k, c in GEMMA_SETTINGS if k.startswith("rb.")],
             [c for k, c in GEMMA_SETTINGS if k.startswith(("ap.", "br.")) or k == "lor"]))
    names = {("qwen", "rb"): r"Qwen, \sys", ("qwen", "base"): "Qwen, baselines",
             ("gemma", "rb"): r"Gemma, \sys", ("gemma", "base"): "Gemma, baselines"}
    seeds = ("1", "2", "3")
    rows = []
    for fam, sp, pts, group, rates, rb, base in fams:
        if sp["group"] != group or sp["rules_sha256"] != rules_sha:
            raise SystemExit(f"spread_{fam}.json: group {sp['group']} or rules differ")
        out[f"spread.{fam}.partial"] = str(sp["partial_skipped"])      # nonzero only in an interim build
        runs: dict[tuple[str, str], list[dict]] = {}
        aside: dict[tuple[str, str], list[dict]] = {}
        own = set_aside(fam)
        for r in sp["repeated"]:
            cfg, trace = r["config"], r["trace"]
            if (trace, cfg) not in own and (trace not in rates or (cfg not in rb and cfg not in base)):
                raise SystemExit(f"spread_{fam}.json: {cfg} at {trace} is neither a preset, a baseline setting, nor a "
                                 f"run its own section reads")
            first, s1 = pts.get(trace, cfg), r["seeds"]["1"]
            summ = pts.cell[(trace, cfg)]["summary"]
            off = summ["off_instance_ms"]
            wait = ((summ.get("overhead_ms") or {}).get("round_wait_ms") or {}).get("mean")
            if (s1["quality"], s1["e2e_mean_s"], s1["cost_per_1k"], s1["off_instance_mean_s"],
                    s1.get("round_wait_mean_s")) != (first[0], first[1], first[3], off["mean"] / 1000.0 if off else None,
                                                     wait / 1000.0 if wait is not None else None):
                raise SystemExit(f"spread_{fam}.json: {cfg} at {trace}, seed 1 is not the first-seed report's run")
            (aside if (trace, cfg) in own else runs)[(trace, cfg)] = [r["seeds"][x] for x in seeds]
        SPREAD_RUNS[fam], SPREAD_ASIDE[fam] = dict(runs), aside

        full = [t for t in rates if all((t, c) in runs for c in rb)]      # rates with every preset repeated
        if not full:
            raise SystemExit(f"spread_{fam}.json: no rate with every preset repeated")
        # the spread is reported over those rates only, and the text names them (every rate once complete): a
        # rate with a preset missing would make the table's counts disagree with "every preset"
        repeated_rb = sum(k[1] in rb for k in runs)                       # before the filter: the pending mark
        runs = {k: v for k, v in runs.items() if k[0] in full}
        names_ = [rate(t) for t in full]
        listed = "at " + (", ".join(names_[:-1]) + " and " + names_[-1] if len(names_) > 1 else names_[0]) + r"\,req/s"
        out[f"spread.{fam}.at"] = "at each rate" if len(full) == len(rates) else listed
        # the beat count's noun phrase: a total over the rates, so never "... at each rate"
        fam_name = "Qwen" if fam == "qwen" else "Gemma"
        out[f"spread.{fam}.basephrase"] = (f"repeated {fam_name} baseline settings" if len(full) == len(rates) else
                                           f"{fam_name} baseline settings repeated {listed}")

        def rng(k, m, rel):
            xs = [v[m] for v in runs[k]]
            return (max(xs) - min(xs)) / (statistics.mean(xs) if rel else 1.0)

        kinds = {"rb": [k for k in runs if k[1] in rb], "base": [k for k in runs if k[1] in base]}
        for kind, keys in kinds.items():
            if not keys:
                raise SystemExit(f"spread_{fam}.json: no repeated {kind} setting")
            key = f"spread.{fam}.{kind}"
            out[f"{key}.n"] = str(len(keys))
            cells = []
            for m, metric, rel in (("q", "quality", False), ("lat", "e2e_mean_s", True), ("cost", "cost_per_1k", True)):
                col = [rng(k, metric, rel) for k in keys]
                med, top = median(col), max(col)
                text = (f"{med:.3f}", f"{top:.3f}") if m == "q" else (f"{100 * med:.1f}", f"{100 * top:.1f}")
                # the table rounds to the nearest; the text cites RouteBalance's largest as "at most": rounded up
                out[f"{key}.{m}.med"] = text[0]
                out[f"{key}.{m}.max"] = (text[1] if kind != "rb" else
                                         f"{up(top, 3):.3f}" if m == "q" else bound_up(100 * top, 1))   # a bound
                # beside Table 8
                SPREAD_NUM[(fam, kind, m)] = (med, top)
                cells.append(f"{text[0]} / {text[1]}")
            rows.append(f"{names[(fam, kind)]} & {len(keys)} & " + " & ".join(cells) + r" \\")
        def at(k, i):
            v = runs[k][i]
            return v["quality"], v["e2e_mean_s"], None, v["cost_per_1k"]

        # complete: every preset repeated at every rate, no setting with a seed missing, and the group's parity
        # passed; otherwise only an --interim build, which writes a \pending mark (prints nothing; check_rulings.py
        # counts it, 0 at submission) so the draft reads as final while the repetitions run
        complete = not sp["partial_skipped"] and len(full) == len(rates) and sp["parity"] == "passed"
        if not complete and not interim:
            raise SystemExit(f"spread_{fam}.json: not complete ({sp['partial_skipped']} partial, {len(full)} of "
                             f"{len(rates)} rates with every preset, parity {sp['parity']}); pass --interim")
        out[f"spread.{fam}.interim"] = "" if complete else (
            f"\\pending{{{fam.capitalize()}'s seeds 2--3 are partly run: {repeated_rb} of "
            f"{len(rb) * len(rates)} preset points, parity {sp['parity']}.}}")
        out[f"spread.{fam}.full"] = str(len(full))
        out[f"spread.{fam}.beaten.n"] = str(sum(k[0] in full for k in kinds["base"]))
        for i, x in enumerate(seeds):
            def pt(k):
                v = runs[k][i]
                return v["quality"], v["e2e_mean_s"], None, v["cost_per_1k"]
            beaten = sum(any(beats(pt(r), pt(b)) for r in kinds["rb"] if r[0] == b[0])
                         for b in kinds["base"] if b[0] in full)
            rbbeaten = sum(any(beats(pt(b), pt(r)) for b in kinds["base"] if b[0] == r[0])
                           for r in kinds["rb"] if r[0] in full)
            out[f"spread.{fam}.beaten.s{x}"] = str(beaten)
            out[f"spread.{fam}.rbbeaten.s{x}"] = str(rbbeaten)
            SPREAD_NUM[(fam, "beaten", x)] = (beaten, rbbeaten)
        # a baseline setting beaten at some seeds and not at others: at each seed where no preset beats it, some
        # preset of its rate is at least as good on two measures and misses the third by a margin, relative to the
        # baseline's value; per such seed the smallest such margin over the presets and measures counts, under its
        # measure; the text cites each measure's largest
        miss = {"q": 0.0, "lat": 0.0, "cost": 0.0}
        flips = 0
        for b in (k for k in kinds["base"] if k[0] in full):
            pre = [r for r in kinds["rb"] if r[0] == b[0]]
            won = [any(beats(at(r, i), at(b, i)) for r in pre) for i in range(3)]
            if all(won) or not any(won):
                continue
            flips += 1
            for i in (i for i in range(3) if not won[i]):
                alone: dict[str, list[float]] = {"q": [], "lat": [], "cost": []}
                for r in pre:
                    a, c = at(r, i), at(b, i)
                    m = {"q": max(1 - a[0] / c[0], 0.0), "lat": max(a[1] / c[1] - 1, 0.0),
                         "cost": max(a[3] / c[3] - 1, 0.0)}
                    if sum(v > 0 for v in m.values()) == 1:
                        k_ = next(k_ for k_, v in m.items() if v > 0)
                        alone[k_].append(m[k_])
                best = min(((min(v), k_) for k_, v in alone.items() if v), default=None)
                if best is None:
                    raise SystemExit(f"spread_{fam}.json: {b} at seed {i + 1} is not within one measure of a preset")
                miss[best[1]] = max(miss[best[1]], best[0])
        out[f"spread.{fam}.flip.n"] = str(flips)
        for k_ in ("q", "lat", "cost"):                 # bounds ("within"): rounded up; a measure with no case
            out[f"spread.{fam}.flip.{k_}"] = (f"{up(100 * miss[k_], 1):.1f}" if miss[k_] > 0 else r"\resnone")
        if fam == "gemma" and len({SPREAD_NUM[("gemma", "beaten", x)][0] for x in seeds}) != 1:
            raise SystemExit("spread_gemma.json: the text says the presets beat the same number at every seed")
        # equal quality, paired by seed (the same arrivals and prompt order): each seed's presets read at that seed's
        # baseline quality, as equal_quality(); a setting is compared when it is comparable at all three seeds
        red_by: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for k in (k for k in kinds["base"] if k[0] in full):
            per = []
            for i in range(3):
                presets, v = [runs[(k[0], c)][i] for c in rb], runs[k][i]
                l = per_metric([(p["quality"], p["e2e_mean_s"]) for p in presets], v["quality"], tolerance)
                c = per_metric([(p["quality"], p["cost_per_1k"]) for p in presets], v["quality"], tolerance)
                if l is None or c is None:
                    break
                per.append((l[0] / v["e2e_mean_s"] - 1, c[0] / v["cost_per_1k"] - 1))
            if len(per) == 3:
                red_by[k] = per
        if not red_by:
            raise SystemExit(f"spread_{fam}.json: no baseline setting comparable at equal quality at every seed")
        for i, x in enumerate(seeds):
            out[f"spread.{fam}.eq.lat.s{x}"] = red(median([p[i][0] for p in red_by.values()]))
            out[f"spread.{fam}.eq.cost.s{x}"] = red(median([p[i][1] for p in red_by.values()]))
        out[f"spread.{fam}.pair.n"] = str(len(red_by))
        out[f"spread.{fam}.pair.faster"] = str(sum(all(p[0] < 0 for p in per) for per in red_by.values()))
        SPREAD_NUM[(fam, "pair")] = red_by
        # the text: a router sends each prompt to the same model at every seed; on Gemma no baseline setting beats a
        # preset at any seed; the largest baseline range is a router's
        if any(rng(k, "quality", False) != 0 for k in kinds["base"] if k[1].split(".")[1] in ("ap", "br")):
            raise SystemExit(f"spread_{fam}.json: a router's quality differs between seeds")
        if fam == "gemma" and any(beats(at(b, i), at(r, i)) for i in range(3) for r in kinds["rb"]
                                  for b in kinds["base"] if b[0] == r[0]):        # every repeated pair, any rate
            raise SystemExit("spread_gemma.json: a baseline setting beats a preset at some seed")
        top = max(kinds["base"], key=lambda k: rng(k, "e2e_mean_s", True))
        if top[1].split(".")[1] not in ("ap", "br"):
            raise SystemExit(f"spread_{fam}.json: the largest baseline latency range is not a router's")
        out[f"spread.{fam}.base.top.name"] = label(top[1])
        out[f"spread.{fam}.base.top.rate"] = rate(top[0])
        out[f"spread.{fam}.base.top.lat"] = f"{rng(top, 'e2e_mean_s', False):.2f}"
        e2e = [v["e2e_mean_s"] for v in runs[top]]
        hi, lo = e2e.index(max(e2e)), e2e.index(min(e2e))
        out[f"spread.{fam}.base.top.off"] = (            # the same two seeds as the E2E range
            f"{runs[top][hi]['off_instance_mean_s'] - runs[top][lo]['off_instance_mean_s']:.2f}")
        if top not in red_by:
            raise SystemExit(f"spread_{fam}.json: the largest baseline range is not compared at equal quality")
        SPREAD_NUM[(fam, "top")] = top
        lats = [p[0] for p in red_by[top]]
        out[f"spread.{fam}.base.top.red.lo"], out[f"spread.{fam}.base.top.red.hi"] = (red_down(max(lats)),   # "X--Y%
                                                                                      red_up(min(lats)))    # faster
        # ... at every seed": a bound, rounded outward
    if rate(SPREAD_NUM[("qwen", "top")][0]) == rate(SPREAD_NUM[("gemma", "top")][0]):   # the text: "both at R req/s"
        out["spread.top.rate"] = rate(SPREAD_NUM[("qwen", "top")][0])
    qat = out.res["spread.qwen.at"]
    return table(
        "tab:spread", r"Spread over seeds 1--3 (median / largest"
        + ("" if qat == "at each rate" else f", Qwen {qat}") + r"): the range of quality, and the ranges of mean E2E latency and of cost as a share "
        r"of their mean.",
        "lrccc", r"Settings & $n$ & Quality & E2E (\%) & Cost (\%)", rows)


SPREAD_NUM: dict = {}
SPREAD_RUNS: dict = {}       # family -> (trace, config) -> the three seeds' values: the presets and baseline settings
SPREAD_ASIDE: dict = {}      # the same for the runs their own sections read: RouteBalance's variants and the bursts


def set_aside(fam: str) -> set[tuple[str, str]]:
    """The repeated runs the spread leaves to their own sections, by exact name: RouteBalance's own variants
    (RB-decoupled, RB-single, the per-request normalization) at the family's rates, the burst traces' systems, and
    round robin and RouteBalance at the vLLM-SR table's rates.
    Any other repeated run outside the presets and baseline settings stops the build."""
    if fam == "qwen":
        own = {(t, f"qwen.{v}.{w}") for t in RATES for w in EIGHT for v in ("rb-static-dec", "rb-static-reqnorm")}
        own |= {(t, SINGLE + w) for t in RATES for w in EIGHT}
        own |= {(f"{g}-{t}", c) for g, _ in BURSTS for t in THREE for _, c in BURST_SYSTEMS}
        own |= {(t, c) for t in VSR_RATES for k, c in VSR_SYSTEMS if k != "vsr"}      # the vLLM-SR table's checks
        return own
    return {(t, c + w) for t in GEMMA_RATES for w in GEMMA_W for c in ("gemma.rb-static-dec.", GEMMA_SINGLE)}


def extras(q: Points, g: Points, vsr_summary: dict, corpus: dict, out: Out) -> None:
    """The values the text cites beyond the tables."""
    big = "Qwen/Qwen2.5-72B"
    n = {c["summary"]["n_requests"] for (t, cfg), c in q.cell.items() if t in RATES and cfg in
         [f"qwen.rb-static.{w}" for w in EIGHT] + main_baselines()}
    if len(n) != 1:
        raise SystemExit(f"main-grid runs with different request counts: {n}")
    out["runs.qwen.requests"] = f"{n.pop():,}".replace(",", "{,}")
    gn = {c["summary"]["n_requests"] for c in g.cell.values()}
    if len(gn) != 1:
        raise SystemExit(f"Gemma runs with different request counts: {gn}")
    out["runs.gemma.requests"] = f"{gn.pop():,}".replace(",", "{,}")
    shares = [q.cell[(t, "qwen.ap.pw080.lor")]["model_mix"].get(big, 0.0) for t in RATES]
    if share(min(shares)) != share(max(shares)):
        raise SystemExit("AvengersPro 0.80's 72B share differs between rates at the precision the text gives")
    out["ap80.mix72"] = share(max(shares))
    out["ap80.latx"] = f"{q.get('l42', 'qwen.ap.pw080.lor')[1] / q.get('l12', 'qwen.ap.pw080.lor')[1]:.1f}"
    for t in RATES:
        out[f"ap80.lat.{rate(t)}"] = f"{q.get(t, 'qwen.ap.pw080.lor')[1]:.2f}"
        if not q.get(t, "qwen.rb-static.q80l10c10")[0] > q.get(t, "qwen.ap.pw080.lor")[0]:
            raise SystemExit(f"{t}: RouteBalance 80/10/10 is not above AvengersPro 0.80 in quality")
    rb80 = [q.get(t, "qwen.rb-static.q80l10c10")[1] for t in RATES]
    out["rb80.lat.min"], out["rb80.lat.max"] = f"{down(min(rb80), 2):.2f}", f"{up(max(rb80), 2):.2f}"   # "between"
    small = ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B")
    br0 = [sum(q.cell[(t, f"qwen.br.t00.{d}")]["model_mix"].get(m, 0.0) for m in small)
           for t in RATES for d in ROUTER_DISPATCH]
    out["br0.small.min"] = f"{int(down(100 * min(br0), 0))}"                     # "at least": rounded down
    for t in THREE:
        out[f"br0.lor.lat.{rate(t)}"] = f"{q.get(t, 'qwen.br.t00.lor')[1]:.2f}"
        out[f"br0.wait.{rate(t)}"] = f"{q.cell[(t, 'qwen.br.t00.lor')]['summary']['overhead_ms']['router_ms']['mean']:.0f}"
        out[f"br0.wait.{rate(t)}s"] = f"{q.cell[(t, 'qwen.br.t00.lor')]['summary']['overhead_ms']['router_ms']['mean'] / 1000:.1f}"
        for k, cfg in (("ap", "qwen.ap.pw080.lor-shipped"), ("br", "qwen.br.t05.lor-shipped")):
            w = q.cell[(t, cfg)]["summary"]["overhead_ms"]["router_ms"]["mean"] / 1000
            out[f"ovh.{rate(t)}.{k}.shipped"] = f"{w:.1f}"
            pooled = q.cell[(t, cfg.replace("-shipped", ""))]["summary"]["overhead_ms"]["router_ms"]["mean"] / 1000
            if t in ("l30", "l42") and not w > pooled:          # the appendix: "waits longer at 30 and 42 req/s"
                raise SystemExit(f"{cfg} at {t}: one call at a time does not wait longer than 32 at once")
    waits = [(q.cell[(t, b)]["summary"]["overhead_ms"]["router_ms"]["mean"], b, t) for t in RATES
             for b in main_baselines() if ".disp." not in b]
    w, b, t = max(waits)
    out["ovh.router.max"], out["ovh.router.max.rate"] = f"{w / 1000:.1f}", rate(t)
    out["ovh.router.max.who"] = label(b)
    # per-request normalization: what the eight presets beat when each request is normalized alone
    nb = 0
    for t in RATES:
        rn = [q.get(t, f"qwen.rb-static-reqnorm.{w}") for w in EIGHT]
        nb += sum(any(beats(v, q.get(t, b)) for v in rn) for b in main_baselines())
    out["dom.reqnorm.n"] = str(nb)
    # bursts: ranges of mean latency over every burstiness and rate (seed 1), and over seeds 1-3 where they were run
    # (the burst traces at every seed; the Poisson row at seeds 2 and 3 only where the main grid repeated it)
    for k, cfg in BURST_SYSTEMS:
        vals = [q.get(f"{gg}-{t}" if gg else t, cfg)[1] for gg in ("ga025", "ga05", "ga2", "") for t in THREE]
        out[f"burst.{k}.min"], out[f"burst.{k}.max"] = f"{min(vals):.1f}", f"{max(vals):.1f}"
        allv, poisson = list(vals), []
        for gg in ("ga025", "ga05", "ga2", ""):
            for t in THREE:
                trace = f"{gg}-{t}" if gg else t
                rep = SPREAD_ASIDE["qwen"].get((trace, cfg)) or SPREAD_RUNS["qwen"].get((trace, cfg))
                if rep is None and gg:
                    raise SystemExit(f"{cfg} at {trace}: the burst trace has no seeds 2 and 3")
                if not gg:
                    poisson.append(rep is not None)
                if rep is not None:
                    allv += [rep[i]["e2e_mean_s"] for i in (1, 2)]
        expected = [k != "ap"] * len(THREE)          # the main grid repeats every system's Poisson runs but AvengersPro 0.80's
        if poisson != expected:
            raise SystemExit(f"{cfg}: its Poisson runs at seeds 2 and 3 are {poisson}, the text expects {expected}")
        out[f"burst.{k}.seeds.min"], out[f"burst.{k}.seeds.max"] = (f"{down(min(allv), 1):.1f}",      # every value
                                                                    f"{up(max(allv), 1):.1f}")         # inside
        out[f"burst.{k}.poisson.seeds"] = "1--3" if all(poisson) else "1"
    # vLLM-SR throughput at 6 req/s
    out["vsr.vsr.6.rps"] = f"{vsr_summary['qwen.vsr.mom-balance__l6__s1']['summary']['throughput_rps']:.1f}"
    # Gemma: against RB-decoupled at the same weights; the trained routers under load
    lat, cost, dq = [], [], []
    for w in ("q80l10c10", "q50l30c20", "uniform"):
        for t in GEMMA_RATES:
            a, d = g.get(t, f"gemma.rb-static.{w}"), g.get(t, f"gemma.rb-static-dec.{w}")
            lat.append(a[1] / d[1] - 1)
            cost.append(a[3] / d[3] - 1)
            dq.append(a[0] / d[0] - 1)
    if max(lat) >= 0 or max(cost) >= 0:
        raise SystemExit("Gemma: RouteBalance is not faster and cheaper than RB-decoupled everywhere")
    out["gemma.dec.lat.lo"], out["gemma.dec.lat.hi"] = red(max(lat)), red(min(lat))
    out["gemma.dec.cost.lo"], out["gemma.dec.cost.hi"] = red(max(cost)), red(min(cost))
    out["gemma.dec.q.lo"], out["gemma.dec.q.hi"] = red(max(dq), 1), red(min(dq), 1)
    t = "gl42"
    rb = g.get(t, "gemma.rb-static.q80l10c10")
    ratios = [g.get(t, c)[1] / rb[1] for c in ("gemma.ap.pw053.lor", "gemma.br.t03.lor")]
    out["gemma.42.trained.lo"], out["gemma.42.trained.hi"] = f"{min(ratios):.1f}", f"{max(ratios):.1f}"
    out["gemma.42.br0.vs.uniform"] = f"{g.get(t, 'gemma.br.t00.lor')[1] / g.get(t, 'gemma.rb-static.uniform')[1]:.1f}"
    rbs = [c for k, c in GEMMA_SETTINGS if k.startswith("rb.")]          # by key: RB-single's names share the prefix
    for t in ("gl30", "gl42"):
        for c in ("gemma.ap.pw053.lor", "gemma.br.t03.lor", "gemma.br.t00.lor", "gemma.disp.lor"):
            v = g.get(t, c)
            if not any(g.get(t, r)[0] >= v[0] and g.get(t, r)[1] < v[1] for r in rbs):
                raise SystemExit(f"Gemma {t}: no RouteBalance setting of at least {c}'s quality is faster")
    if out.res["gemma.rb.beaten"] != "0":
        raise SystemExit("Gemma: a baseline setting beats a RouteBalance setting")
    for t in GEMMA_RATES:
        c = g.get(t, "gemma.rb-static.q80l10c10")[3] / g.get(t, "gemma.ap.pw080.lor")[3] - 1
        out[f"gemma.{rate(t)}.rbap.costgap"] = gap_up(c, 1)                       # "within": rounded up
        out[f"gemma.{rate(t)}.rbap.costred"] = red(c)
        mix = {c: g.cell[(t, c)]["model_mix"] for c in ("gemma.br.t05.lor", "gemma.rb-static.q80l10c10")}
        if not mix["gemma.br.t05.lor"]["google/gemma-3-27b-it"] > mix["gemma.rb-static.q80l10c10"]["google/gemma-3-27b-it"]:
            raise SystemExit(f"Gemma {t}: BEST-Route 0.5 does not send more requests to the 27B")
        for w in ("q80l10c10", "q50l30c20", "uniform"):
            a = g.cell[(t, f"gemma.rb-static.{w}")]["model_mix"].get("google/gemma-3-1b-it", 0.0)
            d = g.cell[(t, f"gemma.rb-static-dec.{w}")]["model_mix"].get("google/gemma-3-1b-it", 0.0)
            if not a > d:
                raise SystemExit(f"Gemma {t} {w}: RouteBalance does not send more requests to the 1B than RB-decoupled")
    out["gemma.rbap.q.max"] = gap_up(max(GEMMA_NUM["rbap.q"]), 1)
    out["gemma.br05.latx.lo"] = f"{min(GEMMA_NUM['br05.latx']):.1f}"
    out["gemma.br05.latx.hi"] = f"{max(GEMMA_NUM['br05.latx']):.1f}"
    out["gemma.br05.costx.lo"] = f"{min(GEMMA_NUM['br05.costx']):.1f}"
    out["gemma.br05.costx.hi"] = f"{max(GEMMA_NUM['br05.costx']):.1f}"
    out["gemma.br05.q.lo"] = inc(min(GEMMA_NUM["br05.q"]), 1)
    out["gemma.br05.q.hi"] = inc(max(GEMMA_NUM["br05.q"]), 1)
    # the corpora: per-model mean quality, and how often Gemma's 12B is as good as its 27B
    gm, qm = corpus["gemma"]["means"], corpus["qwen"]["means"]
    out["corpus.gemma.1b"] = f"{gm['google/gemma-3-1b-it']:.3f}"
    larger = [gm[m] for m in gm if not m.endswith("1b-it")]
    out["corpus.gemma.larger.lo"], out["corpus.gemma.larger.hi"] = f"{min(larger):.3f}", f"{max(larger):.3f}"
    out["corpus.qwen.3b"] = f"{qm['Qwen/Qwen2.5-3B']:.3f}"
    out["corpus.qwen.7b"] = f"{qm['Qwen/Qwen2.5-7B']:.3f}"
    out["corpus.gemma.12b27b.share"] = share(corpus["gemma"]["at_least"]["google/gemma-3-12b-it>=google/gemma-3-27b-it"])
    out["corpus.gemma.12b27b.gap"] = f"{gm['google/gemma-3-27b-it'] - gm['google/gemma-3-12b-it']:.3f}"


def shipped(q: Points, out: Out) -> str:
    """The routers one call at a time (as released) against 32 calls at once, on LOR at 12/30/42 (appendix)."""
    rows = []
    for kind, vals in (("ap", AP), ("br", BR)):
        for v in vals:
            cells = []
            for t in THREE:
                pooled, serial = f"qwen.{kind}.{v}.lor", f"qwen.{kind}.{v}.lor-shipped"
                wp = q.cell[(t, pooled)]["summary"]["overhead_ms"]["router_ms"]["mean"] / 1000
                ws = q.cell[(t, serial)]["summary"]["overhead_ms"]["router_ms"]["mean"] / 1000
                cells += [f"{q.get(t, pooled)[1]:.2f}", f"{q.get(t, serial)[1]:.2f}", f"{wp:.2f}", f"{ws:.2f}"]
                if t in ("l30", "l42") and not round(ws, 2) > round(wp, 2):     # the appendix: "one call at a time
                    raise SystemExit(f"{serial} at {t}: one call at a time does not wait longer, as printed "  # waits
                                     f"({ws:.2f} against {wp:.2f} s)")                    # longer at 30 and 42 req/s"
            rows.append(f"{label(f'qwen.{kind}.{v}.lor').replace(' / LOR', '')} & " + " & ".join(cells) + r" \\")
    return table(
        "tab:shipped", r"The routers on LOR with up to 32 calls at once, as in the main comparison (32), and one call "
        r"at a time (1): mean E2E latency and mean router wait, in seconds.",
        "l" + "rrrr" * 3, r"Router & \multicolumn{4}{c}{12\,req/s} & \multicolumn{4}{c}{30\,req/s} & "
        r"\multicolumn{4}{c}{42\,req/s} \\ & E2E 32 & E2E 1 & wait 32 & wait 1 & E2E 32 & E2E 1 & wait 32 & wait 1 & "
        r"E2E 32 & E2E 1 & wait 32 & wait 1", rows, wide=True)


def reqnorm(q: Points, out: Out) -> str:
    """In-batch against per-request normalization at the eight presets, beside RB-decoupled (appendix); the text's
    range of the per-request variant's latency against RB-decoupled's over the eight presets at the three rates."""
    big = "Qwen/Qwen2.5-72B"
    rows, rel = [], []
    for w in EIGHT:
        cells = []
        for t in THREE:
            a, b = q.get(t, f"qwen.rb-static.{w}"), q.get(t, f"qwen.rb-static-reqnorm.{w}")
            d = q.get(t, f"qwen.rb-static-dec.{w}")
            ma = q.cell[(t, f"qwen.rb-static.{w}")]["model_mix"].get(big, 0.0)
            mb = q.cell[(t, f"qwen.rb-static-reqnorm.{w}")]["model_mix"].get(big, 0.0)
            cells += [f"{a[1]:.2f}", f"{b[1]:.2f}", f"{d[1]:.2f}", share(ma), share(mb)]
            rel.append(b[1] / d[1] - 1)
        rows.append(f"{wlabel(w)} & " + " & ".join(cells) + r" \\")
    out["reqnorm.dec.lat.lo"], out["reqnorm.dec.lat.hi"] = pct(min(rel)), pct(max(rel))
    # the text: at 80/10/10 the per-request variant sends the 72B the share RB-decoupled does, at every rate
    if {share(q.cell[(t, "qwen.rb-static-reqnorm.q80l10c10")]["model_mix"][big]) for t in RATES} != \
            {share(q.cell[(t, "qwen.rb-static-dec.q80l10c10")]["model_mix"][big]) for t in RATES}:
        raise SystemExit("the text says the per-request variant at 80/10/10 sends the 72B the share RB-decoupled does")
    return table(
        "tab:reqnorm", r"\sys with its scores normalized over the round and over each request's own candidates, and "
        r"RB-decoupled (Dec.): mean E2E latency (s) and the 72B model's share of requests (\%).",
        "l" + "rrrrr" * 3, r"Preset & \multicolumn{5}{c}{12\,req/s} & \multicolumn{5}{c}{30\,req/s} & "
        r"\multicolumn{5}{c}{42\,req/s} \\ \cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16} "
        r"& \multicolumn{3}{c}{E2E} & \multicolumn{2}{c}{72B} & "
        r"\multicolumn{3}{c}{E2E} & \multicolumn{2}{c}{72B} & \multicolumn{3}{c}{E2E} & \multicolumn{2}{c}{72B} \\ "
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}\cmidrule(lr){12-14}"
        r"\cmidrule(lr){15-16} "
        r"& round & request & Dec. & round & request & round & request & Dec. & round & request & round & request & "
        r"Dec. & round & request", rows, wide=True, colsep="4pt")


def calibration(fleet: dict, tpot: dict, prefill: dict, out: Out) -> str:
    """The TPOT calibration per tier (appendix): the levels held, the samples, the medians per level, the constant."""
    rows = []
    levels = sorted({b for v in tpot["provenance"]["curve_samples"].values() for b, _ in v})
    for tier in TIER_ORDER:
        model = fleet["tiers"][tier]["model"].split("/")[-1]
        measured = dict((b, ms_) for b, ms_ in tpot["provenance"]["curve_measured"][tier])
        cells = [f"{measured[b]:.1f}" if b in measured else "--" for b in levels]
        pf = prefill["prefill"][tier]
        rows.append(f"{model} & " + " & ".join(cells) + f" & {tpot['constant'][tier]:.1f} & "
                    f"{tpot['provenance']['constant_samples'][tier]:,}".replace(",", "{,}")
                    + f" & {pf['r2']:.2f} \\\\")
    out["calib.levels"] = ", ".join(str(b) for b in levels[:-1]) + f" and {levels[-1]}"
    big, small = (dict(tpot["provenance"]["curve_measured"][t]) for t in ("qwen2.5-14b_v100", "qwen2.5-7b_a30"))
    shared = sorted(set(big) & set(small))
    faster = [b for b in shared if big[b] < small[b]]
    if not faster or faster != shared[:len(faster)]:                   # the background: "the 14B on four V100s decodes
        raise SystemExit(f"the 14B is faster than the 7B at levels {faster}, not at every level up to one")  # faster
    out["calib.14b.faster.upto"] = str(faster[-1])                     # than the 7B on one A30 with up to N running"
    missing = {(fleet["tiers"][tier]["model"].split("/")[-1], b) for tier in TIER_ORDER for b in levels
               if b not in dict(tpot["provenance"]["curve_measured"][tier])}
    if missing != {("Qwen2.5-72B", 96)}:                                 # the appendix: "Every level holds except
        raise SystemExit(f"the calibration's missing levels are {sorted(missing)}, not the 72B at 96 alone")  # 96 ..."
    return table(
        "tab:calib", r"The TPOT calibration: the median per-token decode time (ms) with $n$ requests running on each "
        r"instance, the per-token decode time $\tau$ that \sys uses (the median of all the tier's samples over every "
        r"level), the number of samples, and the $R^2$ of the prefill line.",
        "l" + "r" * (len(levels) + 3), r"Model & " + " & ".join(f"$n{{=}}{b}$" for b in levels)
        + r" & $\tau$ & Samples & Prefill $R^2$", rows, wide=True)   # nine columns: across the two-column appendix


def no_ties(t: str, here: dict) -> None:
    """No two points of a rate tie on quality, mean E2E and cost. The figures' legend reads "filled: beaten by no other
    point", exact with or without ties. The check stays: a tie would also hide one marker under another."""
    seen = {}
    for c, v in here.items():
        key = (v[0], v[1], v[3])
        if key in seen:
            raise SystemExit(f"{t}: {c} and {seen[key]} tie on quality, mean E2E and cost: the legend would be inexact")
        seen[key] = c


def gemma_flags(g: Points, plot: dict) -> None:
    """For the Gemma figure: RouteBalance's settings and the baseline settings (RB-decoupled and RB-single left out, as
    in the Qwen figure) at each rate, and whether any of them beats each point."""
    cfgs = [c for k, c in GEMMA_SETTINGS if not k.startswith(("dec.", "single."))]
    pts = {}
    for t in GEMMA_RATES:
        here = {c: g.get(t, c) for c in cfgs}
        no_ties(t, here)
        pts[t] = {c: {"q": v[0], "e2e": v[1], "p99": v[2], "cost": v[3],
                      "beaten": any(beats(o, v) for oc, o in here.items() if oc != c)} for c, v in here.items()}
    plot["gemma_points"] = pts


def frontier_flags(q: Points, plot: dict) -> None:
    """For the figures: every main-grid point at each rate, and whether any point of the rate beats it."""
    pts = {}
    for t in RATES:
        cfgs = [f"qwen.rb-static.{w}" for w in EIGHT] + main_baselines()
        here = {c: q.get(t, c) for c in cfgs}
        no_ties(t, here)
        pts[t] = {c: {"q": v[0], "e2e": v[1], "p99": v[2], "cost": v[3],
                      "beaten": any(beats(o, v) for oc, o in here.items() if oc != c)} for c, v in here.items()}
    plot["points"] = pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interim", action="store_true",
                    help="allow a spread file whose repetitions are not complete (a hidden \\pending mark, counted)")
    a = ap.parse_args()
    d = verified(a.data)
    for rep_name in ("qwen_report.json", "gemma_report.json"):
        if d[rep_name]["provenance"]["rules"]["sha256"] != d["prereg_rules.json.sha256"]:
            raise SystemExit(f"{rep_name} was made with other rules than data/prereg_rules.json")
    tolerance = d["prereg_rules.json"]["equal_quality"]["tolerance"]
    parity = d["parity.json"]["results"]
    for key, (rep_name, group) in PARITY_REQUIRED.items():
        if parity[key]["rc"] != 0 or not parity[key]["output"][-1].startswith("PASS"):
            raise SystemExit(f"parity {key}: not passed")
        runs = {cid for cid, c in d[rep_name]["cells"].items() if c["group"] == group}
        if set(parity[key]["runs"]) != runs:
            raise SystemExit(f"parity {key} covers other runs than the report's {group} group")
    q = Points(d["qwen_report.json"], ("qwen", "qwen-ablation", "qwen-budget-30", "qwen-budget-45", "qwen-budget-75",
                                       "qwen-vsr"))
    g = Points(d["gemma_report.json"], ("gemma",))
    out, plot = Out(), {}
    out["eq.tolerance"] = f"{tolerance:g}"          # the preregistered tolerance above the highest preset (§5.1)
    dominance(q, out, plot)
    equal_quality(q, out, tolerance)
    spread_table = spread(d, q, g, d["prereg_rules.json.sha256"], tolerance, a.interim, out)
    single(q, g, out)                               # its guards and values; its rows are decoupled()'s now
    tables = {
        "gen_main": main_table(q, out),
        "gen_open": open_table(q, plot),
        "gen_decoupled": decoupled(q, out, plot),
        "gen_overhead": overhead(q, out),
        "gen_ablation": ablations(q, out),
        "gen_burst": bursts(q, out),
        "gen_budget": budget(q, out),
        "gen_gemma": gemma(g, out, plot),
        "gen_vsr": vsr(d["vsr_summary.json"], q, out),
        "gen_cluster": cluster(d["fleet_qwen.json"], d["tpot_qwen.json"], d["prefill_qwen.json"], out),
        "gen_shipped": shipped(q, out),
        "gen_reqnorm": reqnorm(q, out),
        "gen_calib": calibration(d["fleet_qwen.json"], d["tpot_qwen.json"], d["prefill_qwen.json"], out),
        "gen_spread": spread_table,
        "gen_tail": tail(q, g, tolerance, out),
        "gen_roundrange": round_range(d["round_range.json"], d["fleet_qwen.json"], q, out),
    }
    for key, rs in (("dec.q80l10c10.{}.cost", THREE), ("dec.q80l10c10.{}.lat", THREE), ("single.q80.{}.lat", THREE),
                    ("abl.wlat0.{}.latinc", THREE), ("abl.wlat0.{}.costinc", THREE),
                    ("abl.promptblind.{}.latx", THREE), ("ovh.{}.total", RATES),
                    ("tail.qwen.{}.median", RATES)):                         # ranges the text gives as "A--B"
        vals = [float(out.res[key.format(rate(t))].replace("{,}", "")) for t in rs]
        if min(vals) != vals[0] or max(vals) != vals[-1]:
            raise SystemExit(f"{key}: the text gives the range from {rate(rs[0])} to {rate(rs[-1])} req/s, and a rate "
                             f"in between lies outside it: {vals}")
    for k in ("lat", "cost"):                       # the spread's seed 1 must give the first-seed report's counts
        for n in (("dec", f"dec.{k}.lower"), ("single", f"single.{k}.lower")):
            if out.res[f"{n[1]}.s1"] != out.res[n[1]]:
                raise SystemExit(f"{n[1]}: seed 1 from the spread file differs from the first-seed report's")
    for fam in ("qwen", "gemma"):        # the repeated runs' summary in §5.6: each per-seed value's lowest and highest
        for kind in ("eq.lat", "beaten"):
            vals = sorted((out.res[f"spread.{fam}.{kind}.s{x}"] for x in (1, 2, 3)), key=lambda v: float(v.replace("{,}", "")))
            out[f"spread.{fam}.{kind}.lo"], out[f"spread.{fam}.{kind}.hi"] = vals[0], vals[-1]
    if len({out.res[f"spread.gemma.beaten.s{x}"] for x in (1, 2, 3)}) != 1:
        raise SystemExit("the text says the presets beat the same number of Gemma settings at every seed")
    if out.res["dec.n"] != out.res["single.n"]:          # one pair count for both variants in §5.6's sentence
        raise SystemExit("RB-decoupled and RB-single have different numbers of preset-rate pairs")
    for n in ("dec.lat.lower", "single.lat.lower"):      # "faster ... in at least N of the pairs" at every seed
        out[f"{n}.min3"] = str(min(int(out.res[f"{n}.s{x}"]) for x in (1, 2, 3)))
    mix(q, out, plot)
    round_accounting(d["round_shares.json"], d["round_range.json"], d["fleet_qwen.json"], q, out)
    zero = d["budget_judge.json"]["stores"]["store.s2-zero.jsonl"]          # the served texts the judge could not score
    out["budget.judge.zero"] = str(zero["records"])
    extras(q, g, d["vsr_summary.json"], d["corpus_stats.json"], out)
    vsr_failure(d["vsr_failure.json"], d["vsr_summary.json"], out)
    num = lambda key: float(out.res[key].replace("{,}", ""))
    for what, keys in (("the latency reduction at equal quality grows with the load", [f"eq.{rate(t)}.lat.median" for t in RATES]),
                       ("AvengersPro 0.80's latency rises with the load", [f"ap80.lat.{rate(t)}" for t in RATES]),
                       ("Gemma's latency advantage over AvengersPro grows with the load",
                        [f"gemma.{rate(t)}.rbap.lat" for t in GEMMA_RATES])):
        vals = [num(k) for k in keys]
        if any(b <= a for a, b in zip(vals, vals[1:])):
            raise SystemExit(f"the text says {what}: {vals}")
    for what, keys, down_ in (
            ("the 72B share at 80/10/10 falls from 12 to 42 req/s", [f"mix.rb.q80l10c10.{rate(t)}" for t in THREE], True),
            ("AvengersPro 0.80 waits longer under load", [f"ovh.{rate(t)}.ap" for t in RATES], False)):
        vals = [num(k) for k in keys]                  # from the first rate to the last, never back
        if not ((vals[0] > vals[-1]) if down_ else (vals[0] < vals[-1])) or any(
                (b > a) if down_ else (b < a) for a, b in zip(vals, vals[1:])):
            raise SystemExit(f"the text says {what}: {vals}")
    if out.res["ovh.br.peak.rate"] == "42" or not (num("ovh.12.br") < num("ovh.br.peak") > num("ovh.42.br")) \
            or not num("ovh.42.br") > num("ovh.12.br"):                 # a router "waits longer under load"
        raise SystemExit("the text gives BEST-Route 0.5's wait rising to a peak below 42 req/s and falling at 42")
    if not num("tpot.14b") < num("tpot.7b"):
        raise SystemExit("the background says the 14B on four V100s decodes faster than the 7B on one A30")
    if not num("budget.b75.rb.ex") > num("budget.b75.filter.ex"):
        raise SystemExit("the text says the budget filter lowers the exhaustion rate")
    for tier in d["fleet_gemma.json"]["tiers"].values():
        size = tier["model"].split("-")[2]
        out[f"price.gemma.{size}"] = f"\\${1e6 * tier['cost_in']:.2f}/\\${1e6 * tier['cost_out']:.2f}"
    for k, v in OPEN_OUT.items():
        out[k] = v
    frontier_flags(q, plot)
    gemma_flags(g, plot)
    lines = ["% Generated by tools/gen_results.py from data/ (R12). Do not edit; \\res{key} reads these."]
    lines += [f"\\resdef{{{k}}}{{{v}}}" for k, v in sorted(out.res.items())]
    (a.out / "generated" / "results.tex").write_text("\n".join(lines) + "\n")
    (a.out / "generated" / "plot_points.json").write_text(json.dumps(plot, indent=1, sort_keys=True) + "\n")
    for name, text in tables.items():
        (a.out / "tables" / f"{name}.tex").write_text(text)
    print(f"{len(out.res)} values, {len(tables)} tables")


if __name__ == "__main__":
    main()
