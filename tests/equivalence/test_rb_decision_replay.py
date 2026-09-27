"""The rebuilt objective picks v3's model on every banked RouteBalance decision.

v3 logged, per request, the latency that entered each candidate's score and the expected length it came
from (decision log); the cell holds the per-model quality estimates (all_model_scores) and the served model.
Re-scoring with routebalance.schedule.score (max_divide, v3's normalization) must choose the served model for
every request. The instance within the model is decided by queue depth and recency, which v3 did not log,
so the check is on the model; a request whose best score is shared by two models would be ambiguous and
fails the test.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from routebalance.schedule.score import Candidate, score

V3_SCHEDULER_CONFIG = ("scheduler_config_s2.json", "002413dae0ae8c1534e90d31e7ecdd12")   # found by name, pinned by md5
RATES = (6, 30)
_PRESET = re.compile(r"^wq([\d.]+)_wl([\d.]+)_wc([\d.]+)_lambda(\d+)_n3534\.json$")


def _tiers(oracle) -> dict[str, tuple[str, float]]:
    name, md5 = V3_SCHEDULER_CONFIG
    found = [p for p in (oracle.root / "Block" / "block" / "config").rglob(name)
             if hashlib.md5(p.read_bytes()).hexdigest() == md5]
    assert found, f"no {name} with md5 {md5} under the oracle"
    path = found[0]
    meta = json.loads(path.read_text())["instance_metadata"]
    return {itype: (m["model_name"], float(m["cost_per_output_token"])) for itype, m in meta.items()}


def _decisions(cells: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(cells.glob("decisions_*.jsonl")):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            if d.get("req"):
                out[d["req"]] = d["cand"]
    return out


def _cells(oracle, rate: int) -> list[Path]:
    return sorted(p for p in oracle.cells_dir(f"s2_p1_l{rate}").iterdir() if _PRESET.match(p.name))


@pytest.mark.parametrize("rate", RATES)
def test_every_banked_decision_picks_the_served_model(oracle, rate):
    tiers = _tiers(oracle)
    cells = _cells(oracle, rate)
    assert len(cells) >= 8, f"expected the 8 banked presets at λ{rate}, found {[c.name for c in cells]}"
    decisions = _decisions(oracle.cells_dir(f"s2_p1_l{rate}"))
    report = {}
    for path in cells:
        wq, wl, wc, _ = _PRESET.match(path.name).groups()
        rows = json.loads(path.read_text())["response_details"]
        checked = mismatched = ambiguous = missing = 0
        for r in rows:
            if r.get("error"):
                continue
            cand = decisions.get(r["request_id"])
            if cand is None:
                missing += 1
                continue
            cands = []
            for c in cand:
                model, cost_out = tiers[c["itype"]]
                cands.append(Candidate(c["inst"], model, float(r["all_model_scores"][model]),
                                       float(c["exp_len"]) * cost_out, float(c["lat_s"]), None, 0, 0.0))
            scored = score(cands, w_quality=float(wq), w_cost=float(wc), w_latency=float(wl),
                           normalization="max_divide")
            best = max(s.score for s in scored)
            winners = {s.candidate.model for s in scored if s.score == best}
            checked += 1
            ambiguous += len(winners) > 1
            mismatched += winners != {r["model"]}
        report[path.name] = dict(checked=checked, mismatched=mismatched, ambiguous=ambiguous, missing=missing)
    bad = {k: v for k, v in report.items() if v["mismatched"] or v["ambiguous"] or v["missing"] or not v["checked"]}
    assert not bad, f"λ{rate}: {bad}"
    assert sum(v["checked"] for v in report.values()) >= 8 * 3500
