"""analysis.metrics re-aggregates the banked v3 grid cells to the v3 DOMINANCE tables.

Every row of DOMINANCE_l30.csv (31 cells, including the five fixtures of the port spec) and DOMINANCE_l6.csv
(39 cells) is recomputed from its cell's response_details and duration with the rebuild's summary. Counts,
quality, coverage, mean and p99 latency and throughput must be bit-exact. Decode cost must agree to 1e-12
relative: v3 priced in USD per 1M tokens and summed sequentially, the rebuild prices per token from the fleet
config and sums exactly (math.fsum), so the two differ in the last bits. The only v3 convention used is its
price table.
"""
import pytest

from analysis.metrics import summarize
from analysis.quality import ScoredCorpus
from tests.equivalence.banked_cells import (JUDGE_KEY, TEST_CORPUS, V3_PRICES, banked_outcomes,
                                            load_banked_cell)

FIXTURES = {"wq0.5_wl0.3_wc0.2_lambda30_n3534", "pt_sq_l30_n3534", "pt_rand_l30_n3534", "ap_pw07_rr_l30_n3534",
            "br4_t07_rr_l30_n3534"}
COST_REL = 1e-12
COUNTS = ("n", "err", "n_zero")
EXACT = ("q", "coverage", "mean_ms", "p99_ms", "thru")


@pytest.fixture(scope="module")
def corpus(oracle):
    return ScoredCorpus.load([oracle.data_file(TEST_CORPUS)], JUDGE_KEY)


def _rebuilt(name, cell, corpus):
    s = summarize(name, banked_outcomes(cell), cell["duration"], corpus, V3_PRICES, None, [])     # no SLO in v3's table
    return s, {"n": s.n_ok, "err": s.n_failed, "n_zero": s.n_zero, "q": s.quality, "coverage": s.coverage,
               "cost_decode": s.cost_decode_per_1k, "mean_ms": s.e2e_ms.mean, "p99_ms": s.e2e_ms.p99,
               "thru": s.throughput_rps}


@pytest.mark.parametrize("cells_dir,rate,n_cells", [("s2_p1_l30", 30, 31), ("s2_p1_l6", 6, 39)])
def test_the_dominance_table_is_reproduced(oracle, corpus, cells_dir, rate, n_cells):
    rows = oracle.dominance_rows(cells_dir, rate)
    assert len(rows) == n_cells
    if rate == 30:
        assert FIXTURES <= {r["cell"] for r in rows}
    wrong, worst_cost = [], 0.0
    for row in rows:
        s, got = _rebuilt(row["cell"], load_banked_cell(oracle.cells_dir(cells_dir) / f"{row['cell']}.json"), corpus)
        if s.problems:
            wrong.append((row["cell"], "problems", s.problems))
        wrong += [(row["cell"], k, got[k], row[k]) for k in COUNTS if got[k] != int(row[k])]
        wrong += [(row["cell"], k, got[k], row[k]) for k in EXACT if got[k] != float(row[k])]
        rel = abs(got["cost_decode"] - float(row["cost_decode"])) / float(row["cost_decode"])
        worst_cost = max(worst_cost, rel)
        if rel > COST_REL:
            wrong.append((row["cell"], "cost_decode", got["cost_decode"], row["cost_decode"]))
    assert wrong == [], f"{len(wrong)} mismatches: {wrong[:10]}"
    print(f"{cells_dir}: {len(rows)}/{len(rows)} rows reproduced; {', '.join(COUNTS + EXACT)} bit-exact; "
          f"worst decode-cost relative error {worst_cost:.2e}")
