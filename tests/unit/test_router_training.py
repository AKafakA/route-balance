"""The routers' training inputs and AvengersPro artifacts (D68-D72): the row filter, upstream AvengersPro's
per-cluster balance ranking with per-query costs, and the BEST-Route candidates ordered by price."""
import json

import numpy as np
import pytest

from routebalance.config import ConfigError
from tools import router_training as rt

KEY = rt.JUDGE_KEY
QWEN = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"]
GEMMA = ["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]


def _row(i, scores, p=10, outs=None, models=QWEN):
    outs = outs or {m: 5 for m in models}
    return {"request_id": f"r{i}", "prompt": f"p{i}", "input_len": p,
            "models": {m: {"response": f"{m}-{i}", "output_length": outs[m],
                           **({"llm_judge_scores": {KEY: scores[m]}} if scores.get(m) is not None else {})}
                       for m in models}}


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_the_real_fleets_prices_and_gemmas_price_order():
    qwen = rt.fleet_prices("configs/fleet_qwen.json")
    assert qwen["Qwen/Qwen2.5-72B"] == (3.8e-07, 4e-07) and set(qwen) == set(QWEN)
    assert rt.price_order(QWEN, qwen) == QWEN                        # Qwen: price follows size (v3's order)
    gemma = rt.fleet_prices("configs/fleet_gemma.json")
    assert rt.price_order(GEMMA, gemma) == GEMMA                     # Gemma (D208's prices): size order too


def test_the_price_order_reorders_by_mean_price_and_keeps_ties_in_the_given_order():
    # a ladder whose prices do not follow size, with a tie (as v3's Gemma prices: 4B cheapest, 1B = 12B by mean)
    prices = {GEMMA[0]: (1e-07, 1e-07), GEMMA[1]: (4e-08, 8e-08), GEMMA[2]: (5e-08, 1.5e-07), GEMMA[3]: (8e-08, 1.6e-07)}
    assert rt.price_order(GEMMA, prices) == [GEMMA[1], GEMMA[0], GEMMA[2], GEMMA[3]]
    assert rt.price_order([GEMMA[2], GEMMA[0], GEMMA[1], GEMMA[3]], prices) == [GEMMA[1], GEMMA[2], GEMMA[0], GEMMA[3]]

def test_a_model_with_two_prices_or_models_not_the_fleets_are_refused(tmp_path):
    f = tmp_path / "fleet.json"
    f.write_text(json.dumps({"tiers": {"a": {"model": "m", "cost_in": 1e-7, "cost_out": 1e-7},
                                       "b": {"model": "m", "cost_in": 2e-7, "cost_out": 1e-7}}, "instances": []}))
    with pytest.raises(ConfigError, match="two prices"):
        rt.fleet_prices(f)
    with pytest.raises(ConfigError, match="once"):
        rt.check_models(QWEN[:3], rt.fleet_prices("configs/fleet_qwen.json"))


def test_filter_keeps_only_prompts_every_model_scored():
    rows = [_row(0, dict.fromkeys(QWEN, 0.5)), _row(1, {**dict.fromkeys(QWEN, 0.5), QWEN[2]: None})]
    kept, dropped = rt.keep_scored(rows, QWEN, KEY)
    assert [r["request_id"] for r in kept] == ["r0"] and dropped == ["r1"]


def test_the_cost_is_per_query_at_the_fleets_prices():
    prices = {m: (1e-7 * (j + 1), 2e-7 * (j + 1)) for j, m in enumerate(QWEN)}
    q, c = rt.matrices([_row(0, dict.fromkeys(QWEN, 0.25), p=100, outs={m: 10 * (j + 1) for j, m in enumerate(QWEN)})],
                       QWEN, prices, KEY)
    assert q.tolist() == [[0.25] * 4]
    assert c[0].tolist() == pytest.approx([100 * 1e-7 * k + 10 * k * 2e-7 * k for k in (1, 2, 3, 4)])


def test_balance_ranking_is_upstreams_score():
    m = ["a", "b", "c"]
    quality = np.array([[0.2, 0.5, 0.8], [0.4, 0.5, 0.6]])       # means 0.3 0.5 0.7 -> normalized 0, 0.5, 1
    cost = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])          # cost/max 0.25 0.5 1 -> cost score 0.75 0.5 0
    assert rt.balance_ranking(quality, cost, m, 1.0) == ["c", "b", "a"]      # accuracy only
    assert rt.balance_ranking(quality, cost, m, 0.0) == ["a", "b", "c"]      # cost only
    # pw 0.5: a 0.375, b 0.5, c 0.5: b and c tie, and a tie keeps the given order
    assert rt.balance_ranking(quality, cost, m, 0.5) == ["b", "c", "a"]
    same = np.full((2, 3), 0.5)                                  # all equal accuracy -> 1.0 each (upstream)
    assert rt.balance_ranking(same, cost, m, 0.9) == ["a", "b", "c"]


def test_the_means_are_upstreams_so_equal_accuracies_stay_equal():
    # Codex review: eight prompts where every model averages exactly 0.2; a column reduction gives 0.19999999999999998
    # for three of them, and min-max normalization would make 7B the clear winner. Upstream ranks 3B first.
    quality = np.array([[0.2, 0.0, 0.2, 0.2]] * 4 + [[0.2, 0.4, 0.2, 0.2]] * 4)
    prices = rt.fleet_prices("configs/fleet_qwen.json")
    cost = np.array([[10 * prices[m][0] + 5 * prices[m][1] for m in QWEN]] * 8)
    assert rt.balance_ranking(quality, cost, QWEN, 0.70) == QWEN


def test_every_cluster_needs_prompts():
    labels = np.array([0, 0, 2])
    with pytest.raises(ConfigError, match="cluster 1 is empty"):
        rt.cluster_rankings(labels, np.ones((3, 2)), np.ones((3, 2)), ["a", "b"], 0.5, 3)


def test_the_best_route_candidates_put_the_cheapest_first_so_a_tie_labels_it():
    gemma = rt.fleet_prices("configs/fleet_gemma.json")
    order = rt.price_order(GEMMA, gemma)
    row = _row(0, {GEMMA[0]: 0.9, GEMMA[1]: 0.9, GEMMA[2]: 0.4, GEMMA[3]: 0.9}, models=GEMMA)
    ex = rt.best_route_example(row, 0, order, GEMMA[-1], KEY)
    assert [c["model"] for c in ex["candidates"]] == ["gemma-3-1b-it_bo1", "gemma-3-4b-it_bo1",
                                                      "gemma-3-12b-it_bo1", "gemma-3-27b-it_bo1"]
    scores = [c["scores"]["armoRM_scores"] for c in ex["candidates"]]
    assert max(range(len(scores)), key=lambda i: scores[i]) == 0   # v3's wrapper labels the first best: 1B
    assert ex["output"] == f"{GEMMA[-1]}-0" and ex["instruction"] == "p0" and ex["input"] == ""
    assert ex["candidates"][0]["token_num_prompt"] == 10 and ex["candidates"][0]["token_num_responses"] == 5


def test_the_validation_set_is_a_seeded_tenth_of_the_training_prompts():
    rows = [{"id": i} for i in range(100)]
    train, val = rt.split(rows, 0.1, 42)
    assert len(val) == 10 and len(train) == 90 and not {r["id"] for r in train} & {r["id"] for r in val}
    assert rt.split(rows, 0.1, 42) == (train, val) and rt.split(rows, 0.1, 7) != (train, val)


def test_the_commands_write_new_files_only(tmp_path, monkeypatch):
    rows = [_row(i, {m: (i % 3) / 2 for m in QWEN}) for i in range(12)] + [_row(12, {m: None for m in QWEN})]
    corpus = _jsonl(tmp_path / "scored.jsonl", rows)
    out = tmp_path / "rows.jsonl"
    key = ["--judge-key", "deepeval-llama3.1-8b-it_reference"]
    argv = ["filter", "--corpus", str(corpus), "--fleet", "configs/fleet_qwen.json", *key, "--out", str(out)]
    assert rt.main(argv) == 0 and len(out.read_text().splitlines()) == 12
    with pytest.raises(ConfigError, match="exists"):
        rt.main(argv)
    br = ["best-route-data", "--corpus", str(out), "--fleet", "configs/fleet_qwen.json", *key, "--models", *QWEN,
          "--out", str(tmp_path / "br")]
    assert rt.main(br) == 0
    labels = json.loads((tmp_path / "br" / "labels.json").read_text())
    assert labels["label_models"] == QWEN and labels["train"] + labels["val"] == 12
    with pytest.raises(ConfigError, match="lack a score"):
        rt.main([*br[:2], str(corpus), *br[3:-1], str(tmp_path / "br2")])

    fake_kmeans = {"stands in for": "the fitted k-means"}      # picklable, like the real one
    monkeypatch.setattr(rt, "embed_and_cluster", lambda prompts, e, r, k: (fake_kmeans, np.arange(len(prompts)) % k))
    av = ["avengers-pro", "--corpus", str(out), "--fleet", "configs/fleet_qwen.json", *key, "--models", *QWEN,
          "--family", "qwen", "--pw", "0.25", "0.80", "--embedder", "e", "--embedder-revision", "r", "--n-clusters", "4",
          "--out-root", str(tmp_path / "ap")]
    pytest.importorskip("joblib")
    assert rt.main(av) == 0
    d = tmp_path / "ap" / "avengers_pro_qwen_pw0.25"
    cfg = json.loads((d / "config.json").read_text())
    ranks = json.loads((d / "cluster_rankings.json").read_text())
    assert cfg["cost"] == "per_query_per_cluster" and cfg["train_rows"] == 12 and cfg["cost_sensitivity"] == 0.75
    assert set(ranks) == {"0", "1", "2", "3"} and all(sorted(r) == sorted(QWEN) for r in ranks.values())
    assert (d / "kmeans.joblib").is_file() and (tmp_path / "ap" / "avengers_pro_qwen_pw0.80").is_dir()
    with pytest.raises(ConfigError, match="exist"):
        rt.main(av)



def test_a_row_without_an_id_or_a_string_price_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no request_id"):
        rt.row_id({"prompt": "p"}, 3)
    fleet = tmp_path / "fleet.json"
    fleet.write_text(json.dumps({"tiers": {"t": {"model": "m", "cost_in": "1e-7", "cost_out": 2e-7}}}))
    with pytest.raises(ConfigError, match="numeric prices"):
        rt.fleet_prices(fleet)
