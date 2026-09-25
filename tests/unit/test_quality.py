"""The scored corpus looks a request up by (prompt, model) and keeps its three outcomes apart."""
import json

import pytest

from analysis.quality import ScoredCorpus
from routebalance.config import ConfigError

KEY = "judge-ref"


def _rec(prompt, **scores):
    return {"prompt": prompt, "models": {m: {"llm_judge_scores": {} if s is None else {KEY: s, "other": 0.9}}
                                         for m, s in scores.items()}}


def _corpus(tmp_path, *records, name="c.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_zero_is_a_score_and_gaps_are_not(tmp_path):
    c = ScoredCorpus.load([_corpus(tmp_path, _rec("p1", small=0.0, large=0.8), _rec("p2", small=None, large=0.5))], KEY)
    assert c.lookup("p1", "small") == ("matched", 0.0)
    assert c.lookup("p1", "large") == ("matched", 0.8)
    assert c.lookup("p2", "small") == ("corpus_gap", None)
    assert c.lookup("p2", "medium") == ("corpus_gap", None)
    assert c.lookup("p3", "small") == ("unmatched", None)


def test_prompts_match_on_exact_text(tmp_path):
    c = ScoredCorpus.load([_corpus(tmp_path, _rec("p1", small=0.3))], KEY)
    assert c.lookup(" p1", "small")[0] == "unmatched"
    assert c.lookup("p1\n", "small")[0] == "unmatched"


def test_a_record_with_no_scored_model_is_unmatched(tmp_path):
    c = ScoredCorpus.load([_corpus(tmp_path, _rec("p1", small=None, large=None))], KEY)
    assert c.lookup("p1", "small") == ("unmatched", None) and len(c) == 0


def test_corpora_load_together(tmp_path):
    c = ScoredCorpus.load([_corpus(tmp_path, _rec("a", small=0.1), name="train.jsonl"),
                           _corpus(tmp_path, _rec("b", small=0.2), name="test.jsonl")], KEY)
    assert c.lookup("a", "small") == ("matched", 0.1) and c.lookup("b", "small") == ("matched", 0.2)


def test_a_prompt_in_two_records_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="already"):
        ScoredCorpus.load([_corpus(tmp_path, _rec("a", small=0.1), name="1.jsonl"),
                           _corpus(tmp_path, _rec("a", small=0.1), name="2.jsonl")], KEY)


@pytest.mark.parametrize("record", [
    _rec("a", small=1.5), _rec("a", small=-0.1), _rec("a", small=True), _rec("a", small="0.5"),
    {"prompt": "", "models": {}}, {"prompt": "a"}, {"prompt": "a", "models": {"small": {}}},
    {"prompt": "a", "models": {"small": None}},
])
def test_malformed_records_fail(tmp_path, record):
    with pytest.raises(ConfigError):
        ScoredCorpus.load([_corpus(tmp_path, record)], KEY)
