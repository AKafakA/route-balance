"""The score store scores the text actually served (D39); the v3 pair corpus keeps its three outcomes apart."""
import hashlib
import json

import pytest

from analysis.quality import ScoreStore, V3PairCorpus, text_sha256
from routebalance.config import ConfigError

KEY = "judge-ref"


def _line(prompt, model, text, **scores):
    return {"prompt": prompt, "model": model, "text_sha256": text_sha256(text), "text": text, "scores": scores}


def _store(tmp_path, *lines, name="store.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    return path


# ---------------------------------------------------------------- the served-response store (D39)

def test_a_score_belongs_to_one_byte_identical_text(tmp_path):
    s = ScoreStore.load([_store(tmp_path, _line("p1", "small", "Paris.", **{KEY: 0.0}),
                                _line("p1", "large", "Paris is the capital.", **{KEY: 0.9}))], KEY)
    assert s.lookup("p1", "small", "Paris.") == ("matched", 0.0)          # zero is a real score
    assert s.lookup("p1", "large", "Paris is the capital.") == ("matched", 0.9)
    assert s.lookup("p1", "small", "Paris. ") == ("unscored", None)       # one byte more
    assert s.lookup("p1", "small", "Paris is the capital.") == ("unscored", None)   # another model's text
    assert s.lookup("p2", "small", "Paris.") == ("unscored", None)
    assert s.models == frozenset({"small", "large"}) and len(s) == 2


def test_no_other_text_of_the_pair_stands_in(tmp_path):
    s = ScoreStore.load([_store(tmp_path, _line("p1", "small", "a", **{KEY: 0.4}),
                                _line("p1", "small", "b", **{KEY: 0.6}))], KEY)
    assert s.lookup("p1", "small", "a") == ("matched", 0.4) and s.lookup("p1", "small", "b") == ("matched", 0.6)
    assert s.lookup("p1", "small", "c") == ("unscored", None)


def test_the_hash_is_of_the_utf8_bytes():
    assert text_sha256("é") == hashlib.sha256("é".encode("utf-8")).hexdigest()


def test_a_text_scored_only_by_another_judge_is_unscored(tmp_path):
    s = ScoreStore.load([_store(tmp_path, _line("p1", "small", "a", other=0.4))], KEY)
    assert s.lookup("p1", "small", "a") == ("unscored", None) and s.models == frozenset()


def test_the_lookup_needs_the_text(tmp_path):
    s = ScoreStore.load([_store(tmp_path, _line("p1", "small", "a", **{KEY: 0.4}))], KEY)
    with pytest.raises(ValueError, match="served text"):
        s.lookup("p1", "small", None)


def test_store_files_load_together_and_a_key_twice_is_an_error(tmp_path):
    a = _store(tmp_path, _line("p1", "small", "a", **{KEY: 0.4}), name="1.jsonl")
    b = _store(tmp_path, _line("p2", "small", "a", **{KEY: 0.5}), name="2.jsonl")
    assert len(ScoreStore.load([a, b], KEY)) == 2
    c = _store(tmp_path, _line("p1", "small", "a", **{KEY: 0.4}), name="3.jsonl")
    with pytest.raises(ConfigError, match="already in the store"):
        ScoreStore.load([a, c], KEY)
    second_judge = _store(tmp_path, _line("p1", "small", "a", other=0.7), name="4.jsonl")
    assert ScoreStore.load([a, second_judge], KEY).lookup("p1", "small", "a") == ("matched", 0.4)
    assert ScoreStore.load([a, second_judge], "other").lookup("p1", "small", "a") == ("matched", 0.7)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(text_sha256="0" * 64),
    lambda r: r.update(text="changed"),
    lambda r: r.update(scores={KEY: 1.5}),
    lambda r: r.update(scores={KEY: True}),
    lambda r: r.update(scores={KEY: "0.5"}),
    lambda r: r.update(scores={}),
    lambda r: r.update(prompt=""),
    lambda r: r.update(extra=1),
    lambda r: r.pop("text"),
])
def test_malformed_store_lines_fail(tmp_path, mutate):
    rec = _line("p1", "small", "a", **{KEY: 0.4})
    mutate(rec)
    with pytest.raises(ConfigError):
        ScoreStore.load([_store(tmp_path, rec)], KEY)


# ---------------------------------------------------------------- the v3 pair corpus (banked v3 cells only)

def _rec(prompt, **scores):
    return {"prompt": prompt, "models": {m: {"llm_judge_scores": {} if s is None else {KEY: s, "other": 0.9}}
                                         for m, s in scores.items()}}


def _corpus(tmp_path, *records, name="c.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_v3_zero_is_a_score_and_gaps_are_not(tmp_path):
    c = V3PairCorpus.load([_corpus(tmp_path, _rec("p1", small=0.0, large=0.8), _rec("p2", small=None, large=0.5))], KEY)
    assert c.lookup("p1", "small", None) == ("matched", 0.0)
    assert c.lookup("p1", "large", "whatever was served") == ("matched", 0.8)
    assert c.lookup("p2", "small", None) == ("corpus_gap", None)
    assert c.lookup("p2", "medium", None) == ("corpus_gap", None)
    assert c.lookup("p3", "small", None) == ("unmatched", None)


def test_v3_prompts_match_on_exact_text(tmp_path):
    c = V3PairCorpus.load([_corpus(tmp_path, _rec("p1", small=0.3))], KEY)
    assert c.lookup(" p1", "small", None)[0] == "unmatched"
    assert c.lookup("p1\n", "small", None)[0] == "unmatched"


def test_v3_a_record_with_no_scored_model_is_unmatched(tmp_path):
    c = V3PairCorpus.load([_corpus(tmp_path, _rec("p1", small=None, large=None))], KEY)
    assert c.lookup("p1", "small", None) == ("unmatched", None) and len(c) == 0


def test_v3_corpora_load_together(tmp_path):
    c = V3PairCorpus.load([_corpus(tmp_path, _rec("a", small=0.1), name="train.jsonl"),
                           _corpus(tmp_path, _rec("b", small=0.2), name="test.jsonl")], KEY)
    assert c.lookup("a", "small", None) == ("matched", 0.1) and c.lookup("b", "small", None) == ("matched", 0.2)


def test_v3_a_prompt_in_two_records_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="already"):
        V3PairCorpus.load([_corpus(tmp_path, _rec("a", small=0.1), name="1.jsonl"),
                           _corpus(tmp_path, _rec("a", small=0.1), name="2.jsonl")], KEY)


@pytest.mark.parametrize("record", [
    _rec("a", small=1.5), _rec("a", small=-0.1), _rec("a", small=True), _rec("a", small="0.5"),
    {"prompt": "", "models": {}}, {"prompt": "a"}, {"prompt": "a", "models": {"small": {}}},
    {"prompt": "a", "models": {"small": None}},
])
def test_v3_malformed_records_fail(tmp_path, record):
    with pytest.raises(ConfigError):
        V3PairCorpus.load([_corpus(tmp_path, record)], KEY)
