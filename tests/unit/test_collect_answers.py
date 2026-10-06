"""the new scored corpus keeps each row's prompt fields and replaces every model's answer."""
import json

import pytest

from routebalance.config import ConfigError
from tools import collect_answers as ca

ROW = {"request_id": "r0", "prompt": "p0", "input_len": 20, "dataset": "gsm8k", "source": "s", "is_harmful": False,
       "reference_text": "ref", "models": {m: {"response": "old", "output_length": 9, "llm_judge_scores": {"k": 0.5},
                                              "similarity_score": 0.1, "is_truncated": False} for m in ("A", "B")}}


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _answer(prompt, model, text="new", ok=True):
    return {"prompt_id": "x", "prompt": prompt, "model": model, "success": ok, "error": "" if ok else "boom",
            "text": text, "output_tokens": 3}


def test_merge_replaces_the_answers_and_drops_what_the_old_text_decided(tmp_path):
    other = dict(ROW, request_id="r1", prompt="p1")
    corpus = _jsonl(tmp_path / "c.jsonl", [ROW, other])
    answers = _jsonl(tmp_path / "a.jsonl", [_answer("p0", "A", "a0"), _answer("p0", "B", "b0"), _answer("p1", "A")])
    assert ca.merge(corpus, answers, tmp_path / "new.jsonl") == (1, 1)        # p1 lacks B's answer: left out
    (row,) = [json.loads(x) for x in (tmp_path / "new.jsonl").read_text().splitlines()]
    assert {k: row[k] for k in ca.PROMPT_FIELDS} == {k: ROW[k] for k in ca.PROMPT_FIELDS}
    assert row["models"] == {"A": {"response": "a0", "output_length": 3, "llm_judge_scores": {}},
                             "B": {"response": "b0", "output_length": 3, "llm_judge_scores": {}}}


def test_merge_keeps_a_corpus_without_is_harmful_as_it_is(tmp_path):
    """Gemma's corpora carry no is_harmful: the merge copies it only when the corpus has it."""
    row = {k: v for k, v in ROW.items() if k != "is_harmful"}
    corpus = _jsonl(tmp_path / "c.jsonl", [row])
    answers = _jsonl(tmp_path / "a.jsonl", [_answer("p0", "A", "a0"), _answer("p0", "B", "b0")])
    assert ca.merge(corpus, answers, tmp_path / "new.jsonl") == (1, 0)
    (new,) = [json.loads(x) for x in (tmp_path / "new.jsonl").read_text().splitlines()]
    assert "is_harmful" not in new and {k: new[k] for k in row if k != "models"} == {k: row[k] for k in row if k != "models"}


def test_merge_refuses_a_corpus_with_is_harmful_on_some_rows_only(tmp_path):
    other = {k: v for k, v in dict(ROW, request_id="r1", prompt="p1").items() if k != "is_harmful"}
    corpus = _jsonl(tmp_path / "c.jsonl", [ROW, other])
    answers = _jsonl(tmp_path / "a.jsonl", [_answer(p, m) for p in ("p0", "p1") for m in ("A", "B")])
    with pytest.raises(ConfigError, match="some rows only"):
        ca.merge(corpus, answers, tmp_path / "new.jsonl")


def test_merge_still_needs_every_other_prompt_field(tmp_path):
    corpus = _jsonl(tmp_path / "c.jsonl", [{k: v for k, v in ROW.items() if k != "dataset"}])
    answers = _jsonl(tmp_path / "a.jsonl", [_answer("p0", "A"), _answer("p0", "B")])
    with pytest.raises(KeyError):
        ca.merge(corpus, answers, tmp_path / "new.jsonl")


@pytest.mark.parametrize("answers,match", [
    ([_answer("p0", "A", ok=False)], "failed answer"),
    ([_answer("zz", "A")], "not in"),
    ([_answer("p0", "C")], "not a model"),
    ([_answer("p0", "A"), _answer("p0", "A")], "twice"),
])
def test_merge_refuses_answers_it_cannot_place(tmp_path, answers, match):
    corpus = _jsonl(tmp_path / "c.jsonl", [ROW])
    with pytest.raises(ConfigError, match=match):
        ca.merge(corpus, _jsonl(tmp_path / "a.jsonl", answers), tmp_path / "new.jsonl")


def test_merge_refuses_a_corpus_with_a_duplicate_prompt(tmp_path):
    with pytest.raises(ConfigError, match="duplicate"):
        ca.merge(_jsonl(tmp_path / "c.jsonl", [ROW, ROW]), _jsonl(tmp_path / "a.jsonl", []), tmp_path / "n.jsonl")
