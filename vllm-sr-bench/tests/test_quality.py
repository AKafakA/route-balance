import json

import pytest

from vsrbench.quality import training_means

KEY = "deepeval-llama3.1-8b-it_reference"


def rows(tmp_path, scores):
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps({"models": {m: {"llm_judge_scores": {KEY: s}} for m, s in r.items()}}) + "\n"
                         for r in scores))
    return p


def test_means_over_every_row(tmp_path):
    got = training_means(rows(tmp_path, [{"a": 0.2, "b": 1.0}, {"a": 0.4, "b": 0.0}]), ["a", "b"])
    assert got["means"] == pytest.approx({"a": 0.3, "b": 0.5}) and got["rows"] == 2 and len(got["sha256"]) == 64


def test_a_row_without_a_score_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no deepeval"):
        training_means(rows(tmp_path, [{"a": 0.2, "b": 1.0}, {"a": 0.4}]), ["a", "b"])


def test_a_score_outside_the_range_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not in"):
        training_means(rows(tmp_path, [{"a": 1.5}]), ["a"])


def test_a_json_string_holding_a_line_separator_stays_one_row(tmp_path):
    """U+2028 (and U+2029, U+0085) inside a JSON string is not a line end: the 28 Sep Qwen training rows hold 37 (10-02)."""
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps({"response": f"a{sep}b", "models": {"a": {"llm_judge_scores": {KEY: 0.5}}}},
                                    ensure_ascii=False) + "\n" for sep in (" ", " ", "\u0085")), encoding="utf-8")
    got = training_means(p, ["a"])
    assert got["rows"] == 3 and got["means"] == pytest.approx({"a": 0.5})
