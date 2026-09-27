"""Prompt order, raw prompts and the prompt token count must be exactly as documented in bench/dataset.py."""
import hashlib
import json
import random

import pytest

from bench.dataset import chat_prompt_tokens, load_prompts, order_digest, prompt_order, read_rows
from routebalance.config import ConfigError
from tests.unit.fakes import FakeTokenizer

ROWS = [{"id": f"p{i}", "prompt": f"question {i}?", "source": "s"} for i in range(12)]
ROWS[3]["prompt"] = "  keeps outer whitespace\nand inner newlines  "


def _write(tmp_path, rows=ROWS, text=None):
    path = tmp_path / "eval.jsonl"
    path.write_text(text if text is not None else "".join(json.dumps(r) + "\n" for r in rows))
    return path


def _shuffled(n, seed):
    rows = list(range(n))
    random.Random(seed).shuffle(rows)
    return rows


def test_fixed_order_is_the_seed_zero_shuffle_for_every_seed():
    for seed in (0, 1, 2, 7):
        assert prompt_order(12, "fixed", seed) == _shuffled(12, 0)


def test_per_seed_order_starts_at_the_v3_order():
    assert prompt_order(12, "per_seed", 1) == prompt_order(12, "fixed", 1)
    assert prompt_order(12, "per_seed", 2) == _shuffled(12, 1) != prompt_order(12, "fixed", 2)
    assert sorted(prompt_order(12, "per_seed", 3)) == list(range(12))
    with pytest.raises(ValueError):
        prompt_order(12, "per_seed", 0)
    with pytest.raises(ValueError):
        prompt_order(12, "random", 1)


def test_load_prompts_sends_raw_text_in_order(tmp_path):
    path = _write(tmp_path)
    tok = FakeTokenizer()
    ps = load_prompts(path, 10, "fixed", 1, tok, "tok-id")
    order = _shuffled(12, 0)[:10]
    assert [p.row for p in ps.prompts] == order
    assert [p.text for p in ps.prompts] == [ROWS[i]["prompt"] for i in order]      # unchanged, not templated
    assert [p.prompt_id for p in ps.prompts] == [ROWS[i]["id"] for i in order]
    assert ps.dataset_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (ps.order, ps.seed, ps.tokenizer_id) == ("fixed", 1, "tok-id")


def test_prompt_tokens_count_one_templated_user_message(tmp_path):
    tok = FakeTokenizer()
    ps = load_prompts(_write(tmp_path), 12, "per_seed", 2, tok, "tok-id")
    for p in ps.prompts:
        assert p.prompt_tokens == len("<u>" + p.text + "</u><a>")
    conversation, add_generation_prompt, tokenize = tok.calls[0]
    assert conversation == [{"role": "user", "content": ps.prompts[0].text}]
    assert add_generation_prompt is True and tokenize is True
    assert chat_prompt_tokens(tok, "abc") == len("<u>abc</u><a>")


@pytest.mark.parametrize("n", [0, 13])
def test_request_count_must_fit_the_dataset(tmp_path, n):
    with pytest.raises(ConfigError):
        load_prompts(_write(tmp_path), n, "fixed", 1, FakeTokenizer(), "tok-id")


@pytest.mark.parametrize("text", [
    '{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n',       # duplicate id
    '{"id": "a", "prompt": "x"}\n{"id": "b", "prompt": "x"}\n',       # duplicate prompt
    '{"id": "a"}\n',                                                   # no prompt
    '{"id": "a", "prompt": ""}\n',                                     # empty prompt
    '{"id": 1, "prompt": "x"}\n',                                      # id not a string
    '["a", "x"]\n',                                                    # not an object
    '{"id": "a", "prompt": "x"}\n\n{"id": "b", "prompt": "y"}\n',     # blank line
    '{"id": "a", "prompt": "x", "id": "b"}\n',                        # duplicate key
    '{"id": "a", "prompt": "x"\n',                                     # broken JSON
])
def test_malformed_datasets_fail(tmp_path, text):
    with pytest.raises(ConfigError):
        read_rows(_write(tmp_path, text=text))


def test_final_newline_is_optional(tmp_path):
    assert read_rows(_write(tmp_path, text='{"id": "a", "prompt": "x"}')) == [("a", "x")]


def test_order_digest_identifies_the_sequence():
    assert order_digest([0, 1, 2]) == order_digest([0, 1, 2]) != order_digest([0, 2, 1])
    assert order_digest([1, 23]) != order_digest([12, 3])
