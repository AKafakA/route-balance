"""The estimator's pure-Python pieces: the chat-template unwrap and md5-pinned artifact reads.

The numeric KNN tests need numpy and faiss and live in tests/equivalence (test_knn_numeric.py).
"""
import hashlib

import pytest

from routebalance.estimate.artifacts import ArtifactError, read_verified
from routebalance.estimate.prompt import unwrap_chat_prompt

QWEN = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
GEMMA = "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"


@pytest.mark.parametrize("wrapped", [QWEN.format("  What is 2+2?\n"), GEMMA.format("What is 2+2?  "),
                                     "  What is 2+2?\n"])
def test_unwrap_returns_the_stripped_user_text(wrapped):
    assert unwrap_chat_prompt(wrapped) == "What is 2+2?"


def test_unwrap_is_idempotent():
    once = unwrap_chat_prompt(QWEN.format("Explain KV caches."))
    assert unwrap_chat_prompt(once) == once == "Explain KV caches."


def test_unwrap_keeps_only_the_first_user_turn():
    two_turns = QWEN.format("first") + "<|im_start|>user\nsecond<|im_end|>\n"
    assert unwrap_chat_prompt(two_turns) == "first"


def test_unwrap_without_an_end_tag_keeps_the_whole_text():
    # v3 behaviour: no closing tag means no match, and the templated text is used as it is (stripped)
    truncated = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nno end tag "
    assert unwrap_chat_prompt(truncated) == truncated.strip()


def test_unwrap_prefers_the_qwen_template_when_both_markers_appear():
    text = QWEN.format("<start_of_turn>user\ninner<end_of_turn>")
    assert unwrap_chat_prompt(text) == "<start_of_turn>user\ninner<end_of_turn>"


def test_unwrap_multiline_user_text():
    assert unwrap_chat_prompt(QWEN.format("line 1\nline 2")) == "line 1\nline 2"


def test_empty_prompt_is_returned_as_is():
    assert unwrap_chat_prompt("") == ""


def test_read_verified_returns_the_bytes_that_match(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"payload")
    assert read_verified(f, hashlib.md5(b"payload").hexdigest()) == b"payload"


def test_read_verified_rejects_a_different_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"payload")
    with pytest.raises(ArtifactError, match="does not match"):
        read_verified(f, hashlib.md5(b"other").hexdigest())


def test_read_verified_reports_a_missing_file(tmp_path):
    with pytest.raises(ArtifactError, match="cannot read"):
        read_verified(tmp_path / "absent.pkl", "0" * 32)
