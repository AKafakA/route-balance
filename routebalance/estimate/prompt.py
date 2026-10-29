"""The estimator reads the user's text, not the chat template around it.

The KNN index was built from raw prompts, while served prompts arrive chat-templated; embedding the
template destroys neighbour discrimination. This is the only
place served text is turned back into estimator text, so no caller can reach the index with a template.

The rules: only the first user turn is kept and the system
turn is dropped; if the first user turn has no end tag the whole text is used; the result is stripped.
Idempotent on raw prompts.
"""
from __future__ import annotations

import re

_QWEN_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
_GEMMA_USER = re.compile(r"<start_of_turn>user\n(.*?)<end_of_turn>", re.S)


def unwrap_chat_prompt(p: str) -> str:
    if not p:
        return p
    for marker, user_turn in (("<|im_start|>user", _QWEN_USER), ("<start_of_turn>user", _GEMMA_USER)):
        if marker in p:
            m = user_turn.search(p)
            return (m.group(1) if m else p).strip()
    return p.strip()
