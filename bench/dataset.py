"""The evaluation prompts in send order, and the prompt token count every system receives.

Send order. v3 read the JSONL in file order and then ran `random.seed(0); random.shuffle(rows)` (the vLLM
fork's CustomDataset), with seed 0 for every run whatever the run's --seed: the dataset loader was never
given the seed. A shuffle depends only on the list length, so the v3 order is the permutation
Random(0).shuffle(range(n_rows)); a run of n requests sends its first n entries (no oversampling).
  fixed     that order for every seed; reproduces every banked v3 cell
  per_seed  run seed s uses Random(s - 1): seed 1 is the v3 order, seeds 2, 3, ... permute it
            (owner decision for the MLSys runs: seeds vary the prompt order as well as the arrivals)

Templating (owner decision 2026-09-24). The client sends the RAW prompt; the scheduler passes it to vLLM's
chat endpoint as the single user message and vLLM applies the chat template once. v3 templated in the client
and again on the server, which added 19 prompt tokens to every request with the Qwen2.5 template.

Prompt token count. `prompt_tokens` is the number of tokens the serving engine prefills for that message:
    len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      tokenize=True))
with the dataset's reference tokenizer (named in the scope). It is computed once before the run and sent with
every request; the scheduler's latency and cost estimates read it and no system counts tokens itself, so
every system sees the same number. With the Qwen2.5 tokenizer it equals v3's recorded input_len on all 3534
prompts (v3 counted the once-templated text), and it is now also what the backend prefills. The backend's own
count (vLLM usage.prompt_tokens) comes back in every reply, so the analysis can check that the two agree.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from bench.schema import Strict
from routebalance.config import ConfigError, loads_strict

Order = Literal["fixed", "per_seed"]


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, str]], *, add_generation_prompt: bool,
                            tokenize: bool) -> Any: ...


class Prompt(Strict):
    row: int = Field(ge=0)               # 0-based line of the dataset file
    prompt_id: str                       # the row's "id"
    text: str                            # the raw user prompt, exactly as in the file
    prompt_tokens: int = Field(ge=1)


class PromptSet(Strict):
    dataset_sha256: str
    tokenizer_id: str
    order: Order
    seed: int
    prompts: list[Prompt]


def read_rows(path: Path) -> list[tuple[str, str]]:
    """(id, prompt) for every line, in file order. Every line is a JSON object with a string id and a
    non-empty string prompt; ids and prompts are unique (quality is looked up by prompt text)."""
    rows: list[tuple[str, str]] = []
    try:
        lines = Path(path).read_bytes().decode("utf-8").split("\n")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if lines and lines[-1] == "":
        lines.pop()                      # the final newline
    for lineno, line in enumerate(lines, start=1):
        try:
            rec = loads_strict(line)
        except ConfigError as e:
            raise ConfigError(f"{path}:{lineno}: {e}") from e
        if not isinstance(rec, dict):
            raise ConfigError(f"{path}:{lineno}: not a JSON object")
        pid, text = rec.get("id"), rec.get("prompt")
        if not isinstance(pid, str) or not isinstance(text, str) or not text:
            raise ConfigError(f"{path}:{lineno}: needs a string 'id' and a non-empty string 'prompt'")
        rows.append((pid, text))
    for what, values in (("id", [r[0] for r in rows]), ("prompt", [r[1] for r in rows])):
        if len(set(values)) != len(values):
            raise ConfigError(f"{path}: duplicate {what} values")
    return rows


def prompt_order(n_rows: int, order: Order, seed: int) -> list[int]:
    """The send order as dataset row indices (a permutation of range(n_rows))."""
    if order == "fixed":
        shuffle_seed = 0
    elif order == "per_seed":
        if seed < 1:
            raise ValueError(f"per_seed order needs a run seed >= 1, got {seed}")
        shuffle_seed = seed - 1
    else:
        raise ValueError(f"unknown prompt order {order!r}")
    rows = list(range(n_rows))
    random.Random(shuffle_seed).shuffle(rows)
    return rows


def order_digest(rows: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()[:16]


def chat_prompt_tokens(tokenizer: ChatTokenizer, text: str) -> int:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                        tokenize=True)
    return len(ids)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompts(path: Path, n_requests: int, order: Order, seed: int, tokenizer: ChatTokenizer,
                 tokenizer_id: str) -> PromptSet:
    rows = read_rows(path)
    if not 1 <= n_requests <= len(rows):
        raise ConfigError(f"{path} has {len(rows)} prompts; a run of {n_requests} requests cannot be served")
    chosen = prompt_order(len(rows), order, seed)[:n_requests]
    prompts = [Prompt(row=i, prompt_id=rows[i][0], text=rows[i][1],
                      prompt_tokens=chat_prompt_tokens(tokenizer, rows[i][1])) for i in chosen]
    return PromptSet(dataset_sha256=file_sha256(path), tokenizer_id=tokenizer_id, order=order, seed=seed,
                     prompts=prompts)


def load_hf_tokenizer(tokenizer_id: str) -> ChatTokenizer:
    from transformers import AutoTokenizer  # imported here: only the benchmark host needs transformers
    return AutoTokenizer.from_pretrained(tokenizer_id)
