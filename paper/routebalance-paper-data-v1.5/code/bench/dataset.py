"""The evaluation prompts in send order, and the prompt token count every system receives.

Send order. The rows are read in file order and permuted by Random(0).shuffle(range(n_rows)), as
`random.seed(0); random.shuffle(rows)` would; a run of n requests sends its first n entries (no oversampling).
  fixed     that order for every seed
  per_seed  run seed s uses Random(s - 1): seed 1 is the fixed order, seeds 2, 3, ... permute it
            (the paper's runs: seeds vary the prompt order as well as the arrivals)

Templating. The client sends the RAW prompt; the scheduler passes it to vLLM's
chat endpoint as the single user message and vLLM applies the chat template once.

Prompt token count. `prompt_tokens` is the number of tokens the serving engine prefills for that message:
    len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      tokenize=True))
with the dataset's reference tokenizer (named in the scope). It is computed once before the run and sent with
every request; the scheduler's latency and cost estimates read it and no system counts tokens itself, so
every system sees the same number, and it is what the backend prefills. The backend's own
count (vLLM usage.prompt_tokens) comes back in every reply, so the analysis can check that the two agree.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from pydantic import Field

from bench.schema import Strict, read_jsonl
from routebalance.config import ConfigError

Order = Literal["fixed", "per_seed"]


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, str]], *, add_generation_prompt: bool,
                            tokenize: bool) -> Any: ...


class Prompt(Strict):
    row: int = Field(ge=0)               # 0-based line of the dataset file
    prompt_id: str                       # the row's "id"
    text: str                            # the raw user prompt, exactly as in the file
    prompt_tokens: int = Field(ge=1)
    budget_cost: Optional[float]         # null unless the dataset has a budget table (W7b)


class PromptSet(Strict):
    dataset_sha256: str
    tokenizer_id: str
    order: Order
    seed: int
    prompts: list[Prompt]
    budgets_sha256: Optional[str] = None  # default-ok: set by attach_budgets, the budget sweep only


def read_rows(path: Path) -> list[tuple[str, str]]:
    """(id, prompt) for every line, in file order. Every line is a JSON object with a string id and a
    non-empty string prompt; ids and prompts are unique (quality is looked up by prompt text)."""
    rows: list[tuple[str, str]] = []
    for lineno, rec in read_jsonl(path):
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
                      prompt_tokens=chat_prompt_tokens(tokenizer, rows[i][1]), budget_cost=None) for i in chosen]
    return PromptSet(dataset_sha256=file_sha256(path), tokenizer_id=tokenizer_id, order=order, seed=seed,
                     prompts=prompts)


def load_hf_tokenizer(tokenizer_id: str) -> ChatTokenizer:
    from transformers import AutoTokenizer  # imported here: only the benchmark host needs transformers
    return AutoTokenizer.from_pretrained(tokenizer_id)


def attach_budgets(prompts: PromptSet, budgets: Path, dataset: Path) -> PromptSet:
    """Each prompt's cost budget (USD) from a budget table: JSONL, one line per prompt of the dataset, in the
    dataset's order, {"id": <the row's id>, "budget_cost": USD > 0, ...} (its other
    fields document how the budget was built and are not read). A table that does not cover exactly
    the dataset's ids, in its order, or a budget that is not a positive finite number, is refused."""
    ids = [rid for rid, _ in read_rows(dataset)]
    table: list[tuple[str, float]] = []
    for lineno, rec in read_jsonl(budgets):
        cost = rec.get("budget_cost") if isinstance(rec, dict) else None
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str) or isinstance(cost, bool) \
                or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ConfigError(f"{budgets}:{lineno}: a budget line needs a string id and a positive budget_cost")
        table.append((rec["id"], float(cost)))
    if [rid for rid, _ in table] != ids:
        raise ConfigError(f"{budgets} does not list exactly the ids of {dataset}, in its order")
    by_id = dict(table)
    return prompts.model_copy(update={
        "budgets_sha256": file_sha256(budgets),
        "prompts": [p.model_copy(update={"budget_cost": by_id[p.prompt_id]}) for p in prompts.prompts]})
