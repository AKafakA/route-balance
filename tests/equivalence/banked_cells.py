"""Read access to the banked v3 data, and the v3 conventions needed only to reproduce v3 numbers.

The oracle root mirrors the source trees: <root>/rb-exp/v3-exp/cells/<dir>/ holds v3 cells and their
DOMINANCE_l<rate>.csv; the eval set and the scored corpora are found by file name under <root>/Block/data and
pinned by md5 (the digests the v3 cell provenance records).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from analysis.metrics import Outcome

EVAL_SET = ("best-route-v3-test-3534-eval.jsonl", "15ed0e67ffa50e0d17955127c144a566")
TEST_CORPUS = ("test_scored_filtered.jsonl", "0318916065c1c7713cd20c0002574d0e")
JUDGE_KEY = "deepeval-llama3.1-8b-it_reference"
V3_TOKENIZER = "Qwen/Qwen2.5-3B"

# v3's price table (USD per 1M output tokens, paper Table 1), used only here to reproduce v3's cost column;
# the rebuild reads prices from the fleet config in USD per token.
V3_PRICE_OUT_PER_1M = {"Qwen/Qwen2.5-3B": 0.06, "Qwen/Qwen2.5-7B": 0.07, "Qwen/Qwen2.5-14B": 0.15,
                       "Qwen/Qwen2.5-72B": 0.40}
V3_PRICES = {m: p / 1e6 for m, p in V3_PRICE_OUT_PER_1M.items()}

# v3 sent the client-templated prompt; its analysis recovered the user text with this regex (dominance.py)
_USER_BLOCK = re.compile(r"<\|im_start\|>user\n(.*?)(?:\n?<\|im_end\|>|\Z)", re.DOTALL)


def unwrap_chat(prompt: str) -> str:
    if "<|im_start|>user" not in prompt:
        return prompt.strip()
    m = _USER_BLOCK.search(prompt)
    return (m.group(1) if m else prompt).strip()


class Oracle:
    def __init__(self, root: Path):
        self.root = Path(root)

    def cells_dir(self, name: str) -> Path:
        return self.root / "rb-exp" / "v3-exp" / "cells" / name

    def data_file(self, spec: tuple[str, str]) -> Path:
        name, md5 = spec
        found = sorted((self.root / "Block" / "data").rglob(name))
        assert len(found) == 1, f"expected exactly one {name} under {self.root}/Block/data, found {found}"
        assert hashlib.md5(found[0].read_bytes()).hexdigest() == md5, f"{found[0]} is not the v3 file"
        return found[0]

    def dominance_rows(self, cells_dir: str, rate: int) -> list[dict]:
        with open(self.cells_dir(cells_dir) / f"DOMINANCE_l{rate}.csv", newline="") as fh:
            return list(csv.DictReader(fh))


def load_banked_cell(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def banked_outcomes(cell: dict) -> list[Outcome]:
    """A v3 cell as the rebuild's metrics see it: v3 took a request as successful when its error was empty."""
    out = []
    for r in cell["response_details"]:
        ok = not r["error"]
        out.append(Outcome(prompt=unwrap_chat(r["prompt"]), ok=ok, model=r["model"] if ok else None,
                           text=None,        # v3 scored the (prompt, model) pair (V3PairCorpus), not the served text
                           output_tokens=r["output_len"] if ok else None, e2e_s=r["e2el"], server_latency_s=None,
                           timings_ms=None, prompt_tokens_sent=None, prompt_tokens_served=None))
    return out
