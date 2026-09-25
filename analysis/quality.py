"""Judged quality of a served response: the DeepEval score of record, zero-safe (D14, D39).

The score store (D39) holds one judged score per byte-identical response. JSONL, one line per text:
  {"prompt": str, "model": str, "text_sha256": <sha256 hex of the text's UTF-8 bytes>, "text": str,
   "scores": {<judge_key>: score in [0, 1], ...}}
tools/judge_served.py writes it (merging the judge's output into a new store file); the W0 references
(tools/reference_generation.py) seed it before the first cell. A served response is looked up by
(prompt, model, sha256(text)):
  matched   that exact text has a score under the judge key. 0.0 is a real score.
  unscored  it has none. No other text's score stands in, not even one for the same prompt and model:
            the cell's quality is unavailable (analysis/metrics.py) until this text is judged.
Prompts and model names match exactly (the client sends the dataset text unchanged; the served name is the
scored name). A key scored twice under one judge key across the loaded files is an error (a second judge adds
its own key, in the same line or another), and so is a line whose text does not hash to its text_sha256. A
text scored under other judge keys only is unscored for this key.

V3PairCorpus reproduces the banked v3 tables only (tests/equivalence): v3 looked a score up by (prompt,
model) in a scored corpus, whatever text had been served, with these outcomes:
  matched     the prompt is in a corpus and that model has a score under the judge key
  corpus_gap  the prompt is present but that model has no score: counted, never averaged, not a failure
  unmatched   the prompt is in no corpus, or no model of its record is scored: the cell's quality is refused
A scored corpus is JSONL, one prompt per line:
  {"prompt": str, "models": {<model>: {"llm_judge_scores": {<judge_key>: score, ...}, ...}, ...}, ...}
A prompt in two records, across all corpora loaded together, is an error. It is never used for an MLSys
cell: analysis.metrics.summarize_cell takes a ScoreStore only.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence

from routebalance.config import ConfigError, loads_strict

Outcome = Literal["matched", "unscored", "corpus_gap", "unmatched"]
STORE_KEYS = frozenset({"prompt", "model", "text_sha256", "text", "scores"})


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class QualitySource(Protocol):
    models: frozenset[str]      # models with at least one score

    def lookup(self, prompt: str, model: str, text: Optional[str]) -> tuple[Outcome, Optional[float]]: ...


def _score(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{where}: score {value!r} is not a number in [0, 1]")
    return float(value)


def read_jsonl(path: Path):
    """(line number, record) for every line; the final newline is optional."""
    try:
        lines = Path(path).read_bytes().decode("utf-8").split("\n")   # not splitlines(): U+2028 may sit in a string
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if lines and lines[-1] == "":
        lines.pop()                                        # the final newline
    for lineno, line in enumerate(lines, start=1):
        try:
            yield lineno, loads_strict(line)
        except ConfigError as e:
            raise ConfigError(f"{path}:{lineno}: {e}") from e


def store_key(prompt: str, model: str, sha: str) -> tuple[str, str, str]:
    return prompt, model, sha


def parse_store_line(rec: object, where: str) -> tuple[tuple[str, str, str], str, dict[str, float]]:
    """(key, text, scores) of one store line; anything malformed raises."""
    if not isinstance(rec, dict) or set(rec) != STORE_KEYS:
        raise ConfigError(f"{where}: a store line has exactly the keys {sorted(STORE_KEYS)}")
    prompt, model, sha, text, scores = (rec[k] for k in ("prompt", "model", "text_sha256", "text", "scores"))
    if not (isinstance(prompt, str) and prompt) or not (isinstance(model, str) and model):
        raise ConfigError(f"{where}: prompt and model must be non-empty strings")
    if not isinstance(text, str) or not isinstance(sha, str) or text_sha256(text) != sha:
        raise ConfigError(f"{where}: the text does not hash to its text_sha256")
    if not isinstance(scores, dict) or not scores:
        raise ConfigError(f"{where}: scores must be a non-empty object")
    return store_key(prompt, model, sha), text, {k: _score(v, f"{where} {k}") for k, v in scores.items()}


class ScoreStore:
    """Scores under one judge key, by (prompt, model, text hash)."""

    def __init__(self, scores: dict[tuple[str, str, str], float]):
        self._scores = scores
        self.models = frozenset(m for _, m, _ in scores)

    def __len__(self) -> int:
        return len(self._scores)

    def lookup(self, prompt: str, model: str, text: Optional[str]) -> tuple[Outcome, Optional[float]]:
        if text is None:
            raise ValueError("the score store looks up the served text; none was given")
        score = self._scores.get(store_key(prompt, model, text_sha256(text)))
        return ("unscored", None) if score is None else ("matched", score)

    @classmethod
    def load(cls, paths: Sequence[Path], judge_key: str) -> "ScoreStore":
        """A text may be scored by several judges, in any files; one judge scores a text once."""
        scores: dict[tuple[str, str, str], float] = {}
        seen: set[tuple[tuple[str, str, str], str]] = set()
        for path in paths:
            for lineno, rec in read_jsonl(path):
                key, _, judged = parse_store_line(rec, f"{path}:{lineno}")
                for jk in judged:
                    if (key, jk) in seen:
                        raise ConfigError(f"{path}:{lineno}: this (prompt, model, text) is already in the store "
                                          f"under {jk}")
                    seen.add((key, jk))
                if judge_key in judged:
                    scores[key] = judged[judge_key]
        return cls(scores)


class V3PairCorpus:
    """The v3 lookup by (prompt, model), for re-aggregating banked v3 cells only (see the module docstring)."""

    def __init__(self, scores: dict[str, dict[str, float]]):
        self._scores = scores
        self.models = frozenset(m for per_model in scores.values() for m in per_model)   # scored at least once

    def __len__(self) -> int:
        return len(self._scores)

    def lookup(self, prompt: str, model: str, text: Optional[str]) -> tuple[Outcome, Optional[float]]:
        del text                          # v3 scored the pair, whatever text was served
        per_model = self._scores.get(prompt)
        if per_model is None:
            return "unmatched", None
        score = per_model.get(model)
        if score is None:
            return "corpus_gap", None
        return "matched", score

    @classmethod
    def load(cls, paths: Sequence[Path], judge_key: str) -> "V3PairCorpus":
        scores: dict[str, dict[str, float]] = {}
        seen: set[str] = set()
        for path in paths:
            for lineno, rec in read_jsonl(path):
                where = f"{path}:{lineno}"
                prompt, per_model = _scored(rec, judge_key, where)
                if prompt in seen:
                    raise ConfigError(f"{where}: prompt already in a scored record")
                seen.add(prompt)
                if per_model:
                    scores[prompt] = per_model
        return cls(scores)


def _scored(rec, judge_key: str, where: str) -> tuple[str, dict[str, float]]:
    if not isinstance(rec, dict) or not isinstance(rec.get("prompt"), str) or not rec["prompt"]:
        raise ConfigError(f"{where}: a record needs a non-empty string 'prompt'")
    models = rec.get("models")
    if not isinstance(models, dict):
        raise ConfigError(f"{where}: 'models' must be an object")
    per_model: dict[str, float] = {}
    for model, mrec in models.items():
        judged = mrec.get("llm_judge_scores") if isinstance(mrec, dict) else None
        if not isinstance(judged, dict):
            raise ConfigError(f"{where}: model {model!r} has no 'llm_judge_scores' object")
        if judge_key not in judged:
            continue                                       # a corpus gap for this model
        per_model[model] = _score(judged[judge_key], f"{where}: {model!r} {judge_key}")
    return rec["prompt"], per_model
