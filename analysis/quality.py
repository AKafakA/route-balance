"""Judged quality of a served response: the DeepEval score of (prompt, model) in the scored corpus (D14, D62).

As in v2 and v3, each model answered each prompt once offline and the judge scored that answer; a served
request is looked up by (prompt, model), whatever text was served (the score is taken as load-independent):
  matched     the prompt is in a corpus and that model has a score under the judge key. 0.0 is a real score.
  corpus_gap  the prompt is present but that model has no score: counted, never averaged, not a failure
  unmatched   the prompt is in no corpus, or no model of its record is scored: the cell's quality is refused
Prompts and model names match exactly (the client sends the dataset text unchanged; the served name is the
scored name). A scored corpus is JSONL, one prompt per line:
  {"prompt": str, "models": {<model>: {"llm_judge_scores": {<judge_key>: score in [0, 1], ...}, ...}, ...}, ...}
A prompt in two records, across all corpora loaded together, is an error. The campaign render pins each
family's corpus (scope.json scored_corpora); tests/equivalence re-aggregates the banked v3 cells with it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional, Sequence

from routebalance.config import ConfigError, loads_strict

Outcome = Literal["matched", "corpus_gap", "unmatched"]


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


class ScoredCorpus:
    """Scores under one judge key, by (prompt, model)."""

    def __init__(self, scores: dict[str, dict[str, float]]):
        self._scores = scores
        self.models = frozenset(m for per_model in scores.values() for m in per_model)   # scored at least once

    def __len__(self) -> int:
        return len(self._scores)

    def lookup(self, prompt: str, model: str) -> tuple[Outcome, Optional[float]]:
        per_model = self._scores.get(prompt)
        if per_model is None:
            return "unmatched", None
        score = per_model.get(model)
        if score is None:
            return "corpus_gap", None
        return "matched", score

    @classmethod
    def load(cls, paths: Sequence[Path], judge_key: str) -> "ScoredCorpus":
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
