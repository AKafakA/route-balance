"""Each model's quality evidence for vLLM-SR: its mean judged score on the training split, the labels
F-MRLB's estimator and the other baselines train on."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vsrbench.constants import JUDGE_KEY


def training_means(rows_path: Path, models: list[str], judge_key: str = JUDGE_KEY) -> dict:
    """{"means": {model: mean score}, "rows": n, "sha256": the file's}. Every row must score every model: a row
    without a model's score is an error, never a silent skip."""
    raw = Path(rows_path).read_bytes()
    sums, n = {m: 0.0 for m in models}, 0
    # newlines only: str.splitlines also breaks at U+2028 and other separators, which a JSON string may hold (the Qwen
    # training rows hold 37 in their answers)
    for i, line in enumerate(raw.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for m in models:
            try:
                score = row["models"][m]["llm_judge_scores"][judge_key]
            except (KeyError, TypeError) as e:
                raise ValueError(f"{rows_path}:{i}: no {judge_key} score for {m}") from e
            if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                raise ValueError(f"{rows_path}:{i}: {m} score {score!r} is not in [0, 1]")
            sums[m] += float(score)
        n += 1
    if n == 0:
        raise ValueError(f"{rows_path} holds no rows")
    return {"means": {m: sums[m] / n for m in models}, "rows": n, "sha256": hashlib.sha256(raw).hexdigest(),
            "judge_key": judge_key}
