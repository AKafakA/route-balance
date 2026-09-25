"""T3: replaying v3's own estimator batches reproduces what every banked v3 lambda=30 RouteBalance cell recorded.

Each response records all_model_scores (the KNN quality of every model) and predicted_length (the chosen
model's expected length), computed from the chat-templated prompt, so this also covers the unwrap. v3 estimated
each scheduler batch in one call, and each response records its batch (batch_id, batch_size). The batch matters:
faiss switches kernels at 20 queries, and the kernels differ by ~1e-7 in similarity, which the 1/(d + 1e-6)
neighbour weight amplifies when a corpus prompt is at distance ~0. So every batch is replayed as v3 ran it (its
requests in request order).

Pass: |diff| <= 1e-6 on quality and <= 1e-3 tokens on length (NaN fails) for at least 99.5% of rows, and every
other row is explained by one of:
  a near-tie between rank k and k + 1 (similarity gap < 1e-5): a neighbour-set flip;
  our estimate equals what v3 itself recorded for the same prompt in another cell (qualities within 1e-6 and,
  when this row's length is off, that cell chose the same model and recorded our length within 1e-3): v3 did
  not reproduce its own value here;
  a nearest neighbour at distance < 1e-5, where the weight turns float noise between machines into a weight
  change, with the difference bounded (quality <= 1e-5, length <= 1e-2 tokens).
"""
import json

import numpy as np

from routebalance.estimate.prompt import unwrap_chat_prompt
from tests.equivalence.encoder import real_estimator


def _rb_cells(oracle):
    cells = sorted(oracle.cells_dir("s2_p1_l30").glob("wq*_lambda30_n3534.json"))
    assert len(cells) >= 8, cells
    return cells


def _batches(cell) -> list:
    """The cell's estimator batches, each in request order, checked against the recorded batch size."""
    batches: dict = {}
    for r in json.loads(cell.read_text())["response_details"]:
        if not r["error"]:
            batches.setdefault(r["scheduling_overhead_breakdown"]["batch_id"], []).append(r)
    for members in batches.values():
        members.sort(key=lambda r: int(r["request_id"].rsplit("-", 1)[1]))
        assert len(members) == members[0]["scheduling_overhead_breakdown"]["batch_size"]
    return list(batches.values())


def _nearest(est, prompt):
    q = est._encoder.encode([unwrap_chat_prompt(prompt)], normalize_embeddings=True, batch_size=32)
    sims, _ = est._faiss.search(np.asarray(q, dtype=np.float32), est.k + 1)
    d = 1.0 - sims[0].astype(np.float64)
    return float(d[0]), float(d[est.k] - d[est.k - 1])


def _quality_diff(e, scores) -> float:
    """Largest |ours - recorded| over the models; NaN if any value is NaN (np.max propagates it)."""
    return float(np.max(np.abs(np.array([e[m].quality - q for m, q in scores.items()]))))


def _equals_v3_elsewhere(e, cell, r, dl, recorded) -> bool:
    for other_cell, other in recorded:
        if other_cell == cell or not _quality_diff(e, other["all_model_scores"]) <= 1e-6:
            continue
        if dl <= 1e-3 or (other["model"] == r["model"] and abs(e[r["model"]].length - other["predicted_length"]) <= 1e-3):
            return True
    return False


def test_banked_scores_and_lengths(oracle):
    est = real_estimator(oracle)
    rows, over, bitwise = 0, [], 0
    recorded: dict = {}
    for cell in _rb_cells(oracle):
        for members in _batches(cell):
            for r, e in zip(members, est.estimate_batch([r["prompt"] for r in members], 256), strict=True):
                recorded.setdefault(r["prompt"], []).append((cell.name, r))
                dq = _quality_diff(e, r["all_model_scores"])
                dl = abs(e[r["model"]].length - r["predicted_length"])
                rows += 1
                bitwise += dq == 0.0 and dl == 0.0
                if not (dq <= 1e-6 and dl <= 1e-3):
                    over.append((cell.name, r, e, dq, dl))
    print(f"T3: {rows} rows replayed in v3's batches; {bitwise} bitwise equal; {len(over)} outside tolerance")
    assert len(over) <= 0.005 * rows
    for cell, r, e, dq, dl in over:
        prompt = r["prompt"]
        assert np.isfinite(dq) and np.isfinite(dl), f"{cell}: non-finite estimate for {prompt[:80]!r}"
        v3_elsewhere = _equals_v3_elsewhere(e, cell, r, dl, recorded[prompt])
        d1, gap = _nearest(est, prompt)
        print(f"   {cell}: dq {dq:.3g}, dlen {dl:.3g}; equals v3 in another cell: {v3_elsewhere}; nearest distance "
              f"{d1:.3g}; rank-k gap {gap:.3g}")
        near_duplicate = d1 < 1e-5 and dq <= 1e-5 and dl <= 1e-2
        assert gap < 1e-5 or v3_elsewhere or near_duplicate, f"{cell}: unexplained difference on {prompt[:80]!r}"
