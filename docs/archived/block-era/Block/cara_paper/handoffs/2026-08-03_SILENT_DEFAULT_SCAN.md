# Silent decision-feeding defaults — class scan (author-ordered 2026-08-03 ~19:00Z)

**The class**: a failure/missing-data path substituting a neutral-looking value that then
feeds a routing/dispatch/scoring DECISION, silently. Family: the `or`-zero score drop, the
`{}` stats masking (fixed), v1 BR first-listed fallback (fixed), v1 AP largest-model fallback
(fixed), era SQ head-pinning (the newest member, found today).

**Scan result**: 88 default-substitution sites in `cara_serve.py`; ~20 touch scheduling state
(`num_running`, `num_waiting`, `pending_decode_tokens`, `kv_cache_utilization`) that feeds
min/max/sort/score. Raw inventory: `grep -nE '\.get\(...,(0|0.0)\)|or 0' cara_serve.py`.

## Status of the class today

1. **Instance-level blindness: DEAD in current code.** `_fetch_all_instance_stats` raises on
   ANY missing/empty instance (`:997-1003`), so no `.get(...,0)` site ever sees an absent
   instance. The era head-pin path cannot execute under fail-loud — it returns ONLY if the
   parity toggle is chosen, scoped to those 16 cells.

2. **Key-level schema drift: THE LIVE RESIDUAL.** The fetch validates that each instance
   returned a non-empty dict — it does NOT validate per-key completeness. A backend
   returning a partial stats dict (schema drift, vLLM version skew) would flow through, and
   every `.get("num_waiting", 0)`-style site silently reads 0 → queue-blind dispatch/scoring
   with nothing recording it. Same shape as the instance-level bug, one level finer.

3. **Dead-reckoning sites** (`:1077, :1089, :1094`) initialize the local in-memory model —
   legitimate as initialization, but they inherit key-level zeros from (2) if drift occurs.

## Queued fixes (post-window, author-gated — NO serving-code edits mid-window)

* **Boundary schema check**: `_fetch_all_instance_stats` validates the required key set per
  instance (the same single-boundary discipline as the zero-call arm); missing keys raise.
* **Random blind-dispatch fallback** (author's design call): where dispatch must proceed
  without state — parity mode only — fall back to RANDOM, never `min()` over fabricated
  equal values. CR note queued.
* **Static-gate CLASS 4**: detector for `.get(<sched-state-key>, 0)` reachable from decision
  functions, so the class cannot silently regrow.

## Part 2 — deterministic-instance concentration on error/edge paths (author-ordered ~19:20Z)

**The narrower class**: failure or edge handling that concentrates load on a SPECIFIC
instance rather than spreading it. Audit of every instance-selection site:

| site | mechanism | verdict |
|---|---|---|
| `cara_serve.py:1642` — budget filter rejects ALL → `min(all_preds, key=worst_cost)` | same-tier instances share identical prices → cost ties → **the first instance of the cheapest tier, every time**, for every all-rejected request | **CLASS HIT.** Logged but concentrating. Dormant in S2 cells (pluggable filter off) — **LIVE in the paper's budget experiments** (`tab:budget_violations` ran this path) |
| `dispatch/shortest_queue.py` — strict `q < best_q` | integer queue depths tie constantly at low load → **ties keep first-in-list** → standing first-instance skew even with healthy telemetry | **CLASS HIT (mild, always-on).** The RB path fixed exactly this with a score→queue→LRU tie-break (`:2019`, "eliminates the first-listed-always-wins skew") — pipeline SQ never received the fix |
| `cara_serve.py:1766-1776` — STRICT filter empties candidates → fall back to SQ over all | strategy substitution with a logged warning; inherits SQ's tie bias | borderline — logged, not fixed-instance per se |
| era `{}`-blind SQ head-pin | dead under fail-loud; returns only under the parity toggle, scoped | documented (Part 1) |
| `dispatch/po2_queue.py` | seeded `rng.sample(2)`; tie keeps first *sampled* (itself random) | **clean** |
| RB scoring `:2019-2021` | explicit LRU tertiary tie-break | **clean — the pattern the others should copy** |
| `:1791/:1806` llumnix_minus / qlen ablation modes | min/max ties → first; experimental, not in S2 cells | class hit, dormant |
| broadcast `:496/:520` | `random.shuffle`/`random.choice` | clean |

**Queued fixes (post-window, author-gated):**
1. Budget all-rejected fallback: random-or-LRU among tied-cheapest instances, never bare `min()` — matters for any future budget experiment.
2. Pipeline SQ: copy the RB LRU tie-break. Note: this CHANGES SQ baseline behavior slightly (fairer at low load) — a semantics decision for the next campaign, not a patch.
3. (from Part 1) parity blind-dispatch → random; boundary key-schema check; static-gate CLASS 4.
