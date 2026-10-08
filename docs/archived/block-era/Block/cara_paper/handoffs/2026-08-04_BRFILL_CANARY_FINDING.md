> **⛔ SUPERSEDED 2026-08-04 08:42Z — this finding was a MISDIAGNOSIS. Do not act on it.**
> The 358/3534 stop was NOT fd exhaustion. Root cause: the campaign fail-loud stats gate
> (`_get_instance_stats` raises for sq/po2, which fetch per-request stats; rr doesn't) plus
> head root `/` at 100% truncating the write. Fixes: `_get_instance_stats` masks under
> `stats_blind_absorb` (grid/Q1 keep fail-loud); output → `/mydata`. Canary then completed
> 3534/3534 and the full B fill is running. See CLOSEGATE §3b (corrected) and the
> 2026-07-30_v3_execution_log.md 08:30Z entry. Retained below for the forensic record only.

# BR fill canary — FINDING (endgame, 2026-08-04 ~07:31Z) — ⛔ SUPERSEDED (see banner)

**The B fill (complete the grid's BR sq/po2 at λ≥18) is INFEASIBLE on the v3 scheduler.**
The author-approved canary tested exactly this; it stopped itself, by design.

## What happened
`br4_t03_po2_queue_B_l24` (blind_policy=random, stats_blind_absorb=1) completed **358 / 3534**
requests, then the scheduler **died** (fd/connection exhaustion under BR's ~1,500-deep
backlog). The launcher STOPPED and ran no further cells. Partial cell marked
`.CANARY_INCOMPLETE_358` (never valid). Grid untouched (separate `_BRFILL` dir).

## Why absorb did not help — the mechanism, precisely
Absorb handles the stats-fetch **timeout** (proceeds instead of raising). It does NOT reduce
the **connection load**: each dispatch still opens per-instance stats connections, and with
~1,500 in-flight requests those connections exhaust the scheduler's fds and kill the process
(`aiohttp … Bad file descriptor`). The bottleneck is fd exhaustion, which absorb cannot touch.

## Why rr completes but sq/po2 die (consistent with the paper's §265 collapse)
Round-robin needs **no** queue state → it fetches **no** per-request stats → no connection
churn → it completes (banked the collapse: 47–56 s means at λ30). shortest-queue and
power-of-two **do** fetch per-request stats → churn → fd exhaustion → death at ~358 requests.
So the grid's BR sweep at λ≥18 is legitimately **rr-only**: sq/po2 have **no sustainable
operating point** there — that is the finding, not a data gap.

## Consequences (all honest, none a workaround)
1. **The A/B comparison is MOOT** — neither A nor B completes at λ24, so they cannot be
   compared. The blocker is the scheduler fd limit, not the blind policy.
2. **POST3's BR-Po2 numbers STAND** — they came from the ERA scheduler, whose different
   connection handling let those cells complete. The v3 fail-loud scheduler cannot reproduce
   them, which is itself consistent with "BR collapses under stateful dispatch at high λ."
3. **Completing these cells needs the connector hardening** (fd limits / stats-fetch batching /
   pooling) — the queued POST-WINDOW fix (UNALLOWABLE-BUG-#1 mitigation). NOT attempted
   mid-window (author principle: no self-decided workaround).

## Deliverable state (unaffected)
* **Grid: 4 rates strictly dominated (30/24/18/12), λ6 honest non-dominance** (low-load
  overlap). BR represented by rr at every rate. COMPLETE.
* **Q1: POSTED.**
* Endgame BR fill: reported infeasible; nothing to integrate.

**For the author:** the sq/po2 collapse at λ≥18 is now characterized down to the mechanism
(stateful-dispatch stats-fetch fd exhaustion). It strengthens, not weakens, the paper's
"decoupled router/dispatcher collapses under load" story. The A/B comparison cannot be run on
this scheduler; that awaits the post-window connector fix.
