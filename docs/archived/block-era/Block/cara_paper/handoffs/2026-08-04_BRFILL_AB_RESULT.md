# BR fill A/B result + POST3 λ24 reconciliation (2026-08-04 09:27Z)

**The BR sq/po2 fill completed** (root-caused per CLOSEGATE §3b): λ24 full (8/8 B cells),
λ18 partial (5/8), cutoff before λ30. 13 B cells + 1 era-parity A cell banked, two-location
(head /mydata + VPS `s2_cluster/cells/brfill/`).

## A/B on POST3's t0.3/po2/λ24 (posted mean e2e λ24 = 37426 ms)

Same seed / same arrivals; only the dispatch blind-policy differs.

| draw | mean e2e λ24 | p99 ms | vs posted 37426 |
|---|---:|---:|---|
| POST3 posted (W3 table) | 37426 | — | — |
| A — counter (era-parity) | 41373 | 127542 | **+10.5%** |
| B — random (corrected) | 42642 | 128408 | +13.9% |
| **B vs A (the dispatch fix)** | — | — | **+3.1%** |

**Column note (corrected a mis-read):** in the W3 table `| … mean e2e ms λ12 | λ24 |`, the
value **2727 is the λ12 mean** and **37426 is the λ24 mean** — NOT a mean/p99 pair. There is
no order-of-magnitude discrepancy.

## Reading

1. **Dispatch fix immaterial:** A→B = **+3.1%**, below the author's 5% rule. The fixed-instance
   pin (BR high-rate/budget) does NOT materially move the latency — the concern that prompted
   the rerun is cleared.
2. **Fresh draw +10.5% hotter than posted** (era-parity A, not the fix): collapse-regime run
   variance at λ24, non-reproducible, SAME regime. The absorb-mask is faithful, not the cause —
   fresh A (41373) lands on the ORIGINAL Jul-29 posted draw (43289 ms, `OPENREVIEW_AS_POSTED`
   line 280), reproducing the collapse.
3. **Claim intact either way:** posted 14.3× (37426 vs RB 2624); fresh ≈15.8× — if anything
   stronger. No claim is contradicted.

## Author decision (queued — does NOT block the closegate)

Keep POST3's **37426** (dispatch fix immaterial; it is a real earlier draw) OR refresh to the
fresh collapsed draw. Recommendation: **keep 37426** — the +10.5% is inherent collapse variance,
and the collapse claim is safe at both values. Never re-assert the two basis-dependent claims
(posting rail).

## λ6 + λ12 FRAMING — author decision 2026-08-04 09:44Z (extended 09:46Z)

**Both low-load rates are near-parity, not strict-dominance** (verified 2026-08-04: AP-pw0.70's
own-best-by-argmax stays high-quality at λ6/λ12, so RB can't 4-axis-dominate it; at λ18/24/30
AP's own-best collapses to a low-quality point and RB DOES strictly dominate). Representative
RB point at both = **RB (0.6,0.2,0.2)**. Framed as **cost + latency dominance with a minor
quality trail** (honest — quality trails):

| λ6 decode | quality | cost | mean ms | p99 ms |
|---|---:|---:|---:|---:|
| **RB (0.6,0.2,0.2)** | 0.4210 | **0.0296** | **3705** | **22141** |
| AP-pw0.70 / po2 | 0.4273 | 0.0379 | 4352 | 31062 |
| RB vs AP | **−1.5%** (minor) | **−21.9%** | **−14.9%** | **−28.7%** |

| λ12 decode | quality | cost | mean ms | p99 ms |
|---|---:|---:|---:|---:|
| **RB (0.6,0.2,0.2)** | 0.4200 | **0.0298** | **3896** | **22982** |
| AP-pw0.70 / po2 | 0.4273 | 0.0377 | 4479 | 29286 |
| RB vs AP | **−1.7%** (minor) | **−21.0%** | **−13.0%** | **−21.5%** |

RB (0.6,0.2,0.2) **dominates AP on cost, mean, and p99** while **matching AP's quality within
0.0063 (1.5%)**. The eff-0.70 retune (q0.4241, cost 0.0377 ≈ AP) CONFIRMED no cost-feasible
weight closes the quality gap at λ6 — so the honest characterization is **near-quality-parity
with a decisive cost+latency advantage**, the RB-spans-a-frontier story, not a strict Pareto
dominance claim. Consistent with the posted "reaches AvengersPro's quality band at lower cost."
State the decode-only cost basis wherever the numbers are used (basis-sensitive per dominance.py).

## 2026-08-04T09:44Z — window pass (folded): λ6 done + framed; release readiness

**No STOP.** λ6 eff-0.70 retune COMPLETE + synced (two-location); scheduler idle, no orphans;
monitor fresh (104s, 0 findings); self-test 162/89/0. Disk head root 92%/5.1G, /mydata 807G;
VPS ~95% (pack post-release from VPS copy). Deliverable + brfill + A/B + λ6 all banked, verified.
**~28 min to lease release (10:12Z), ~2.25 h to posting close (11:59Z HARD).**
