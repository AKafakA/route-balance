# Rebuttal packet — versions

Convention from 2026-07-29 03:25Z (author instruction): every rewrite creates a NEW file and
the previous one is committed first. Never edit a shipped version in place.

| file | estimator comparator | burst section | chars | status |
|---|---|---|---|---|
| `2026-07-29_REBUTTAL_v1_nominal-prior.md` | nominal per-tier constants carried in the implementation | present (5 rates) | 39,763 | fallback |
| `2026-07-29_REBUTTAL_v2_calibrated-prior.md` | per-tier medians measured across the sweep, table published | dropped | 37,268 | superseded |
| `2026-07-29_REBUTTAL_v3.md` | same as v2 | dropped | 36,328 | superseded |
| `2026-07-29_REBUTTAL_v4.md` | calibrated prior, cost-only claim | dropped | 35,980 | superseded |
| `2026-07-29_REBUTTAL_v5.md` | v4 + POST 3 split to fit the 10k gate | dropped | 35,873 | superseded |
| `2026-07-29_REBUTTAL_v6.md` | v5 + 20 corrections from a 3-agent closed-rebuttal review | dropped | 37,099 | superseded |
| `2026-07-29_REBUTTAL_v7.md` | v6 + preset naming, per-post bases, Table 4 wording | dropped | 38,087 | superseded |
| `2026-07-29_REBUTTAL_v8.md` | v7 + third time-matched latency rate (18/18 cells) | dropped | 38,289 | **current** |

## Why two versions exist
v2 reports the learned head as never slower and ~8% faster at high load, against a comparator
calibrated to measured medians. That is stronger and simpler than v1, which reported a tie on
latency plus a separate burst argument.

**v2 is provisional on one measurement.** Its learned cells are from the Jul-27 bridge run and
its calibrated-static cells are from the night of Jul-28/29 — not time-matched. The A/B
control at 00:33Z showed a 5.9% swing at lambda=28 attributable to run timing alone, which is
most of v2's claimed latency margin. A six-cell time-matched run at lambda 12/24/28 decides it:

- margin survives -> v2 ships
- margin collapses -> v1 ships (tie on latency, burst section carries the latency argument)

## v3 changes (author decisions, 2026-07-29 ~03:50Z)
- CUT all three self-narrations, including the commitment to correct the abstract and appendix
- KEEP lambda=28 as a 3-seed mean. Single-seed would flip the sign: seed 1 alone gives the
  learned head 1.9% SLOWER where the 3-seed mean gives 8.1% faster (seeds -1.9 / +4.6 / +19.8).
  Reporting one run there would be reporting noise.
- KEEP the AvengersPro topology rows, for consistency with the second-family table
- NO further runs at lambda 6/8/10/18; single-run basis stands, matching the submission

## Open
1. time-match result (18 cells running) -> decides v3 vs v1
2. "Po2 beats round-robin wherever round-robin was measured" is false against our own topology
   table (BEST-Route lambda12: RR 2,596 vs Po2 2,624). Inherited from the trimmed base, which
   predates the fix. Author has not yet said whether to correct it.


## v4 — the time-matched result (2026-07-29 ~02:30Z)

The 18-cell time-matched control settled v2/v3 against v1, and it went against v3.

Latency, both arms measured in ONE session:
    lambda=24  learned 2,458  static 2,487   -1.1%
    lambda=28  learned 3,143  static 3,079   +2.1%
v3 claimed ~8% from cross-session cells. That margin was an artefact of comparing Jul-27
learned cells against Jul-29 static cells -- the same confound the 00:33Z A/B control exposed.
v3's latency claim is dead.

Cost SURVIVES time-matching and is the strongest number in the packet:
    lambda=24  0.02334 / 0.02576   9.4% cheaper
    lambda=28  0.02328 / 0.02568   9.3% cheaper
consistent with the cross-session estimate, as expected for a routing-mix effect that does not
depend on timing. Tier mix 14B 68->56, 7B 29->38.

v4 therefore makes ONE claim: against a prior calibrated to measured medians, the learned head
is 8.1-9.8% cheaper at equal quality, equal latency and equal throughput. No latency claim, no
burst section, no unexplained constants.

Burst was NOT restored. Those cells carry the same cross-session confound, and re-measuring
them time-matched (~40 min) was judged not worth it: stationary latency ties at both lambda 24
and 28, so the burst axis would most likely tie too.


## v5 -- fits the posting gate (2026-07-29)

Per-post character counts against the 10,000-char limit. v4 passed on four posts and failed on
one: POST 3 (faAU) was 17,655 -- 7,655 over.

Trimming 7.7k would have cut evidence. Instead POST 3 is split into three comments along the
three concerns the reviewer actually raised, which OpenReview permits:

    POST 0   1,652   general response + meta-review
    POST 1   6,914   287j  writing, clarity, batching
    POST 2   6,831   gTfM  estimator justification, deployment model
    POST 3a  5,914   faAU  general-purpose dispatchers   (meta-review key #2)
    POST 3b  5,278   faAU  generality -- topology
    POST 3c  6,657   faAU  generality -- model family; tuning and budget
    POST 4   2,627   AC    short, AC-first ordering

Word-level diff confirms the split dropped exactly five words: the "W1: Generality (model
family, topology)" section header, which is now the 3b/3c post titles. No evidence removed.

Posting order: POST 0 first (it carries the two scoping notes the other posts rely on), then
2 and 3a (the two meta-review keys), then 1, 3b, 3c, then 4 last.


## v6 -- corrections from the closed-rebuttal review (2026-07-29)

Three independent agents read v5: simulated gTfM, simulated faAU (both under "the rebuttal is
closed, you may not ask new questions, decide your score"), and a cross-post consistency
auditor. Twenty items fixed. The ones that were actually WRONG, not merely loose:

1. POST 3a said the static prior costs "about 7%". POST 2's own table gives 8.1-9.8% cheaper
   for the learned head = 8.8-10.8% penalty for the static one. No arithmetic yields 7%, and
   the error flattered our own onboarding story. Both reviewers caught it independently.
2. POST 3c said shortest-queue's tail "was the worst" on the Qwen topology. POST 3b's table
   says the opposite: SQ p99 11,046 is the BEST of the three dispatchers (Po2 11,160, RR
   11,254). Contradicted our own table one comment earlier.
3. POST 2 said scheduler load "grows 5x" over lambda 6 -> 24. That is 4x. A 5x span from
   lambda=6 implies the rate we removed from the packet.
4. POST 3b attributed shortest-queue's 11.0 s p99 to its 4.7% 72B share. The same table shows
   round-robin carrying 15% on 72B for a p99 2% higher -- p99 is nearly flat in 72B share.
5. POST 3c called the preset sweep "sweeping the quality weight alone". All three simplex
   weights move: (1/3,1/3,1/3) -> (0.5,0.3,0.2) -> (0.8,0.1,0.1).
6. POST 3c used "the balanced preset" for two different presets three lines apart; +0.0112 is
   the (0.5,0.3,0.2) row.
7. POST 3a's onboarding path claimed static priors need only "brief median probes at a moderate
   operating rate" -- POST 2 states the constants are medians across the WHOLE sweep. The
   two-stage story contradicted the experiment it cited.
8. POST 2's "(below)" pointed at nothing: the third arm's numbers lived only in the AC post,
   while POST 1 told 287j to find a "three-arm isolation" in a two-arm reply. The third arm is
   now stated in POST 2 where it was promised.
9. "8 of 9" understated the Gemma result -- all nine point estimates favour us, eight
   significantly.
10. POST 2's mechanism claimed per-INSTANCE resolution as the differentiator. The static arm
    retains a within-tier queue tiebreak, so it already discriminates between replicas; what
    the adjacent table actually shows is LOAD dependence (14B drifts 15.4 -> 18.6 ms/token).
    Reworded to the mechanism the evidence supports.

VERIFIED, not assumed: the third arm's "69.1% smallest-tier, quality 0.4899" was flagged
because 0.4899 also appears as dispatcher-only/SQ in POST 3b. Recomputed the v1exact arm2 mix
from the cells: 3B 69.1 / 7B 22.3 / 14B 8.1 / 72B 0.5, IDENTICAL at lambda 6, 12, 24, 30 --
the load-blind signature. 0.4899 recurs across arm1/arm2/arm3 rows in V1_REAGG_BOTH.csv, so
the collision is metric quantisation, not a copy error. Claim stands.

Scores from the simulated panel, rebuttal closed: gTfM 4 -> 5 (moved by the explicit scoping
concession); faAU 4 -> 4 with confidence raised and "would not oppose acceptance".


## v7 -- the remainder of the audit, plus one standing instruction (2026-07-29)

- POST 1 said "an updated Table 4". The author's standing instruction is not to claim the table
  is already updated: it now reports the new Po2/round-robin measurements and commits to the
  full sweep in revision.
- POST 4 still carried "8 of 9" (v6 fixed 3c only). All nine favour us, eight significantly.
- "The quality-leaning preset" named (0.5,0.3,0.2) in POST 4 and (0.8,0.1,0.1) in 3c;
  "balanced" and "uniform" both named (1/3,1/3,1/3). Every ambiguous mention now carries its
  weight triple.
- 3b and 3c reported cost and quality with the basis statement sitting in POST 2 -- a different
  reviewer's comment. Both now carry it, so nobody compares our input+output costs against the
  paper's output-only axis.
- The Gemma table's load-bearing caveat (static latency priors, not learned heads) moved from
  the general response to directly beside the table, and now says what that scopes: the
  formulation generalises to a new family; the learned estimator was not tested there.
- POST 2's 2,458 ms and 3a's 2,408 ms are the same configuration and rate from different
  sessions. POST 2 says so.

## Per-post character counts (v7, all pass the 10,000 gate)

    POST 0   1,652     POST 3a  6,038     POST 4   2,910
    POST 1   6,988     POST 3b  5,628
    POST 2   7,634     POST 3c  7,218

## Known-and-accepted, not fixed

- The BEST-Route collapse is explained in 3a, 3b and POST 4; the KV/TTFT scope answer appears
  in both POST 1 and POST 2. Two reviewers asked, so both need an answer, and every post passes
  the character gate -- trimming was judged not worth the risk of touching settled text.
- Tier-mix percentages round to 99 or 101 in seven rows.


## v8 -- the time-matched control completed, 18/18 (2026-07-29 02:58Z)

The lambda=12 block finished, so the latency tie now rests on three paired rates instead of
two. Both simulated reviewers had named that as the weak point of the claim.

    lambda   learned   static    difference        stationarity screen (fail > 2.0)
    12       2,243     2,224     learned +0.8%     1.01 1.02 0.99 / 0.96 0.96 0.95
    24       2,458     2,487     learned -1.1%     0.96 0.99 0.97 / 0.98 1.01 0.95
    28       3,143     3,079     learned +2.1%     1.45 0.89 0.98 / 1.27 1.05 0.96

Three seeds per cell, all 18 pass the screen. The gap is within 2.1% everywhere and CHANGES
SIGN across rates, which is what a tie looks like; a real effect would not flip. lambda=12 is
the tightest pair in the set -- the learned seeds land at 2,243 / 2,243 / 2,241.

Note for anyone re-reading these cells: at lambda=28 seed 1 is elevated in BOTH arms (4,102 and
3,428 against ~2,650 and ~2,900 for the other seeds). That is a session-order effect at the top
of the sweep, it hits both arms, and it is exactly what pairing is for. All three seeds are kept
-- no replicate is dropped for being an outlier.

The run had to be resumed twice. It died mid-lambda=12 because it was not session-independent,
and the first resume silently did nothing: the inline ssh command contained the scheduler
process name in its own argv, so its `pkill -f` matched the invoking shell. Fixed with
`node0_scripts/tm_resume.sh`, which keeps the pattern inside a file.

Cells synced to both locations (node0 + repo), 18/18.
