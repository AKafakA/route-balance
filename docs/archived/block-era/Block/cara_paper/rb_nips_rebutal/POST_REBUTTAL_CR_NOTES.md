# Post-rebuttal notes — corrections owed at camera-ready

Not surfaced during the rebuttal. Each item below is a real discrepancy between what the
submission reports and what the code computes. Author's standing policy: **silent in the
rebuttal, truthful if a reviewer asks directly, explicit change note at camera-ready.**

None of these change a claim we make. Every rebuttal number is a paired comparison between arms
computed the same way, so the differences and intervals are unaffected; what moves is absolute
level, and only on the axes named below.

---

## CR-1 — Evaluator drops zero scores (quality basis)

The aggregator's `or`-chain match treats a DeepEval score of exactly 0.0 as missing, so those
(prompt, model) pairs fall out of the average instead of counting as zero. Effect: coverage
lands near 76% instead of 100%, and quality reads about 0.507 where the zero-safe computation
gives about 0.385.

- Fixed in `aggregate_v2_grid.py`; **`aggregate_overnight_grid.py` still carries the bug.**
- Every rebuttal cell is scored on the submitted basis, so the new numbers and the submitted
  tables are mutually consistent — which is the property the rebuttal actually asserts.
- Checked both ways: on either basis all seven paired quality CIs span zero, so the
  "statistically indistinguishable" claim holds regardless.
- **Owed at CR:** re-report quality on the zero-safe basis, with a change note stating the
  absolute level moved and why, and confirming which claims were unaffected.
- Removed from POST 2 before posting: the sentence naming ~76% judged coverage, which pointed
  straight at this.

## CR-2 — Cost axis: input+output vs output-only

The rebuttal's cost columns price **input+output** tokens at the Table-1 rates. The
submission's cost axis prices **output only**. Absolute $/1k figures therefore do not line up
between the two, though every comparison within the rebuttal is like-for-like.

- The 8.1–9.8% claim is a ratio between two arms priced identically, so it does not depend on
  which convention is used.
- **Owed at CR:** put both the paper and the new cells on one convention and say which, with a
  change note.
- Removed from POST 2 before posting: the clause contrasting the two conventions.

## CR-3 — KNN retrieval parameters (already on the list)

The paper reports k=50 over 14,919 examples. The deployed v1 artifact is the Mar-15 pickle with
k=10 baked in over 14,966 examples; the configured k=50 was dead for `type="default"`.
**Owed at CR:** correct k and the corpus size.

## CR-4 — TTFT phrasing

Manuscript wording implies TTFT is modeled in the assignment score. It is not; the score is
decode-dominated. POST 2 states this to the reviewer. **Owed at CR:** fix the phrasing.

## CR-5 — Vidur citation promised in POST 2

POST 2 tells Reviewer gTfM that we will cite the simulation-stack line of work in the design
section, and characterises it: Vidur fits random forests to per-kernel durations and composes
them over layer executions; ours is a single XGBoost head per (model, GPU) tier fitted directly
to TPOT — the same idea at coarser granularity.

**Owed at CR:** add the citation and a sentence placing our estimator against it. This is a
commitment made to a reviewer, not an optional edit. Verify the characterisation against the
Vidur paper before writing it — it came from the author, not from a source I checked.

## CR-6 — safety rubric is not applied to the Gemma corpus (optional enhancement, NOT a defect)

Raised 2026-07-29, then **downgraded after checking the code**. My first reading called this a
scoring inconsistency. It is not, and the author's reading was correct.

`score_with_deepeval.py` scores with a single DeepEval `GEval` rubric and never reads
`is_harmful` — the parameter does not appear in that path. The safety-template branch lives in
`llm_judge_scorer.py` and `prometheus_judge.py`, which produce a different judge
(`Qwen_Qwen2.5-7B-Instruct` and a Prometheus 1-5 scorer). Neither is present in the Gemma
corpus and neither is the metric of record.

**Every reported quality figure is `deepeval-llama3.1-8b-it_reference`, and deepeval scored the
Qwen and Gemma corpora identically. The cross-family comparison is fair.** Nothing in the paper
or the posted replies is affected.

Where a safety rubric would matter is the corpus RouteBalance *learns* from — the KNN index —
not the yardstick every system is measured by. All systems are evaluated on the same deepeval
axis, so a rubric difference upstream cannot bias the comparison between them.

**Optional future enhancement, not owed at CR:** score safety-source prompts under a dedicated
rubric to give the estimator a better signal on refusal-appropriate responses. That is a
possible RouteBalance improvement, not a correction. `is_harmful` is now reconstructed in the
staged HF corpus (a pure function of `source`, matching the Qwen rule), so the option is open.

**Worth fixing regardless:** `llm_judge_scorer.score_pairs()` defaults a missing `is_harmful`
to `[False] * len(pairs)` silently. Any corpus that omits the field changes rubric with no
warning. It should warn.

## CR-7 — W2-B onboarding paragraph attributes two Qwen numbers to Gemma

Found 2026-07-30. The posted W2-B paragraph opens *"We use the Gemma onboarding process as the
example"* and then reports four figures, two of which are Qwen's:

| posted figure | actually |
|---|---|
| ~6 h to generate 18,549 prompts | Gemma ✓ |
| ~2 h to judge 10,065×4 = 40,260 pairs | Gemma ✓ |
| "the KNN index took only 52.3 s to build" | **Qwen** — the 14,966-example build (`2026-07-25_EXECUTION_HANDBOOK.md:153`), dated three days before Gemma generation existed |
| 404,929 requests / 11.1 h profiling | Qwen — correctly labeled "the original family", but inside a Gemma-framed paragraph |

Gemma's own index build was measured at **24 s** over 8,086 examples
(`logs/gemma_pipeline_node0/_gemma_knn.log`: `KNN build DONE rc=0 wall=24s`). We therefore
quoted a figure 2.2× larger than the truth **for our own cheapest step** — conservative, not an
overclaim, and the "index build is negligible" point holds either way.

**Author's call 2026-07-30: 52.3 s vs 24 s is immaterial — both are negligible against the 6 h
and 2 h phases, and the sentence's point survives unchanged. Not a correction owed.** What
remains owed is the family attribution: the paragraph opens on Gemma and must not present the
original family's figures as that example. Fix the framing at CR and use Gemma's own numbers
throughout, including the 208 s prior probe, which is the figure that actually carries the
argument and was never posted.

The compression to 5,000 characters caused it: the uncut draft (`POST3a_full.md:35`) separated
the families cleanly and said *"After filtering, judging the surviving corpus"*. Both the
attribution and the filter word were lost in the cut.

**Also never posted, though measured:** the low-load prior probe that *replaced* profiling for
Gemma cost **208 s** (`2026-07-28_G5_DECISION.md`). That figure is what makes the argument, and
it is strictly better than what went up.

**Owed at CR** — the all-Gemma version, every figure from this family:

> 6 h generate → 12 min filter (719 s) → 2 h judge → 24 s index → 208 s priors, against
> 404,929 requests and 11.1 h of latency profiling for the original family.

**Verified while checking this, and it holds:** POST 3 claims Gemma ran on *measured static
per-(model,GPU) priors*. That claim had a live failure path — `cara_serve.py` matched tiers
against a hardcoded Qwen ladder, and `"3b"` never matches `google/gemma-3-1b-it`, so
`CARA_STATIC_LAT` would have been silently ignored with no error. It was caught before any cell
ran and fixed via Port #4. Re-verified 2026-07-30 end to end: deployed config md5
`5020c3cd30bfb1404e6e9374a4a68783` equals the audit-gate constant, carries
4.799 / 12.933 / 21.376 / 25.470 ms/tok, and `cell_audit.sh:57` asserts `--static-tpot` on the
live process via `/proc/$PID/cmdline`. The config was `/tmp`-only on node0 and is now in
`configs/node0_runtime/cara/gemma_jul27/`.

## CR-8 — The Gemma corpus is the cap-filter survivors, and the filter is not domain-neutral

The posted W2-B states both 18,549 generated and 10,065 judged, so the 46% drop is visible, but
the reason was cut in compression. `prepare_benchmark_data.py --filter-truncated
--max-output-tokens 1024` drops any prompt where **any** of the four models hit the cap, because
the quality label is `0.5·similarity_score + 0.5·mean(judge)` and similarity is measured against
the 27B reference — a truncated response makes that term meaningless. Measured on the raw
broadcast: 8,467 records (45.6%) had ≥1 model truncated; 12B 35.6%, 27B 31.0%, 4B 28.8%, 1B
13.0%. Gemma-3-it is simply more verbose than Qwen2.5 at the same cap.

The filter is **not** domain-neutral:

| domain | retained | corpus share before → after |
|---|---:|---|
| squad | 78.8% | 15.8% → 23.0% |
| gsm8k | 67.9% | 15.5% → 19.4% |
| lmsys | 55.9% | 13.5% → 13.9% |
| mix_instruct | 50.6% | 15.6% → 14.5% |
| reward_bench | 47.9% | 9.0% → 7.9% |
| beaver_tails | 45.0% | 15.7% → 13.1% |
| code_ultra_feedback | **29.9%** | 15.0% → **8.2%** |

**This does not affect the W1-B table**: all five systems there run the identical 1,979 held-out
prompts, so the comparison is internally valid whatever the mix, and the packet makes no
cross-family numeric comparison.

**The cap itself is defensible and should be stated as such, not apologised for.** 1,024 is the
`max_tokens` every online cell serves under (`CARA_PROJECT_STATUS.md:835` — "match benchmark
setting"), so corpus and cells share one response-length regime; a record truncated at the cap
describes a generation the deployed system never produces. It is also **permissive by the
harness's own conventions**: our benchmark is vLLM's `benchmark_serving`, and since we supply our
own corpus it runs the custom-dataset path, whose output length defaults to 256 tokens — random
defaults to 128, sonnet 150, ShareGPT to each reference completion's own length
(`vllm/benchmarks/datasets.py:1380/1408/1422/1451` in the pinned fork `2894a9872` that served
every cell). At 1,024 we allow 4× the default of the path we run. It is also where useful output ends
on this data: degenerate repetition rises from 0.2% at 50–200 output tokens to 34.5% beyond 800
(same file, :825), so a longer cap would admit repetition whose judge score measures degeneracy
rather than capability. And the construction is family-neutral — same script, flag and value
built the Qwen corpus, and **neither published corpus contains a single truncated (prompt, model)
pair** (0 of 59,676 Qwen, 0 of 32,344 Gemma, verified 2026-07-30). Gemma's higher loss rate is a
property of the family, which is what a second-family experiment exists to surface.

**Owed at CR:** state the cap, its benchmark provenance, and this composition shift wherever the
Gemma family table appears, so a cross-family reading of the absolute quality levels is not
invited. `it_truncation_report.py` was written to produce exactly this table and was never run to
a saved output; the numbers above were recomputed 2026-07-30 from `gemma_gen_broadcast.json.zst`,
now in the pack.

---

## Rails

1. Never re-assert either of the two basis-dependent claims identified in the July disclosure
   audit — not in the rebuttal, not in the CR, not in a talk.
2. If any reviewer asks about coverage, the cost convention, or k, answer truthfully and
   completely. These are unstated, not deniable.
3. CR must carry an explicit change note covering CR-1 through CR-4 together, so the corrections
   read as one disclosed revision rather than four scattered edits.
5. CR-7 and CR-8 are compression casualties, not measurement errors — every underlying number was
   measured correctly and is logged in the pack. Both are conservative or neutral in direction.
   Same disposition as CR-1/CR-2: silent now, corrected at CR, truthful if asked.
6. When a reply is compressed, re-check that every figure still carries its family/basis label.
   Both CR-7 and CR-8 were introduced by the 5,000-character cut, not by the drafting.
4. Before the CR build, re-run the aggregation with the fixed script and re-check every number
   that survives from the submission.
