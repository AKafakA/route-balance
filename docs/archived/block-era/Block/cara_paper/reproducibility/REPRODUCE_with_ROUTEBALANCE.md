# INSTRUCTION DOC 2 — Reproduce v2 via the route-balance repo
Uses `~/Code/llm/route-balance/v2-reproduce/` (v2 code synced into the public paper repo) against this
pack's `models/`, `data/`, `results/`. `PK=/home/wd312/Code/llm/v2-exp`, `RB=~/Code/llm/route-balance/v2-reproduce`.

> Note: route-balance has its own layout (no top-level `block/` package). The **analysis (L1) is fully
> self-contained** in `RB/scripts/` (pure python + numpy over the pack's data/results). For **L2 serving**,
> the cara stack (`RB/serving/cara/`) imports from the broader `block.*` package, so the cleanest serving
> deploy still uses the **Block repo** (`REPRODUCE_with_BLOCK.md`); route-balance drives the orchestration
> + analysis. The two paths produce the SAME numbers from the SAME pack.

## L1 — ANALYSIS SMOKE (no cluster, ~2 min) — fully self-contained from route-balance
The scripts have hard-coded paths to the Block tree; for the route-balance path, run them against the pack:
```bash
cd $RB
# point the scripts at this pack (env override or edit the BASE/SCORED constants at the top of the scripts):
export V2_BASE=$PK/results            # the 442 collected cells + _aggregate/
export V2_SCORED=$PK/data/cara/scored/test_scored_filtered.jsonl
export V2_GEMMA=$PK/data/cara/scored/gemma3_scored_full.jsonl
python3 scripts/aggregate_v2_grid.py     # -> $PK/results/_aggregate/AGGREGATE.csv (287 rows; dispatcher_only=21)
python3 scripts/consolidate_v2.py        # -> ALL §6 numbers
```
(If the scripts don't yet read the env vars, edit the `ROOT/BASE/SCORED` constants at the top of
`scripts/aggregate_v2_grid.py` and `scripts/consolidate_v2.py` to the `$PK` paths — they are otherwise identical
to the Block copies.) PASS = matches `docs/V2_CHANGELOG_AND_SUMMARY.md §6`.

## L2 — FULL CLUSTER RE-RUN
### 0. Restore from pack (same as Block path)
```bash
# acquire 13-node topology (RB/configs/model_deployment.json), setup.sh per node.
rsync -a $PK/models/cara/  <serving-host>:~/Block/models/cara/      # deployed predictors+baselines
cp $PK/configs/* <serving-host>:~/Block/block/config/cara/          # v2 configs
# re-pull Qwen pool from HF; launch cara_serve from the BLOCK repo (the block.* package is needed to run the
# serving stack — RB/serving/cara is the same source, archived for reference).
```
### Orchestration (RB/orchestration/ = same scripts as Block's v2_orchestration/)
Run the same per-artifact scripts as in `REPRODUCE_with_BLOCK.md`:
`v2_baseline_maygrid.sh` → `v2_phase2.sh` → `v2_cara_costextreme.sh` (§5.2); `v2_phase3_batching.sh` (§5.5);
`v2_budget_cara_native.sh` (§5.4); `v2_multiseed_cara.sh`; `vllm_sr/vsr_relaunch.sh` + `v2_vllm_sr_grid.sh`;
judge-alt (vllm 0.10.2 cu128 + transformers 4.57.6, unsloth/gemma-3-12b-it bf16, score_with_deepeval).
### After each track → aggregate + compare to `$PK/results` ground truth (quality matches; latency shape matches).

### Sanity gates (same as Block path)
`dispatcher_only==21`; cara-peak λ12 ≈ 0.419 not ~0.52; routing by DISTRIBUTION; no `a or b` zero-score drop.

## Targets — `docs/V2_CHANGELOG_AND_SUMMARY.md §6`
cara peak 0.4190 #1; cost-extreme cheapest 1.68e-5; br4 collapse 21× (64204ms) vs cara 2389ms @λ30;
bootstrap [0.4089,0.4288]; overhead 134ms; vLLM-SR collapse 89–97% @λ≥18; judge-alt cara #1 both judges, r=0.555.
