# Fix verification checklist — 2026-07-31

Every change made tonight, with the exact command to verify it. Nothing here asks you to trust a
summary. Run the command, compare to "expect".

Repo: `/home/wd312/Code/llm/Block` (branch `v3-campaign`).
node0: `ssh asdwb@d8545-10s10505.wisc.cloudlab.us`.

---

## A. Which tree ran which cells (check this first — everything else depends on it)

| cells | tree | config | notes |
|---|---|---|---|
| POST-2 six cells `/mydata/v3/post2_minmax/v2_*` | `~/Block` (**v2**) | `/mydata/v3/scheduler_config_minmax.json` | minmax, 0 P100 refs |
| W2-A `/mydata/v3/post3_w2a/w2a_*` | `~/Block` (**v2**) | `cfg_w2a_{full,f50,promptblind}.json` | index swap only |
| earlier arm-4 variants `/mydata/v3/arm4_*` | `~/Block_v1x_arm4fix` (**v1exact+patch**) | v1 carafast / minmax | superseded |
| P1 `/mydata/v3/p1/` (17 cells, paused) | `~/Block_v3` (**v3**) | `scheduler_config_v3.json` | pre-dates minmax port |

```bash
# expect: v2 tree has GPU binding + _lat_val numerator + quality_norm
ssh asdwb@d8545-10s10505.wisc.cloudlab.us \
 'f=~/Block/block/global_scheduler/cara/cara_serve.py
  echo "HOST_GPU=$(grep -c _HOST_GPU $f) latnum=$(grep -c "lat_score  = 1.0 - (_lat_val" $f) qnorm=$(grep -c quality_norm $f)"'
# expect: HOST_GPU=2 latnum=1 qnorm=2
```

---

## B. The two defects that changed your numbers — both PRE-EXISTING

Neither originates in this session. Verify against the frozen submitted tree.

```bash
# B1. _lat_val feeds ONLY the normalizer in v1exact (numerator stays learned)
git show v1exact-submitted:block/global_scheduler/cara/cara_serve.py | grep -n "_lat_val\|lat_score  ="
# expect: 1474 def _lat_val | 1483 e2es=[_lat_val...] | 1532 lat_score = 1.0 - (pred["e2e_latency"]/...)
#         i.e. _lat_val appears at 1483 only; 1532 uses pred["e2e_latency"]

# B2. instance->head mapping matches on model_name alone, first hit wins
git show v1exact-submitted:block/global_scheduler/cara/cara_serve.py | sed -n '2660,2670p'
# expect: `if meta.get("model_name") == inst._model_name:` ... `break`   (no gpu_type test)

# B3. v2 fixed B2 in June
git show rebuttal-jul25-window:block/global_scheduler/cara/cara_serve.py | sed -n '2753,2769p'
# expect: _HOST_GPU prefix map, then `model_name == ... and (not gpu or meta.get("gpu_type")==gpu)`
```

**Measured consequence of B2** (this is the 6.2x):
```bash
ssh asdwb@d8545-10s10505.wisc.cloudlab.us 'python3 /tmp/tpot_means.py'
# expect: 3b learned mean ~64.4 ms/tok vs measured 10.40; 7b/14b/72b accurate to ~2%
```

---

## C. Code fixes made this session (v3 tree)

```bash
git log --oneline v1exact-submitted..HEAD -- block/ | cat
git diff --stat v1exact-submitted HEAD -- block/ | tail -3
```

| # | fix | verify | expect |
|---|---|---|---|
| C1 | estimator fails closed on missing/empty artifact | `grep -n "does not exist\|exposes ZERO" block/predictor/cara/model_estimator.py` | 2 raises |
| C2 | startup assert: score_type=deepeval requires the array | `grep -n "carry no.*array" block/predictor/cara/model_estimator.py` | 1 hit |
| C3 | BR requires explicit `fallback_model` | `grep -n "requires an explicit fallback_model" block/global_scheduler/cara/routers/best_route_4way.py` | 1 hit |
| C4 | `/v1/router` + `router_sha` observable | `grep -n "router_sha\|/v1/router" block/global_scheduler/cara/cara_serve.py` | ≥3 hits |
| C5 | `predict_all_models` returns `deepeval_score` | `grep -n deepeval_score block/predictor/cara/estimators/knn_estimator.py` | 1 hit |
| C6 | evaluator raises instead of defaulting to 0.0 | `grep -n "no 'deepeval_score'" block/predictor/cara/offline_training/train_knn_estimator.py` | 1 hit |
| C7 | **all** silent numeric defaults removed | `grep -nE "0\.05\)\|return 1\.0  # fallback\|return 0\.5  # fallback\|return 128\.0" block/predictor/cara/cara_learned_predictor.py block/predictor/cara/estimators/pfs_estimator.py` | **no matches** |
| C8 | `quality_norm` ported from v2 | `grep -n "_qnorm\|quality_norm" block/global_scheduler/cara/cara_serve.py` | ≥4 hits |
| C9 | RULE-0 gate hard-refuses non-v3 trees | `ssh … '~/cell_audit_v3.sh pre current arm1; echo rc=$?'` | AUDIT-FAIL, rc=1 |

---

## D. P100 purge

```bash
grep -rc p100 block/config/cara/predictor_xgboost_3model.json \
              block/config/cara/predictor_config_learned.json \
              cara_paper/handoffs/window_setup_work/v3_configs/scheduler_config_v3.json
# expect: 0 0 0
ssh asdwb@d8545-10s10505.wisc.cloudlab.us 'grep -c p100 /mydata/v3/scheduler_config_minmax.json'
# expect: 0
```
**Still present (deliberate, not yet purged):** node0 `~/Block/block/config/cara/predictor_xgboost_3model.json`
(5 refs) — left so arms 1/2/4 shared one predictor config. Nothing can bind to it (no p100 entry in
`instance_metadata`), but it should be purged before P1.
Also present: standalone P100-only files under `block/config/cara/*p100*` and `block/exp/cara/*p100*`
(11 files, not loaded by v3) — awaiting your call.

---

## E. Things I got wrong tonight and corrected — check these hardest

| claim I made | status | why it was wrong |
|---|---|---|
| "static prior arm 3B collapse proves learned head superiority" | **retracted** | first variant also deleted the queue envelope; answered the wrong ablation |
| "the 14B head under-predicts by 20%" | **retracted** | derived from a synthetic feature vector I invented, not real serving state |
| "152.9% MAPE means the 14B head is uncalibrated" | **retracted** | MAE is 0.9 ms on a ~16 ms target; MAPE inflated by near-zero targets |
| "the constants that ran (12.3/…) are closer to truth than the posted ones" | **retracted** | my `(e2el−ttft)` proxy folded queueing into decode; median ITL says posted constants are right (0.22 ms/tok) |
| "environment drift explains the AP rebuild mismatch" | **retracted** | pinning node0's exact stack gave 7/32 vs 8/32 — no better |
| "CR should stay on raw normalization" | **withdrawn** | v2's own cells show raw + fixed binding sends 56-58% to 3B, 72B unused |

---

## F. Process failures (no data lost, but they cost window time)

- 3 duplicate-launcher incidents (two concurrent benchmarks). Now guarded by `/tmp/runlock.sh`
  (`flock` + benchmark-count + per-runner checks). Verify: `ssh … 'bash /tmp/runlock.sh /tmp/w2a.sh /tmp/x.log'`
  while something runs → expect `REFUSED`.
- P1 output initially written to root (63 G, 92% full) at 21 MB/cell. Relocated to `/mydata` via
  symlink at cell 1. Verify: `ssh … 'df -h ~/Block_v3/experiment_output/v3_p1 | tail -1'` → `/dev/nvme0n1`.
- 4 self-matching `pgrep` errors (pattern text appearing in my own command line).
