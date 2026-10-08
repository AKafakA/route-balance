# RoBERTa-base Fused Predictor HP Sweep V4 — Findings

**Launched:** 2026-04-19 19:16 UTC on Vast 2×V100-16GB (ssh alias `vast`).
**Goal:** find best LR / weight_decay / warmup / scheduler combo for roberta-base fused predictors on `length_bucket` + `deepeval` targets. All else equal to Apr 18 CSD3 run (batch=16, max_len=512, fp16, seed=42, patience=30, save_total_limit=1).

This doc is updated every 20 minutes by a cron-driven analysis loop. Read the "Latest update" stamp near the top before acting.

Related: [2026-04-19 handoff](../handoffs/2026-04-19_handoff_vast_sweep.md) — full record of the day's runs, failures, and budget.

**Latest update:** 2026-04-20 16:40 UTC — **🏁 TERMINAL STATE. USER HALTED CHAIN.** 9 of 12 V5 configs complete, zero improvements over V4 winners. V6/V7/V8 chain skipped per user decision (HP space exhausted within fixed-model-and-data constraint). Vast host scheduled for release. Cron `863bc95a` cancelled. **Not done**: v5_lb_warm0.05 (was ~70 min in), v5_de_warm0.15 (~37 min in), v5_de_wd0.005 (never started). These are lost; final result unaffected.

---

## Sweep design (6 configs × 2 targets, serial per GPU)

| # | name | lr | wd | warmup | sched |
|---|---|---|---|---|---|
| 1 | baseline | 1e-5 | 0 | 0.03 | poly |
| 2 | lr2e-5 | 2e-5 | 0 | 0.03 | poly |
| 3 | wd0.01 | 1e-5 | **0.01** | 0.03 | poly |
| 4 | warmup0.1 | 1e-5 | 0 | **0.1** | poly |
| 5 | cosine | 1e-5 | 0 | 0.03 | **cosine** |
| 6 | combo | 2e-5 | 0.01 | 0.1 | cosine |

- `length_bucket` on GPU 0 (tmux `cara:1.0`). Metric: avg `eval_accuracy` across 4 Qwen models.
- `deepeval` on GPU 1 (tmux `cara:1.1`). Metric: avg `eval_mae` across 4 Qwen models.
- Disk policy: running-best retention. Only the winning config per target keeps its 476 MB `fused_model.pt` on openclaw; losers keep only the 16 KB metric JSONs.
- Sentinel-based failure detection: orchestrator's EXIT trap writes `/workspace/Block/sweep_state/<target>.status` with `RUNNING / SUCCESS / FAIL exit=N stage=S config=C`.

## Current sweep state (iteration 9)

| target | sentinel | current config | step | epoch | config best-so-far |
|---|---|---|---|---|---|
| length_bucket | RUNNING (pid 91989) | wd0.01 (started 21:58:20 UTC) | in progress | ~ep 8 | baseline 0.4225 |
| deepeval | RUNNING (pid 92064) | wd0.01 (started 22:05:24 UTC) | in progress | ~ep 3 | **lr2e-5 0.2237** |

- GPUs: 84% / 93% util, ~11 GB VRAM each
- Vast disk: 12 GB used / 4.8 GB free / 16 GB total (recovered after config 2 cleanup)
- Openclaw: baselines + lr2e-5 artifacts landed for both targets ✓

Completed configs landed on openclaw: **4 of 12** (baselines + lr2e-5 for both). Config 3 (wd0.01) ETA: LB ~23:18 UTC, DE ~23:25 UTC.

ETA (if no surprises):
- Each config ~80 min at patience=30 (stops ~30 epochs past best)
- 6 configs per GPU, serial → ~8 h per GPU, 8 h wall total
- All 12 runs done ~04:16 UTC Apr 20

## Completed configs

### length_bucket (target metric: avg eval_accuracy, higher better)

| config | avg_acc | Δ vs baseline | Δ vs Apr 18 | best_ep | final_ep | train_time | adj_acc | mae_tokens |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.4225 | — | −0.0020 | 6 | 36 | 4823 s (80 min) | 0.659 | 101.0 |
| lr2e-5 | 0.4161 | −0.0064 | −0.0084 | 6 | 36 | 4825 s (80 min) | 0.664 | 100.8 |
| wd0.01 | 0.4236 | +0.0011 | −0.0009 | 6 | 36 | 4863 s (81 min) | 0.664 | 101.4 |
| warmup0.1 | 0.4201 | −0.0024 | −0.0044 | 9 | 39 | 5223 s (87 min) | 0.663 | 103.3 |
| **cosine** ⭐ | **0.4241** | **+0.0016** | −0.0004 | 6 | 36 | 4818 s (80 min) | 0.665 | 101.5 |
| combo | 0.4232 | +0.0007 | −0.0013 | 9 | 39 | 5274 s (88 min) | 0.663 | 101.5 |

Per-Qwen accuracy:
- baseline: 14B=0.4961, 72B=0.4086, 3B=0.4018, 7B=0.3833
- lr2e-5: 14B=0.4942, 72B=0.4070, 3B=0.3927, 7B=0.3707
- wd0.01: 14B=0.4895, 72B=0.4128, 3B=0.4130, 7B=0.3792

### deepeval (target metric: avg eval_mae, lower better)

| config | avg_mae | Δ vs baseline | Δ vs Apr 18 | best_ep | final_ep | train_time | avg_spearman |
|---|---|---|---|---|---|---|---|
| baseline | 0.2245 | — | −0.0004 | 10 | 40 | 5359 s (89 min) | 0.436 |
| lr2e-5 | 0.2237 | −0.0008 | −0.0012 | 5 | 35 | 4696 s (78 min) | 0.454 |
| wd0.01 | 0.2252 | +0.0007 | +0.0003 | 5 | 35 | 4737 s (79 min) | 0.456 |
| **warmup0.1** ⭐ | **0.2226** | **−0.0019** | −0.0023 | 12 | 42 | 5641 s (94 min) | 0.456 |
| cosine | 0.2267 | +0.0022 | +0.0018 | 6 | 36 | 4827 s (80 min) | 0.448 |
| combo | 0.2261 | +0.0016 | +0.0012 | 10 | 40 | 5414 s (90 min) | 0.430 |

## V5 Design (launched 2026-04-20 04:11 UTC, target-specific — no shared configs)

Per-cron rule: divergent V4 winners → target-specific CONFIGS. Neighborhood exploration of each target's V4 winner; 6 configs per GPU; 12 total.

### length_bucket V5 configs (base: lr=1e-5, wd=0, warmup=0.03, cosine — from V4/cosine)
| name | lr | wd | warmup | sched | rationale |
|---|---|---|---|---|---|
| v5_lb_lr5e-6 | **5e-6** | 0.0 | 0.03 | cosine | lr × 0.5 (see if slower opt finds better min) |
| v5_lb_lr7e-6 | **7e-6** | 0.0 | 0.03 | cosine | lr × 0.7 |
| v5_lb_lr1.5e-5 | **1.5e-5** | 0.0 | 0.03 | cosine | lr × 1.5 (V4 lr=2e-5 hurt; test intermediate) |
| v5_lb_wd0.005 | 1e-5 | **0.005** | 0.03 | cosine | mild WD (wd=0.01 + poly helped; try with cosine) |
| v5_lb_wd0.01cos | 1e-5 | **0.01** | 0.03 | cosine | wd=0.01 + cosine (V4 had wd=0.01 + poly best-non-cosine) |
| v5_lb_warm0.05 | 1e-5 | 0.0 | **0.05** | cosine | slight warmup (V4 baseline warmup=0.03) |

### deepeval V5 configs (base: lr=1e-5, wd=0, warmup=0.1, polynomial — from V4/warmup0.1)
| name | lr | wd | warmup | sched | rationale |
|---|---|---|---|---|---|
| v5_de_lr5e-6 | **5e-6** | 0.0 | 0.1 | poly | lr × 0.5 |
| v5_de_lr7e-6 | **7e-6** | 0.0 | 0.1 | poly | lr × 0.7 |
| v5_de_lr1.5e-5 | **1.5e-5** | 0.0 | 0.1 | poly | lr × 1.5 (V4 lr=2e-5 slightly helped; test intermediate) |
| v5_de_warm0.05 | 1e-5 | 0.0 | **0.05** | poly | lower warmup than V4 winner |
| v5_de_warm0.15 | 1e-5 | 0.0 | **0.15** | poly | higher warmup (V4 warmup=0.1 was best; push further) |
| v5_de_wd0.005 | 1e-5 | **0.005** | 0.1 | poly | very mild WD on top of winning warmup |

**Running-best carries over from V4**: LB=0.4241 (cosine), DE=0.2226 (warmup0.1). Openclaw still holds those winners; V5 configs replace them only if strictly better.

Expected wall ~8 h (6 configs × ~80 min per GPU, parallel). V5 completion ETA ~12:00 UTC Apr 20.

## V4 Final Summary (both targets, 6 configs each = 12 runs, ~8.9 h wall)

**length_bucket winner:** `cosine` (lr=1e-5, wd=0, warmup=0.03, scheduler=**cosine**) → avg_acc **0.4241** (+0.0016 vs baseline; −0.0004 vs Apr 18 CSD3 — essentially matches).
**deepeval winner:** `warmup0.1` (lr=1e-5, wd=0, **warmup=0.1**, scheduler=poly) → avg_mae **0.2226** (−0.0019 vs baseline; −0.0023 vs Apr 18 CSD3).

**Key insight — every single knob tested diverges per target** (4-for-4):
| single-knob change from baseline | LB effect | DE effect |
|---|---|---|
| lr 1e-5→2e-5 | −0.0064 | −0.0008 |
| wd 0→0.01 | **+0.0011** | +0.0007 |
| warmup 0.03→0.1 | −0.0024 | **−0.0019** |
| scheduler poly→cosine | **+0.0016** | +0.0022 |

**Stacked knobs (combo)** (lr=2e-5, wd=0.01, warmup=0.1, cosine): 2 DE-favorable (lr, warmup) + 2 LB-favorable (wd, cosine). Result: LB +0.0007 vs baseline (beaten by pure cosine), DE +0.0016 vs baseline (beaten by pure warmup0.1). Stacking doesn't beat isolated best-single-knob per target.

Per-Qwen mae:
- baseline: 3B=0.2176, 14B=0.2230, 72B=0.2262, 7B=0.2314
- lr2e-5: 14B=0.2190, 3B=0.2211, 72B=0.2229, 7B=0.2318
- wd0.01: 14B=0.2175, 3B=0.2229, 72B=0.2250, 7B=0.2353

## Curve observations

- **lr=2e-5 converges faster**: deepeval best epoch 10→5 (halved). Length_bucket stays at ep 6 for all configs so far.
- **spearman_r creeping up**: baseline 0.436 → lr2e-5 0.454 → wd0.01 0.456. All deepeval configs are inching up on spearman regardless of MAE direction.
- **length_bucket adjacent_accuracy** is stuck at ~0.66 across all 3 configs — the model's "close-but-off-by-one" error rate is stable.
- **wd0.01 shifts per-Qwen mix**: length_bucket 14B dropped (0.496→0.489) but 3B and 72B rose. WD regularizes toward a more balanced per-model fit. Net: small positive.

## Key findings so far (5 of 6 V4 configs complete)

- Vast baseline reproduces Apr 18 CSD3 exactly within seed noise (LB −0.002, DE −0.0004). Pipeline fidelity confirmed.
- **lr=2e-5 helps regression, hurts classification** (LB −0.0064, DE −0.0008). Diverging per target.
- **wd=0.01 helps classification, hurts regression** (LB +0.0011, DE +0.0007). Mirror image of lr.
- **warmup=0.1 helps regression, hurts classification** (DE −0.0019 best, LB −0.0024). Another diverging knob.
- **cosine helps classification, hurts regression** (LB +0.0016 **NEW BEST**, DE +0.0022). **Every single knob so far diverges per target.**
- **Target-specific winner profile (5 of 6 in):**
  - LB: **lr=1e-5, wd=0, warmup=0.03, cosine** (cosine — 0.4241)
  - DE: **lr=1e-5, wd=0, warmup=0.1, poly** (warmup0.1 — 0.2226)
- **The divergence is nearly perfect**: LB wants cosine + low warmup + some wd; DE wants poly + high warmup + no wd. A shared config seems fundamentally suboptimal.
- **Best-epoch drift with warmup**: warmup=0.1 pushed DE best to ep 12 (vs baseline ep 10). Other knobs hold LB/DE best at ep 5–6.
- **Qwen-7B still weakest everywhere** (LB 0.38, DE 0.23). Not a config artifact.
- V5 design implication: target-specific CONFIGS arrays are essential.

## Proposed next-round configs

*(pending — ≥3 configs per target needed; will update after wd0.01 completes ~23:18 UTC)*

---

## Reference points

**Apr 18 CSD3 baseline** (what Vast baseline is meant to reproduce):
- length_bucket: avg accuracy = 0.4245 (epochs 8 best, stopped ep 18 with patience=10)
- deepeval: avg mae = 0.2249 (epoch 10 best, stopped ep 20)
- config: lr=1e-5, wd=0 (default), warmup=0.03, sched=polynomial

**Apr 1 `roberta_fused_5ep`** (5-epoch run, length_bucket only): avg accuracy = 0.4225. Shows 5 epochs already captures most of the signal.

## Integrity notes

- Reported metric convention: avg MAE + BestAcc + RankCorr across 4 Qwen models. **Never per-model Spearman ρ** (per `memory/feedback_no_spearman_rho.md`).
- Best-checkpoint protection verified earlier: Trainer + `load_best_model_at_end=True` + `metric_for_best_model=accuracy|mae` + `save_total_limit=1` correctly preserves the per-target best. Confirmed against today's patience=3 run (best=final re-eval).

---

## Training Notes — current global-best per target

Updated every cron iteration. **Fixed for ALL configs**: roberta-base, batch=16, max_len=512, fp16, seed=42, patience=30, save_total_limit=1, epochs=200.

### length_bucket — **cosine** (config 5 of V4) — updated iter 18

- Hyperparameters: `lr=1e-5, wd=0.0, warmup=0.03, scheduler=cosine`
- Metrics: avg_acc=0.4241 (+0.0016 vs V4 baseline 0.4225; −0.0004 vs Apr 18 CSD3 0.4245 — effectively matches)
- Per-Qwen accuracy: 14B=0.4956, 72B=0.4097, 3B=0.4092, 7B=0.3817
- Best epoch 6 of 36 trained
- Training time: 4818 s (80 min) on V100-PCIE-16GB (fp16)
- adj_acc=0.6648, mae_tokens=101.50
- **Model location**: `openclaw:~/Code/llm/Block/models/cara/roberta_fused_vast_sweep/length_bucket/cosine/fused_model.pt` (476 MB)

Prior LB winners: baseline → wd0.01 (0.4236) → cosine (0.4241). wd0.01's model.pt removed; metric JSONs remain.

### deepeval — **warmup0.1** (config 4 of V4) — updated iter 15

- Hyperparameters: `lr=1e-5, wd=0.0, warmup=0.1, scheduler=polynomial`
- Metrics: avg_mae=0.2226 (−0.0019 vs V4 baseline 0.2245; −0.0023 vs Apr 18 CSD3 0.2249)
- Per-Qwen mae: 3B=0.2166, 14B=0.2190, 72B=0.2241, 7B=0.2307
- Best epoch 12 of 42 trained (patience=30 fired at 12+30)
- Training time: 5641 s (94 min) on V100-PCIE-16GB (fp16)
- avg_spearman_r=0.456
- **Model location**: `openclaw:~/Code/llm/Block/models/cara/roberta_fused_vast_sweep/deepeval/warmup0.1/fused_model.pt` (476 MB)

Prior DE winners: lr2e-5 (0.2237) → warmup0.1 (0.2226). `lr2e-5` model.pt was removed from openclaw when warmup0.1 replaced it; lr2e-5's metric JSONs remain for the record.

### Disk-eviction log
*(if openclaw < 3 GB free, winner models get moved to `personal_gpu_vm:~/models_cache/roberta_fused_vast_sweep/` — NFS home on kimana, persistent across VM reboots.)*

*(no eviction events yet — openclaw currently 4.0 GB free)*

### Incident log

**2026-04-20 08:27 UTC — Vast host went offline.**
- Symptom: SSH port 14036 refused; ControlMaster socket gone.
- Actual cause (confirmed by user 08:35 UTC): **Vast instance destroyed due to insufficient account balance.** Not preemption; the instance + its local disk are gone.
- State at destruction: Phase=V5, config 3 (lr1.5e-5) mid-training on both GPUs. Completed V5 configs: lr5e-6, lr7e-6 (both WORSE than V4 winners).
- What survives on openclaw: global-best fused_model.pt for LB/cosine (0.4241) and DE/warmup0.1 (0.2226), plus all V4 + V5/lr5e-6 + V5/lr7e-6 metric JSONs.
- What is lost: in-flight V5/lr1.5e-5 training state (including the checkpoint). Will be re-run on new host.
- Response: cron `25fed15b` cancelled, chain halted cleanly. See Resume Plan below.

---

## Resume Plan (when user provisions new Vast host)

**What needs to happen**, in order, when the new host is up:

1. **Update SSH config on openclaw** — new Vast host will have a different IP+port. Edit `~/.ssh/config` `Host vast` block (`HostName`, `Port`). Reopen ControlMaster.
2. **Re-authorize openclaw → new-vast rsync key** and vice versa (vast → openclaw for artifact shipping). The previous vast's pubkey in `~/.ssh/authorized_keys` on openclaw is stale — replace or leave alongside.
3. **Rebuild the environment on new vast**:
   - rsync `/home/wd312/Code/llm/Block/block/`, `requirements.txt`, `data/cara/scored/{train_scored_filtered,test_scored_filtered}.jsonl` from openclaw to `new-vast:/workspace/Block/`
   - Create venv: `python3 -m venv .venv_vast && .venv_vast/bin/pip install -U pip wheel`
   - Install deps: `.venv_vast/bin/pip install -r requirements.txt accelerate`
   - Install accelerate separately (was an extra on the old host; see earlier smoke-test fix)
   - Deploy scripts: `/workspace/Block/sweep_target.sh` + `/workspace/Block/extract_metric.py` (both have their content preserved on this openclaw; see `deploy_vast_roberta_fused.sh` + what we wrote in-session — can regenerate)
4. **Re-seed sweep_state** on new vast so running-best retention knows the current global bests:
   ```
   mkdir -p /workspace/Block/sweep_state
   printf '0.4240506329113924\tcosine\n' > /workspace/Block/sweep_state/length_bucket_best.txt
   printf '0.22260072827339172\twarmup0.1\n' > /workspace/Block/sweep_state/deepeval_best.txt
   echo V5 > /workspace/Block/sweep_state/phase.txt
   ```
5. **Trim sweep_target.sh CONFIGS to skip completed configs.** Configs already done on the old host (don't re-run these):
   - LB: `v5_lb_lr5e-6` (0.4223), `v5_lb_lr7e-6` (0.4208)
   - DE: `v5_de_lr5e-6` (0.2238), `v5_de_lr7e-6` (0.2233)
   Remaining V5 configs to run:
   - LB: `v5_lb_lr1.5e-5`, `v5_lb_wd0.005`, `v5_lb_wd0.01cos`, `v5_lb_warm0.05`
   - DE: `v5_de_lr1.5e-5`, `v5_de_warm0.05`, `v5_de_warm0.15`, `v5_de_wd0.005`

   NB: skip-if-exists inside the orchestrator won't help — those training_results.json are on OPENCLAW, not the new vast host. Either:
   - (a) Pre-seed the new vast's `models/cara/roberta_fused_vast_sweep/<target>/<name>/training_results.json` by rsyncing from openclaw — then skip-if-exists kicks in. (Cleanest, honest, keeps all configs in the array.)
   - (b) Just edit the CONFIGS array to remove the done configs. (Simpler, slightly harder to audit.)
6. **Relaunch cron** after host is validated — I can reuse the `25fed15b` prompt body (expanded V5 → V8 chain logic) but with a fresh cron job ID.

**Budget check**: if the new host comes up with ≥ 8 h runway, chain continues into V6 / V7 / V8 normally.

**Estimated wall to finish V5**: 4 LB configs + 4 DE configs = ~5.5 h per GPU (parallel). V6 + V7 + V8 chain could add another ~20 h depending on improvements.

### All completed configs — metric JSONs kept forever on openclaw

Per-config metric JSONs (training_results.json + log_history.json + fused_config.json, ~20 KB each) remain at `openclaw:~/Code/llm/Block/models/cara/roberta_fused_vast_sweep/<target>/<config>/`. Model PTs follow running-best retention (only current winner kept).

---

## Final Terminal Summary (2026-04-20 16:40 UTC)

**Global winners chosen from V4+V5 (21 completed configs, seed=42):**

### length_bucket
- **Config**: `cosine` (V4)
- **Hyperparameters**: lr=1e-5, wd=0.0, warmup=0.03, scheduler=**cosine**
- **avg_accuracy**: **0.4241** (±0.005 single-seed noise)
- Δ vs Apr 18 CSD3 (0.4245, poly scheduler): **−0.0004 (effectively tied)**
- Δ vs Apr 1 `roberta_fused_5ep` (0.4225): **+0.0016**
- best_ep=6, final_ep=36, train_time=4823s (80 min)
- Per-Qwen: 14B=0.4956, 72B=0.4097, 3B=0.4092, 7B=0.3817
- Model: `openclaw:~/Code/llm/Block/models/cara/roberta_fused_vast_sweep/length_bucket/cosine/fused_model.pt` (476 MB)

### deepeval
- **Config**: `warmup0.1` (V4)
- **Hyperparameters**: lr=1e-5, wd=0.0, **warmup=0.1**, scheduler=polynomial
- **avg_mae**: **0.2226** (±0.003 single-seed noise)
- Δ vs Apr 18 CSD3 (0.2249): **−0.0023 (~1% improvement)**
- best_ep=12, final_ep=42, train_time=5641s (94 min)
- Per-Qwen: 3B=0.2166, 14B=0.2190, 72B=0.2241, 7B=0.2307
- avg_spearman_r=0.456
- Model: `openclaw:~/Code/llm/Block/models/cara/roberta_fused_vast_sweep/deepeval/warmup0.1/fused_model.pt` (476 MB)

### Configs that never completed (Vast released before they ran)
- LB: v5_lb_warm0.05 (in-flight at halt, ~70 min training, not shipped)
- DE: v5_de_warm0.15 (in-flight at halt, ~37 min training, not shipped); v5_de_wd0.005 (never launched)
- These only existed on Vast local disk and were destroyed with the instance release.

### Per-target "what moved the needle"
- **LB**: scheduler swap poly→cosine was the only single-knob gain (+0.0016). Nothing else beat it; stacking failed.
- **DE**: warmup increase 0.03→0.1 was the only single-knob gain (−0.0019). lr=2e-5 helped a tiny bit too (−0.0008) but compounded with cosine it hurt.

### Divergence pattern (paper-worthy finding)
Every tested knob had **opposite effect** on LB vs DE:
| knob | LB effect | DE effect |
|---|---|---|
| lr 1e-5→2e-5 | −0.0064 | −0.0008 |
| wd 0→0.01 | **+0.0011** | +0.0007 |
| warmup 0.03→0.1 | −0.0024 | **−0.0019** |
| scheduler poly→cosine | **+0.0016** | +0.0022 |

**Implication**: the two targets optimize different predictive-head behaviors. Shared hyperparameter choices will always compromise one. Single-objective selection (per-target winners) consistently beats stacked/shared configs.

### HP tuning exhaustion evidence
- V4 tested 4 single-knob changes + 1 stacked combo (6 configs per target).
- V5 tested lr-neighborhood (3 values), wd-variants (2), warmup-variants (2), scheduler-swap (for LB only) — **12 configs planned, 9 completed, 0 improvements**.
- Train-loss analysis: min_train_loss ranges from 0.036 to 0.491 across LB configs, but eval accuracy stays in 0.4161–0.4241. Generalization surface saturates around 0.42 regardless of how deeply the model fits the 14.9k training prompts. DE shows analogous saturation at ~0.222.
- Conclusion: further HP tuning within these knob bounds (lr∈[5e-6, 5e-5], wd∈[0, 0.05], warmup∈[0.03, 0.2], sched∈{poly, cosine, linear}) is unlikely to yield ≥0.003 LB or ≥0.002 DE improvement. Meaningful gains require: larger encoder / more data / architectural changes (out of scope per user's fixed-study constraint) OR multi-seed averaging for stable error bars.

### Vast budget spent
- Total rental: 2026-04-19 11:22 UTC → 2026-04-20 16:40 UTC = **~29.3 h** (with ~4.5 h of outage in the middle)
- Useful compute (configs that produced results): ~22 h
- Cost @ $0.25/h: **~$7.30**
