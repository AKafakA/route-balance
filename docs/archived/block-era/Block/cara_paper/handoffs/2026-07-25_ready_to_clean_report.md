# Ready-to-clean report — duplicated / stale raw data beside the v1+v2 paper-ready sets (2026-07-25)

**⛔ NOTHING HAS BEEN DELETED. This is a MARKED LIST awaiting explicit author approval, item by item
or by group.** Context: v1 basis now packed at `/home/wd312/Code/llm/rb-exp/v1-exp/` (hardlinked pack,
manifest inside); v2 basis packed at `/home/wd312/Code/llm/v2-exp/`. Disk: 15G free / 116G.
Recommended order if approved: tar+rsync an off-machine copy of both packs FIRST, then clean by group.

## KEEP — never candidates (paper-ready or deployed; do not touch)
- `cara_paper/.../main_table_full_3534/cara_final_e2e/` 3.6G — the submitted 243-config grid (now also
  referenced by the v1-exp pack via hardlinks).
- `experiment_output/v2_sweep_jun06/` 11G — the v2/EuroSys grid (see consolidation OPTION below).
- `data/cara/train_scored_filtered.jsonl` 90M + `data/cara/scored/` 59M — quality lookups (also on HF).
- Deployed models ×2 locations: `models/cara/latency/deploy_*`, `quality/judge/deploy_knn_k10`,
  `length_bucket`, `length_regression` (v2-exp gap CLOSED 2026-07-25 via hardlink), `best_route_4way_qwen`,
  avengers artifacts; `models/cara/quality/judge/deploy_knn` (Jun-6 k50 reconstruction — v1-exp pack).
- Both packs; `data/cara/best-route-v3*` benches; `qwen_lora_fused` (LoRA-keep rule).

## Group A — superseded PRE-FINAL v1 result runs (~1.05G) [confidence: high]
| path (under `cara_paper/smoke_test_apr_13/results/main_table_full_3534/`) | size | why stale |
|---|---|---|
| `cara-knn-full/` | 395M | May-4 KNN-vs-RoBERTa comparison era; superseded by cara_final_e2e |
| `cara-roberta-full/` | 319M | May-4; RoBERTa arm absent from the submitted paper |
| `filter_ablation/` | 110M | May-2 precursor; final slo_ablation track lives inside cara_final_e2e |
| `budget_control/` | 101M | May-2 precursor; final budget_calibrated_demo inside cara_final_e2e |
| `weights_ablation/` | 55M | May-2 precursor of the weight sweep |

## Group B — April methodology-era results (~750M) [high]
`results/ablation_3634` 484M · `results/main_table` 92M · `results/ablation` 83M ·
`results/sat_{discovery,down,low}_apr26` ~63M · `results/latency_sweep_overnight_apr26` 26M
(the apr26 sweeps fed the **never-deployed** apr28 retrain). All superseded by the final May grid.

## Group C — model artifacts (~3.5G) [medium-high; per-item notes]
| path (under `models/cara/`) | size | note |
|---|---|---|
| `roberta_fused_5ep/` | 1.9G | archived RoBERTa path, not in any submitted result; retrainable |
| `roberta_fused_vast_sweep/` | 963M | same era, sweep variants |
| `quality/judge/ablation/` | 573M | ablation index builds (R5-era inputs preserved in aggregates) |
| `latency_predictors_apr28/` | 65M | **the NEVER-DEPLOYED trap** — cleaning it also removes the hazard; trap warning stays in the audits |
| `quality/judge/deploy/` | 24M | older 14,966-example build; note: its `train.log` holds the cited 52.3 s index-build timing — extract the log to v1-exp/docs before cleaning |
| `latency/deploy_lstm_v2/` | 156K | historical (the May-3 fabrication-incident artifact); tiny |
| `best_route_wrapper_qwen/` | 544M | DeBERTa wrapper scaffold; the deployed baseline is best_route_4way_qwen (kept ×2) — confirm not referenced by v1 configs before cleaning |

## Group D — data intermediates (~2.9G) [medium; clean LAST]
| path (under `data/cara/`) | size | note |
|---|---|---|
| `latency_data/` | 1.6G | raw per-instance QPS sweeps (head training data). Heads are deployed+kept ×2; paper policy = latency data regenerated, not redistributed. Keep marked until after the window (cheap insurance if heads ever need retraining at the v1 recipe) |
| `broadcast_v3_20k.json` | 598M | raw pre-scoring broadcast; superseded by scored files |
| `training_data/` + `training_data_with_ref/` | 687M | scoring intermediates; superseded by *_scored_filtered |
| `vast_backup_march31/` | 114M | training-era backup |
| `training_logs/` | 32M | logs |

## Group E — misc (~150M) [high]
`experiment_output/v2_repro_smoke/` 146M — post-pack validation smokes, purpose served.

## Group F — ⚠ Block-project (NON-cara) data — excluded from this approval; confirm separately
`data/profiling` 276M · `data/length_estimation` 227M · `data/trace_data` 93M — belong to the
Block/SoCC project, not RouteBalance.

## OPTION (not a deletion): hardlink-dedup the v2 duplicate (~frees up to ~9-11G)
`experiment_output/v2_sweep_jun06` (11G) and `v2-exp/results` largely duplicate each other on the SAME
disk. Replacing md5-identical file pairs with hardlinks keeps BOTH paths intact while freeing the
duplicate blocks. Requires a verification pass (md5 match per file) + your approval; zero data loss by
construction. Biggest single space win available.

**Totals if A+B+C+D+E approved: ≈ 8.4G freed** (+ up to ~11G more via the dedup option).
