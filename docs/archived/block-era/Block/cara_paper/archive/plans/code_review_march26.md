# Cara Code Review — March 26, 2026

## Summary
Review of all Cara code (predictor, scheduler, benchmark, training, deployment).
Builds on the March 24 code review (34 issues) — checks fix status and adds new findings.

## Status of March 24 CRITICAL Bugs

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | `train_llm_predictor.py` IndentationError | **FIXED on VPS** (March 24) | `self.examples.append` moved back to correct indentation. Synced to GPU VM + CSD3. |
| 2 | `train_bert_predictor.py` data leakage | **LOW RISK** | Only triggers without `--test-input`. All our runs use `--test-input`. |
| 3 | `judge_class` mapping broken | **MITIGATED** (March 24) | Safety-aware `_get_judge_score()` always returns 0-1 range, so the discontinuity at score=1.0 never triggers. |
| 4 | `Instance.py` output_tokens missing | **FIXED on VPS** (March 23) | Not merged to cara-small-a100 branch. Must merge before CloudLab. |
| 5 | `generate_latency_benchmark.py` tuple unpack | **FIXED on VPS** (March 23) | `sample_real_requests` returns 4 values, callers updated. |
| 6 | `fix_safety_scores.py` empty response | **UNFIXED** | Not used in training pipeline. Low priority. |
| 7 | `host_configs.json` missing non-A30 hosts | **UNFIXED** | Must generate for new CloudLab reservation hosts. |
| 8 | `deploy_cara.py` argparse bool | **UNFIXED** | Must fix before CloudLab. |
| 9 | `cara_serve.py` sampling param override | **UNFIXED** | Silently corrupts ALL benchmarks. Must fix before CloudLab. |

**Fixed on VPS (not merged to branch):** #1, #4, #5. Must merge before CloudLab.
**Mitigated:** #2, #3. Low risk for our usage.
**Still UNFIXED (blocking CloudLab):** #7, #8, #9, #18/#26 (QPS control).

## New Findings (March 26)

### Code Architecture

**Positive:**
- Clean separation: predictor, scheduler, benchmark, training
- Modular estimator design (knn, lstm, mlp, xgboost, linear, roofline)
- Evaluation framework with adapters pattern
- Config-driven deployment

**Negative:**

19. **No tests** — Zero unit tests, zero integration tests. No way to verify correctness after changes.

20. **No Cara-specific requirements.txt** — Uses Block's `requirements.txt` which lacks `sentence-transformers`, `peft`, `faiss-cpu`. Manual dependency management.

21. **Hardcoded paths** — `predictor_deployment_config.json` has `/users/asdwb/` paths. Not portable.

22. **No README** in cara subdirectories. No docstrings on public APIs.

### Runtime Concerns

23. **`cara_learned_predictor.py` GPU conflict** — Loads ModernBERT to GPU unconditionally. On serving nodes sharing GPU with vLLM, this causes OOM. Need CPU-only option or lazy loading.

24. **`aiohttp.ClientSession` per request** in `ScheduleTraceClient` — Creates and destroys session per call. File descriptor leak under load. Should use a persistent session.

25. **`cara_serve.py` global mutable state** — Global dicts for pending requests, instance tracking. Breaks with `workers > 1` (uvicorn multiprocessing). Either pin workers=1 or use shared state.

26. **`run_e2e_benchmark.py` QPS control** (#18 from March 24) — Still broken. `asyncio.gather` launches ALL requests simultaneously regardless of QPS setting. Must fix before any e2e evaluation.

27. **`run_e2e_benchmark.py` switch_scheduler** (#19) — Still a no-op stub. Multi-scheduler comparison loop shows different labels but identical behavior.

### Training Pipeline

28. **No unified experiment runner** — Each training script is independent with different CLI interfaces. No single script to reproduce all predictor training. Need a `train_all.sh` or similar.

29. **Checkpoint management** — `train_bert_predictor.py` now auto-cleans checkpoints (fixed on VPS). But other training scripts (LSTM, MLP) don't clean up. Disk fills quickly on GPU VM.

30. **XGBoost temporal leakage** (#21 from March 24) — Uses random split for time-series latency data. Records from the same QPS level and time window leak between train/test. Should split by QPS level or time.

### Data Pipeline

31. **HuggingFace datasets stale** — `cara_latency_prediction` has concurrency-capped data (known buggy). `cara_model_estimator` missing `dataset` and `is_harmful` fields. Both need re-upload before paper submission.

32. **Actual output tokens recovery** — Formula `round((e2e-ttft)/tpot)+1` verified exact on 4,690 records. All local data tagged. But the recovery formula should be documented in the code, not just in memory docs.

## Fix Priority for April 26 CloudLab

### BLOCKING (experiments won't work without these)
1. **#26** `run_e2e_benchmark.py` QPS control — `asyncio.create_task()` not bare coroutine. **Without this, all requests fire simultaneously regardless of QPS setting. No valid E2E evaluation.**
2. **#9** `cara_serve.py` sampling param override — remove forced `frequency_penalty=1.2`. Corrupts ALL benchmark results silently.
3. **#7** `host_configs.json` — generate for all 18 new CloudLab hosts.
4. **#27** `switch_scheduler` stub — implement actual scheduler switching for baseline comparison.
5. **#4** `Instance.py` output_tokens — merge VPS fix to branch.
6. **#8** `deploy_cara.py` argparse bool — `type=bool` → `action="store_false"`.
7. **#23** `cara_learned_predictor.py` GPU conflict — `.cuda()` on line 118 will OOM on serving nodes sharing GPU with vLLM. Add `--device cpu` option or lazy loading.

### Must Fix (correct results)
8. **#24** `ScheduleTraceClient` — persistent aiohttp.ClientSession instead of per-request. FD leak under load.
9. **#25** `cara_serve.py` global mutable state — pin `workers=1` in deployment scripts.
10. **#10** LSTM off-by-one — already fixed on VPS, verify on CSD3.

### Already Fixed on VPS (merge to branch before CloudLab)
- **#1** `train_llm_predictor.py` indentation — FIXED March 24
- **#3** `judge_class` mapping — MITIGATED (safety-aware 0-1 range)
- **#4** `Instance.py` output_tokens — FIXED March 23
- **#5** `generate_latency_benchmark.py` tuple — FIXED March 23

### Should Fix (artifact quality)
11. Add smoke tests for each predictor
12. Create `requirements-cara.txt`
13. Remove hardcoded `/users/asdwb/` paths
14. Fix GPU conflict in `cara_learned_predictor.py`
