# Failed Experiments — To Address Later

## 1. RoBERTa fused × 3 targets (length, similarity, judge_class)
- **Error**: `RuntimeError: CUDA error: device-side assert triggered` in `transformers/masking_utils.py` at step 21
- **Cause**: transformers 5.3.0 bug — `eager_mask` still calls `sdpa_mask` internally, CUDA assert on certain padding patterns
- **Tried**: `attn_implementation="eager"` — didn't help (same code path)
- **Fix plan**: `pip install transformers==4.50.3`, run RoBERTa experiments, then `pip install transformers==5.3.0` to restore
- **Status**: Skipped, move on

## 2. XGBoost E2E actual (first attempt)
- **Error**: `--use-actual-output-len` flag parsed but not passed to `prepare_features_and_targets()`
- **Fix**: Added `use_actual_output_len=args.use_actual_output_len` to the call on line 281
- **Status**: Fixed, re-run succeeded (MAE 0.025-0.946s, huge improvement)

## 3. Qwen-0.5B LoRA (early study)
- **Error**: `ModuleNotFoundError: No module named 'peft'`
- **Fix**: `pip install peft`
- **Status**: Fixed, running in resume study

## 4. Early study stopped at Qwen-0.5B failure
- **Error**: `set -e` caused script to stop at first failure
- **Fix**: Removed `set -e`, added `|| echo "FAILED"` per experiment
- **Status**: Fixed in resume script

## 5. Judge score extraction mixed quality + safety scores
- **Error**: For harmful prompts, `llm_judge_scores` contains BOTH Qwen quality judge AND protectai safety classifier. Code averaged them together, giving wrong labels.
- **Correct behavior**: Harmful prompts → use protectai score only (refusal = high). Safe prompts → use Qwen judge only (quality).
- **Fix**: Added `_get_judge_score(m_data, is_harmful)` helper to `train_bert_predictor.py`, `train_llm_predictor.py`, `evaluate_predictors.py`, `knn_estimator.py`
- **Affected completed runs**: ModernBERT fused judge_class (early study ×2), ModernBERT 7B judge_class (full study) — need re-run
- **Unaffected**: All similarity runs, all length runs, Qwen-0.5B similarity (running)
- **Status**: Fixed, synced. ModernBERT judge_class re-runs needed later.

## 6. Qwen-0.5B LoRA judge — IndentationError (March 24)
- **Error**: My logging addition broke indentation — `self.examples.append()` moved inside the `if target in ("judge",...)` block instead of the inner `for` loop. For judge target, logging ran on empty list → crash.
- **Fix**: Moved `self.examples.append()` back to correct indentation, logging after the loop.
- **Status**: Fixed and synced. But judge process started 3 seconds before sync → crashed. **Need re-run.**
- **Note**: length and similarity completed OK (loaded old code before sync broke it)

## 7. LSTM latency — off-by-one target (code_review #10)
- **Error**: `y_windows.append(all_targets[i-1])` uses last item IN the window (already in input), not the NEXT item after window.
- **Fix**: Changed to `all_targets[i]`
- **Status**: Fixed, synced. But LSTM latency already running with old code (started 18:28). **Results will be wrong. Need re-run.**

## 8. LSTM quality — `--test-input` ignored + no seed + wrong judge scores (code_review #11, #12)
- **Error**: Always uses random 10% split from training data, ignores `--test-input`. No seed. Used wrong averaged judge scores.
- **Fix**: Added test_input loading, seed=42, safety-aware judge extraction.
- **Status**: Fixed, synced. LSTM quality hasn't run yet — will use correct code.

## 9. MLP — no seed for train/val split
- **Error**: `mlp_estimator.py` used `np.random.permutation(n)` without seed
- **Fix**: Changed to `np.random.RandomState(42).permutation(n)`
- **Status**: Fixed, synced. MLP already ran with old code. Minor impact (quick to re-run, ~1 min).

## Re-runs completed
1. ✓ Qwen-0.5B LoRA judge — eval_loss=0.135
2. ✓ LSTM latency (CUDA) — 12 min total, all 5 instance types
3. ✓ MLP (with seed) — Quality MAE 0.087-0.119

## Still need re-run (GPU contention OOM)
4. ModernBERT fused judge_class — OOM (another user had 21.7GB on GPU 0). ~30 min.
5. ModernBERT 7B judge_class — Same OOM. ~20 min.
- **Plan A**: Run in gap between long training runs on GPU VM
- **Plan B**: CSD3 `intr` QoS (1h session, instant grant) — needs user MFA auth
