# T3 calibrated-budget demo (2026-05-06) — λ=16, per-prompt calibrated budgets
| cell | router/filter | rps | mean_e2el_ms | dist (3B/7B/14B/72B) | violations | exhausted | clamped |
|---|---|---|---|---|---|---|---|
| cara_filter_calibrated_l16 | cara+cara_tiered | 15.72 | 1072 | 3B=69.4% 7B=17.7% 14B=12.8% 72B=0.1% | 1146/3534 (32.4%) | 1154 | 2172 |
| br4_argmax_calibrated_l16 | br4+none | 15.73 | 2138 | 3B=39.8% 7B=9.8% 14B=44.8% 72B=5.6% | 1457/3534 (41.2%) | 1464 | 2367 |
