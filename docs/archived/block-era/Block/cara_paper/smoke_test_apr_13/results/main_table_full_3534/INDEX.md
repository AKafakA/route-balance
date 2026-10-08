# main_table_full_3534/ index — last updated 2026-05-04 01:36 UTC

Topology: 13-host cluster (3 A30-3B + 5 A30-7B + 3 V100-14B + 2 A100-72B), APC OFF for sensitivity sweep, P100 dropped (Pareto-dominated).

## Phase 1: cara sensitivity sweep (24 cells, APC=OFF)
| Cell | Predictor | w_qual | w_lat | w_cost | λ | N | Status | rps | mean_e2el_ms | duration_s |
|---|---|---|---|---|---|---|---|---|---|---|
| cara-knn-full/wq0.5_lambda6_n3534.json | knn | 0.5 | 0.3 | 0.2 | 6 | 3534 | error_C3524_F10 | 5.96 | 1772 | 591.7 |
| cara-knn-full/wq0.5_lambda8_n3534.json | knn | 0.5 | 0.3 | 0.2 | 8 | 3534 | error_C3494_F40 | 7.85 | 2064 | 445.0 |
| cara-knn-full/wq0.5_lambda10_n3534.json | knn | 0.5 | 0.3 | 0.2 | 10 | 3534 | NO_FILE | - | - | - |
| cara-knn-full/wq0.5_lambda12_n3534.json | knn | 0.5 | 0.3 | 0.2 | 12 | 3534 | error_C3499_F35 | 9.56 | 36038 | 366.1 |
| cara-knn-full/wq0.7_lambda6_n3534.json | knn | 0.7 | 0.2 | 0.1 | 6 | 3534 | error_C3525_F9 | 5.97 | 2147 | 590.7 |
| cara-knn-full/wq0.7_lambda8_n3534.json | knn | 0.7 | 0.2 | 0.1 | 8 | 3534 | error_C3513_F21 | 7.82 | 2735 | 449.0 |
| cara-knn-full/wq0.7_lambda10_n3534.json | knn | 0.7 | 0.2 | 0.1 | 10 | 3534 | NO_FILE | - | - | - |
| cara-knn-full/wq0.7_lambda12_n3534.json | knn | 0.7 | 0.2 | 0.1 | 12 | 3534 | error_C3510_F24 | 8.50 | 57015 | 413.2 |
| cara-knn-full/wq0.8_lambda6_n3534.json | knn | 0.8 | 0.1 | 0.1 | 6 | 3534 | error_C3521_F13 | 5.92 | 2432 | 594.4 |
| cara-knn-full/wq0.8_lambda8_n3534.json | knn | 0.8 | 0.1 | 0.1 | 8 | 3534 | error_C3518_F16 | 7.83 | 3685 | 449.5 |
| cara-knn-full/wq0.8_lambda10_n3534.json | knn | 0.8 | 0.1 | 0.1 | 10 | 3534 | NO_FILE | - | - | - |
| cara-knn-full/wq0.8_lambda12_n3534.json | knn | 0.8 | 0.1 | 0.1 | 12 | 3534 | error_C3505_F29 | 8.39 | 60437 | 418.0 |
| cara-roberta-full/wq0.5_lambda6_n3534.json | roberta | 0.5 | 0.3 | 0.2 | 6 | 3534 | ok | 5.97 | 1747 | 591.7 |
| cara-roberta-full/wq0.5_lambda8_n3534.json | roberta | 0.5 | 0.3 | 0.2 | 8 | 3534 | ok | 7.94 | 1904 | 445.1 |
| cara-roberta-full/wq0.5_lambda10_n3534.json | roberta | 0.5 | 0.3 | 0.2 | 10 | 3534 | NO_FILE | - | - | - |
| cara-roberta-full/wq0.5_lambda12_n3534.json | roberta | 0.5 | 0.3 | 0.2 | 12 | 3534 | ok | 10.43 | 23587 | 338.8 |
| cara-roberta-full/wq0.7_lambda6_n3534.json | roberta | 0.7 | 0.2 | 0.1 | 6 | 3534 | ok | 5.97 | 1791 | 591.7 |
| cara-roberta-full/wq0.7_lambda8_n3534.json | roberta | 0.7 | 0.2 | 0.1 | 8 | 3534 | ok | 7.95 | 2069 | 444.6 |
| cara-roberta-full/wq0.7_lambda10_n3534.json | roberta | 0.7 | 0.2 | 0.1 | 10 | 3534 | NO_FILE | - | - | - |
| cara-roberta-full/wq0.7_lambda12_n3534.json | roberta | 0.7 | 0.2 | 0.1 | 12 | 3534 | ok | 9.50 | 40187 | 371.8 |
| cara-roberta-full/wq0.8_lambda6_n3534.json | roberta | 0.8 | 0.1 | 0.1 | 6 | 3534 | ok | 5.97 | 1863 | 591.7 |
| cara-roberta-full/wq0.8_lambda8_n3534.json | roberta | 0.8 | 0.1 | 0.1 | 8 | 3534 | ok | 7.96 | 2374 | 444.1 |
| cara-roberta-full/wq0.8_lambda10_n3534.json | roberta | 0.8 | 0.1 | 0.1 | 10 | 3534 | NO_FILE | - | - | - |
| cara-roberta-full/wq0.8_lambda12_n3534.json | roberta | 0.8 | 0.1 | 0.1 | 12 | 3534 | ok | 8.96 | 49375 | 394.6 |
