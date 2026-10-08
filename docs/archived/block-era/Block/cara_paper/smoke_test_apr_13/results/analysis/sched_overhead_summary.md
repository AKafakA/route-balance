# Cara Scheduler Overhead Breakdown

Per-request scheduling cost (ms) from `scheduling_overhead_breakdown` in each response.

Run class = run prefix. `cara` scheduler runs the full batch + scoring pipeline;
`random` / `round_robin` / `shortest_queue` skip most stages.

| run | n | total p50 | total p95 | batch_wait p50 | estimator p50 | stats_fetch p50 | sidecar p50 | scoring p50 |
|-----|--|:---------:|:---------:|:--------------:|:-------------:|:---------------:|:-----------:|:-----------:|
| apr14_comp_filter_cara_tiered_20260414_164228 | 50 | 242.4 | 377.34 | 108.61 | 84.64 | 11.07 | 1.25 | 0.11 |
| apr14_comp_filter_none_20260414_164228 | 50 | 284.5 | 498.23 | 100.91 | 101.93 | 10.4 | 1.42 | 0.11 |
| apr14_comp_filter_polyserve_20260414_164228 | 50 | 241.94 | 366.49 | 110.55 | 95.16 | 11.4 | 1.26 | 0.1 |
| apr14_comp_filter_qlm_20260414_164228 | 50 | 248.81 | 357.66 | 111.97 | 99.51 | 12.85 | 1.24 | 0.1 |
| apr14_comp_filter_slos_serve_20260414_164228 | 50 | 229.57 | 410.68 | 101.08 | 97.92 | 11.54 | 1.29 | 0.11 |
| apr14_comp_qps_12_20260414_164228 | 100 | 1644.78 | 3758.89 | 116.84 | 110.88 | 2.07 | 1.61 | 0.08 |
| apr14_comp_qps_16_20260414_164228 | 200 | 1290.26 | 10074.31 | 119.6 | 106.05 | 1.73 | 1.58 | 0.08 |
| apr14_comp_qps_20_20260414_164228 | 200 | 1765.66 | 10230.87 | 134.94 | 95.45 | 2.14 | 1.6 | 0.08 |
| apr14_comp_qps_2_20260414_164228 | 50 | 196.62 | 362.9 | 100.17 | 103.98 | 11.45 | 1.33 | 0.11 |
| apr14_comp_qps_4_20260414_164228 | 50 | 226.21 | 382.37 | 100.97 | 91.94 | 12.55 | 1.29 | 0.11 |
| apr14_comp_qps_8_20260414_164228 | 100 | 627.09 | 1664.31 | 121.09 | 96.56 | 3.54 | 1.61 | 0.09 |
| apr14_comp_scheduler_cara_20260414_164228 | 200 | 1126.64 | 9152.23 | 126.56 | 107.64 | 2.48 | 1.6 | 0.08 |
| day1.4_filter_cara_tiered | 50 | 251.38 | 386.45 | 111.89 | 96.59 | 12.75 | 1.26 | 0.1 |
| day1.4_filter_none | 50 | 249.88 | 420.0 | 111.97 | 102.38 | 11.34 | 1.3 | 0.1 |
| day1.4_filter_polyserve | 50 | 231.7 | 389.69 | 100.95 | 89.6 | 11.12 | 1.3 | 0.11 |
| day1.4_filter_qlm | 50 | 246.85 | 382.84 | 110.18 | 88.49 | 13.14 | 1.25 | 0.1 |
| day1.4_filter_slos_serve | 50 | 242.88 | 380.55 | 107.41 | 84.4 | 10.85 | 1.35 | 0.11 |
| day2.2_latency_high_20260413_201758 | 200 | 1659.65 | 9559.26 | 118.38 | 96.34 | 2.1 | 1.55 | 0.08 |
| day2.2_latency_low_20260413_201758 | 50 | 217.06 | 366.47 | 101.06 | 79.81 | 11.61 | 1.25 | 0.1 |
| day2.2_latency_med_20260413_201758 | 200 | 970.59 | 5442.4 | 130.97 | 86.73 | 3.15 | 1.55 | 0.08 |
| day2.4_baseline_cara_20260413_203237_baselines | 200 | 2096.84 | 5413.78 | 163.22 | 137.58 | 2.22 | 1.65 | 0.09 |

## Observations worth putting in the paper

- **cara total overhead:** p50=2096.84 ms, p95=5413.78 ms over 200 reqs
- **cara estimator (predictor) cost:** p50=137.58 ms — this is the dominant component
