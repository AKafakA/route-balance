# Phase 2/3 vs revision divergence

Generated: 2026-04-20T09:25Z

## Phase 2: n_ablation (N=4,6,8 est @ QPS=30) vs revision/po2

Current runs: 3; Prior runs: 22

| Key (cur) | Match | throughput Δ | mean_e2e Δ | TTFT P99 Δ |
|---|---|---|---|---|
| sharegpt/min_new_request_latency/qps_30_num_queries_10000_n_ | NOT FOUND | — | — | — |
| sharegpt/min_new_request_latency/qps_30_num_queries_10000_n_ | NOT FOUND | — | — | — |
| sharegpt/min_new_request_latency/qps_30_num_queries_10000_n_ | NOT FOUND | — | — | — |

## Phase 3.1a: burstiness Po2+Llumnix vs revision/burstiness

Current runs: 8; Prior runs: 8

| Key (cur) | Match | throughput Δ | mean_e2e Δ | TTFT P99 Δ |
|---|---|---|---|---|
| burstiness_0.25/sharegpt/min_lunmnix_load/qps_30_num_queries | ✓ | +1.3% | -1.0% | -52.0% |
| burstiness_0.25/sharegpt/min_new_request_latency/qps_30_num_ | ✓ | +1.0% | -1.8% | +4.0% |
| burstiness_0.5/sharegpt/min_lunmnix_load/qps_30_num_queries_ | ✓ | +1.5% | +0.1% | -15.5% |
| burstiness_0.5/sharegpt/min_new_request_latency/qps_30_num_q | ✓ | +0.8% | +1.1% | +233.1% |
| burstiness_1.0/sharegpt/min_lunmnix_load/qps_30_num_queries_ | ✓ | +0.8% | +0.8% | +28.2% |
| burstiness_1.0/sharegpt/min_new_request_latency/qps_30_num_q | ✓ | +1.2% | +0.6% | +320.2% |
| burstiness_2.0/sharegpt/min_lunmnix_load/qps_30_num_queries_ | ✓ | +0.0% | +1.2% | +48.1% |
| burstiness_2.0/sharegpt/min_new_request_latency/qps_30_num_q | ✓ | +1.5% | +2.3% | +353.1% |

## Phase 3.1b: burstiness Fanout-est vs revision/burstiness

Current runs: 4; Prior runs: 8

| Key (cur) | Match | throughput Δ | mean_e2e Δ | TTFT P99 Δ |
|---|---|---|---|---|
| burstiness_0.25/sharegpt/min_new_request_latency/qps_30_num_ | ✓ | +1.6% | -2.3% | -78.9% |
| burstiness_0.5/sharegpt/min_new_request_latency/qps_30_num_q | ✓ | +1.0% | -0.3% | -52.1% |
| burstiness_1.0/sharegpt/min_new_request_latency/qps_30_num_q | ✓ | +1.4% | +0.6% | -12.1% |
| burstiness_2.0/sharegpt/min_new_request_latency/qps_30_num_q | ✓ | +1.6% | +1.7% | +10.4% |

## Phase 3.2a: error_heatmap Po2+Llumnix vs revision/prediction_error

Current runs: 17; Prior runs: 20

| Key (cur) | Match | throughput Δ | mean_e2e Δ | TTFT P99 Δ |
|---|---|---|---|---|
| len_err_0_lat_err_0/sharegpt/min_lunmnix_load/qps_30_num_que | ✓ | -0.1% | +0.5% | +61.7% |
| len_err_0_lat_err_0/sharegpt/min_new_request_latency/qps_30_ | ✓ | +1.6% | +1.3% | +333.4% |
| len_err_0_lat_err_100/sharegpt/min_new_request_latency/qps_3 | ✓ | +1.1% | -0.6% | +79.2% |
| len_err_0_lat_err_25/sharegpt/min_new_request_latency/qps_30 | ✓ | +0.9% | -1.1% | +295.6% |
| len_err_0_lat_err_50/sharegpt/min_new_request_latency/qps_30 | ✓ | +0.7% | -0.0% | +59.6% |
| len_err_100_lat_err_0/sharegpt/min_new_request_latency/qps_3 | ✓ | +0.1% | +1.5% | +446.0% |
| len_err_100_lat_err_100/sharegpt/min_new_request_latency/qps | ✓ | +3.1% | -0.4% | +206.7% |
| len_err_100_lat_err_25/sharegpt/min_new_request_latency/qps_ | ✓ | +0.5% | +0.9% | +435.9% |
| len_err_100_lat_err_50/sharegpt/min_new_request_latency/qps_ | ✓ | -0.1% | -2.3% | -6.0% |
| len_err_25_lat_err_0/sharegpt/min_new_request_latency/qps_30 | ✓ | +1.5% | +1.5% | +364.6% |
| len_err_25_lat_err_100/sharegpt/min_new_request_latency/qps_ | ✓ | +1.0% | -0.1% | +148.4% |
| len_err_25_lat_err_25/sharegpt/min_new_request_latency/qps_3 | ✓ | +0.5% | -0.3% | +454.7% |
| len_err_25_lat_err_50/sharegpt/min_new_request_latency/qps_3 | ✓ | +2.0% | -1.6% | +1.5% |
| len_err_50_lat_err_0/sharegpt/min_new_request_latency/qps_30 | ✓ | +0.8% | +0.2% | +259.6% |
| len_err_50_lat_err_100/sharegpt/min_new_request_latency/qps_ | ✓ | +0.9% | -0.2% | +172.8% |
| len_err_50_lat_err_25/sharegpt/min_new_request_latency/qps_3 | ✓ | +1.3% | +0.5% | +362.1% |
| len_err_50_lat_err_50/sharegpt/min_new_request_latency/qps_3 | ✓ | +0.2% | -1.1% | +12.0% |

## Phase 3.2b: error_heatmap Fanout-est vs revision/prediction_error

Current runs: 16; Prior runs: 20

| Key (cur) | Match | throughput Δ | mean_e2e Δ | TTFT P99 Δ |
|---|---|---|---|---|
| len_err_0_lat_err_0/sharegpt/min_new_request_latency/qps_30_ | ✓ | +0.8% | +0.5% | -8.6% |
| len_err_0_lat_err_100/sharegpt/min_new_request_latency/qps_3 | ✓ | +1.3% | -0.7% | -47.9% |
| len_err_0_lat_err_25/sharegpt/min_new_request_latency/qps_30 | ✓ | +1.3% | -1.2% | -4.1% |
| len_err_0_lat_err_50/sharegpt/min_new_request_latency/qps_30 | ✓ | +0.6% | -1.2% | -66.6% |
| len_err_100_lat_err_0/sharegpt/min_new_request_latency/qps_3 | ✓ | -0.3% | -0.3% | -6.3% |
| len_err_100_lat_err_100/sharegpt/min_new_request_latency/qps | ✓ | +3.2% | -0.7% | -36.9% |
| len_err_100_lat_err_25/sharegpt/min_new_request_latency/qps_ | ✓ | +1.0% | +0.1% | +13.5% |
| len_err_100_lat_err_50/sharegpt/min_new_request_latency/qps_ | ✓ | +1.1% | -2.1% | -75.0% |
| len_err_25_lat_err_0/sharegpt/min_new_request_latency/qps_30 | ✓ | +0.9% | +0.2% | -0.6% |
| len_err_25_lat_err_100/sharegpt/min_new_request_latency/qps_ | ✓ | +0.3% | -0.6% | -29.0% |
| len_err_25_lat_err_25/sharegpt/min_new_request_latency/qps_3 | ✓ | +0.7% | -0.6% | +5.3% |
| len_err_25_lat_err_50/sharegpt/min_new_request_latency/qps_3 | ✓ | +1.8% | -1.3% | -71.3% |
| len_err_50_lat_err_0/sharegpt/min_new_request_latency/qps_30 | ✓ | -0.3% | -0.3% | -4.8% |
| len_err_50_lat_err_100/sharegpt/min_new_request_latency/qps_ | ✓ | +0.4% | -0.6% | -36.0% |
| len_err_50_lat_err_25/sharegpt/min_new_request_latency/qps_3 | ✓ | +0.6% | -0.7% | +6.7% |
| len_err_50_lat_err_50/sharegpt/min_new_request_latency/qps_3 | ✓ | +0.7% | -1.9% | -75.2% |
