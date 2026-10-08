# Related Work Survey — March 25, 2026

## Papers Added to Related Work Section

### LLM Model Routing
- PORT [2509.02718] — Training-free ANNS, aggregate budget, NeurIPS 2025
- OmniRouter [2502.20576] — Lagrangian, per-request + aggregate
- PILOT [2508.21141] — Contextual bandit, aggregate
- KNN routing [2505.12601] — KNN beats learned routers
- Dynamic Quality-Latency [2508.11291] — Joint quality+latency, edge networks

### Scheduling
- k-LPM [2502.04677] — TTFT/TPOT NP-hard, NeurIPS 2025
- PARS [2510.03243] — Prompt-aware SJF via length ranking
- Proteus (ASPLOS 2024) — Accuracy scaling under load
- BucketServe [2507.17120] — Input-length bucketing (disambiguate from our output bucketing)

### Budget/Length Control
- BudgetThinker [2508.17196] — Control tokens + RL per-request
- BeLLMan [2510.15330] — Infrastructure congestion signals
- TimeBill [2512.21859] — Time-budgeted inference with length prediction
- Length Representations [2507.20398] — How LLMs encode length internally
- Length Feedback [2601.01768] — Dynamic feedback for length regulation
- Plan-and-Write [2511.01807] — Structure-guided length control

### Heterogeneous
- Demystifying Het. GPU [2502.00722] — MILP for mixed GPU scheduling

## Papers Discussed but NOT in Related Work (future work / notes)
- xRouter [2510.08439] — RL routing, per-request (subsumed by OmniRouter discussion)
- Cascadia [2506.04203] — Cascade serving (subsumed by FrugalGPT)
- AugServe [2512.04013] — Tool-augmented scheduling
- Aragog [2511.20975] — Agentic workflow routing
- OI-MAS [2601.04861] — Multi-agent confidence routing
- AgentBalance [2512.11426] — Multi-agent budget
- Thinking Budget [2512.19585] — Reasoning token budget (future work)
- FusionFactory [2507.10540] — Multi-LLM fusion

## Key Positioning
Cara uniquely combines quality-aware routing + load-aware scheduling with:
1. Per-request probabilistic budget compliance (bucket distributions) — novel
2. TTFT+TPOT decomposition avoiding length dependency — validated by k-LPM NP-hardness
3. Works with unmodified off-the-shelf models — unlike BudgetThinker/BeLLMan
4. Heterogeneous multi-model multi-GPU — beyond single-model routers or schedulers
