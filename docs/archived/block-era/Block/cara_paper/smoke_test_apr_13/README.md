# Cara 4× A30 Integration Smoke Test — Workspace (Apr 13–15, 2026)

Single workspace for all configs, scripts, results, and docs for the 3-day CloudLab
A30 smoke test on reservation `asdwb-299799`.

## Layout

```
cara_paper/smoke_test_apr_13/
├── README.md                 this file
├── PLAN.md                   3-day plan (authoritative; editable as we learn)
├── TRACKER.md                live status tracker — updated at each checkpoint
├── manifests/
│   └── cl_manifest_a30_apr_12.xml   source of truth for node identities
├── scripts/
│   ├── parse_manifest.py     manifest XML → normalized node records (JSON on stdout)
│   ├── generate_hosts.py     manifest → configs/{hosts,ampere_hosts}
│   ├── generate_deployment_config.py  manifest + role map → configs/model_deployment_a30_smoketest.json
│   ├── generate_host_configs.py       manifest → configs/host_configs.json (IPs + backend/predictor ports)
│   ├── run_setup.sh          stage generated configs into block/config/, then bash block/exp/cara/setup.sh
│   ├── a30_smoke_test.sh     Day 1 sanity runner
│   ├── a30_latency_sweep.sh  Day 2 low/med/high runner
│   ├── a30_broadcasting.sh   Day 2 broadcasting runner (reduced sweep)
│   ├── a30_baselines.sh      Day 2 baseline sweep
│   ├── a30_pd_smoke.sh       Day 3 P-D runner
│   └── validate_latency_schema.py  Day 2.3 HF schema dry-run
├── configs/                  GENERATED — do not hand-edit; regenerate from manifest
├── results/
│   ├── day1/
│   ├── day2/
│   └── day3/
├── logs/                     stdout/stderr from each run (timestamped)
└── handoff/
    └── HANDOFF_APRIL15.md    end-of-window handoff (written Day 3)
```

## Principles

1. **Manifest is the source of truth.** Node hostnames, roles, and counts are parsed
   from `manifests/cl_manifest_a30_apr_12.xml`. Never hand-edit `configs/`.
2. **Single workspace.** Everything related to this 3-day run lives here. The only
   files touched outside this directory are `block/config/{hosts,ampere_hosts}`
   (staged by `scripts/run_setup.sh`, with backups) and the existing
   `block/exp/cara/setup.sh` that setup expects.
3. **Idempotent generation.** Rerunning `scripts/generate_*.py` reproduces identical
   outputs from the same manifest.
4. **No silent changes.** Decisions from the plan file (layout, QPS, P-D split) are
   embedded as CLI defaults in the scripts. Changes require editing `PLAN.md` first
   and flagging in `TRACKER.md`.

## Quick start

```bash
cd /home/wd312/Code/llm/Block

# 1. Regenerate configs from manifest (idempotent)
python3 cara_paper/smoke_test_apr_13/scripts/generate_hosts.py
python3 cara_paper/smoke_test_apr_13/scripts/generate_deployment_config.py

# 2. Day 1 bring-up (stages hosts, runs setup.sh, reboots)
bash cara_paper/smoke_test_apr_13/scripts/run_setup.sh

# 3. Day 1 sanity
bash cara_paper/smoke_test_apr_13/scripts/a30_smoke_test.sh

# ... see PLAN.md for the full sequence
```

Always consult `TRACKER.md` for current state before starting a new phase.
