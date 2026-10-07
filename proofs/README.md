# Proofs: the ref branch does what the code that ran did

The ref branch rewrites the code of the submission branch (commit 984eb1a, the code the paper's runs executed) for
reading. These proofs check, on the archived runs themselves, that the rewrite changes nothing a run did. They run on a
machine that holds the run archive, the artifacts the runs loaded and a scheduler environment. Each tree is called through
an adapter with the same methods and the same plain-data results (`--adapter`): `adapters/submission.py` for the code
that ran, `adapters/ref.py` for this tree.

## Inputs

| input | what | checked against |
|---|---|---|
| the run archive | the 1,457 runs' folders (`runs/qwen/`, `runs/gemma/`), the bundle's views, reports and inputs | the release records |
| the artifacts | every kNN index, router folder, calibration table and latency head the runs loaded | the md5 of every file each run's manifest recorded |
| the upstream checkouts | AvengersPro at its pinned commit with its one recorded patch, best-route-llm at its pinned commit | `scripts/setup.sh`'s pins, and the runs' recorded module md5s |
| the encoders | the two Hub models at their pinned revisions | the runs' recorded file md5s |
| the environment | Python 3.10 with `pip install -e` of the code that ran (the pinned dependencies) | `pip check` |
| the GPU run's engines | vLLM 0.18.1 from PyPI on Python 3.10, as `scripts/setup.sh` installs it, and Qwen2.5-3B and 7B at the engine arguments' revisions | `configs/engine_args_qwen.json` |

## The proofs

| proof | command | compares |
|---|---|---|
| evidence | `python -m proofs.evidence` | every run's record and evidence parsed by the tree's strict schema, its verdict's record and engine checks recomputed, and both run gates rerun, against the run's archived verdict |
| estimates | `python -m proofs.replay run estimates` | the estimator on every RouteBalance round's requests, batched as the round was: digests between two trees, and the logged lengths |
| rounds | `python -m proofs.replay run rounds` | every RouteBalance round (`schedule_round`) on the same estimates and a load view rebuilt from the log: digests between two trees |
| estimator | `python -m proofs.replay run estimator` | the estimator on each request of the pipeline runs that used it: digests between two trees, and the logged decode and feature tokens |
| routers | `python -m proofs.replay run routers` | AvengersPro, BEST-Route and RB-decoupled on every request: digests between two trees, and the logged model, score and reason |
| dispatchers | `python -m proofs.replay run dispatchers` | every pipeline run's dispatcher on its own router picks: digests between two trees, and the logged instance for round robin and random behind the passthrough router |
| reports | `proofs/reports.sh` | the four reports rebuilt from the runs, after the tree's own parity gate ran over every comparison group in a scratch copy: every file and log against the bundle's |
| GPU end to end | `proofs/gpu/run.sh` | every run of a small render on two real vLLM engines, through each tree's own runner (`proofs/gpu/e2e.py` lists what stands in for the lease): each run's verdict, and the choices that do not depend on timing (RB-decoupled's tier and score as its decision reason records them, round robin's and random's instances), request by request, between the two trees |

`python -m proofs.replay compare A.jsonl B.jsonl` compares two trees' digests run by run.

## What the logs allow, and what they do not

- A decision line records the round, every candidate's terms, score and latency estimate, the per-model lengths and
  the chosen instance, but not every instance's load at the round's start nor the queues behind the tie-breaks. So the
  rounds are replayed on a load view rebuilt from the log, and the two trees are compared on it; the scoring of what
  was logged is checked by the arm gate, which the evidence proof reruns on every run.
- The encoders (the estimator's and AvengersPro's) run on the CPU (the code pins `device="cpu"`), and their results
  depend in the last bits on the CPU, the thread count and the batch: on the replay machine at its default 8 threads
  about half of one run's logged lengths come out exact and the rest within about 1e-7, while at 4 threads most
  differ, up to a few percent, as other neighbours are picked. BEST-Route's ranker differs from the logs in the sixth
  decimal of its score. So the estimator and the routers are proven by the two trees giving the same results on the
  same machine, the same thread count and the same batches, bit for bit; the log comparisons are reported beside.
- Every replay process runs with `PYTHONHASHSEED=0` and fixed `OMP_NUM_THREADS` and `MKL_NUM_THREADS`, the same for
  the two trees, and the replay is run twice on one tree to show it is deterministic.
