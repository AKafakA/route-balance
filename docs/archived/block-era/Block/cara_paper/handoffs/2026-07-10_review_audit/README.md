# 2026-07-10 review-audit evidence (workflow wf_1c2221cf-e94)

Five read-only auditors + one synthesis verified every claim in the 2026-07-10 ChatGPT EuroSys
review (`../2026-07-10_chatgpt_eurosys_review_RAW.md`) against the paper tex, the rendered PDF,
code + v2 experiment results, local plans/handoffs, and the web. Files:

- `tex_audit.md`  — per-item findings vs `route-balance-paper/eurosys_2027/main.tex` + tables + references.bib (verbatim quotes + line numbers).
- `pdf_audit.md`  — rendered-PDF (pdftoppm) findings: Conference'17 placeholder CONFIRMED p.1; author block IS anonymous (review wrong there); radar grayscale-illegible; Fig 2 OK.
- `code_audit.md` — which asks are offline-computable vs need GPU. KEY: review's "Decoupled-SameEstimator" == existing arm-2 (cara_serve.py:1624-31, CARA_TIEBREAK=queue + w_lat=0); arm-4 static prior (…:1476-97); R1/R2 served-text assets; steady-state fields in every trace; gpuhour row; normalization = quality minmax + cost/lat max-divide; public REPRODUCE k=50/B=1000 bugs confirmed.
- `plans_audit.md` — existing Exp A/B specs, already-run dynamics cells, cells/day capacity evidence, standing-rule constraints (incl. the no-TTFT-claim rule).
- `web_audit.md`  — RouterWise (arXiv:2604.10907) REAL/planner; BOute (arXiv:2602.10729, MLSys'26) REAL/planner; OmniRouter real = Feb-2025/KDD-25 (our bib mis-dates Dec-2026); EuroSys 2027 CFP VERBATIM confirms different-title+different-name for arXiv-exposed submissions + AI-disclosure + Sep 17/24 deadlines; narrowed novelty NOT contradicted by anything found.
- `synthesis.md`  — per-item dispositions + draft two-track plan (superseded by the approved plan in `../2026-07-26_eurosys_revision_experiments.md`). Known error in synthesis: it claims the arXiv source is missing from the repo — WRONG, source of truth is `route-balance-paper/eurosys_2027/main.tex` (tag arxiv-2026-06-16); its cara_neurips.tex references are the stale Block-repo copy, use the eurosys main.tex line numbers from tex_audit.md instead.

Recovery note: pdf/plans/tex auditors hit StructuredOutput size limits; their findings were
recovered verbatim from agent transcripts — content is complete, format is raw JSON-ish.
