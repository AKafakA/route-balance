## POST 0 — "Thank All Reviewers and General Response" (top-level Official Comment; also
## addresses the Meta Review)

We sincerely thank all reviewers and the Area Chair for their time and constructive feedback! Following the meta-review's two named priorities — justifying the learned (XGBoost) estimator and comparing against general-purpose dispatchers — we have run new experiments for both, on the submitted configuration, with the same scheduler, the same predictor stack and the same evaluation harness throughout. Two scoping notes we state up front rather than in a footnote: the topology experiment deliberately re-provisions the cluster (that is what it tests), and the second model family was served on measured static per-(model, GPU) latency priors rather than newly trained latency heads. Every other new measurement is on the submitted cluster and configuration. Responses are posted separately for each reviewer; we would be glad to discuss further.

To help recall the submission, the strengths highlighted by the reviewers:

[Problem & formulation]
- Reviewer 287j: "The paper aims to solve the right problem and probably does a good job in
solving it."
- Reviewer faAU: "Sharp observation and elegant formulation ... insightful and elegant."
- Reviewer gTfM: "this is the first work to demonstrate this capability."

[Empirical performance]
- Reviewer faAU: "Strong empirical performance ... its scheduling overhead stays sub-linear
as load grows 5× ... demonstrating the effectiveness of batched amortization."

[Design]
- Reviewer faAU: "Pragmatic engineering design ... precisely targeted, no extraneous
complexity."
