We thank the Area Chair for the meta-review. It identifies two priorities: better justification of design choices, including the XGBoost estimator and adaptive batching, and comparison with general-purpose dispatchers. We conducted new experiments and added clarifications for both. Full results are provided in the reviewer-specific replies.

**(1) Design-choice justification.** We implemented the calibrated static alternative proposed by Reviewer gTfM: the same objective, with the learned head replaced by a fixed per-tier TPOT constant derived from median decode latency across the load sweep. For Reviewer 287j, we also clarify the role of adaptive batching in global scheduling, explain how it fits into the serving stack, and position it more clearly relative to prior work.

**(2) General-purpose dispatchers.** Following Reviewer faAU, we added power-of-two choices alongside shortest-queue and round-robin.

To further strengthen the evaluation scope raised in the meta-review, we added two generality experiments: a re-provisioned cluster topology and end-to-end onboarding of a second model family, Gemma-3, including measured onboarding cost. We also revise the framing and limitations to incorporate the reviewers' suggestions and clearly separate current contributions from future work.
