{
 "findings": [
  {
   "item": "web-1-RouterWise",
   "status": "ASSET-EXISTS",
   "evidence": "REAL. arXiv:2604.10907 'RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving' by Hossein Hosseini Kasnavieh, Gholamreza Haffari, Christopher Leckie, Adel N. Toosi, submitted 13 Apr 2026. Abstract: 'Multi-model LLM routing has emerged as an effective approach for reducing serving cost and latency... prior routing methods typically assume that each model has a fixed latency. In real deployments, this assumption is inaccurate.' It 'searches over feasible system setups and, for each fixed setup, computes the best routing policy under the latency target' = planning/deployment-time, NOT per-request online live-state. Review's characterization (deployment/planning routing, no online adaptation) is REVIEW-CORRECT."
  },
  {
   "item": "web-2-BOute",
   "status": "ASSET-EXISTS",
   "evidence": "REAL. arXiv:2602.10729 'BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization' by Youhe Jiang, Fangcheng Fu, Eiko Yoneki, submitted 11 Feb 2026, MLSys 2026 (listed on mlsys.org/virtual/2026 and cl.cam.ac.uk/~ey204/pubs/2026_MLSYS_BOUTE.pdf). Abstract: uses 'a multi-objective Bayesian optimization (MOBO) framework to co-optimize the routing strategy and model deployment.' Review-correct: it is MLSys 2026 and co-optimizes quality-aware routing + heterogeneous deployment. Note: abstract mentions NO online per-request adaptation, so it is a planner \u2014 consistent with the narrowed-novelty repositioning."
  },
  {
   "item": "web-3-OmniRouter",
   "status": "REVIEW-CORRECT",
   "evidence": "REAL. arXiv:2502.20576 'OmniRouter: Budget and Performance Controllable Multi-LLM Routing' by Kai Mei, Wujiang Xu, Minghao Guo, Shuhang Lin, Yongfeng Zhang, v1 submitted 27 Feb 2025; published KDD-25 / ACM SIGKDD Explorations (dl.acm.org/doi/10.1145/3787470.3787480). The RouteBalance paper cites it 'dated 2026' (per arxiv.org/html/2606.17949v1), which is WRONG \u2014 the real work is Feb 2025 / KDD-25. Bib mis-dating is REVIEW-CORRECT; the specific 'Dec 2026' month could not be extracted from the HTML, so that granularity is UNVERIFIABLE."
  },
  {
   "item": "web-4-EuroSys2027-CFP",
   "status": "REVIEW-CORRECT",
   "evidence": "2027.eurosys.org/cfp.html verbatim: 'The review process is double-blind, meaning that the authors' identities will be concealed from the reviewers and vice versa.' AND critically: 'Publication in non-peer-reviewed contexts, such as arXiv papers... does not count as concurrent submissions. However, in such cases, the submitted version should have a substantially different title and use a different system/tool name, if applicable.' Page limit: 'a maximum of 12 pages of technical content, plus... references.' AI disclosure: 'any use of AI tools (such as ChatGPT) must be fully disclosed in the submission.' The DIFFERENT-TITLE + DIFFERENT-SYSTEM-NAME claim (C1) is CONFIRMED for EuroSys 2027."
  },
  {
   "item": "C1-fall-deadlines",
   "status": "REVIEW-CORRECT",
   "evidence": "2027.eurosys.org/cfp.html shows two cycles. Fall cycle verbatim: 'Paper titles and abstracts due: Thursday, September 17, 2026' and 'Full paper submissions due: Thursday, September 24, 2026' (Notification: Friday, January 29, 2027). Matches the review's Sep 17 (abstract) / Sep 24 (paper) 2026 claim exactly. (Spring cycle: May 7 / May 14, 2026.)"
  },
  {
   "item": "C1-arXiv-exposure",
   "status": "REVIEW-CORRECT",
   "evidence": "The paper is PUBLICLY on arXiv fully de-anonymized: arxiv.org/html/2606.17949v1 'RouteBalance: Fused Model Routing and Load Balancing for Heterogeneous LLM Serving' by 'Wei Da and Evangelia Kalyvianaki from University of Cambridge.' This exposes authors + institution + the SAME system name 'RouteBalance' + same title \u2014 directly violating the EuroSys 2027 requirement of a 'substantially different title and... different system/tool name.' The 'Conference'17' placeholder and DOI text could not be surfaced from the HTML render, so those specific artifacts are UNVERIFIABLE from web (likely present in the PDF/LaTeX)."
  },
  {
   "item": "web-6-EuroSys2026-acceptance",
   "status": "UNVERIFIABLE",
   "evidence": "Could NOT confirm the review's 59/319 (18.5%). openaccept.org/c/sys/eurosys shows only EuroSys 2025: '696 submissions... 85 papers accepted... 12.21% acceptance rate' \u2014 no 2026 figure listed. A search synthesis returned 'EuroSys 2026 has an acceptance rate of 16.9%' but with no authoritative primary source. 59/319=18.5% neither matches the 16.9% synthesis nor is independently confirmable; treat the specific numbers as UNVERIFIABLE."
  },
  {
   "item": "web-7-competing-systems",
   "status": "REVIEW-CORRECT",
   "evidence": "No system found that does joint quality+latency+cost request-to-instance routing using LIVE instance state for a fixed deployment. Llumnix (arXiv:2406.03243, OSDI'24) does live 'runtime rescheduling across multiple model instances' via 'live migration' but is single-model load-balancing/priority, NOT quality-aware model selection. MuxServe 'derives model colocation strategies based on offline profiling'; QLM uses model swapping. RouterWise and BOute are planning-time. This supports the narrowed novelty claim (online live-state per-request for fixed deployment) as defensible \u2014 nothing directly contradicts it."
  },
  {
   "item": "FACT-testbed-28GPU",
   "status": "REVIEW-CORRECT",
   "evidence": "arxiv.org/html/2606.17949v1: '13-instance, 28-GPU' testbed: 'Qwen2.5-72B on A100\u00d74 (2 instances), Qwen2.5-14B on V100\u00d74 (3 instances), Qwen2.5-7B on A30\u00d71 (5 instances), Qwen2.5-3B on A30\u00d71 (3 instances)', 72B/14B use 'tensor parallelism TP=4.' Arithmetic: 2*4(A100)+3*4(V100)+5*1+3*1(A30)=8+12+8=28 GPUs; 2+3+5+3=13 instances; four Qwen2.5 tiers. All fact-checks CONFIRMED (N4 note: single model family Qwen2.5, single topology)."
  },
  {
   "item": "FACT-scale-configs-requests",
   "status": "REVIEW-CORRECT",
   "evidence": "arxiv.org/html/2606.17949v1 reports '442 configurations, \u22481.5M requests.' Matches review's '442 configurations' and '~1.5M requests.'"
  },
  {
   "item": "FACT-quality-numbers",
   "status": "REVIEW-CORRECT",
   "evidence": "arxiv.org/html/2606.17949v1: RouteBalance peak quality '0.419' vs BEST-Route '0.406', i.e. '+0.013' with '95% CI [+0.005, +0.022].' Matches review's 0.406->0.419 / +0.013. (N7 concern about practical significance of +0.013 stands on merits.)"
  },
  {
   "item": "FACT-23x-and-tiers",
   "status": "REVIEW-CORRECT",
   "evidence": "arxiv.org/html/2606.17949v1: at 30 req/s, '2.6-4.1x ahead of enhanced BEST-Route' and '23x' faster than 'published BEST-Route deployment.' Confirms review's '23x serial comparison.' Four Qwen2.5 tiers / 13 instances confirmed (see testbed finding)."
  },
  {
   "item": "FACT-k-and-bootstrap",
   "status": "PARTIALLY-CORRECT",
   "evidence": "arxiv.org/html/2606.17949v1: KNN estimator 'k=10' and 'Bootstrap analysis employed 10,000 replicates' \u2014 confirming the PAPER's stated values (k=10, 10,000). The P6 inconsistency is the claim that a REPRODUCE command uses k=50 and 1,000 resamples; those are in the artifact/repo, not the arXiv paper, so the mismatch itself is UNVERIFIABLE from web \u2014 but the paper-side values are REVIEW-CORRECT."
  }
 ],
 "summary": "Web verification of the review's external claims is largely favorable to the review. (1) RouterWise is REAL: arXiv:2604.10907 (Kasnavieh, Haffari, Leckie, Toosi; 13 Apr 2026), a deployment/planning-time joint resource-allocation-and-routing system with no per-request online live-state adaptation \u2014 matching the review's characterization. (2) BOute is REAL: arXiv:2602.10729 (Jiang, Fu, Yoneki; 11 Feb 2026, MLSys 2026), co-optimizing quality-aware routing + heterogeneous GPU deployment via multi-objective Bayesian optimization, with no stated online adaptation \u2014 a planner, consistent with the proposed repositioning. (3) OmniRouter is REAL: arXiv:2502.20576 (Mei et al.), v1 Feb 27 2025, published KDD-25 / SIGKDD Explorations; the RouteBalance paper mis-cites it as '2026', confirming the bib-dating error (exact 'Dec 2026' month unverifiable from HTML). (4) The EuroSys 2027 CFP verbatim confirms double-blind review, a 12-page limit, mandatory AI-tool disclosure, and \u2014 critically \u2014 that arXiv-exposed submissions 'should have a substantially different title and use a different system/tool name'; fall-cycle deadlines are exactly Sep 17 (abstract) / Sep 24 (paper) 2026. (C1) The paper is publicly on arXiv fully de-anonymized as 'RouteBalance' by Wei Da and Evangelia Kalyvianaki (University of Cambridge), reusing the same title and system name \u2014 a direct compliance problem under that CFP clause. All internal FACT-CHECKS reproduce from the public arXiv HTML: 28-GPU/13-instance testbed (2xA100 TP4 + 3xV100 TP4 + 8xA30, four Qwen2.5 tiers), 442 configurations, ~1.5M requests, 0.406->0.419 (+0.013, 95% CI [+0.005,+0.022]), '23x' vs published BEST-Route, k=10 KNN, and 10,000 bootstrap replicates. Two items remain unconfirmed: the EuroSys 2026 acceptance figure 59/319 (18.5%) could not be verified against a primary source (openaccept lists only EuroSys 2025 at 85/696=12.21%; a non-authoritative 16.9% synthesis appeared), and the P6 artifact-command discrepancies (k=50, 1,000 resamples) live in the repo/artifact and are not visible from the web. On narrowed novelty (item 7): Llumnix does live cross-instance rescheduling but is single-model load-balancing, not quality-aware selection; MuxServe/QLM are offline/swap-based; RouterWise/BOute are planners \u2014 so nothing found directly contradicts the online live-state, fixed-deployment, joint quality/latency/cost request-to-instance framing."
}