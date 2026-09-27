"""Predictions the scheduler scores with.

  prompt.py         chat-template unwrap: the one place served text becomes estimator text
  artifacts.py      md5-pinned reads of every artifact file
  knn.py            per-model quality and output length from the prompt's nearest neighbours
  latency.py        TPOT sources x envelopes: the latency value that enters the objective
  xgboost_tpot.py   the learned TPOT source (per-tier boosters); only the learned arm imports it
"""
