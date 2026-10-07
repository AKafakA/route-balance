"""The proofs that the ref branch does what the code that ran the paper's experiments did.

Each proof reads the archived runs (the 1,457 runs' folders: records, decision logs, configurations, manifests) and
the artifacts those runs loaded, and checks one property:

  archive.py           the archive, read with the standard library only (no code tree's reader)
  adapters/            one module per code tree's interfaces; a dump written through either is the same format
  replay.py            estimator, rounds, routers, dispatchers: each tree's outputs on the runs' own inputs, as
                       per-run digests (compare two trees' files), plus the checks the logs allow directly
  evidence.py          every record, manifest and engine reading parsed by the tree's own schema, the record and
                       engine checks of the verdict, and the run gates, against each run's archived verdict

README.md says what each proof compares and how to run it.
"""
