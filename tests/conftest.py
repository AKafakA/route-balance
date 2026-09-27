"""Shared pytest options.

--oracle-root is the read-only copy of the banked v3 data (cells, scored corpora, eval set) and of the v3
source and artifacts (KNN indexes, TPOT heads) that the equivalence suite compares the rebuild with.
--artifact-root holds the rebuild's own artifacts that suite checks (retrained TPOT heads, calibration
table). That suite runs on the analysis machine only (make check-heavy); without the options its tests
fail, they never skip.
"""


def pytest_addoption(parser):
    parser.addoption("--oracle-root", action="store", default=None,
                     help="root of the read-only v3 oracle data, required by tests/equivalence")
    parser.addoption("--artifact-root", action="store", default=None,
                     help="root of the rebuild's artifacts (xgb_served/, tpot_table_v3cells_d47.json), required by "
                          "the estimator tests in tests/equivalence")
