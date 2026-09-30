PY ?= .venv-uvicorn/bin/python
ORACLE ?=
ARTIFACTS ?=
THREADS ?= 6
SHELLCHECK ?= shellcheck

.PHONY: check check-heavy lint hygiene static freeze unit contract equivalence gates

# Everything a commit must pass. Offline, CPU-only, no network.
# A suite joins CHECK in the step that adds its first tests, so an empty suite can never pass silently.
CHECK = lint hygiene static freeze unit contract gates
check: $(CHECK)

# The heavy suite: CHECK (which gains contract with its first contract tests) plus the v3 equivalence
# tests. Runs on the analysis machine (dev-gpu) only: it reads the banked v3 data and artifacts under
# ORACLE and the rebuild's own artifacts (retrained TPOT heads, calibration table) under ARTIFACTS, and
# loads a tokenizer and the MiniLM encoder; THREADS caps every numeric library on that shared machine.
#   make check-heavy PY=<venv python> ORACLE=<oracle root> ARTIFACTS=<artifact root>
check-heavy: $(CHECK) equivalence

lint:
	$(PY) -m ruff check .
	$(PY) -m compileall -q routebalance bench analysis gates tools
	for f in scripts/*.sh; do bash -n "$$f" || exit 1; done
	@if command -v $(SHELLCHECK) >/dev/null 2>&1; then echo "$(SHELLCHECK) scripts/*.sh"; $(SHELLCHECK) -x -P scripts scripts/*.sh; \
	else echo "lint: $(SHELLCHECK) not found, scripts/*.sh were only syntax-checked (make lint SHELLCHECK=<path>)"; fi

hygiene:
	$(PY) -m gates.hygiene .

static:
	$(PY) -m gates.static_defects routebalance bench analysis tools

# after the freeze tag, the scope and presets change only by a logged owner decision (gates/freeze.py)
freeze:
	$(PY) -m gates.freeze . --tag lease-2026-09-28

unit:
	$(PY) -m pytest -q tests/unit

contract:
	$(PY) -m pytest -q tests/contract

equivalence:
	OMP_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS) MKL_NUM_THREADS=$(THREADS) \
		$(PY) -m pytest -q tests/equivalence --oracle-root "$(ORACLE)" --artifact-root "$(ARTIFACTS)"

gates:
	$(PY) -m pytest -q tests/gates
