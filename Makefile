PY ?= .venv-uvicorn/bin/python
SHELLCHECK ?= shellcheck

.PHONY: check lint hygiene static unit contract gates

# Everything a commit must pass. Offline, CPU-only, no network.
# A suite joins CHECK in the step that adds its first tests, so an empty suite can never pass silently.
CHECK = lint hygiene static unit contract gates
check: $(CHECK)

lint:
	$(PY) -m ruff check .
	$(PY) -m compileall -q routebalance bench analysis gates tools
	for f in scripts/*.sh; do bash -n "$$f" || exit 1; done
	@if command -v $(SHELLCHECK) >/dev/null 2>&1; then echo "$(SHELLCHECK) scripts/*.sh"; $(SHELLCHECK) -x -P scripts scripts/*.sh; \
	else echo "lint: $(SHELLCHECK) not found, scripts/*.sh were only syntax-checked (make lint SHELLCHECK=<path>)"; fi

hygiene:
	$(PY) -m gates.hygiene $(if $(TERMS),--terms $(TERMS)) .

static:
	$(PY) -m gates.static_defects routebalance bench analysis tools

unit:
	$(PY) -m pytest -q tests/unit

contract:
	$(PY) -m pytest -q tests/contract

gates:
	$(PY) -m pytest -q tests/gates
