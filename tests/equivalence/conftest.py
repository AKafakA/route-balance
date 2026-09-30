import hashlib
import sys
from pathlib import Path

import pytest

from tests.equivalence.banked_cells import Oracle
from tests.equivalence.router_oracle import V3Tree
from tools.venv_parity import differences, packages


@pytest.fixture(scope="session", autouse=True)
def _setup_built_venv_is_current() -> None:
    """In a venv setup.sh built (a CloudLab node's test venv), every equivalence test first checks the venv: built
    from this tree and its current pyproject, and still the scheduler venv's stack plus the dev extras
    (tools/venv_parity.py, against its sibling scheduler venv: venv-test* -> venv-scheduler*). A stale or drifted
    venv would test the new code on another stack and report a pass for one that no longer serves. A development
    machine's venv has no marker."""
    marker = Path(sys.prefix) / "rb-setup.spec"
    if not marker.is_file():
        return
    repo = Path(__file__).resolve().parents[2]
    sha = hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest()[:16]
    want = f"complete test {repo} pyproject.sha256={sha}"
    if marker.read_text().strip() != want:
        pytest.fail(f"{sys.prefix} was built as '{marker.read_text().strip()}', not '{want}': rerun setup.sh --with-tests")
    name = Path(sys.prefix).name
    if not name.startswith("venv-test"):
        pytest.fail(f"{sys.prefix}: a test venv is named venv-test*, paired with its venv-scheduler* sibling")
    sched = Path(sys.prefix).parent / name.replace("venv-test", "venv-scheduler", 1) / "rb-setup.freeze"
    if not sched.is_file():
        pytest.fail(f"no {sched}: the scheduler venv this test venv mirrors is missing")
    diff = differences(packages(sched.read_text().splitlines()),
                       packages((Path(sys.prefix) / "rb-setup.freeze").read_text().splitlines()))
    if diff:
        pytest.fail(f"{sys.prefix} is not the scheduler venv's stack: {diff}")


@pytest.fixture(scope="session")
def oracle(request) -> Oracle:
    root = request.config.getoption("--oracle-root")
    if not root or not Path(root).is_dir():
        pytest.fail(f"the equivalence suite needs --oracle-root <dir of the banked v3 data>; got {root!r}")
    return Oracle(Path(root))


@pytest.fixture(scope="session")
def v3(oracle) -> V3Tree:
    """The v3 scheduler tree, router artifacts and upstream AvengersPro under the same root."""
    return V3Tree.under(oracle.root)


@pytest.fixture(scope="session")
def artifacts(request) -> Path:
    root = request.config.getoption("--artifact-root")
    if not root or not Path(root).is_dir():
        pytest.fail(f"the estimator tests need --artifact-root <dir of the rebuild's artifacts>; got {root!r}")
    return Path(root)
