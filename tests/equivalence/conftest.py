from pathlib import Path

import pytest

from tests.equivalence.banked_cells import Oracle
from tests.equivalence.router_oracle import V3Tree


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
