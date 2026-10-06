"""The test machine's test profile inputs that the hygiene gate keeps out of the tree (a private address): the
hosts file of the two test engines, written at runtime.

Usage: python -m vsrbench.testprofile hosts OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from vsrbench import constants as C


def hosts() -> dict:
    """The test engines listen on the host; the stack reaches them through Docker's bridge gateway, the drivers on
    loopback. The node is labelled an A30 because the shared fleet loader knows only the lease's hardware."""
    return {"test-node": {"address": "127.0.0.1", "hw": "d7525", "service_address": C.TEST_GATEWAY}}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "hosts":
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[1])
    if out.exists():
        print(f"{out} exists", file=sys.stderr)
        return 1
    out.write_text(json.dumps(hosts(), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
