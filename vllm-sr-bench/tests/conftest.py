import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
REPO = MODULE.parent
for p in (str(REPO), str(MODULE)):
    if p not in sys.path:
        sys.path.insert(0, p)
