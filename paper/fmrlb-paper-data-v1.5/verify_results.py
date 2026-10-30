#!/usr/bin/env python3
"""The comparison step of reproduce_results.sh.

    python3 verify_results.py compare OUT       every reference output under expected/ against OUT, byte for byte

Exit 0 only when every output matches.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# every output reproduce_results.sh writes, each with its reference under expected/
OUTPUTS = (
    'generated/results.tex', 'generated/plot_points.json', 'generated/datasets.tex',
    'tables/gen_ablation.tex', 'tables/gen_budget.tex', 'tables/gen_burst.tex', 'tables/gen_calib.tex',
    'tables/gen_cluster.tex', 'tables/gen_decoupled.tex', 'tables/gen_gemma.tex', 'tables/gen_main.tex',
    'tables/gen_open.tex', 'tables/gen_overhead.tex', 'tables/gen_reqnorm.tex', 'tables/gen_roundrange.tex',
    'tables/gen_shipped.tex', 'tables/gen_spread.tex', 'tables/gen_tail.tex', 'tables/gen_vsr.tex',
    'figures/frontier3d.pdf', 'figures/frontier2x6.pdf', 'figures/variants.pdf', 'figures/frontier_gemma.pdf',
    'figures/tail_qwen.pdf', 'figures/tail_gemma.pdf',
)


def compare(out: Path) -> int:
    ref = HERE / "expected"
    present = sorted(str(p.relative_to(ref)) for p in ref.rglob("*") if p.is_file())
    if present != sorted(OUTPUTS):
        print(f"FAIL: expected/ holds {len(present)} references, not the {len(OUTPUTS)} outputs: "
              f"missing {sorted(set(OUTPUTS) - set(present))}, extra {sorted(set(present) - set(OUTPUTS))}")
        return 1
    refs = sorted(OUTPUTS)
    differ = []
    for rel in refs:
        got = out / rel
        if got.is_file() and got.read_bytes() == (ref / rel).read_bytes():
            print(f"same     {rel}")
        else:
            print(f"DIFFERS  {rel}" + ("" if got.is_file() else " (not written)"))
            differ.append(rel)
    if differ:
        print(f"FAIL: {len(differ)} of {len(refs)} reference outputs differ")
        return 1
    print(f"PASS: {len(refs)} reference outputs match")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["compare"] and len(argv) == 2:
        return compare(Path(argv[1]).resolve())
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
