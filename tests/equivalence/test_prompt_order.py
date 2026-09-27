"""bench.dataset reproduces the v3 send order of every banked grid cell (seed 1, rates 30 and 6)."""
import pytest

from bench.dataset import prompt_order, read_rows
from tests.equivalence.banked_cells import EVAL_SET, load_banked_cell, unwrap_chat


@pytest.mark.parametrize("cells_dir,rate,n_cells", [("s2_p1_l30", 30, 31), ("s2_p1_l6", 6, 39)])
def test_every_banked_cell_sent_the_prompts_in_the_rebuilt_order(oracle, cells_dir, rate, n_cells):
    rows = read_rows(oracle.data_file(EVAL_SET))
    fixed = [rows[i][1] for i in prompt_order(len(rows), "fixed", 1)]
    assert [rows[i][1] for i in prompt_order(len(rows), "per_seed", 1)] == fixed   # seed 1 is the v3 order
    cells = [r["cell"] for r in oracle.dominance_rows(cells_dir, rate)]
    assert len(cells) == n_cells
    wrong = []
    for name in cells:
        sent = [unwrap_chat(r["prompt"]) for r in load_banked_cell(oracle.cells_dir(cells_dir) / f"{name}.json")
                ["response_details"]]
        if sent != fixed:
            wrong.append((name, sum(a != b for a, b in zip(sent, fixed, strict=False)), len(sent)))
    assert wrong == [], f"cells sent in another order: {wrong}"
    print(f"prompt order reproduced: {len(cells)}/{len(cells)} cells x {len(fixed)} prompts ({cells_dir})")
