import json
import os
from pathlib import Path

import pytest

from routebalance.config import ConfigError
from tools import cell_driver
from vsrbench import block

RENDER = Path(__file__).resolve().parents[2] / "local/lease/render15"
pytestmark = pytest.mark.skipif(not RENDER.is_dir(), reason="the lease render is local to the operator")


def publish(root: Path, cid: str, valid: bool, mtime: float = 0.0) -> None:
    (root / cid / "gates").mkdir(parents=True)
    f = root / cid / "gates" / "verdict.json"
    f.write_text(json.dumps({"valid": valid}))
    if mtime:
        os.utime(f, (mtime, mtime))


def test_the_block_runs_seed_1_first_on_the_round_robin_templates():
    scope, _ = cell_driver.load(RENDER)
    order = block.plan(scope, "qwen", ["l12", "l30"], [1, 2])
    assert order == [("qwen.vsr.mom-balance__l12__s1", "qwen.disp.rr__l12__s1", 1),
                     ("qwen.vsr.mom-balance__l30__s1", "qwen.disp.rr__l30__s1", 1),
                     ("qwen.vsr.mom-balance__l12__s2", "qwen.disp.rr__l12__s1", 2),
                     ("qwen.vsr.mom-balance__l30__s2", "qwen.disp.rr__l30__s1", 2)]
    with pytest.raises(ConfigError):
        block.plan(scope, "qwen", ["l99"], [1])


def test_another_seed_sends_what_the_main_grid_sends_at_that_seed():
    """D157 renders the main grid's seeds 2-3 only at the knee points: a run at seed 2 sends the seed-1 template's trace
    at seed 2, the arrivals and prompt order of the main grid's seed-2 run at that rate (render 15 still has one)."""
    scope, _ = cell_driver.load(RENDER)
    got = block.expected(scope, "qwen", "qwen.disp.rr__l30__s1", 2, "0" * 64)
    main = cell_driver.expected(scope, "qwen.disp.rr__l30__s2")
    assert got.cell_id == "qwen.vsr.mom-balance__l30__s2" and got.seed == 2
    assert (got.arrival_digest(), got.order_digest()) == (main.arrival_digest(), main.order_digest())
    assert (got.trace, got.dataset, got.prompt_order, got.group) == (main.trace, main.dataset, main.prompt_order, main.group)
    assert got.arrival_digest() != cell_driver.expected(scope, "qwen.disp.rr__l30__s1").arrival_digest()


def test_a_render_with_seed_1_alone_still_plans_three_seeds(monkeypatch):
    scope, _ = cell_driver.load(RENDER)
    seed1 = [c for c in block.expected_cells(scope) if c.seed == 1]
    monkeypatch.setattr(block, "expected_cells", lambda scope: seed1)
    order = block.plan(scope, "qwen", ["l12"], [1, 2, 3])
    assert order == [(f"qwen.vsr.mom-balance__l12__s{s}", "qwen.disp.rr__l12__s1", s) for s in (1, 2, 3)]


def test_a_block_with_two_versions_is_flagged(tmp_path):
    order = [(f"qwen.vsr.mom-balance__l12__s{s}", "qwen.disp.rr__l12__s1", s) for s in (1, 2, 3)]
    assert block.mixed_versions(tmp_path, order) == []
    for s, docker in ((1, "29.1.3"), (2, "29.1.3"), (3, "29.1.4")):
        (tmp_path / order[s - 1][0] / "vsr").mkdir(parents=True)
        (tmp_path / order[s - 1][0] / "vsr" / "harness.json").write_text(json.dumps({"versions": {"docker": docker}}))
    assert block.mixed_versions(tmp_path, order) == ["qwen.vsr.mom-balance__l12__s3"]


def test_a_verdict_alone_is_not_a_valid_run(tmp_path):
    assert block.verdict(tmp_path, "x") == "missing"
    publish(tmp_path, "qwen.vsr.mom-balance__l12__s1", True)            # a verdict without its record and evidence
    assert block.verdict(tmp_path, "qwen.vsr.mom-balance__l12__s1") == "invalid"


def test_the_reference_is_a_run_the_main_grid_counts_done(tmp_path, monkeypatch):
    scope, _ = cell_driver.load(RENDER)
    done = {"qwen.disp.rr__l12__s1"}
    monkeypatch.setattr(cell_driver, "status", lambda scope, exp, root: "done" if exp.cell_id in done else "invalid")
    assert block.reference_for(scope, tmp_path, "qwen", "qwen.disp.rr__l12__s1") == "qwen.disp.rr__l12__s1"
    publish(tmp_path, "qwen.disp.lor__l30__s1", True, mtime=2e9)
    publish(tmp_path, "qwen.disp.po2__l30__s1", True, mtime=1e9)
    done |= {"qwen.disp.lor__l30__s1", "qwen.disp.po2__l30__s1"}
    assert block.reference_for(scope, tmp_path, "qwen", "qwen.disp.rr__l12__s2") == "qwen.disp.lor__l30__s1"
    with pytest.raises(ConfigError):
        block.reference_for(scope, tmp_path / "empty", "qwen", "qwen.disp.rr__l12__s2")


def test_a_block_whose_code_changed_is_flagged(tmp_path):
    order = [(f"qwen.vsr.mom-balance__l12__s{s}", "qwen.disp.rr__l12__s1", s) for s in (1, 2)]
    for (cid, _, _), code in zip(order, ("aa", "bb"), strict=True):
        (tmp_path / cid / "vsr").mkdir(parents=True)
        (tmp_path / cid / "vsr" / "harness.json").write_text(json.dumps(
            {"versions": {"docker": "29.1.3"}, "module": {"vsrbench/client.py": code}, "shared": {}}))
    assert block.mixed_versions(tmp_path, order) == ["qwen.vsr.mom-balance__l12__s2"]


def test_the_block_refuses_flags_it_sets_for_each_run(tmp_path):
    with pytest.raises(ConfigError, match="set by the block"):
        block.main(["run", "--family", "qwen", "--render", str(RENDER), "--traces", "l12", "--seeds", "1",
                    "--cells-root", str(tmp_path), "--reference-cells", str(tmp_path), "--mirror", str(tmp_path),
                    "--", "--seed", "2"])
    for tok in ("--mirror=/elsewhere", "--seed=2", "--mir", "--reference-cell=x", "--templ"):
        with pytest.raises(ConfigError, match="set by the block"):
            block.main(["run", "--family", "qwen", "--render", str(RENDER), "--traces", "l12", "--seeds", "1",
                        "--cells-root", str(tmp_path), "--reference-cells", str(tmp_path), "--mirror", str(tmp_path),
                        "--", tok])


def test_run_cell_takes_no_abbreviated_flags():
    from vsrbench import run_cell
    full = ["--profile", "test", "--render", "r", "--template-cell", "t", "--family", "qwen", "--fleet", "f",
            "--hosts", "h", "--engine-args", "e", "--engines-record", "er", "--data-root", "d", "--cells-root", "c",
            "--vsr", "v", "--templates", "tp", "--models-cache", "m", "--work", "w"]
    assert run_cell._parser().parse_args(full + ["--mirror", "x"]).mirror == "x"          # the control parses
    with pytest.raises(SystemExit):
        run_cell._parser().parse_args(full + ["--mir", "x"])


def test_a_flag_repeated_after_the_separator_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="more than once"):
        block.main(["run", "--family", "qwen", "--render", str(RENDER), "--traces", "l12", "--seeds", "1",
                    "--cells-root", str(tmp_path), "--reference-cells", str(tmp_path), "--mirror", str(tmp_path),
                    "--", "--vsr", "/a/vllm-sr", "--vsr=/b/vllm-sr"])


def test_forwarded_paths_are_checked_in_both_forms(tmp_path):
    base = ["run", "--family", "qwen", "--render", str(RENDER), "--traces", "l12", "--seeds", "1",
            "--cells-root", str(tmp_path), "--reference-cells", str(tmp_path), "--mirror", str(tmp_path), "--"]
    for rest in (["--work", "work"], ["--work=work"]):
        with pytest.raises(ConfigError, match="absolute paths"):
            block.main(base + rest)
    assert block._arg(["--vsr", "/abs/vllm-sr"], "--vsr") == "/abs/vllm-sr"
