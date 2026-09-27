"""The node's test venv is the serving stack plus the dev extras (tools/venv_parity.py, setup.sh --with-tests)."""
from tools.venv_parity import constraints, differences, main, packages

SCHED = ["-e ../route-balance", "fsspec==2026.7.0", "Llm_Blender @ git+https://github.com/y/LLM-Blender.git@aaa",
         "pip==24.2", "torch==2.9.0"]


def test_the_same_stack_plus_the_dev_extras_passes():
    test = SCHED + ["pytest==8.3.3", "iniconfig==2.0.0", "pluggy==1.5.0", "ruff==0.6.9"]
    assert differences(packages(SCHED), packages(test)) == []


def test_a_drifted_package_a_missing_one_another_commit_or_an_extra_one_fail():
    for test, word in ((["fsspec==2026.9.0"] + SCHED[2:], "fsspec"), (SCHED[:1] + SCHED[2:], "fsspec"),
                       (SCHED[:2] + ["llm-blender @ git+https://x/LLM-Blender.git@bbb"] + SCHED[3:], "llm-blender"),
                       (SCHED + ["sympy==1.13.1"], "sympy")):
        diff = differences(packages(SCHED), packages(test))
        assert diff and word in diff[0], (test, diff)


def test_constraints_are_the_pins_only():
    assert constraints(SCHED) == ["fsspec==2026.7.0", "pip==24.2", "torch==2.9.0"]   # no editable, no direct URL


def test_the_command_line_exits_1_on_a_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("\n".join(SCHED))
    b.write_text("\n".join(SCHED + ["pytest==8.3.3"]))
    assert main([str(a), str(b)]) == 0
    b.write_text("\n".join(SCHED[:-1] + ["torch==2.10.0"]))
    assert main([str(a), str(b)]) == 1
