"""The static defect gate must flag each defect class that produced silently wrong v3 cells."""
import textwrap
from pathlib import Path

import pytest

from gates.static_defects import main, scan_package, scan_paths


def _pkg(root: Path, files: dict) -> Path:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return root


def _kinds(root: Path) -> set:
    return {f.kind for f in scan_package(root)}


def test_clean_package_passes(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"schedule/score.py": """
        def score(stats, key):
            if key not in stats or stats[key] is None:
                raise KeyError(key)
            return stats["num_running"] + stats["num_waiting"]

        def any_busy(xs):
            return [x for x in xs if x.busy or x.waiting]
    """})
    assert scan_package(pkg) == []


@pytest.mark.parametrize("src,kind", [
    # `or` used as a value: a real 0 becomes the fallback (the v3 aggregator zero-score bug)
    ("def f(s):\n    return s.get('x') or 0.5\n", "falsy_or"),
    ("def f(v):\n    return float(v or 0.01)\n", "falsy_or"),
    ("def f(v):\n    return v or -1.0\n", "falsy_or"),
    ("DEFAULT_QUALITY = 0.5\ndef f(d):\n    return d.get('q') or DEFAULT_QUALITY\n", "falsy_or"),
    ("def f(d, m):\n    return d.get(m) or d.get('score_fallback')\n", "falsy_or"),
    # invented values when data is missing (v3: quality 0.5, price 0.01, TPOT 0.05)
    ("def f(meta):\n    return meta.get('cost_per_token', 0.01)\n", "fabricated_default"),
    ("def f(meta):\n    return meta.get('instance_type', '')\n", "fabricated_default"),
    ("def f(d):\n    return d.get('k', -1.0)\n", "fabricated_default"),
    ("DEFAULT_PRICE = 0.01\ndef f(d):\n    return d.get('k', DEFAULT_PRICE)\n", "fabricated_default"),
    ("def f(d):\n    return d.get('k', float('nan'))\n", "fabricated_default"),
    ("def f(d):\n    return d.pop('k', 0.01)\n", "fabricated_default"),
    ("def f(d):\n    return d.setdefault('k', 0.5)\n", "fabricated_default"),
    ("def f(r):\n    return getattr(r, 'budget_cost', 0.0)\n", "fabricated_default"),
    # experiment parameters from the environment, however imported (v3 arm-4 incident)
    ("import os\ndef f():\n    return os.environ.get('RB_STATIC_TPOT')\n", "env_param"),
    ("import os\ndef f():\n    return os.getenv('X')\n", "env_param"),
    ("from os import getenv\ndef f():\n    return float(getenv('RB_TPOT', '0.05'))\n", "env_param"),
    ("from os import environ\ndef f():\n    return environ['X']\n", "env_param"),
    ("import os as _os\ndef f():\n    return _os.environ['X']\n", "env_param"),
    # hardware knowledge outside routebalance/hw.py, including single-entry and split maps
    ("GPU = {'d7525': 'a30', 'c4130': 'v100'}\n", "gpu_map"),
    ("GPU = {'d7525': 'a30'}\n", "gpu_map"),
    ("P100 = {'c240g5': 'p100'}\n", "gpu_map"),
    ("def f(h):\n    return h.startswith('d8545')\n", "gpu_map"),
    # hard-coded model lists and model-keyed dicts silently ignored a second family
    ("LADDER = ['Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B']\n", "model_ladder"),
    ("PRICE = {'Qwen/Qwen2.5-3B': 0.1, 'Qwen/Qwen2.5-7B': 0.2}\n", "model_ladder"),
])
def test_each_defect_class_fails(tmp_path, src, kind):
    pkg = _pkg(tmp_path / "routebalance", {"schedule/x.py": src})
    assert kind in _kinds(pkg)


def test_gpu_names_are_not_hardware_types(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": "GPUS = ('a30', 'v100', 'a100')\n"})
    assert scan_package(pkg) == []


def test_single_map_in_hw_is_allowed(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"hw.py": "HW_GPU = {'d7525': 'a30', 'c4130': 'v100', 'd8545': 'a100'}\n"})
    assert scan_package(pkg) == []


def test_two_maps_in_hw_fail(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"hw.py": "A = {'d7525': 'a30'}\nB = {'d7525': 'v100'}\n"})
    assert "gpu_map" in _kinds(pkg)


def test_http_module_get_is_not_a_default(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": """
        import requests
        import requests as req
        async def f(session, url):
            requests.get('https://example.org/status', {'detail': 'full'})
            req.get('https://example.org/status', {'detail': 'full'})
            return await session.get(url, params={'a': 1})
    """})
    assert scan_package(pkg) == []


@pytest.mark.parametrize("src", [
    "class A:\n    def f(self, rid):\n        return self.requests.get(rid, 0.01)\n",
    "def f(latency_by_client, c):\n    return latency_by_client.get(c, 0.05)\n",
    "def f(client):\n    return client.get('quality', 0.5)\n",
])
def test_http_like_names_are_not_exempt(tmp_path, src):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": src})
    assert "fabricated_default" in _kinds(pkg)


def test_waiver_needs_a_reason(tmp_path):
    ok = _pkg(tmp_path / "a" / "routebalance",
              {"x.py": "def f(d):\n    return d.get('k', 0)  # default-ok: counters start at zero\n"})
    bad = _pkg(tmp_path / "b" / "routebalance", {"x.py": "def f(d):\n    return d.get('k', 0)  # default-ok:\n"})
    assert scan_package(ok) == []
    assert "fabricated_default" in _kinds(bad)


def test_env_waiver_is_separate(tmp_path):
    pkg = _pkg(tmp_path / "routebalance",
               {"x.py": "import os\nROOT = os.environ['RB_MODELS']  # env-ok: artifact root path only\n"})
    assert scan_package(pkg) == []


def test_get_with_none_default_is_fine(tmp_path):
    src = "def f(d):\n    v = d.get('k')\n    return d.get('j', None) if v else v\n"
    pkg = _pkg(tmp_path / "routebalance", {"x.py": src})
    assert scan_package(pkg) == []


def test_all_production_packages_are_scanned(tmp_path):
    root = _pkg(tmp_path, {"analysis/agg.py": "def f(s):\n    return s.get('quality') or 0.5\n",
                           "bench/load.py": "import os\nRATE = os.environ['RATE']\n"})
    found = scan_paths([root / "analysis", root / "bench"], root)
    assert {f.path for f in found} == {"analysis/agg.py", "bench/load.py"}


def test_missing_directory_is_a_usage_error(tmp_path):
    assert main([str(tmp_path / "absent")]) == 2


def test_overlapping_directories_scan_each_file_once(tmp_path):
    root = _pkg(tmp_path, {"analysis/agg.py": "def f(s):\n    return s.get('quality') or 0.5\n"})
    found = scan_paths([root / "analysis", root], root)
    assert len(found) == 1


@pytest.mark.parametrize("src,kind", [
    ("import os.path\nTPOT = float(os.environ['RB_STATIC_TPOT'])\n", "env_param"),
    ("SIZES = {'small': 'Qwen/Qwen2.5-3B', 'large': 'Qwen/Qwen2.5-72B'}\n", "model_ladder"),
    ("PAIRS = [('Qwen/Qwen2.5-3B', 0.1), ('Qwen/Qwen2.5-7B', 0.2)]\n", "model_ladder"),
    ("from collections import defaultdict\nQ = defaultdict(lambda: 0.5)\n", "fabricated_default"),
    ("def f(df):\n    return df['quality'].fillna(0.5)\n", "fabricated_default"),
    ("def f(xs):\n    return max(xs, default=0.0)\n", "fabricated_default"),
    ("def f(it):\n    return next(it, 0.05)\n", "fabricated_default"),
    ("GPU = dict(d7525='a30', c4130='v100')\n", "gpu_map"),
])
def test_round2_defects_fail(tmp_path, src, kind):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": src})
    assert kind in _kinds(pkg)


def test_dict_keyword_maps_count_toward_the_single_map(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"hw.py": "A = dict(d7525='a30')\nB = {'c4130': 'v100'}\n"})
    assert "gpu_map" in _kinds(pkg)


@pytest.mark.parametrize("src", [
    "def f(x):\n    match x:\n        case [y, *_] if y.draining or y.failed:\n            return 1\n    return 0\n",
    "def f(s):\n    return bool(s.draining or s.failed)\n",
])
def test_boolean_contexts_are_not_values(tmp_path, src):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": src})
    assert scan_package(pkg) == []


def test_model_ladder_reported_once_per_literal(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": "L = [('Qwen/Qwen2.5-3B', 1), ('Qwen/Qwen2.5-7B', 2)]\n"})
    assert sum(f.kind == "model_ladder" for f in scan_package(pkg)) == 1


@pytest.mark.parametrize("src", [
    "def f(row):\n    return row.get('ttft', default=0.0)\n",
    "def f(df):\n    return df.pivot_table(index='a', values='b', fill_value=0)\n",
    "def f(df, idx):\n    return df.reindex(idx, fill_value=0.0)\n",
    "def f(df):\n    return df.unstack(fill_value=0)\n",
    "import numpy as np\ndef f(x):\n    return np.nan_to_num(x)\n",
    "def f(df):\n    return df.ffill()\n",
    "import collections\nQ = collections.defaultdict(lambda: 0.5)\n",
    "from collections import defaultdict\nQ = defaultdict(float)\n",
    "from pydantic import BaseModel\nclass Cfg(BaseModel):\n    price_per_token: float = 0.01\n",
    "from dataclasses import dataclass\n@dataclass\nclass C:\n    k: int = -1\n",
    "from pydantic import Field\nX = Field(default=0.5)\n",
    "import requests as req\ndef quality(req):\n    return req.get('quality', 0.5)\n",
])
def test_round3_defaults_fail(tmp_path, src):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": src})
    assert "fabricated_default" in _kinds(pkg)


def test_optional_none_field_defaults_are_fine(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": """
        from typing import Optional
        from pydantic import BaseModel, Field
        class Rounds(BaseModel):
            max_batch: Optional[int] = Field(default=None, ge=1)
            timeout_ms: Optional[float] = None
    """})
    assert scan_package(pkg) == []


def test_default_waiver_does_not_silence_hardware_or_model_checks(tmp_path):
    pkg = _pkg(tmp_path / "routebalance", {"x.py": "def f(h):\n    return h == 'd7525'  # default-ok: nope\n"})
    assert "gpu_map" in _kinds(pkg)


def test_cli_works_from_an_unrelated_directory(tmp_path, monkeypatch):
    root = _pkg(tmp_path / "repo", {"routebalance/hw.py": "HW = {'d7525': 'a30'}\n", "routebalance/x.py": "X = 1\n"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main([str(root / "routebalance")]) == 0
