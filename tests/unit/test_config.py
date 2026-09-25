"""Config loading must fail loudly on every malformed or inconsistent input."""
import copy
import json

import pytest
from pydantic import ValidationError

from routebalance.config import ConfigError, config_digest, load_fleet, parse_scheduler

FLEET = {
    "tiers": {
        "qwen2.5-3b_a30": {"model": "Qwen/Qwen2.5-3B", "gpu": "a30", "cost_in": 6e-08, "cost_out": 6e-08,
                           "max_num_seqs": 256},
        "qwen2.5-14b_v100": {"model": "Qwen/Qwen2.5-14B", "gpu": "v100", "cost_in": 2e-07, "cost_out": 2e-07,
                             "max_num_seqs": 256},
    },
    "instances": [
        {"instance_id": "Qwen-2.5-14B_0", "tier": "qwen2.5-14b_v100", "node": "node1", "port": 8000},
        {"instance_id": "Qwen-2.5-3B_0", "tier": "qwen2.5-3b_a30", "node": "node2", "port": 8000},
    ],
}
HOSTS = {"node1": {"address": "h1", "hw": "c4130"}, "node2": {"address": "h2", "hw": "d7525"}}

RB = {
    "mode": "routebalance",
    "weights": {"w_quality": 0.5, "w_latency": 0.3, "w_cost": 0.2},
    "rounds": {"adaptive": True},
    "lpt": "max",
    "pending_decode": "own_length",
    "dead_reckoning": True,
    "telemetry_read": "per_round",
    "latency": {"source": "profiled_curve", "envelope": "batch_aware", "normalization": "fixed_ref",
                "table": "calib/qwen_tpot_curve.json", "table_md5": "1" * 32},
    "estimator": {"index_dir": "knn/qwen", "index_md5": "0" * 32, "encoder_revision": "a" * 40, "k": 50,
                  "score_type": "deepeval", "prior": "knn"},
    "budget": {"admission_filter": False, "confidence": 0.5},
    "telemetry": {"poll_ms": 50.0, "max_age_ms": 500.0, "timeout_s": 3.0},
    "pipeline": None,
}


def _pipeline(router="best_route", dispatch="least_outstanding"):
    d = copy.deepcopy(RB)
    d["mode"] = "pipeline"
    d["weights"] = None
    d["pending_decode"] = None
    d["telemetry_read"] = "poll"
    d["pipeline"] = {"router": {"type": router, "kwargs": {}}, "dispatch": {"type": dispatch, "seed": 1}}
    return d


def _write(tmp_path, fleet=FLEET, hosts=HOSTS):
    (tmp_path / "fleet.json").write_text(json.dumps(fleet))
    (tmp_path / "hosts.json").write_text(json.dumps(hosts))
    return tmp_path / "fleet.json", tmp_path / "hosts.json"


# ---------------------------------------------------------------- fleet

def test_fleet_loads_in_file_order(tmp_path):
    insts = load_fleet(*_write(tmp_path))
    assert [i.instance_id for i in insts] == ["Qwen-2.5-14B_0", "Qwen-2.5-3B_0"]
    assert insts[0].tier.gpu == "v100" and insts[0].base_url == "http://h1:8000"


@pytest.mark.parametrize("mutate,msg", [
    (lambda f, h: f["instances"][0].update(tier="nope"), "unknown tier"),
    (lambda f, h: f["instances"][0].update(node="node9"), "missing from"),
    (lambda f, h: h["node1"].update(hw="d7525"), "expects v100"),
    (lambda f, h: h["node1"].update(hw="r7525"), "not in routebalance.hw"),
    (lambda f, h: f["instances"].append(dict(f["instances"][0])), "duplicate instance_id"),
    (lambda f, h: (h.update(node3={"address": "h2", "hw": "d7525"}),
                   f["instances"].append({"instance_id": "Qwen-2.5-3B_1", "tier": "qwen2.5-3b_a30",
                                          "node": "node3", "port": 8000})), "already used"),
    (lambda f, h: f["tiers"]["qwen2.5-3b_a30"].update(cost_out=0.0), "invalid fleet"),
    (lambda f, h: f["tiers"]["qwen2.5-3b_a30"].update(extra=1), "invalid fleet"),
    (lambda f, h: f["instances"][0].update(port="8000"), "invalid fleet"),
    (lambda f, h: f["instances"].pop(), "tiers with no instance"),
])
def test_fleet_errors(tmp_path, mutate, msg):
    fleet, hosts = copy.deepcopy(FLEET), copy.deepcopy(HOSTS)
    mutate(fleet, hosts)
    with pytest.raises(ConfigError, match=msg):
        load_fleet(*_write(tmp_path, fleet, hosts))


def test_missing_hosts_file_is_an_error(tmp_path):
    fleet_path, _ = _write(tmp_path)
    with pytest.raises(ConfigError, match="cannot read"):
        load_fleet(fleet_path, tmp_path / "absent.json")


# ---------------------------------------------------------------- scheduler

def test_valid_configs_parse():
    assert parse_scheduler(RB).weights.w_latency == 0.3
    assert parse_scheduler(_pipeline()).pipeline.dispatch.type == "least_outstanding"
    d = _pipeline(router="rb_decoupled")
    d["weights"] = dict(RB["weights"])
    assert parse_scheduler(d).weights is not None


@pytest.mark.parametrize("mutate", [
    lambda d: d["weights"].update(w_cost=0.3),                      # does not sum to 1
    lambda d: d["weights"].pop("w_cost"),                            # incomplete weights (v3 merge leak)
    lambda d: d["weights"].update(w_balance=0.0),                    # unknown key
    lambda d: d.update(quality_norm="raw"),                          # unknown top-level key (v3 misplaced key)
    lambda d: d["rounds"].update(max_batch=8),                       # adaptive rounds take no fixed size
    lambda d: d.update(rounds={"adaptive": False, "max_batch": 8}),  # fixed rounds need both values
    lambda d: d["latency"].update(source="learned"),                 # unknown latency source
    lambda d: d["estimator"].update(k="50"),                         # no coercion from strings
    lambda d: d["estimator"].update(index_md5="abc"),                # malformed md5
    lambda d: d["latency"].update(table_md5="0" * 31),               # malformed table md5
    lambda d: d["latency"].pop("table_md5"),                         # the table is pinned like the index
    lambda d: d["estimator"].update(encoder_revision="main"),        # a branch name is not a pinned commit
    lambda d: d["latency"].update(envelope="queue_aware"),           # a curve has no single TPOT to queue on
    lambda d: d["latency"].update(source="xgboost", envelope="own_length"),     # not an arm that exists
    lambda d: d["telemetry"].update(max_age_ms=10.0),                # stale by construction
    lambda d: d["telemetry"].update(max_age_ms=100.0),               # less than 3x poll_ms
    lambda d: d.update(pipeline={"router": {"type": "passthrough", "kwargs": {}},
                                 "dispatch": {"type": "po2", "seed": 1}}),  # pipeline section in RB mode
    lambda d: d.update(weights=None),                                # RB scoring without weights
    lambda d: d.pop("budget"),                                       # missing section
])
def test_scheduler_errors(mutate):
    d = copy.deepcopy(RB)
    mutate(d)
    with pytest.raises(ConfigError):
        parse_scheduler(d)


def test_pending_decode_is_required_exactly_in_routebalance_mode():
    d = copy.deepcopy(RB)
    d["pending_decode"] = None
    with pytest.raises(ConfigError, match="pending_decode is required"):
        parse_scheduler(d)
    d = _pipeline()
    d["pending_decode"] = "lpt_key"
    with pytest.raises(ConfigError, match="lpt_key is RB-only"):
        parse_scheduler(d)
    d["pending_decode"] = "own_length"                      # the D38 sweep
    assert parse_scheduler(d).pending_decode == "own_length"


@pytest.mark.parametrize("mode,read,dr,dispatch,msg", [
    ("routebalance", "per_round", True, None, None),
    ("routebalance", "poll", True, None, None),                      # the D54 control
    ("routebalance", "per_round", False, None, None),                # W4's dead-reckoning ablation
    ("routebalance", "per_request", True, None, "never per request"),
    ("pipeline", "poll", True, "least_outstanding", None),
    ("pipeline", "per_request", False, "least_outstanding", None),   # v3's probe arm (D55)
    ("pipeline", "per_request", False, "po2", None),
    ("pipeline", "poll", False, "least_outstanding", "dead reckoning off"),
    ("pipeline", "per_request", True, "least_outstanding", "dead reckoning off"),
    ("pipeline", "per_round", True, "least_outstanding", "decides per request"),
    ("pipeline", "per_request", False, "round_robin", "reads no load"),
    ("pipeline", "per_request", False, "random", "reads no load"),
])
def test_how_load_is_read_follows_the_mode(mode, read, dr, dispatch, msg):
    d = copy.deepcopy(RB) if mode == "routebalance" else _pipeline(dispatch=dispatch)
    d["telemetry_read"], d["dead_reckoning"] = read, dr
    if msg is None:
        assert parse_scheduler(d).telemetry_read == read
    else:
        with pytest.raises(ConfigError, match=msg):
            parse_scheduler(d)


def test_pipeline_mode_rejects_the_budget_admission_filter():
    d = _pipeline(router="rb_decoupled")
    d["weights"] = RB["weights"]
    d["budget"] = {"admission_filter": True, "confidence": 0.5}
    with pytest.raises(ConfigError, match="admission filter"):
        parse_scheduler(d)


def test_pipeline_without_rb_scoring_rejects_weights():
    d = _pipeline()
    d["weights"] = dict(RB["weights"])
    with pytest.raises(ConfigError, match="never scores"):
        parse_scheduler(d)


def test_pipeline_needs_its_section():
    d = _pipeline()
    d["pipeline"] = None
    with pytest.raises(ConfigError, match="needs a pipeline section"):
        parse_scheduler(d)


def test_digest_is_stable_and_sensitive():
    a, b = parse_scheduler(RB), parse_scheduler(copy.deepcopy(RB))
    assert config_digest(a) == config_digest(b)
    d = copy.deepcopy(RB)
    d["weights"] = {"w_quality": 0.6, "w_latency": 0.2, "w_cost": 0.2}
    assert config_digest(parse_scheduler(d)) != config_digest(a)


def test_config_objects_are_immutable():
    cfg = parse_scheduler(RB)
    with pytest.raises(ValidationError):
        cfg.weights.w_quality = 0.9


# ---------------------------------------------------------------- review round 1 regressions

from routebalance.config import fleet_digest, loads_strict  # noqa: E402


def _write_text(tmp_path, fleet_text, hosts=HOSTS):
    (tmp_path / "fleet.json").write_text(fleet_text)
    (tmp_path / "hosts.json").write_text(json.dumps(hosts))
    return tmp_path / "fleet.json", tmp_path / "hosts.json"


@pytest.mark.parametrize("text", [
    '{"a": 1, "a": 2}',                          # duplicate key, last-one-wins in plain json.loads
    '{"b": 0, "a": 1, "a": 2}',
    '{"a": {"b": 1, "b": 1}}',
    '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}',
])
def test_loads_strict_rejects_duplicates_and_non_finite_tokens(text):
    with pytest.raises(ConfigError):
        loads_strict(text)


def test_duplicate_keys_in_a_file_are_rejected(tmp_path):
    fleet_text = json.dumps(FLEET)[:-1] + ', "instances": []}'
    with pytest.raises(ConfigError, match="duplicate JSON key"):
        load_fleet(*_write_text(tmp_path, fleet_text))


def test_non_utf8_and_deep_json_are_config_errors(tmp_path):
    (tmp_path / "f.json").write_bytes(b'{"a": "\xff"}')
    (tmp_path / "h.json").write_text(json.dumps(HOSTS))
    with pytest.raises(ConfigError):
        load_fleet(tmp_path / "f.json", tmp_path / "h.json")
    with pytest.raises(ConfigError):
        loads_strict("[" * 100000 + "]" * 100000)


@pytest.mark.parametrize("path,value", [
    (("telemetry", "max_age_ms"), 1e400), (("telemetry", "timeout_s"), float("inf")),
    (("weights", "w_cost"), float("nan")),
])
def test_non_finite_numbers_are_rejected(path, value):
    d = copy.deepcopy(RB)
    d[path[0]][path[1]] = value
    with pytest.raises(ConfigError):
        parse_scheduler(d)


def test_infinite_price_is_rejected(tmp_path):
    fleet = copy.deepcopy(FLEET)
    fleet["tiers"]["qwen2.5-3b_a30"]["cost_out"] = float("inf")
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, fleet))


def test_router_kwargs_are_read_only_and_unaliased():
    doc = _pipeline()
    doc["pipeline"]["router"]["kwargs"] = {"threshold": 0.5}
    cfg = parse_scheduler(doc)
    before = config_digest(cfg)
    opts = cfg.pipeline.router.options()
    opts["threshold"] = 0.9                                      # a copy: the config does not change
    doc["pipeline"]["router"]["kwargs"]["threshold"] = 0.9       # the input is not aliased either
    assert cfg.pipeline.router.options() == {"threshold": 0.5} and config_digest(cfg) == before


def test_pipeline_config_can_be_copied_pickled_and_hashed():
    import pickle
    doc = _pipeline()
    doc["pipeline"]["router"]["kwargs"] = {"threshold": 0.5, "fallback_model": "Qwen/Qwen2.5-3B"}
    cfg = parse_scheduler(doc)
    assert copy.deepcopy(cfg) == cfg and pickle.loads(pickle.dumps(cfg)) == cfg
    assert hash(cfg) == hash(parse_scheduler(copy.deepcopy(doc)))


@pytest.mark.parametrize("kwargs", [{"nested": {"a": 1}}, {"xs": [1, 2]}, {"x": float("inf")}])
def test_router_kwargs_must_be_flat_finite_scalars(kwargs):
    doc = _pipeline()
    doc["pipeline"]["router"]["kwargs"] = kwargs
    with pytest.raises(ConfigError):
        parse_scheduler(doc)


def test_model_copy_is_not_an_update_path():
    with pytest.raises(TypeError, match="replaced whole"):
        parse_scheduler(RB).model_copy(update={"mode": "bogus"})


@pytest.mark.parametrize("address", ["", "H1", "h1.", "http://h1", "h1:9000", " h1", "h1\n", "h1\t"])
def test_addresses_must_be_bare_lowercase_hosts(tmp_path, address):
    hosts = copy.deepcopy(HOSTS)
    hosts["node1"]["address"] = address
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, FLEET, hosts))


def test_one_address_cannot_have_two_hardware_types(tmp_path):
    hosts = copy.deepcopy(HOSTS)
    hosts["node2"]["address"] = "h1"
    with pytest.raises(ConfigError, match="two hardware types"):
        load_fleet(*_write(tmp_path, FLEET, hosts))


@pytest.mark.parametrize("instance_id", ["", "Qwen-2.5-14B_0 ", "Qwen-2.5-14В_0"])   # empty, trailing space, Cyrillic В
def test_instance_ids_are_plain_ascii(tmp_path, instance_id):
    fleet = copy.deepcopy(FLEET)
    fleet["instances"][0]["instance_id"] = instance_id
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, fleet))


def test_two_tiers_cannot_share_model_and_gpu(tmp_path):
    fleet = copy.deepcopy(FLEET)
    fleet["tiers"]["qwen2.5-3b_a30_cheap"] = dict(fleet["tiers"]["qwen2.5-3b_a30"], cost_out=1e-9)
    fleet["instances"].append({"instance_id": "x", "tier": "qwen2.5-3b_a30_cheap", "node": "node3", "port": 8000})
    hosts = dict(HOSTS, node3={"address": "h3", "hw": "d7525"})
    with pytest.raises(ConfigError, match="ambiguous"):
        load_fleet(*_write(tmp_path, fleet, hosts))


@pytest.mark.parametrize("path,value", [(("estimator", "k"), 10**40), (("pipeline", "dispatch", "seed"), -1),
                                        (("pipeline", "dispatch", "seed"), 2**70)])
def test_integer_bounds(path, value):
    d = _pipeline()
    target = d
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError):
        parse_scheduler(d)


def test_negative_zero_does_not_split_the_digest():
    a = copy.deepcopy(RB)
    b = copy.deepcopy(RB)
    a["weights"] = {"w_quality": 0.7, "w_latency": 0.3, "w_cost": 0.0}
    b["weights"] = {"w_quality": 0.7, "w_latency": 0.3, "w_cost": -0.0}
    assert config_digest(parse_scheduler(a)) == config_digest(parse_scheduler(b))


def test_fleet_digest_covers_order_and_prices(tmp_path):
    base = fleet_digest(load_fleet(*_write(tmp_path, FLEET)))
    rev = copy.deepcopy(FLEET)
    rev["instances"].reverse()
    assert fleet_digest(load_fleet(*_write(tmp_path, rev))) != base
    price = copy.deepcopy(FLEET)
    price["tiers"]["qwen2.5-3b_a30"]["cost_out"] = 7e-08
    assert fleet_digest(load_fleet(*_write(tmp_path, price))) != base


@pytest.mark.parametrize("name", ["node1\n", "node 1"])
def test_node_names_are_plain_ascii(tmp_path, name):
    hosts = {name: HOSTS["node1"], "node2": HOSTS["node2"]}
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, FLEET, hosts))


@pytest.mark.parametrize("address", ["127.1", "010.0.0.1", "0x7f.0.0.1", "0x7f000001", "2130706433", "1.2.3"])
def test_numeric_addresses_must_be_canonical(tmp_path, address):
    hosts = copy.deepcopy(HOSTS)
    hosts["node1"]["address"] = address
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, FLEET, hosts))


def test_canonical_ipv4_is_accepted(tmp_path):
    hosts = copy.deepcopy(HOSTS)
    hosts["node1"]["address"] = "127.0.0.1"
    assert load_fleet(*_write(tmp_path, FLEET, hosts))[0].base_url == "http://127.0.0.1:8000"


def test_large_flat_json_parses_in_linear_time():
    import time
    doc = json.dumps({f"k{i}": 0 for i in range(100000)})
    t0 = time.perf_counter()
    assert len(loads_strict(doc)) == 100000
    assert time.perf_counter() - t0 < 5.0


@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-3B ", " Qwen/Qwen2.5-3B", " ", ""])
def test_model_names_have_no_surrounding_whitespace(tmp_path, model):
    fleet = copy.deepcopy(FLEET)
    fleet["tiers"]["qwen2.5-3b_a30"]["model"] = model
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, fleet))


@pytest.mark.parametrize("field,value", [("table", "/etc/passwd"), ("table", "calib/../../x"), ("table", " calib/x"),
                                         ("index_dir", "../../../x"), ("index_dir", "/abs/knn")])
def test_artifact_paths_are_relative(field, value):
    d = copy.deepcopy(RB)
    section = "latency" if field == "table" else "estimator"
    d[section][field] = value
    with pytest.raises(ConfigError):
        parse_scheduler(d)


@pytest.mark.parametrize("value", [0, 5000, "256"])
def test_max_num_seqs_is_a_bounded_integer(tmp_path, value):
    fleet = copy.deepcopy(FLEET)
    fleet["tiers"]["qwen2.5-3b_a30"]["max_num_seqs"] = value
    with pytest.raises(ConfigError):
        load_fleet(*_write(tmp_path, fleet))


def test_fleet_digest_covers_max_num_seqs(tmp_path):
    base = fleet_digest(load_fleet(*_write(tmp_path, FLEET)))
    other = copy.deepcopy(FLEET)
    other["tiers"]["qwen2.5-3b_a30"]["max_num_seqs"] = 128
    assert fleet_digest(load_fleet(*_write(tmp_path, other))) != base


def test_lowest_latency_needs_its_window():
    d = _pipeline(dispatch="lowest_latency")
    with pytest.raises(ConfigError, match="latency_window"):
        parse_scheduler(d)
    d["pipeline"]["dispatch"]["latency_window"] = 10
    assert parse_scheduler(d).pipeline.dispatch.latency_window == 10


@pytest.mark.parametrize("dispatch,window", [("lowest_latency", 0), ("lowest_latency", 10.0), ("lowest_latency", "10"),
                                             ("least_outstanding", 10), ("po2", 1)])
def test_latency_window_range_and_scope(dispatch, window):
    d = _pipeline(dispatch=dispatch)
    d["pipeline"]["dispatch"]["latency_window"] = window
    with pytest.raises(ConfigError):
        parse_scheduler(d)


@pytest.mark.parametrize("source,envelope", [("constant", "own_length"), ("constant", "queue_aware"),
                                             ("xgboost", "queue_aware"), ("profiled_curve", "batch_aware"),
                                             ("constant", "batch_aware")])
def test_supported_latency_pairs_parse(source, envelope):
    d = copy.deepcopy(RB)
    d["latency"].update(source=source, envelope=envelope)
    cfg = parse_scheduler(d)
    assert (cfg.latency.source, cfg.latency.envelope) == (source, envelope)
