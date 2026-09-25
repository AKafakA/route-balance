"""The judge loop: every unscored served text goes to the judge once, in the judge's own input format, and only
the judge's scores for exactly those texts enter a new store file."""
import asyncio
import hashlib
import json

import pytest

from analysis.quality import ScoreStore, text_sha256
from bench.arrivals import constant_trace
from bench.dataset import load_prompts
from bench.records import Sampling, write_cell
from bench.runner import CellSpec, run_cell
from routebalance.config import ConfigError
from tests.unit.fakes import CONFIG_DIGEST, SAMPLING, FakeScheduler, FakeTokenizer, write_dataset
from tools import judge_served as js

KEY = "deepeval-llama3.1-8b-it_reference"


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _cell(tmp_path, n=4):
    prompts = load_prompts(write_dataset(tmp_path, n), n, "fixed", 1, FakeTokenizer(), "tok")
    spec = CellSpec(cell_id="c1__l1__s1", labels={}, trace=constant_trace(n, 200.0, 1.0), seed=1,
                    sampling=Sampling(**SAMPLING), expected_config_digest=CONFIG_DIGEST)
    rec = asyncio.run(run_cell(spec, prompts, FakeScheduler(fail=lambda i: i == 3), "r1"))
    root = tmp_path / "cells"
    (root / "c1__l1__s1").mkdir(parents=True)
    write_cell(rec, root / "c1__l1__s1")
    return root, [p.text for p in prompts.prompts]


def _refs(tmp_path, prompts):
    return _jsonl(tmp_path / "corpus.jsonl", [{"prompt": p, "reference_text": f"ref {p}", "models": {}} for p in prompts])


def _texts(tmp_path, rows):
    return _jsonl(tmp_path / "texts.jsonl", [{"pass": "reference", "success": True, "text_sha256": text_sha256(t),
                                              "prompt": p, "model": m, "text": t} for p, m, t in rows])


def _judge(lines, score=lambda i: 0.5):
    """What the scorer writes: the same lines, a score added under the key (None: the metric returned none)."""
    out = []
    for i, line in enumerate(lines):
        line = json.loads(json.dumps(line))
        s = score(i)
        if s is not None:
            line["models"][js.SLOT]["llm_judge_scores"] = {KEY: s}
        out.append(line)
    return out


def test_collect_sends_each_unscored_text_once_in_the_judges_format(tmp_path):
    root, prompts = _cell(tmp_path)
    refs = _refs(tmp_path, prompts)
    texts = _texts(tmp_path, [(prompts[0], "m-small", "ok"), (prompts[0], "m-large", "other")])
    lines, empty = js.collect(KEY, [], [texts], [root], [refs])
    # the cell served "ok" by m-small for its 3 successful prompts; one of them repeats the reference text
    assert len(lines) == 4 and empty == []
    assert {(ln["prompt"], ln["models"][js.SLOT]["served_model"], ln["models"][js.SLOT]["response"]) for ln in lines} \
        == {(prompts[0], "m-large", "other")} | {(p, "m-small", "ok") for p in prompts[:3]}
    first = lines[0]
    assert set(first) == {"request_id", "prompt", "reference_text", "models"} and list(first["models"]) == [js.SLOT]
    assert first["reference_text"] == f"ref {first['prompt']}"
    assert len({ln["request_id"] for ln in lines}) == 4


def test_collect_skips_texts_the_store_already_scores(tmp_path):
    root, prompts = _cell(tmp_path)
    store = _jsonl(tmp_path / "store.0.jsonl", [{"prompt": prompts[0], "model": "m-small", "text_sha256": text_sha256("ok"),
                                                 "text": "ok", "scores": {KEY: 0.9}}])
    lines, _ = js.collect(KEY, [store], [], [root], [_refs(tmp_path, prompts)])
    assert {ln["prompt"] for ln in lines} == set(prompts[1:3])
    other = js.collect("another-judge", [store], [], [root], [_refs(tmp_path, prompts)])[0]
    assert len(other) == 3                                   # a score under another key does not count


def test_a_prompt_without_a_reference_answer_fails_the_collection(tmp_path):
    root, prompts = _cell(tmp_path)
    with pytest.raises(ConfigError, match="no reference answer"):
        js.collect(KEY, [], [], [root], [_refs(tmp_path, prompts[:1])])


def test_empty_texts_are_listed_not_sent(tmp_path):
    root, prompts = _cell(tmp_path)
    texts = _texts(tmp_path, [(prompts[0], "m-large", "")])
    lines, empty = js.collect(KEY, [], [texts], [root], [_refs(tmp_path, prompts)])
    assert empty == [{"prompt": prompts[0], "model": "m-large", "first_seen": "texts.jsonl:1"}]
    assert all(ln["models"][js.SLOT]["response"] for ln in lines)


def test_merge_adds_exactly_the_judged_texts_to_a_new_store_file(tmp_path):
    root, prompts = _cell(tmp_path)
    lines, _ = js.collect(KEY, [], [], [root], [_refs(tmp_path, prompts)])
    sent = _jsonl(tmp_path / "in.jsonl", lines)
    back = _jsonl(tmp_path / "out.jsonl", _judge(lines, score=lambda i: None if i == 1 else 0.0))
    new, unscored = js.merge(KEY, [], sent, back)
    assert len(new) == 2 and unscored == [lines[1]["request_id"]]
    store = _jsonl(tmp_path / "store.1.jsonl", new)
    s = ScoreStore.load([store], KEY)
    assert s.lookup(lines[0]["prompt"], "m-small", "ok") == ("matched", 0.0)
    with pytest.raises(ConfigError, match="already has"):
        js.merge(KEY, [store], sent, back)


@pytest.mark.parametrize("tamper", [
    lambda out: out.pop(),
    lambda out: out[0].update(request_id="x"),
    lambda out: out[0].update(prompt="changed"),
    lambda out: out[0]["models"][js.SLOT].update(response="changed"),
    lambda out: out[0]["models"].update(other={"response": "x"}),
    lambda out: out[0]["models"][js.SLOT].update(llm_judge_scores={KEY: 1.5}),
    lambda out: out[0]["models"][js.SLOT].update(llm_judge_scores={KEY: 0.5, "stray": 0.1}),
])
def test_merge_refuses_a_judge_output_that_is_not_its_input(tmp_path, tamper):
    root, prompts = _cell(tmp_path)
    lines, _ = js.collect(KEY, [], [], [root], [_refs(tmp_path, prompts)])
    out = _judge(lines)
    tamper(out)
    with pytest.raises(ConfigError):
        js.merge(KEY, [], _jsonl(tmp_path / "in.jsonl", lines), _jsonl(tmp_path / "out.jsonl", out))


def _render(tmp_path, corpus, sha=None):
    render = tmp_path / "render"
    render.mkdir(exist_ok=True)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest() if sha is None else sha
    (render / "plan.json").write_text(json.dumps({"format": "rb-plan/1", "references": {
        "qwen": {"path": corpus.name, "sha256": digest}}}))
    return render


def test_the_command_line_reads_the_pinned_references_and_never_overwrites(tmp_path):
    root, prompts = _cell(tmp_path)
    out = tmp_path / "judge_in.jsonl"
    render = _render(tmp_path, _refs(tmp_path, prompts))
    argv = ["collect", "--judge-key", KEY, "--cells", str(root), "--render", str(render),
            "--data-root", str(tmp_path), "--family", "qwen", "--out", str(out)]
    assert js.main(argv) == 0 and len(out.read_text().splitlines()) == 3
    with pytest.raises(ConfigError, match="exists"):
        js.main(argv)


def test_a_reference_corpus_other_than_the_pinned_one_is_refused(tmp_path):
    root, prompts = _cell(tmp_path)
    render = _render(tmp_path, _refs(tmp_path, prompts), sha="0" * 64)
    with pytest.raises(ConfigError, match="the plan pins"):
        js.main(["collect", "--judge-key", KEY, "--cells", str(root), "--render", str(render), "--data-root",
                 str(tmp_path), "--family", "qwen", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(ConfigError, match="no references for"):
        js.pinned_corpora(json.loads((render / "plan.json").read_text()), ["gemma"], tmp_path)
