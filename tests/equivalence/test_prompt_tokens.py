"""The prompt token count sent to the scheduler, checked against v3 with the real Qwen2.5 tokenizer.

The rebuild's count (one templated user message) equals v3's recorded input_len on every prompt, and the
raw prompt templated once is exactly the text v3's client produced. v3's backend then templated that text a
second time; the last assertion measures what that cost (19 tokens per request with this template).
"""
from bench.dataset import chat_prompt_tokens, load_hf_tokenizer, load_prompts
from tests.equivalence.banked_cells import EVAL_SET, V3_TOKENIZER, load_banked_cell

RB_CELL = "wq0.5_wl0.3_wc0.2_lambda30_n3534.json"


def test_prompt_tokens_equal_v3_input_len(oracle):
    tok = load_hf_tokenizer(V3_TOKENIZER)
    prompts = load_prompts(oracle.data_file(EVAL_SET), 3534, "per_seed", 1, tok, V3_TOKENIZER).prompts
    sent = load_banked_cell(oracle.cells_dir("s2_p1_l30") / RB_CELL)["response_details"]
    assert [p.prompt_tokens for p in prompts] == [r["input_len"] for r in sent]
    templated = [tok.apply_chat_template([{"role": "user", "content": p.text}], add_generation_prompt=True,
                                         tokenize=False) for p in prompts]
    assert templated == [r["prompt"] for r in sent]
    extra = {chat_prompt_tokens(tok, r["prompt"]) - r["input_len"] for r in sent}
    assert extra == {19}
    print(f"prompt tokens == v3 input_len on {len(prompts)}/{len(sent)} prompts; double templating added {extra}")
