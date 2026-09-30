"""Does vLLM stop at Qwen's <|im_end|> when it is sent as the standard OpenAI `stop` string instead of vLLM's
`stop_token_ids`? (owner 09-30: "a then b and stated in the appendix": test (a), the standard field; if it does not
stop, (b) omit it.) vLLM-SR v0.4.0 accepts only the OpenAI chat fields, so `stop_token_ids` [151645] (D102) cannot pass.

Sends each prompt three times to one vLLM engine, with the campaign's sampling (temperature 0, repetition_penalty 1.0,
frequency_penalty 1.2, max_completion_tokens 1024):
  ids   stop_token_ids [151645]      what RouteBalance and every baseline send
  str   stop ["<|im_end|>"]          the standard field vLLM-SR accepts
  none  neither
and reports, per variant, the finish reasons, the output lengths, and how many texts equal the `ids` text.

    python stop_token_probe.py --url http://127.0.0.1:8601 --model <served name> --prompts FILE.jsonl --n 40
"""
import argparse
import asyncio
import json
from collections import Counter

import aiohttp

VARIANTS = {"ids": {"stop_token_ids": [151645]}, "str": {"stop": ["<|im_end|>"]}, "none": {}}


async def one(session, url, model, prompt, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "repetition_penalty": 1.0, "frequency_penalty": 1.2, "max_completion_tokens": 1024, **extra}
    async with session.post(f"{url}/v1/chat/completions", json=body) as r:
        d = await r.json()
        if r.status != 200:
            return {"error": f"{r.status} {str(d)[:200]}"}
        c = d["choices"][0]
        return {"text": c["message"]["content"], "finish": c["finish_reason"], "stop_reason": c.get("stop_reason"),
                "tokens": d["usage"]["completion_tokens"]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    prompts = []
    for line in open(a.prompts):
        row = json.loads(line)
        prompts.append(row.get("prompt") or row.get("text") or row["messages"][0]["content"])
        if len(prompts) == a.n:
            break
    out = {k: [] for k in VARIANTS}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        for p in prompts:                      # one at a time: the texts must not depend on batching
            for k, extra in VARIANTS.items():
                out[k].append(await one(s, a.url, a.model, p, extra))
    for k, rows in out.items():
        errs = [r["error"] for r in rows if "error" in r]
        ok = [r for r in rows if "error" not in r]
        same = sum(1 for r, ref in zip(rows, out["ids"], strict=True)
                   if "text" in r and "text" in ref and r["text"] == ref["text"])
        print(f"{k:5s} ok {len(ok):3d} errors {len(errs)} finish {dict(Counter(r['finish'] for r in ok))} "
              f"stop_reason {dict(Counter(str(r['stop_reason']) for r in ok))} "
              f"tokens mean {sum(r['tokens'] for r in ok) / max(len(ok), 1):.1f} same-as-ids {same}/{len(rows)}"
              + (f" first error: {errs[0]}" if errs else ""))
    json.dump(out, open("stop_token_probe.out.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
