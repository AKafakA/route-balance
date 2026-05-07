"""Simple benchmark client for RouteBalance scheduler smoke test.
No dependency on vllm.benchmarks — uses raw aiohttp against /v1/completions.
"""
import argparse
import asyncio
import json
import time
import random
import aiohttp
import numpy as np


async def send_request(session, url, prompt, max_tokens, semaphore):
    """Send one request and return timing info.
    RouteBalance scheduler returns JSON with ttft, itl, output_tokens, server_latency."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t_start = time.monotonic()

    try:
        async with semaphore:
            async with session.post(url, json=payload) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}: {text[:200]}", "prompt_len": len(prompt.split())}
                data = json.loads(text)

    except Exception as e:
        return {"error": str(e), "prompt_len": len(prompt.split())}

    t_end = time.monotonic()
    client_e2e = t_end - t_start

    if not data.get("success", False):
        return {"error": data.get("error", "unknown"), "prompt_len": len(prompt.split())}

    ttft = data.get("ttft", 0)
    itl = data.get("itl", [])
    output_tokens = data.get("output_tokens", 0)
    server_e2e = data.get("server_latency", client_e2e)
    tpot = np.mean(itl) if itl else 0
    overhead = data.get("scheduling_overhead_breakdown", {})

    return {
        "ttft": ttft,
        "tpot": tpot,
        "e2e": server_e2e,
        "client_e2e": client_e2e,
        "output_tokens": output_tokens,
        "prompt_len": len(prompt.split()),
        "model": data.get("model", ""),
        "overhead": overhead,
    }


async def run_benchmark(host, port, dataset_path, num_prompts, request_rate, output_path):
    """Run benchmark and collect results."""
    # Load prompts
    prompts = []
    with open(dataset_path) as f:
        for line in f:
            item = json.loads(line)
            # Handle different dataset formats
            prompt = item.get("prompt", item.get("input", item.get("text", "")))
            if isinstance(prompt, list):
                # Chat format
                prompt = "\n".join(m.get("content", "") for m in prompt if isinstance(m, dict))
            prompts.append(prompt)

    if num_prompts < len(prompts):
        prompts = prompts[:num_prompts]
    else:
        # Repeat if needed
        prompts = (prompts * (num_prompts // len(prompts) + 1))[:num_prompts]

    url = f"http://{host}:{port}/v1/completions"
    semaphore = asyncio.Semaphore(32)
    results = []

    print(f"Sending {len(prompts)} requests at {request_rate} QPS to {url}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            if request_rate > 0 and i > 0:
                await asyncio.sleep(1.0 / request_rate)
            task = asyncio.create_task(
                send_request(session, url, prompt, max_tokens=128, semaphore=semaphore)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    # Compute stats
    successful = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if successful:
        ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
        e2es = [r["e2e"] for r in successful]
        tpots = [r["tpot"] for r in successful if r["tpot"] > 0]
        output_lens = [r["output_tokens"] for r in successful]

        stats = {
            "num_requests": len(prompts),
            "num_successful": len(successful),
            "num_errors": len(errors),
            "request_rate": request_rate,
            "ttft_p50_ms": float(np.percentile(ttfts, 50) * 1000) if ttfts else 0,
            "ttft_p95_ms": float(np.percentile(ttfts, 95) * 1000) if ttfts else 0,
            "ttft_p99_ms": float(np.percentile(ttfts, 99) * 1000) if ttfts else 0,
            "tpot_p50_ms": float(np.percentile(tpots, 50) * 1000) if tpots else 0,
            "tpot_p95_ms": float(np.percentile(tpots, 95) * 1000) if tpots else 0,
            "e2e_p50_ms": float(np.percentile(e2es, 50) * 1000) if e2es else 0,
            "e2e_p95_ms": float(np.percentile(e2es, 95) * 1000) if e2es else 0,
            "avg_output_tokens": float(np.mean(output_lens)) if output_lens else 0,
            "throughput_rps": len(successful) / (max(e2es) if e2es else 1),
        }

        print(f"\nResults: {len(successful)}/{len(prompts)} successful, {len(errors)} errors")
        print(f"  TTFT P50={stats['ttft_p50_ms']:.0f}ms P95={stats['ttft_p95_ms']:.0f}ms")
        print(f"  TPOT P50={stats['tpot_p50_ms']:.0f}ms P95={stats['tpot_p95_ms']:.0f}ms")
        print(f"  E2E  P50={stats['e2e_p50_ms']:.0f}ms P95={stats['e2e_p95_ms']:.0f}ms")
        print(f"  Avg output tokens: {stats['avg_output_tokens']:.0f}")
    else:
        stats = {"num_requests": len(prompts), "num_successful": 0, "num_errors": len(errors)}
        print(f"\nAll {len(errors)} requests failed!")
        for e in errors[:3]:
            print(f"  Error: {e['error'][:200]}")

    if errors:
        print(f"  Sample errors: {[e['error'][:100] for e in errors[:3]]}")

    # Save
    output = {"stats": stats, "results": [r for r in results], "errors": [e for e in errors]}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--request-rate", type=float, default=2)
    parser.add_argument("--output", default="smoke_result.json")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.host, args.port, args.dataset, args.num_prompts, args.request_rate, args.output))
