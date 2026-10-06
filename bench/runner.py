"""Open-loop benchmark client: every request is sent at its scheduled arrival time, whatever is in flight.

One run of one cell:
  1. read the scheduler's identity (GET /v1/config) and refuse to start unless its configuration digest is
     the one the scope expects for this cell;
  2. for request i, sleep until t0 + arrival_i (monotonic clock), then start it as its own task, so a slow
     reply never delays a later send;
  3. wait for every reply, then read the identity again (a configuration change during the cell shows);
     the cyclic garbage collector is off from the first send to the last reply, and the last request is started
     before the wait over every request is built (a full collection there had delayed the last send by up
     to 204 ms); a cell's records are acyclic and bounded, freed by reference counting;
  4. return one CellRecord (bench/records.py).
The runner sends no warm-up or ready-check requests (v3 sent six through the scheduler, into its decision
log and counters); warming the fleet is the driver's job. A failure is recorded with its error and never
retried. duration_s runs from the start of the arrival loop until the last reply (v3's benchmark
duration), so throughput is successful requests / duration_s. e2e_s is measured on the client, from handing
the request to the HTTP client until the reply body has been read.

Usage: python -m bench.runner --scope <rendered scope.json> --cell <cell_id> --scheduler-url http://HOST:PORT
                              --data-root <dir holding the datasets> --out-dir <cells root>/<cell_id>
The output directory is the cell's own (bench/records.py): it must exist, be named by the cell id and hold
no cell record yet.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from analysis.manifest import cell_problems
from bench.arrivals import Trace, arrival_times, schedule_digest
from bench.client import HttpSchedulerClient, SchedulerClient, parse_reply, request_body
from bench.dataset import Prompt, PromptSet, attach_budgets, load_hf_tokenizer, load_prompts, order_digest
from bench.records import (FORMAT, CellRecord, RequestRecord, Sampling, SchedulerIdentity, cell_path,
                           records_in, write_cell)
from bench.schema import Strict
from bench.scope import ExpectedCell, expected_cells, load_scope
from routebalance.config import ConfigError


class CellSpec(Strict):
    cell_id: str
    labels: dict[str, str]
    trace: Trace
    seed: int
    sampling: Sampling
    expected_config_digest: str


class WrongConfiguration(ConfigError):
    pass


async def run_cell(spec: CellSpec, prompts: PromptSet, client: SchedulerClient, run_id: str) -> CellRecord:
    if prompts.seed != spec.seed or len(prompts.prompts) != spec.trace.n_requests:
        raise ConfigError(f"cell {spec.cell_id}: the prompt set does not match the cell (seed {prompts.seed} vs "
                          f"{spec.seed}, {len(prompts.prompts)} prompts vs {spec.trace.n_requests} requests)")
    arrivals = arrival_times(spec.trace, spec.seed)
    identity_start = await client.identity()
    if identity_start.config_digest != spec.expected_config_digest:
        raise WrongConfiguration(f"cell {spec.cell_id}: the scheduler runs configuration "
                                 f"{identity_start.config_digest}, the scope expects {spec.expected_config_digest}")
    started_unix = time.time()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter()
        tasks = []
        for i, (prompt, arrival) in enumerate(zip(prompts.prompts, arrivals, strict=True)):
            delay = t0 + arrival - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            request_id = f"{spec.cell_id}.{run_id}.{i}"
            tasks.append(asyncio.create_task(_send(client, t0, i, request_id, prompt, arrival, spec.sampling)))
        await asyncio.sleep(0)                  # the last request starts before the wait below is built
        requests = await asyncio.gather(*tasks)
        duration_s = time.perf_counter() - t0
    finally:
        if was_enabled:
            gc.enable()
    ended_unix = time.time()
    identity_end, identity_end_error = await _identity_after(client)
    return CellRecord(
        format=FORMAT, cell_id=spec.cell_id, run_id=run_id, labels=spec.labels, seed=spec.seed, trace=spec.trace,
        arrival_digest=schedule_digest(arrivals), dataset_sha256=prompts.dataset_sha256,
        budgets_sha256=prompts.budgets_sha256,
        tokenizer_id=prompts.tokenizer_id, prompt_order=prompts.order,
        prompt_order_digest=order_digest([p.row for p in prompts.prompts]), sampling=spec.sampling,
        max_connections=client.max_connections, request_timeout_s=client.timeout_s,
        identity_start=identity_start, identity_end=identity_end, identity_end_error=identity_end_error,
        connection_waits=client.connection_waits, started_unix=started_unix, ended_unix=ended_unix,
        duration_s=duration_s, requests=list(requests))


async def _send(client: SchedulerClient, t0: float, index: int, request_id: str, prompt: Prompt,
                arrival: float, sampling: Sampling) -> RequestRecord:
    body = request_body(request_id, prompt, sampling)
    sent = time.perf_counter()
    try:
        status, raw = await client.complete(body)
        e2e = time.perf_counter() - sent
        reply, error = parse_reply(status, raw, request_id)
    except Exception as e:  # every failure is recorded, which invalidates the cell; none is retried
        e2e = time.perf_counter() - sent
        reply, error = None, f"{type(e).__name__}: {e}"
    return RequestRecord(index=index, request_id=request_id, prompt_row=prompt.row, prompt_id=prompt.prompt_id,
                         prompt=prompt.text, prompt_tokens=prompt.prompt_tokens, budget_cost=prompt.budget_cost,
                         arrival_s=arrival,
                         sent_s=sent - t0, e2e_s=e2e, reply=reply, error=error)


async def _identity_after(client: SchedulerClient) -> tuple[Optional[SchedulerIdentity], Optional[str]]:
    """The identity after the last reply; a scheduler that died during the cell still leaves its record."""
    try:
        return await client.identity(), None
    except Exception as e:  # recorded; the analysis marks the cell invalid
        return None, f"{type(e).__name__}: {e}"


async def _run_http(spec: CellSpec, prompts: PromptSet, url: str, max_connections: int, timeout_s: float,
                    run_id: str) -> CellRecord:
    async with HttpSchedulerClient(url, max_connections, timeout_s) as client:
        return await run_cell(spec, prompts, client, run_id)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run one benchmark cell against a live scheduler.")
    ap.add_argument("--scope", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--scheduler-url", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    scope = load_scope(a.scope)
    cells = {c.cell_id: c for c in expected_cells(scope)}
    if a.cell not in cells:
        raise ConfigError(f"cell {a.cell!r} is not in {a.scope}")
    cell = cells[a.cell]
    if not a.out_dir.is_dir():
        raise ConfigError(f"{a.out_dir} is not a directory")
    if a.out_dir.resolve().name != cell.cell_id:
        raise ConfigError(f"{a.out_dir}: a cell is written to its own directory, named {cell.cell_id}")
    if cell_path(a.out_dir, cell.cell_id).exists():
        raise ConfigError(f"{cell_path(a.out_dir, cell.cell_id)} exists; a cell is never overwritten")
    if records_in(a.out_dir):
        raise ConfigError(f"{a.out_dir} already holds a cell record {records_in(a.out_dir)}")
    tokenizer = load_hf_tokenizer(cell.dataset.tokenizer_id)
    prompts = load_prompts(a.data_root / cell.dataset.path, cell.trace.n_requests, cell.prompt_order, cell.seed,
                           tokenizer, cell.dataset.tokenizer_id)
    if cell.dataset.budgets is not None:
        prompts = attach_budgets(prompts, a.data_root / cell.dataset.budgets.path, a.data_root / cell.dataset.path)
    _check_before_sending(cell, prompts)
    spec = CellSpec(cell_id=cell.cell_id, labels=cell.labels, trace=cell.trace, seed=cell.seed,
                    sampling=scope.bench.sampling, expected_config_digest=cell.config.config_digest)
    record = asyncio.run(_run_http(spec, prompts, a.scheduler_url, scope.bench.max_connections,
                                   scope.bench.request_timeout_s, uuid.uuid4().hex[:8]))
    out = write_cell(record, a.out_dir)
    problems = cell_problems(record, cell, scope.bench)
    print(f"{out}: {len(record.requests)} requests in {record.duration_s:.1f} s; "
          + ("valid" if not problems else "INVALID: " + "; ".join(problems)))
    return 0 if not problems else 1


def _check_before_sending(cell: ExpectedCell, prompts: PromptSet) -> None:
    """The dataset and the prompt order about to be sent are the ones the scope expects (the arrivals are
    generated from the cell's own trace and seed)."""
    if prompts.dataset_sha256 != cell.dataset.sha256:
        raise ConfigError(f"dataset {cell.dataset.path} has sha256 {prompts.dataset_sha256}, "
                          f"the scope expects {cell.dataset.sha256}")
    want = None if cell.dataset.budgets is None else cell.dataset.budgets.sha256
    if prompts.budgets_sha256 != want:
        raise ConfigError(f"the budget table has sha256 {prompts.budgets_sha256}, the scope expects {want}")
    if order_digest([p.row for p in prompts.prompts]) != cell.order_digest():
        raise ConfigError(f"the prompt order does not match the scope's (is the dataset's row count "
                          f"{cell.dataset.rows}?)")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
