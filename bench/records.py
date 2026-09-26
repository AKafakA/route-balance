"""The per-cell result record: everything one benchmark run of one cell produced.

One JSON file per cell, `<cell_id>.cell.json`. Only files with that suffix are cells; logs, decision logs and
sidecars never carry it, so they can never be counted as cells (v3 counted provenance sidecars).
In a campaign every cell has its own directory under the cells root, named by its id: `<root>/<cell_id>/`
holds exactly one record, `<cell_id>.cell.json`, beside the evidence the driver collected for the cell
(scripts/run_cell.sh: the configuration posted, the decision-log slice, scheduler stats and manifests, gates).
Times are seconds on the client's monotonic clock from the start of the arrival loop, except the *_unix
wall-clock stamps.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from bench.arrivals import Trace
from bench.dataset import Order
from bench.schema import Strict, parse, read_json
from routebalance.config import ConfigError

FORMAT = "rb-cell/1"
CELL_SUFFIX = ".cell.json"


class Sampling(Strict):
    temperature: float = Field(ge=0)
    repetition_penalty: float = Field(gt=0)
    frequency_penalty: float
    max_tokens: int = Field(ge=1)        # the output cap


class Reply(Strict):
    """A successful scheduler reply; the field meanings are the API contract in bench/client.py."""
    request_id: str
    model: str
    instance_id: str
    generated_text: str
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    ttft_s: float = Field(ge=0)
    server_latency_s: float = Field(ge=0)
    timings_ms: dict[str, float]

    @field_validator("timings_ms")
    @classmethod
    def _non_negative(cls, v: dict[str, float]) -> dict[str, float]:
        bad = sorted(k for k, x in v.items() if x < 0)
        if bad:
            raise ValueError(f"negative timings: {bad}")
        return v


class RequestRecord(Strict):
    index: int = Field(ge=0)             # position in the send order
    request_id: str
    prompt_row: int = Field(ge=0)
    prompt_id: str
    prompt: str                          # the raw prompt that was sent
    prompt_tokens: int = Field(ge=1)     # the count sent to the scheduler (bench/dataset.py)
    budget_cost: Optional[float]         # the budget sent (USD), budget cells only; None everywhere else
    arrival_s: float = Field(ge=0)       # scheduled send time
    sent_s: float = Field(ge=0)          # when the request was handed to the HTTP client
    e2e_s: float = Field(ge=0)           # from hand-off until the reply was read, or the failure
    reply: Optional[Reply]               # None exactly when the request failed
    error: Optional[str]

    @model_validator(mode="after")
    def _reply_xor_error(self):
        if (self.reply is None) == (self.error is None):
            raise ValueError("a request has either a reply or an error")
        return self


class SchedulerIdentity(Strict):
    config_digest: str
    fleet_digest: str


class CellRecord(Strict):
    format: Literal["rb-cell/1"]
    cell_id: str
    run_id: str
    labels: dict[str, str]
    seed: int = Field(ge=0, lt=2**32)
    trace: Trace
    arrival_digest: str
    dataset_sha256: str
    budgets_sha256: Optional[str]        # the budget table the budgets came from; None outside the budget sweep
    tokenizer_id: str
    prompt_order: Order
    prompt_order_digest: str
    sampling: Sampling
    max_connections: int
    request_timeout_s: float
    identity_start: SchedulerIdentity     # read before the first request
    identity_end: Optional[SchedulerIdentity]   # read after the last reply; None if that read failed
    identity_end_error: Optional[str]
    connection_waits: int                 # requests that waited for a pooled connection (> 0: not open loop)
    started_unix: float
    ended_unix: float
    duration_s: float                     # start of the arrival loop until the last reply
    requests: list[RequestRecord]

    @model_validator(mode="after")
    def _identity_end_xor_error(self):
        if (self.identity_end is None) == (self.identity_end_error is None):
            raise ValueError("the end identity is either read or its error is recorded")
        return self


def cell_path(out_dir: Path, cell_id: str) -> Path:
    return Path(out_dir) / f"{cell_id}{CELL_SUFFIX}"


def cell_dir(root: Path, cell_id: str) -> Path:
    """The cell's own directory under a campaign's cells root."""
    return Path(root) / cell_id


def records_in(directory: Path) -> list[str]:
    """Names of the cell records directly inside a directory (empty if it does not exist)."""
    d = Path(directory)
    return sorted(p.name for p in d.iterdir() if p.name.endswith(CELL_SUFFIX)) if d.is_dir() else []


def write_cell(record: CellRecord, out_dir: Path) -> Path:
    """Write atomically; an existing cell is never overwritten. If the final link fails for any other reason
    the temporary file is kept, so a finished run is never lost."""
    final = cell_path(out_dir, record.cell_id)
    tmp = final.with_name(f".{final.name}.{record.run_id}.tmp")
    tmp.write_text(record.model_dump_json())
    try:
        os.link(tmp, final)              # fails if the cell exists
    except FileExistsError:
        tmp.unlink()
        raise ConfigError(f"{final} exists; a cell is never overwritten") from None
    except OSError as e:
        raise ConfigError(f"could not publish {final} ({e}); the record is kept in {tmp}") from e
    tmp.unlink()
    return final


def read_cell(path: Path) -> CellRecord:
    return parse(CellRecord, read_json(path), f"cell {path}")
