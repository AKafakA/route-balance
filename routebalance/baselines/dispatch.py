"""Baseline dispatchers: pick one instance among the router's candidates.

A dispatcher reads only the loads the pipeline hands it, taken from the shared TelemetryView: the engines' own
num_running + num_waiting, from the pollers' 50 ms cache (as llm-d's endpoint picker reads them) or fetched for
the request (v3's per-request probe). Dead reckoning is RouteBalance's own mechanism and no dispatcher uses it
(owner 09-25). A dispatcher never fetches anything itself; lowest_latency also learns from the completions and
timeouts the server reports. Candidates arrive in fleet order, and order is behaviour. least_outstanding is
v3's shortest queue on those counts (D40: one rule, no separate shortest-queue dispatcher).
Ties: least_outstanding gives them to the least recently assigned candidate, then to fleet order (as
RouteBalance's own tie-break, and like the round robin among ties of stock least-connection balancers;
v3's first-candidate rule biased the first instance); po2 keeps the first drawn (v3); lowest_latency
draws among them (LiteLLM).

Randomised dispatchers own a random.Random seeded from the config. A dispatcher is built for every
applied config, so applying a config is the only reseed and the only reset of counters, assignment
times and latency samples.
"""
from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Mapping, Sequence

from routebalance.config import ConfigError, Dispatch as DispatchConfig
from routebalance.telemetry import Load
from routebalance.types import Instance


class Dispatcher(ABC):
    def choose(self, candidates: Sequence[Instance], loads: Mapping[str, Load]) -> Instance:
        """`loads` covers every candidate (instance_id -> Load)."""
        if not candidates:
            raise ValueError(f"{type(self).__name__}: no candidates")
        return self._choose(candidates, loads)

    @abstractmethod
    def _choose(self, candidates: Sequence[Instance], loads: Mapping[str, Load]) -> Instance: ...

    def probe_set(self, candidates: Sequence[Instance]) -> Sequence[Instance]:
        """With the per-request probe: the candidates to fetch now; choose_probed then picks among them on
        their fetched loads. Every candidate by default; power of two draws its pair here, so it fetches only
        the two instances it compares (D59)."""
        return candidates

    def choose_probed(self, probed: Sequence[Instance], loads: Mapping[str, Load]) -> Instance:
        """`probed` is what probe_set returned; `loads` covers each of them."""
        return self.choose(probed, loads)

    def observe_completion(self, instance_id: str, e2e_s: float, output_tokens: int) -> None:
        """Called for every completed request (e2e_s: from sending it to its last token); only
        lowest_latency learns from it."""
        del instance_id, e2e_s, output_tokens

    def observe_timeout(self, instance_id: str) -> None:
        """Called for every request that timed out; only lowest_latency learns from it."""
        del instance_id


def _queue(load: Load) -> int:
    return load.stats.num_running + load.stats.num_waiting


class RoundRobin(Dispatcher):
    """One rotation per model, keyed by the first candidate's model as in v3; under passthrough that
    is a single rotation over the whole fleet."""

    def __init__(self):
        self._next: dict[str, int] = {}

    def _choose(self, candidates, loads):
        key = candidates[0].model
        idx = self._next.get(key, 0) % len(candidates)   # default-ok: a rotation starts at the first candidate
        self._next[key] = idx + 1
        return candidates[idx]


class LeastOutstanding(Dispatcher):
    """Shortest queue: the engine's own running + waiting (v3's shortest queue). Among equal queues the candidate
    assigned least recently (never assigned = 0.0), then the earliest in fleet order."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._assigned: dict[str, float] = {}

    def _choose(self, candidates, loads):
        best = min(candidates, key=lambda c: (_queue(loads[c.instance_id]),
                                              self._assigned.get(c.instance_id, 0.0)))  # default-ok: 0.0 = never
        self._assigned[best.instance_id] = self._clock()
        return best


class PowerOfTwo(Dispatcher):
    """Two distinct candidates drawn uniformly (random.sample, as v3), the shorter queue wins, the first
    drawn on a tie. A single candidate is taken without drawing, so it consumes no randomness. With the probe
    the pair is drawn first and only its two instances are fetched: the same draws as the cache form in every run
    whose load reads succeed. A failed read fails its request, and a run with a failed request enters no
    comparison (analysis/metrics.py); after one, the probe form has consumed a draw the cache form has not."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def _draw(self, candidates):
        return list(candidates) if len(candidates) < 2 else self._rng.sample(candidates, 2)

    @staticmethod
    def _shorter(pair, loads):
        return min(pair, key=lambda i: _queue(loads[i.instance_id]))

    def _choose(self, candidates, loads):
        return self._shorter(self._draw(candidates), loads)

    def probe_set(self, candidates):
        if not candidates:
            raise ValueError("PowerOfTwo: no candidates")
        return self._draw(candidates)

    def choose_probed(self, probed, loads):
        if not 1 <= len(probed) <= 2:
            raise ValueError(f"PowerOfTwo: choose_probed takes the drawn pair, got {len(probed)} instance(s)")
        return self._shorter(probed, loads)


class RandomDispatch(Dispatcher):
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def _choose(self, candidates, loads):
        return self._rng.choice(candidates)


TIMEOUT_SAMPLE_S = 1000.0   # LiteLLM's sample for a timed-out request (seconds, not per token)


class LowestLatency(Dispatcher):
    """Lowest mean of recent latency samples, after LiteLLM's latency-based routing (non-streaming path):
    https://github.com/BerriAI/litellm/blob/7abaf4ed/litellm/router_strategy/lowest_latency.py

    Each instance keeps its last `window` samples (L140-147). A completed request adds e2e_s /
    output_tokens (end-to-end seconds per output token; e2e_s alone if it produced no tokens,
    L120-131), and a request that timed out adds 1000.0; other failures add nothing (L193-228).
    A dispatch takes the candidate with the lowest sample mean, 0.0 for an instance without samples;
    candidates tied at the lowest are drawn uniformly with the seeded RNG (L414, L488-493: a random
    choice among deployments within a buffer of 0 above the minimum), so unseen instances go first,
    at random. Once any candidate has a sample, a dispatch also stores a 0.0 sample for each candidate
    that has none, which then counts in its mean until the window drops it: LiteLLM writes
    {"latency": [0]} for unseen deployments into its cached latency map itself (L394-405). Samples
    start empty for every applied config.
    """

    def __init__(self, window: int, seed: int):
        if window < 1:
            raise ConfigError(f"latency_window must be >= 1, got {window}")
        self._samples: dict[str, deque] = {}
        self._window = window
        self._rng = random.Random(seed)

    def _add(self, instance_id: str, sample: float) -> None:
        if instance_id not in self._samples:
            self._samples[instance_id] = deque(maxlen=self._window)
        self._samples[instance_id].append(sample)

    def observe_completion(self, instance_id, e2e_s, output_tokens):
        if output_tokens < 0 or not 0 <= e2e_s < math.inf:
            raise ValueError(f"{instance_id}: need e2e_s >= 0 and output_tokens >= 0, got {e2e_s}, {output_tokens}")
        self._add(instance_id, e2e_s / output_tokens if output_tokens > 0 else e2e_s)

    def observe_timeout(self, instance_id):
        self._add(instance_id, TIMEOUT_SAMPLE_S)

    def mean(self, instance_id: str) -> float:
        samples = self._samples.get(instance_id)
        return sum(samples) / len(samples) if samples else 0.0

    def _choose(self, candidates, loads):
        if any(c.instance_id in self._samples for c in candidates):
            for c in candidates:
                if c.instance_id not in self._samples:
                    self._add(c.instance_id, 0.0)
        means = [self.mean(c.instance_id) for c in candidates]
        lowest = min(means)
        tied = [c for c, m in zip(candidates, means, strict=True) if m == lowest]
        return tied[0] if len(tied) == 1 else self._rng.choice(tied)


def build_dispatcher(cfg: DispatchConfig) -> Dispatcher:
    """A fresh dispatcher (counters at zero, RNG at the seed) for one applied config."""
    if cfg.type == "round_robin":
        return RoundRobin()
    if cfg.type == "least_outstanding":
        return LeastOutstanding()
    if cfg.type == "po2":
        return PowerOfTwo(cfg.seed)
    if cfg.type == "random":
        return RandomDispatch(cfg.seed)
    if cfg.type == "lowest_latency":
        return LowestLatency(cfg.latency_window, cfg.seed)
    raise ConfigError(f"unknown dispatch type {cfg.type!r}")
