"""The v3 S2 fleet in deployment order (order is behaviour), and a Load builder for handcrafted telemetry.

Dispatchers read only num_running, num_waiting and in_flight, so loads are built from those three and
every other stats field is zero (by introspection, so the builder follows the telemetry schema).
"""
from dataclasses import fields, replace

from routebalance.telemetry import InstanceStats, Load
from routebalance.types import Instance, Tier

TIERS = {
    "14B": Tier("qwen2.5-14b_v100", "Qwen/Qwen2.5-14B", "v100", 2e-7, 2e-7, 256, ()),
    "3B": Tier("qwen2.5-3b_a30", "Qwen/Qwen2.5-3B", "a30", 6e-8, 6e-8, 256, ()),
    "72B": Tier("qwen2.5-72b_a100", "Qwen/Qwen2.5-72B", "a100", 9e-7, 9e-7, 256, ()),
    "7B": Tier("qwen2.5-7b_a30", "Qwen/Qwen2.5-7B", "a30", 1e-7, 1e-7, 256, ()),
}
COUNTS = {"14B": 3, "3B": 3, "72B": 2, "7B": 5}
FLEET = [Instance(f"Qwen-2.5-{size}_{k}", TIERS[size], f"node-{size}-{k}", f"h-{size.lower()}-{k}", 8000)
         for size, n in COUNTS.items() for k in range(n)]
POOL = ["Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-7B"]
_ZERO = InstanceStats(**{f.name: 0 for f in fields(InstanceStats)})


def load(running=0, waiting=0, in_flight=0, age_ms=0.0, engine_waiting=0) -> Load:
    return Load(replace(_ZERO, num_running=running, num_waiting=waiting), age_ms, in_flight, engine_waiting)


def of_model(size: str) -> list[Instance]:
    return [i for i in FLEET if i.tier is TIERS[size]]
