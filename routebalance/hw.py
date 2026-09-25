"""The one map from CloudLab hardware type to GPU class and GPUs per node.

Every other module asks this module; nothing else may hard-code the mapping (gates.static_defects
enforces it). An unknown hardware type is a configuration error, never a guess.
"""
from __future__ import annotations

HW = {                    # hardware type -> (GPU class, GPUs per node)
    "d7525": ("a30", 1),
    "c4130": ("v100", 4),
    "d8545": ("a100", 4),
}
HW_GPU = {hw: gpu for hw, (gpu, _) in HW.items()}


class UnknownHardware(ValueError):
    pass


def _row(hw_type: str) -> tuple[str, int]:
    try:
        return HW[hw_type]
    except KeyError:
        raise UnknownHardware(f"hardware type {hw_type!r} is not in routebalance.hw.HW {sorted(HW)}") from None


def gpu_of(hw_type: str) -> str:
    return _row(hw_type)[0]


def gpus_per_node(hw_type: str) -> int:
    return _row(hw_type)[1]
