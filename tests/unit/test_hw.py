"""The hardware map is the only source of GPU classes; unknown hardware is an error."""
import pytest

from routebalance.hw import HW, UnknownHardware, gpu_of

HW_GPU = {hw: gpu for hw, (gpu, _) in HW.items()}


def test_known_hardware():
    assert gpu_of("d7525") == "a30" and gpu_of("c4130") == "v100" and gpu_of("d8545") == "a100"


def test_unknown_hardware_raises():
    with pytest.raises(UnknownHardware, match="not in routebalance.hw"):
        gpu_of("r7525")


def test_every_gpu_class_is_one_the_config_accepts():
    assert set(HW_GPU.values()) <= {"a30", "v100", "a100"}
