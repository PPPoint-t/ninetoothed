import os
from types import SimpleNamespace

import pytest

from ninetoothed.backends import BackendDeviceContractError, validate_backend_device
from tests import utils


def _devices(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch, "mlu", SimpleNamespace(is_available=lambda: False), raising=False)
    monkeypatch.setattr(utils.torch, "npu", SimpleNamespace(is_available=lambda: True), raising=False)


def test_device_discovery_has_no_backend_environment_side_effect(monkeypatch):
    _devices(monkeypatch)
    monkeypatch.setenv("NINETOOTHED_BACKEND", "cuda")

    devices = utils.get_available_devices()

    assert {str(device) for device in devices} == {"cuda", "npu"}
    assert os.environ["NINETOOTHED_BACKEND"] == "cuda"
    assert utils.backend_for_device(next(device for device in devices if str(device) == "npu")) == "ascend"


def test_environment_cannot_reinterpret_npu_as_cuda(monkeypatch):
    _devices(monkeypatch)
    monkeypatch.setenv("NINETOOTHED_BACKEND", "cuda")
    npu = next(device for device in utils.get_available_devices() if str(device) == "npu")

    assert utils.backend_for_device(npu) == "ascend"
    assert not utils.device_supports_backend(npu, "cuda")


def test_backend_resolution_is_order_independent_across_devices(monkeypatch):
    npu = utils.DeviceSpec("npu", "ascend")
    cuda = utils.DeviceSpec("cuda", "cuda")

    monkeypatch.setenv("NINETOOTHED_BACKEND", "ascend")
    assert utils.backend_for_device(npu) == "ascend"
    assert utils.backend_for_device(cuda) == "cuda"

    monkeypatch.setenv("NINETOOTHED_BACKEND", "cuda")
    assert utils.backend_for_device(npu) == "ascend"
    assert utils.backend_for_device(cuda) == "cuda"


def test_backend_filters_and_pairs_use_production_contract(monkeypatch):
    _devices(monkeypatch)

    assert [str(device) for device in utils.get_available_devices("ascend")] == ["npu"]
    assert [str(device) for device in utils.get_available_devices("cuda")] == ["cuda"]
    assert [str(device) for device in utils.get_available_devices("triton")] == ["cuda"]
    assert [str(device) for device in utils.get_available_devices("tilelang")] == ["cuda"]
    assert {(backend, str(device)) for backend, device in utils.backend_device_pairs(("ascend", "triton"))} == {
        ("ascend", "npu"),
        ("triton", "cuda"),
    }


def test_invalid_backend_device_pair_is_rejected():
    with pytest.raises(BackendDeviceContractError, match="requires device type"):
        validate_backend_device("triton", "npu")
