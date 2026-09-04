import contextlib
import os

import torch


class DeviceSpec(str):
    """A discovered device, retaining its NineToothed backend contract."""

    def __new__(cls, device, backend):
        value = super().__new__(cls, device)
        value.backend = backend
        return value


def get_available_devices():
    """Return device strings annotated with their recommended backend."""
    devices = []

    if torch.cuda.is_available():
        devices.append(DeviceSpec("cuda", "cuda"))

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append(DeviceSpec("mlu", "triton"))

    if hasattr(torch, "npu") and torch.npu.is_available():
        devices.append(DeviceSpec("npu", "ascend"))

    # Keep backend selection centralized: an available NPU is preferred unless
    # the caller explicitly selected a backend before importing test helpers.
    if devices and any(str(device).split(":", 1)[0] == "npu" for device in devices):
        os.environ.setdefault("NINETOOTHED_BACKEND", "ascend")

    return tuple(devices)


def backend_for_device(device):
    """Resolve an explicit backend, preferring the environment override."""
    override = os.environ.get("NINETOOTHED_BACKEND")
    if override:
        return override
    kind = str(device).split(":", 1)[0]
    return getattr(
        device, "backend", {"npu": "ascend", "cuda": "cuda"}.get(kind, "triton")
    )


def get_stream(device):
    """Create a stream using the runtime native to ``device``."""
    if str(device).split(":", 1)[0] == "npu":
        return torch.npu.Stream(device=device)
    if str(device).split(":", 1)[0] == "cuda":
        return torch.cuda.Stream(device=device)
    return contextlib.nullcontext()


def synchronize(device=None):
    """Synchronize the selected device runtime."""
    kind = str(device).split(":", 1)[0] if device is not None else "cuda"
    if kind == "npu":
        return torch.npu.synchronize()
    if kind == "cuda":
        return torch.cuda.synchronize(device=device)
    return None


def device_count(device):
    kind = str(device).split(":", 1)[0]
    if kind == "npu":
        return torch.npu.device_count()
    if kind == "cuda":
        return torch.cuda.device_count()
    return 1


def assert_artifact_ready(kernel, device):
    """Validate binary-backed artifacts only where the backend promises one."""
    if backend_for_device(device) == "ascend":
        return
    assert kernel._library is not None


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401
