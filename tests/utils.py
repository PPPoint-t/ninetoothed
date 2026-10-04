import contextlib
import os

import torch
import pytest

from ninetoothed.backends import (
    backend_supports_device,
    normalize_target,
    supported_device_types,
)


class DeviceSpec(str):
    """A discovered device, retaining its NineToothed backend contract."""

    def __new__(cls, device, backend):
        value = super().__new__(cls, device)
        value.backend = backend
        return value


def get_available_devices(backend=None):
    """Return device strings annotated with their recommended backend."""
    devices = []

    if torch.cuda.is_available():
        devices.append(DeviceSpec("cuda", "cuda"))

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append(DeviceSpec("mlu", "triton"))

    if hasattr(torch, "npu") and torch.npu.is_available():
        devices.append(DeviceSpec("npu", "ascend"))

    if backend is None:
        return tuple(devices)

    normalized = normalize_target(backend)
    return tuple(
        device for device in devices if device_supports_backend(device, normalized)
    )


def backend_for_device(device):
    """Resolve the backend from device metadata or its native device type."""
    kind = str(device).split(":", 1)[0]
    return getattr(
        device, "backend", {"npu": "ascend", "cuda": "cuda"}.get(kind, "triton")
    )


def device_supports_backend(device, backend):
    """Use the production backend/device contract for test parametrization."""
    return backend_supports_device(normalize_target(backend), str(device))


def backend_device_pairs(backends):
    """Return only discovered ``(backend, device)`` pairs allowed by contract."""
    return tuple(
        (normalize_target(backend).value, device)
        for backend in backends
        for device in get_available_devices(backend)
    )


def backend_device_params(backends):
    """Return ``(backend, device)`` pytest parameters, skipping absent hardware."""
    params = []
    for backend in backends:
        normalized = normalize_target(backend).value
        devices = get_available_devices(normalized)
        if devices:
            params.extend(
                pytest.param(normalized, device, id=f"{normalized}-{device}")
                for device in devices
            )
        else:
            params.append(
                pytest.param(
                    normalized,
                    None,
                    id=f"{normalized}-unavailable",
                    marks=pytest.mark.skip(
                        reason=(
                            f"backend `{normalized}` requires device type(s) "
                            f"{', '.join(supported_device_types(normalized))}, "
                            "but no compatible hardware is available"
                        )
                    ),
                )
            )
    return tuple(params)


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
