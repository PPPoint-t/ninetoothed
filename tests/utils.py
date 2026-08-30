import contextlib

import torch


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    npu = getattr(torch, "npu", None)

    if npu is not None and npu.is_available():
        devices.append("npu")

    return tuple(devices)


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401
