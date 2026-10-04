"""Backend registry and artifact contracts for NineToothed."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ninetoothed.ir import Kernel


class Target(str, Enum):
    TRITON = "triton"
    TILELANG = "tilelang"
    CUDA = "cuda"
    ASCEND = "ascend"


class BackendDeviceContractError(ValueError):
    """Raised when a backend is paired with an incompatible device type."""


# This is the backend runtime contract, rather than a hardware availability
# probe.  Hardware discovery belongs to callers; every consumer must use this
# single mapping when validating a backend/device pair.
_BACKEND_DEVICE_TYPES = {
    Target.TRITON: ("cuda",),
    Target.TILELANG: ("cuda",),
    Target.CUDA: ("cuda",),
    Target.ASCEND: ("npu",),
}


_CANONICAL_BACKEND_NAMES = {
    None: Target.TRITON,
    "triton": Target.TRITON,
    "tilelang": Target.TILELANG,
    "cuda": Target.CUDA,
    "ascend": Target.ASCEND,
}


@dataclass(frozen=True, kw_only=True)
class Capability:
    """Human-readable status for a backend implementation."""

    name: Target | str
    emits_source: bool
    can_execute: bool
    requires_external_compiler: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Artifact:
    """The output of lowering a :class:`Kernel` to a backend."""

    backend: Target
    kernel_name: str
    language: str
    sources: Mapping[str, str]
    entrypoint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def primary_source_name(self) -> str:
        return next(iter(self.sources))

    @property
    def primary_source(self) -> str:
        return self.sources[self.primary_source_name]

    def write_to(self, output_dir: str | Path) -> tuple[Path, ...]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = []

        for name, content in self.sources.items():
            path = output_path / name
            path.write_text(content, encoding="utf-8")
            paths.append(path)

        return tuple(paths)


@dataclass(frozen=True, kw_only=True)
class BuiltArtifact:
    """A materialized artifact with a reloadable binary and ABI manifest."""

    source: Artifact
    cache_key: str
    source_path: str
    binary_path: str | None
    manifest_path: str
    abi: Mapping[str, Any]


class Backend:
    """Base class for source and executable backend emitters."""

    name: Target | str
    capability: Capability
    supported_options: frozenset[str] = frozenset()

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize options before target lowering starts."""
        unknown = sorted(set(options) - self.supported_options)

        if unknown:
            names = ", ".join(f"`{name}`" for name in unknown)
            raise TypeError(
                f"Unsupported {backend_id_for(self.name)} backend option(s): {names}."
            )

        return dict(options)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        """Apply deterministic target choices that source emission must observe."""
        return kernel

    def emit(self, kernel: Kernel) -> Artifact:
        raise NotImplementedError


class Registry:
    """Small explicit registry to avoid import-time backend guessing."""

    def __init__(self):
        self._backends: MutableMapping[str, Backend] = {}

    def register(self, backend: Backend, *, replace: bool = False) -> None:
        backend_id = backend_id_for(backend.name)

        if backend_id in self._backends and not replace:
            raise ValueError(f"Backend `{backend_id}` is already registered.")

        self._backends[backend_id] = backend

    def get(self, name: Target | str | None) -> Backend:
        normalized = backend_id_for(Target.TRITON if name is None else name)

        try:
            return self._backends[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unsupported backend `{normalized}`. Available backends: {available}."
            ) from exc

    def capabilities(self) -> tuple[Capability, ...]:
        return tuple(backend.capability for backend in self._backends.values())


def backend_id_for(name: Target | str) -> str:
    """Return a normalized registry id without restricting plugin backends."""
    value = name.value if isinstance(name, Target) else str(name)
    backend_id = value.strip().lower()

    if not backend_id:
        raise ValueError("Backend id must not be empty.")
    return backend_id


def normalize_target(name: Target | str | None) -> Target:
    if isinstance(name, Target):
        return name

    key = None if name is None else str(name).lower()

    try:
        return _CANONICAL_BACKEND_NAMES[key]
    except KeyError as exc:
        supported = ", ".join(
            sorted(
                alias for alias in _CANONICAL_BACKEND_NAMES if isinstance(alias, str)
            )
        )
        raise ValueError(
            f"Unsupported backend `{name}`. Supported backends: {supported}."
        ) from exc


def supported_device_types(target: Target | str | None) -> tuple[str, ...]:
    """Return native PyTorch device types accepted by ``target``."""
    normalized = normalize_target(target)
    return _BACKEND_DEVICE_TYPES[normalized]


def backend_supports_device(
    target: Target | str | None,
    device: str,
) -> bool:
    """Return whether a backend accepts a native device type or device name."""
    device_type = str(device).split(":", 1)[0].lower()
    return device_type in supported_device_types(target)


def validate_backend_device(
    target: Target | str | None,
    device: str,
) -> Target:
    """Validate a backend/device pair and return the normalized target."""
    normalized = normalize_target(target)
    device_type = str(device).split(":", 1)[0].lower()
    supported = supported_device_types(normalized)
    if device_type not in supported:
        expected = ", ".join(supported)
        raise BackendDeviceContractError(
            f"Backend `{normalized.value}` requires device type(s) {expected}; "
            f"received `{device_type}` from device `{device}`."
        )
    return normalized
