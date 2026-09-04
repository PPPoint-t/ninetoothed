"""Ascend backend policy, contracts, and private launch metadata."""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.compiler.layout import analyze_layout_transfer
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import IndexExpr, Kernel, LaunchABI, LaunchBinding, ssa


class UnsupportedBackendOpError(ValueError):
    """Ascend legality rejection with an actionable remediation hint."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "the requested operation is outside the verified Ascend backend contract.",
        suggestion: str = "use a supported layout or select another backend.",
    ):
        self.reason = reason
        self.suggestion = suggestion
        super().__init__(f"{message} Reason: {reason} Suggestion: {suggestion}")


if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


class AscendBackend(Backend):
    """Emit the verified first-tier Ascend Triton source."""

    name = Target.ASCEND
    supported_options = frozenset({"max_core_dim", "soc_version"})
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Unified SSA backend emits FP16, BF16, and FP32 elementwise Ascend Triton source.",
            "Ascend 910B3 elementwise JIT and source reload are capability-gated.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate target options without probing an Ascend runtime."""
        normalized = dict(super().normalize_options(options))

        if "soc_version" in normalized:
            soc_version = normalized["soc_version"]

            if not isinstance(soc_version, str) or not soc_version.strip():
                raise ValueError(
                    "Ascend backend option `soc_version` must be a non-empty string."
                )

            normalized["soc_version"] = soc_version.strip()

        if "max_core_dim" in normalized:
            max_core_dim = normalized["max_core_dim"]

            if isinstance(max_core_dim, bool) or not isinstance(max_core_dim, int):
                raise TypeError(
                    "Ascend backend option `max_core_dim` must be an integer."
                )

            if not 1 <= max_core_dim <= 65535:
                raise ValueError(
                    "Ascend backend option `max_core_dim` must be between 1 and 65535."
                )

        return normalized

    def emit(self, kernel: Kernel) -> Artifact:
        # Delay the emitter import because it consumes this module's private
        # contracts.  The registry can then import the backend without a cycle.
        from ninetoothed.backends.emitters.ascend import emit

        return emit(kernel)

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        """Finalize target schedule choices before Ascend source emission."""
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        kernel = specialize_schedule_tiles(kernel)

        if kernel.ssa is None:
            return kernel

        contract = dict(kernel.ssa.metadata.get("schedule", {})).get(
            "ascend_linalg", {}
        )

        schedule = dict(kernel.ssa.metadata.get("schedule", {}))

        if contract.get("rank") != 3 or "ascend_batched_access_rewrite" in schedule:
            return kernel

        return replace(
            kernel,
            ssa=_rewrite_ascend_batched_matmul_access(kernel.ssa, contract),
        )


ASCEND_ELEMENTWISE_DTYPES = frozenset({"float16", "bfloat16", "float32", "int32"})
ASCEND_RNG_DTYPES = frozenset({"float16", "bfloat16", "float32"})
ASCEND_ATOMIC_DTYPES = frozenset({"float16", "bfloat16", "float32", "int32"})
_SIDECAR_SCHEMA = 3


@dataclass(frozen=True)
class AscendSocProfile:
    """Compile-time resource contract for a verified Ascend SoC."""

    name: str
    cube_cores: int | None
    vector_cores: int | None
    l2_cache_bytes: int | None
    ub_bytes: int | None = None
    l1_bytes: int | None = None
    l0a_bytes: int | None = None
    l0b_bytes: int | None = None
    l0c_bytes: int | None = None


_ASCEND_SOC_PROFILES = {
    "Ascend910B3": AscendSocProfile(
        name="Ascend910B3",
        cube_cores=20,
        vector_cores=40,
        l2_cache_bytes=192 * 1024 * 1024,
    )
}


def ascend_soc_profile(soc_version: str | None = None) -> AscendSocProfile:
    """Return the verified compile-time profile without initializing an NPU."""
    return _ASCEND_SOC_PROFILES.get(
        soc_version or "Ascend910B3",
        AscendSocProfile(
            name=soc_version or "unknown",
            cube_cores=None,
            vector_cores=None,
            l2_cache_bytes=None,
        ),
    )


def ascend_capability_matrix() -> Mapping[str, Any]:
    """Return the stable, backend-private Ascend capability contract.

    The result is data-only so callers can inspect admission policy without
    importing or probing an NPU runtime during collection or lowering.
    """
    return {
        "target": Target.ASCEND.value,
        "devices": ("Ascend910B3",),
        "soc_profiles": {
            name: {
                "cube_cores": profile.cube_cores,
                "vector_cores": profile.vector_cores,
                "l2_cache_bytes": profile.l2_cache_bytes,
                "ub_bytes": profile.ub_bytes,
                "l1_bytes": profile.l1_bytes,
                "l0a_bytes": profile.l0a_bytes,
                "l0b_bytes": profile.l0b_bytes,
                "l0c_bytes": profile.l0c_bytes,
            }
            for name, profile in _ASCEND_SOC_PROFILES.items()
        },
        "dtypes": {
            "elementwise": tuple(sorted(ASCEND_ELEMENTWISE_DTYPES)),
            "rng": tuple(sorted(ASCEND_RNG_DTYPES)),
            "atomic": tuple(sorted(ASCEND_ATOMIC_DTYPES)),
            "fail_closed": {
                "float64": "no-verified-ascend-triton-execution-path",
                "int8": "no-verified-weight-only-dequant-gemm-contract",
                "int4": "no-native-tensor-dtype-or-dequant-lowering-contract",
                "float8_e4m3fn": "CANN-910B3-vector-and-matrix-op-not-supported",
                "float8_e5m2": "CANN-910B3-vector-and-matrix-op-not-supported",
            },
        },
        "layouts": {
            "contiguous_ranks": (0, 1, 2, 3, 4),
            "dynamic_shape": True,
            "dynamic_stride": True,
            "non_contiguous": "verified-positive-stride-non-overlapping",
            "broadcast": "verified-rank1-row-and-column-singleton",
            "jagged": False,
            "negative_stride": False,
            "storage_offset": "zero-only-for-generic-dot-loop",
        },
        "jit": {"runtime_specialization": True, "aot_reload": True},
        "aot": {
            "package": "source-sidecar-manifest",
            "sidecar_schema": _SIDECAR_SCHEMA,
            "native_binary": "not-produced-by-triton-ascend",
            "independent_reload": True,
        },
        "microarchitecture": {
            "double_buffering": "not-emittable-by-triton-ascend-contract",
            "async_hbm_l1_ub_l0": "not-verified",
            "tile_autotuning": "fail-closed-deterministic-tiles-only",
            "verified_gemm_tile": {"m": 16, "n": 16, "k": 64},
        },
        "operations": {
            "elementwise": "verified-fp16-bf16-fp32-int32-positive-stride",
            "matmul": "verified-contiguous-rank-2-fp16-bf16-fp32",
            "batched_matmul": "verified-contiguous-rank-3-fp16-bf16-fp32-static-and-dynamic-shape",
            "dynamic_matmul_mnk": "verified-single-artifact-runtime-specialization-mnk",
            "row_vector_reduction": "verified-static-single-block-axis-sum-min-max-keepdim-and-fused-elementwise",
            "softmax": "verified-last-axis-static-fp16-fp32-block-lte-256",
            "rmsnorm": "verified-last-axis-static-fp16-fp32-fp32-accumulator-block-lte-256",
            "cross_block_multi_axis_reduction": "fail-closed-pending-private-schedule",
            "generic_dot_loop": "verified-static-shape-subset",
            "gemm_epilogue": "verified-silu; source-verified-gelu-leaky-relu-scale-bias-residual-same-kernel",
            "weight_only_matmul": "fail-closed-pending-w8a16-w4a16-dequant-contract",
            "attention": "verified-static-fp32-batch2-head2-seq32-causal-and-noncausal",
            "paged_attention": "fail-closed-pending-block-table-indirect-addressing",
            "varlen_attention": "fail-closed-pending-cu-seqlens-prefix-sum-contract",
            "gqa_mqa": "fail-closed-pending-kv-head-broadcast-contract",
            "standalone_conv2d": "fail-closed",
        },
    }


def ascend_toolchain_version() -> str:
    """Return the configured CANN version without initializing an NPU."""
    configured = os.environ.get("CANN_VERSION")

    if configured:
        return configured

    toolkit = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
        "ASCEND_HOME_PATH"
    )
    paths = (
        *((Path(toolkit) / "share/info/asc-devkit/version.info",) if toolkit else ()),
        Path(
            "/usr/local/Ascend/ascend-toolkit/latest/share/info/asc-devkit/version.info"
        ),
        Path("/usr/local/Ascend/driver/version.info"),
    )

    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")

                if separator and key.strip() in {"Version", "package_version"}:
                    return value.strip()
        except FileNotFoundError:
            continue

    return "unknown"


def normalize_ascend_dtype(dtype: str | None) -> str | None:
    """Return the canonical dtype spelling used by Ascend validation."""
    if dtype is None:
        return None

    value = str(dtype).strip().lower()

    if "." in value:
        value = value.rsplit(".", 1)[-1]

    return {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(value, value)


def unsupported_ascend_elementwise_dtypes(
    dtypes: tuple[str | None, ...],
    *,
    allow_rng_auxiliary: bool = False,
    allow_atomic: bool = False,
    allow_unspecified: bool = False,
) -> tuple[str, ...]:
    """Return dtype names outside the verified Ascend capability tier."""
    return tuple(
        sorted(
            {
                "unspecified" if dtype is None else normalize_ascend_dtype(dtype)
                for dtype in dtypes
                if normalize_ascend_dtype(dtype) not in ASCEND_ELEMENTWISE_DTYPES
                and not (
                    allow_rng_auxiliary and normalize_ascend_dtype(dtype) == "int32"
                )
                and not (
                    allow_atomic
                    and normalize_ascend_dtype(dtype) in ASCEND_ATOMIC_DTYPES
                )
                and not (allow_unspecified and normalize_ascend_dtype(dtype) is None)
            }
        )
    )


def static_forward_view_offset(value: Any) -> int:
    """Resolve the private static-forward logical-view offset contract."""
    expression = IndexExpr.parse(value)

    if expression.op == "constant" and _is_nonnegative_int(expression.value):
        return expression.value

    if expression.op == "symbol" and expression.value == "index":
        return 0

    if (
        expression.op == "add"
        and _is_index(expression.operands[0])
        and _is_nonnegative_int(expression.operands[1].value)
    ):
        return expression.operands[1].value

    raise ValueError(
        "Ascend logical views require a static forward offset: a non-negative "
        "constant, `index`, or `index + k`."
    )


def is_static_forward_view_offset(value: Any) -> bool:
    """Return whether a value satisfies the Ascend static-view contract."""
    try:
        static_forward_view_offset(value)
    except (SyntaxError, ValueError):
        return False

    return True


def validate_build_policy(metadata: Mapping[str, Any], request: Any) -> None:
    """Validate the currently verified private Ascend schedule contract."""
    schedule = dict(metadata.get("ssa_schedule", {}))

    if (
        schedule.get("tile", {}).get("elements") != 256
        or schedule.get("vector_width") != 1
    ):
        raise ValueError(
            "Ascend build requires the deterministic BLOCK=256 scalar SSA schedule."
        )

    if request.num_warps is not None or request.num_stages is not None:
        raise ValueError(
            "Ascend build does not accept runtime warp or stage configuration."
        )

    validate_ascend_profile_admission(metadata, request)


def validate_ascend_profile_admission(
    metadata: Mapping[str, Any], request: Any
) -> None:
    """Require build admission to match the profile chosen by scheduling.

    Unknown resource capacities deliberately remain unconstrained.  Every
    known value is sourced from ``AscendSocProfile`` rather than duplicated in
    the materializer or an ad-hoc hardware check.
    """
    options = dict(getattr(request, "backend_options", {}) or {})
    profile = ascend_soc_profile(options.get("soc_version"))
    matrix = ascend_capability_matrix()

    if profile.name not in matrix["soc_profiles"]:
        raise ValueError(
            f"Ascend build requires a verified SoC profile; received `{profile.name}`."
        )

    selected = dict(metadata.get("ssa_metadata", {})).get("selected_schedule_candidate")
    candidates = tuple(
        dict(metadata.get("ssa_metadata", {})).get("schedule_candidates", ())
    )

    if selected is None or not candidates:
        return

    candidate = next(
        (item for item in candidates if item.get("name") == selected), None
    )

    if candidate is None:
        raise ValueError(
            "Ascend build metadata does not contain its selected schedule candidate."
        )

    constraints = dict(candidate.get("constraints", {}))

    for name, expected in _profile_constraints(profile).items():
        actual = constraints.get(name)

        if actual != expected:
            raise ValueError(
                "Ascend build profile does not match the selected schedule: "
                f"`{name}` is {actual!r}, expected {expected!r} for {profile.name}."
            )


def ascend_cache_key(base_key: str, metadata: Mapping[str, Any]) -> str:
    """Namespace source cache entries by the selected Ascend runtime target."""
    profile = ascend_soc_profile(
        dict(metadata.get("ssa_schedule", {})).get("soc_version") or None
    )
    identity = {
        "base": base_key,
        "soc_version": dict(metadata.get("ssa_schedule", {})).get("soc_version", ""),
        "triton_ascend_arch": os.environ.get("TRITON_ASCEND_ARCH", ""),
        "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES", ""),
        "soc_profile": {
            "name": profile.name,
            "cube_cores": profile.cube_cores,
            "vector_cores": profile.vector_cores,
            "l2_cache_bytes": profile.l2_cache_bytes,
            "ub_bytes": profile.ub_bytes,
            "l1_bytes": profile.l1_bytes,
            "l0a_bytes": profile.l0a_bytes,
            "l0b_bytes": profile.l0b_bytes,
            "l0c_bytes": profile.l0c_bytes,
        },
        "cann_version": ascend_toolchain_version(),
    }

    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def ascend_logical_domain(
    specs, outputs: tuple[str, ...], *, allow_access_template: bool = False
) -> str:
    """Build the private logical-domain expression for an Ascend artifact."""
    by_name = {spec.name: spec for spec in specs}

    if not outputs:
        raise ValueError("Ascend launch planning requires at least one output tensor.")

    output = by_name[outputs[0]]

    if output.ndim == 0:
        return "1"

    dimensions = (
        tuple(output.layout.application_shape) if output.layout else tuple(output.shape)
    )
    source_shape = tuple(output.attrs.get("source_shape", output.shape))

    if len(dimensions) != len(source_shape) and not allow_access_template:
        raise ValueError(
            "Ascend logical-domain requires matching output and source ranks."
        )

    if len(dimensions) == 1:
        return f"min({IndexExpr.parse(dimensions[0]).render()}, ({source_shape[0]}))"

    def product(values) -> str:
        return (
            " * ".join(f"({IndexExpr.parse(value).render()})" for value in values)
            or "1"
        )

    if len(dimensions) != len(source_shape):
        return product(dimensions)

    return f"min({product(dimensions)}, {product(source_shape)})"


def ascend_uses_access_template(metadata: Mapping[str, Any]) -> bool:
    """Identify private schedules that consume a public access template."""
    schedule = dict(metadata.get("ssa_metadata", {})).get("schedule", {})

    return bool(
        dict(schedule).get("ascend_dot_loop")
        or dict(schedule).get("ascend_attention_loop")
    )


def write_ascend_sidecar(
    source_path: Path,
    *,
    abi: LaunchABI,
    specs,
    outputs,
    metadata: Mapping[str, Any],
) -> Path:
    """Persist public launch ABI plus Ascend AOT descriptive metadata."""
    path = _sidecar_path(source_path)
    payload = {
        "schema": _SIDECAR_SCHEMA,
        # This is the exact public LaunchABI from Compilation.  A sidecar must
        # never contain a second ABI with backend-specific launch semantics.
        "launch_abi": _ascend_abi_dict(abi),
        "logical_domain": ascend_logical_domain(
            specs,
            tuple(outputs),
            allow_access_template=ascend_uses_access_template(metadata),
        ),
        "outputs": tuple(outputs),
        "max_core_dim": dict(metadata.get("ssa_schedule", {})).get(
            "core_dim_limit", 65535
        ),
        "reduction_schedule": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("reduction"),
        "linalg_contract": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_linalg"),
        "layout_contract": metadata.get("layout_transfer"),
        "block_meta": dict(metadata.get("ssa_schedule", {})).get(
            "ascend_block_meta", {}
        ),
        "advanced_contract": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_advanced"),
        "dot_loop": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_dot_loop"),
        "attention_loop": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_loop"),
        "toolchain": {"cann_version": ascend_toolchain_version()},
    }
    path.write_text(json.dumps(_json_value(payload), sort_keys=True), encoding="utf-8")

    return path


def read_ascend_sidecar(source_path: Path) -> dict[str, Any]:
    """Load and validate the private Ascend AOT sidecar."""
    path = _sidecar_path(source_path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Ascend artifact sidecar does not exist: {path}."
        ) from exc

    if payload.get("schema") != _SIDECAR_SCHEMA:
        raise ValueError(f"Unsupported Ascend artifact sidecar schema in {path}.")

    return payload


def ascend_abi_from_dict(value: Mapping[str, Any]) -> LaunchABI:
    """Restore the public LaunchABI from a sidecar representation."""
    return LaunchABI(
        public_args=tuple(value.get("public_args", ())),
        kernel_args=tuple(
            LaunchBinding(**dict(binding)) for binding in value.get("kernel_args", ())
        ),
        outputs=tuple(value.get("outputs", ())),
        shape_params=tuple(value.get("shape_params", ())),
    )


def _sidecar_path(source_path: Path) -> Path:
    return source_path.with_suffix(".ascend-launch.json")


def _ascend_abi_dict(abi: LaunchABI) -> dict[str, Any]:
    return {
        "public_args": abi.public_args,
        "kernel_args": tuple(
            {
                "name": binding.name,
                "source": binding.source,
                "kind": binding.kind,
                "dim": binding.dim,
                "value": binding.value,
                "access": binding.access,
            }
            for binding in abi.kernel_args
        ),
        "outputs": abi.outputs,
        "shape_params": abi.shape_params,
    }


def _json_value(value: Any):
    """Convert immutable IR metadata into sidecar-safe JSON primitives."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}

    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise TypeError(
        "Ascend private artifact sidecar contains unsupported metadata value "
        f"of type `{type(value).__name__}`."
    )


def _is_index(expression: IndexExpr) -> bool:
    return expression.op == "symbol" and expression.value == "index"


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _recover_arrangement_layout_transfer(
    program: ssa.Program, context: Context
) -> ssa.Program:
    """Recover the public transfer contract when arrangement metadata masks it."""
    analysis = dict(program.metadata.get("analysis", {}))
    if analysis.get("layout_transfer") is not None:
        return program
    if not any(
        op.opcode == "linalg.transpose"
        for block in program.blocks
        for op in block.operations
    ):
        return program
    logical_specs = tuple(replace(spec, layout=None) for spec in context.tensors)
    transfer = analyze_layout_transfer(program, logical_specs)
    if transfer is None or not transfer.schedulable:
        return program
    return replace(
        program,
        metadata=dict(program.metadata)
        | {
            "analysis": analysis | {"layout_transfer": transfer},
            "schedule": dict(program.metadata.get("schedule", {}))
            | {"granularity": "layout-transfer", "layout_transfer": transfer},
        },
    )


def _attach_private_scan_schedule(program: ssa.Program) -> ssa.Program:
    """Select a private scan schedule for explicit scan SSA operations."""
    operations = tuple(_walk_operations(program.blocks))
    scans = tuple(
        op
        for op in operations
        if op.opcode in {"scan.cumsum", "scan.prefix_sum", "call.cumsum"}
    )
    if not scans:
        return program
    if len(scans) != 1 or not scans[0].results:
        raise ValueError(
            "Ascend prefix-scan currently requires one result-producing scan."
        )
    schedule = dict(program.metadata.get("schedule", {}))
    result_shape = tuple(scans[0].results[0].type.shape)
    extent = result_shape[0] if result_shape else "1"
    schedule.update(
        {
            "granularity": "scan",
            "scan": {"mode": "inclusive", "axis": 0, "extent": str(extent)},
        }
    )
    return replace(program, metadata=dict(program.metadata) | {"schedule": schedule})


def _ascend_advanced_contract(program: ssa.Program) -> Mapping[str, Any] | None:
    """Validate and describe advanced operations without changing public IR."""
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }
    operations = tuple(_walk_operations(program.blocks))
    for operation in operations:
        value_types.update({value.name: value.type for value in operation.results})

    rng = [op for op in operations if op.opcode == "math.rand"]
    atomic = [op for op in operations if op.opcode == "mem.atomic_add"]
    calls = [op for op in operations if op.opcode.startswith("call.")]
    block_dot = [op for op in calls if op.opcode == "call.block_dot"]

    if rng:
        for operation in rng:
            if len(operation.operands) != 2:
                raise ValueError(
                    "Ascend RNG requires exactly (seed, offset) operands; "
                    "unsupported RNG ABI is fail-closed."
                )
            result = operation.results[0] if operation.results else None
            dtype = (
                normalize_ascend_dtype(getattr(result.type, "dtype", None))
                if result
                else None
            )
            if dtype not in ASCEND_RNG_DTYPES:
                raise ValueError(
                    f"Ascend RNG does not support dtype `{dtype}`; supported dtypes "
                    "are float16, bfloat16, and float32."
                )

    if atomic:
        for operation in atomic:
            if len(operation.operands) < 2:
                raise ValueError(
                    "Ascend atomic_add requires destination and value operands."
                )
            dtype = normalize_ascend_dtype(
                getattr(value_types.get(operation.operands[-1]), "dtype", None)
            )
            if dtype not in ASCEND_ATOMIC_DTYPES:
                raise ValueError(
                    f"Ascend atomic_add does not support dtype `{dtype}`; "
                    "supported dtypes are float32 and int32."
                )

    for operation in calls:
        name = operation.opcode.removeprefix("call.")
        if name in {"flash_attention", "attention"}:
            raise ValueError(
                "Ascend attention requires the private block-dot contract; "
                "generic attention calls are unsupported."
            )
        if name == "conv2d":
            raise ValueError(
                "Ascend conv2d is expressed as a generic dot-loop; standalone "
                "call.conv2d is fail-closed."
            )
        if name in {"conv2d", "block_dot"} and not operation.results:
            raise ValueError(f"Ascend {name} lowering requires one result value.")

    if block_dot:
        for operation in block_dot:
            attrs = dict(operation.attrs)
            if attrs.get("causal", False) and attrs.get("axis", -1) not in {-1, 1}:
                raise ValueError(
                    "Ascend block-dot attention supports causal masking only on the "
                    "sequence axis."
                )
        return {
            "kind": "block-dot-attention",
            "causal": any(bool(op.attrs.get("causal", False)) for op in block_dot),
            "online_softmax": True,
            "count": len(block_dot),
        }

    if rng or atomic:
        return {
            "kind": "rng-atomic",
            "rng": bool(rng),
            "atomic": bool(atomic),
            "seed_offset_abi": "seed,offset" if rng else None,
        }
    return None


def _ascend_dot_loop_contract(program: ssa.Program) -> Mapping[str, Any] | None:
    """Recognize the generic dot-reduction loop emitted by the public frontend."""
    loops = [op for op in _walk_operations(program.blocks) if op.opcode == "scf.for"]
    matched = []
    for loop in loops:
        body = loop.regions[0].operations if loop.regions else ()
        opcodes = {op.opcode for op in body}
        if {
            "tensor.extract",
            "linalg.dot",
            "arith.add",
            "scf.yield",
        }.issubset(opcodes):
            matched.append(loop)
    if not matched:
        return None
    if len(matched) != 1:
        raise ValueError(
            "Ascend generic dot-loop lowering supports one reduction loop; "
            "nested or multiple dot loops are not yet scheduled."
        )
    return {
        "version": 1,
        "mode": "generic-dot-loop",
        "loop_carried": bool(dict(matched[0].attrs).get("iter_args")),
        "layout": "public-access-template",
        "tile": {"m": 16, "n": 16, "k": 64},
    }


def _ascend_attention_loop_contract(program: ssa.Program) -> Mapping[str, Any] | None:
    """Recognize public loop-carried online-softmax SSA without new opcodes."""
    loops = [
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    ]
    candidates = []
    for loop in loops:
        body = tuple(_walk_operations(loop.regions))
        opcodes = [operation.opcode for operation in body]
        if (
            opcodes.count("linalg.dot") == 2
            and opcodes.count("math.exp2") >= 2
            and "reduce.max" in opcodes
            and "reduce.sum" in opcodes
            and "scf.if" in opcodes
            and len(tuple(dict(loop.attrs).get("iter_args", ()))) == 3
        ):
            candidates.append(loop)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            "Ascend attention requires exactly one online-softmax loop; "
            "multiple candidate loops are fail-closed."
        )
    return {
        "version": 1,
        "mode": "generic-online-softmax-loop",
        "causal": "public-scf-if",
        "layout": "public-access-template",
        "status": "verified-static-public-online-softmax",
    }


class AscendOptimizeSchedule(OptimizeSchedule):
    """Select conservative schedules for the initial Ascend support tier."""

    name = "ssa.ascend.optimize_schedule"
    supported_backends = (Target.ASCEND,)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        """Reject SSA features whose Ascend semantics are not verified yet."""
        self._validate_options(context)
        program = _recover_arrangement_layout_transfer(program, context)
        program = _attach_private_scan_schedule(program)
        dot_loop = _ascend_dot_loop_contract(program)
        if dot_loop is not None:
            program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {"ascend_dot_loop": dot_loop}
                },
            )
        attention_loop = _ascend_attention_loop_contract(program)
        if attention_loop is not None:
            program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {"ascend_attention_loop": attention_loop}
                },
            )
        advanced = _ascend_advanced_contract(program)
        if advanced is not None:
            program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {"ascend_advanced": advanced}
                },
            )
        self._validate_supported_program(program, context)
        linalg = _ascend_linalg_contract(program)
        program = _bind_ascend_matmul_dimensions(program)

        # Keep Ascend-only analysis adjacent to the backend schedule.  The
        # shared pass registry intentionally has no target-injected analysis
        # hook.
        program = _attach_private_alias_contract(program, context)
        dot_loop_contract = dot_loop
        lowered = super().run(program, context)
        dot_loop = dot_loop_contract

        if linalg is None and dot_loop is None:
            return lowered

        schedule = dict(lowered.metadata.get("schedule", {}))
        if linalg is not None:
            schedule["ascend_linalg"] = linalg
        if dot_loop is not None:
            schedule["ascend_dot_loop"] = dot_loop

        return replace(
            lowered, metadata=dict(lowered.metadata) | {"schedule": schedule}
        )

    def schedule_candidates(
        self,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
        context: Context,
    ) -> tuple[ScheduleCandidate, ...]:
        granularity = schedule.get("granularity")
        profile = _ascend_profile(context)

        if granularity == "parallel-reduction":
            reduction = schedule.get("reduction", {})

            if reduction.get("mode") != "row-vector":
                return ()

            return (
                ScheduleCandidate(
                    name="ascend-row-reduction-256",
                    schedule={
                        "tile": {"elements": 256},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                    },
                    constraints={
                        "dtypes": tuple(sorted(ASCEND_ELEMENTWISE_DTYPES)),
                        "layout": "contiguous",
                        **_profile_constraints(profile),
                    },
                    tags=("reduction", "row-vector", "tail-safe"),
                ),
            )

        if granularity == "blocked-linalg" and analysis.get("has_dot"):
            return (
                ScheduleCandidate(
                    name="ascend-tiled-matmul-16x16x64",
                    schedule={
                        "tile": {"elements": 256},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                    },
                    constraints={
                        "dtypes": tuple(sorted(ASCEND_ELEMENTWISE_DTYPES)),
                        "layout": "contiguous",
                        **_profile_constraints(profile),
                    },
                    tags=("linalg", "matmul", "tiled", "batched", "tail-safe"),
                ),
            )

        if granularity == "layout-transfer":
            transfer = analysis.get("layout_transfer")

            if transfer is None or not transfer.schedulable:
                return ()

            return (
                ScheduleCandidate(
                    name="ascend-layout-transfer-16x16",
                    schedule={
                        "tile": {"elements": 256, "block_m": 16, "block_n": 16},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                        "ascend_block_meta": {"TILE_M": 16, "TILE_N": 16},
                    },
                    constraints={
                        "layout": "strided-non-overlapping",
                        **_profile_constraints(profile),
                    },
                    tags=("layout", "transpose", "private-sidecar-meta"),
                ),
            )

        if granularity == "scan":
            return (
                ScheduleCandidate(
                    name="ascend-prefix-scan-256",
                    schedule={
                        "tile": {"elements": 256},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                        "scan": {"mode": "inclusive", "axis": 0},
                    },
                    constraints={
                        "layout": "contiguous",
                        "rank": (1, 2, 3),
                        **_profile_constraints(profile),
                    },
                    tags=("scan", "prefix-scan", "tail-safe"),
                ),
            )

        if granularity == "exp-reduction-dot-region":
            return (
                ScheduleCandidate(
                    name="ascend-generic-online-softmax-loop",
                    schedule={
                        "tile": {"elements": 256},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                        "ascend_attention_loop": {
                            "mode": "generic-online-softmax-loop",
                            "status": "verified-static-public-online-softmax",
                        },
                    },
                    constraints={
                        "layout": "public-access-template",
                        **_profile_constraints(profile),
                    },
                    tags=("attention", "online-softmax", "generic-ssa"),
                ),
            )

        if granularity != "elementwise-grid":
            return ()

        max_core_dim = _max_core_dim(context)

        return (
            ScheduleCandidate(
                name="fp16-bf16-fp32-elementwise-256",
                schedule={
                    "tile": {"elements": 256},
                    "vector_width": 1,
                    "core_dim_limit": max_core_dim,
                },
                constraints={
                    "dtypes": tuple(sorted(ASCEND_ELEMENTWISE_DTYPES)),
                    "layout": "contiguous",
                    "max_core_dim": max_core_dim,
                    **_profile_constraints(profile),
                },
                tags=("default", "elementwise", "fp16", "bf16", "fp32"),
            ),
        )

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis, schedule

        return {}

    def _validate_options(self, context: Context) -> None:
        compiler_options = context.compiler_options

        for option in ("num_warps", "num_stages"):
            if compiler_options.get(option) is not None:
                raise ValueError(
                    f"Ascend backend does not support `{option}`; use Ascend "
                    "schedule candidates instead."
                )

        schedule_options = dict(_ascend_pass_options(context).get("schedule", {}))

        for option in ("num_warps", "num_stages"):
            if option in schedule_options:
                raise ValueError(
                    f"Ascend backend does not support schedule option `{option}`."
                )

    def _validate_supported_program(
        self, program: ssa.Program, context: Context
    ) -> None:
        analysis = dict(program.metadata.get("analysis", {}))
        granularity = str(analysis.get("granularity", ""))

        if not granularity:
            granularity = _granularity_for_analysis(analysis)

        if granularity not in {
            "elementwise-grid",
            "parallel-reduction",
            "blocked-linalg",
            "layout-transfer",
            "scan",
            "exp-reduction-dot-region",
        }:
            raise ValueError(
                "Ascend backend currently supports only FP16, BF16, and FP32 "
                "elementwise or row-vector reduction SSA; "
                f"received schedule granularity `{granularity}`."
            )

        if granularity == "blocked-linalg":
            _ascend_linalg_contract(program)

        if granularity == "parallel-reduction":
            reduction = analysis.get("reduction_schedule", {})

            if reduction.get("mode") != "row-vector":
                raise ValueError(
                    "Ascend backend currently supports only FP16, BF16, and FP32 "
                    "elementwise or row-vector reduction SSA; "
                    "received schedule granularity `parallel-reduction` with "
                    f"reduction mode `{reduction.get('mode')}`."
                )

        tensor_dtypes = tuple(
            tensor.dtype for tensor in context.tensors if not tensor.constexpr
        ) or tuple(
            value.type.dtype for value in program.inputs if value.type.kind == "tensor"
        )
        advanced = dict(program.metadata.get("schedule", {})).get("ascend_advanced", {})
        unsupported_dtypes = unsupported_ascend_elementwise_dtypes(
            tensor_dtypes,
            allow_rng_auxiliary=bool(advanced.get("rng")),
            allow_atomic=bool(advanced.get("atomic")),
            # A missing annotation is a runtime-specialization marker.  The
            # materializer still rejects concrete dtypes outside the verified
            # Ascend capability set after the public lazy path resolves it.
            allow_unspecified=True,
        )

        if unsupported_dtypes:
            names = ", ".join(unsupported_dtypes)
            raise ValueError(
                "Ascend backend currently supports only FP16, BF16, and FP32 "
                "elementwise SSA; "
                f"received tensor dtypes: {names}."
            )

        unsupported_layouts = [
            tensor.name for tensor in context.tensors if tensor.jagged_dim is not None
        ]

        if unsupported_layouts:
            names = ", ".join(unsupported_layouts)
            raise ValueError(
                "Ascend backend currently does not support jagged tensors; "
                f"unsupported tensors: {names}."
            )

        private_schedule = dict(program.metadata.get("schedule", {}))
        allow_rank4_dot_loop = (
            "ascend_dot_loop" in private_schedule
            or "ascend_attention_loop" in private_schedule
        )
        unsupported_views = [
            tensor.name
            for tensor in context.tensors
            if not tensor.constexpr
            and not _is_supported_logical_view(
                tensor,
                allow_rank4_dot_loop=allow_rank4_dot_loop,
            )
        ]

        if unsupported_views:
            names = ", ".join(unsupported_views)
            raise ValueError(
                "Ascend backend supports only rank 0 through 4 logical views; "
                "concrete stride, overlap, and storage-span admission occurs in the "
                f"Ascend materializer. Unsupported tensors: {names}."
            )


def _attach_private_alias_contract(
    program: ssa.Program, context: Context
) -> ssa.Program:
    """Attach metadata consumed only by the Ascend emitter/materializer."""
    views = {
        tensor.name: _logical_view(tensor)
        for tensor in context.tensors
        if not tensor.constexpr
    }
    outputs = {value.name for value in program.outputs}
    inputs = {value.name for value in program.inputs if value.type.kind == "tensor"}
    # The materializer rejects every writer/reader storage overlap.  Marking
    # declared outputs as writers here is deliberately conservative and avoids
    # a shared effect-analysis dependency.
    access_modes = {
        name: ("write" if name in outputs else "read") for name in inputs | outputs
    }
    return replace(
        program,
        metadata=dict(program.metadata)
        | {
            "ascend_alias_analysis": {
                "policy": "reject-storage-overlap",
                "access_modes": access_modes,
                "logical_views": views,
            }
        },
    )


def _ascend_pass_options(context: Context) -> Mapping[str, Any]:
    options: Mapping[str, Any] = {}

    for name in ("*", "ssa.optimize_schedule", "ssa.ascend.optimize_schedule"):
        options = dict(options) | dict(context.pass_options.get(name, {}))

    return options


def _max_core_dim(context: Context) -> int:
    backend_options = dict(context.compiler_options.get("backend_options", {}))
    max_core_dim = backend_options.get("max_core_dim", 65535)

    if isinstance(max_core_dim, bool) or not isinstance(max_core_dim, int):
        raise TypeError("Ascend backend option `max_core_dim` must be an integer.")

    if not 1 <= max_core_dim <= 65535:
        raise ValueError(
            "Ascend backend option `max_core_dim` must be between 1 and 65535."
        )

    return max_core_dim


def _ascend_profile(context: Context) -> AscendSocProfile:
    options = dict(context.compiler_options.get("backend_options", {}))
    return ascend_soc_profile(options.get("soc_version"))


def _profile_constraints(profile: AscendSocProfile) -> Mapping[str, Any]:
    return {
        "soc_version": profile.name,
        "cube_cores": profile.cube_cores,
        "vector_cores": profile.vector_cores,
        "l2_cache_bytes": profile.l2_cache_bytes,
        "ub_bytes": profile.ub_bytes,
        "l1_bytes": profile.l1_bytes,
        "l0a_bytes": profile.l0a_bytes,
        "l0b_bytes": profile.l0b_bytes,
        "l0c_bytes": profile.l0c_bytes,
    }


def _static_reduction_extent(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _ascend_linalg_contract(program: ssa.Program) -> Mapping[str, Any] | None:
    operations = tuple(_walk_operations(program.blocks))
    linalg = tuple(
        operation
        for operation in operations
        if operation.opcode in {"linalg.dot", "linalg.matmul"}
    )

    if not linalg:
        return None

    if _ascend_attention_loop_contract(program) is not None or (
        any(op.opcode in {"math.exp", "math.exp2"} for op in operations)
        and any(op.opcode == "scf.for" for op in operations)
    ):
        return None

    if len(linalg) == 1 and linalg[0].opcode == "linalg.dot":
        if _ascend_dot_loop_contract(program) is not None:
            return None
    if len(linalg) != 1 or linalg[0].opcode != "linalg.matmul":
        raise ValueError(
            "Ascend basic linalg support requires exactly one `linalg.matmul` "
            "operation."
        )

    operation = linalg[0]
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }

    for nested in operations:
        value_types.update({result.name: result.type for result in nested.results})

    if len(operation.operands) != 2 or len(operation.results) != 1:
        raise ValueError("Ascend matmul requires two inputs and one result.")

    lhs = value_types.get(operation.operands[0])
    rhs = value_types.get(operation.operands[1])
    result = operation.results[0].type

    if (
        lhs is None
        or rhs is None
        or lhs.kind != "tensor"
        or rhs.kind != "tensor"
        or result.kind != "tensor"
        or len(lhs.shape) not in {2, 3}
        or len(rhs.shape) != len(lhs.shape)
        or len(result.shape) != len(lhs.shape)
    ):
        raise ValueError(
            "Ascend matmul supports only rank-2 or rank-3 contiguous matrix operands and output."
        )

    rank = len(lhs.shape)
    if rank == 2:
        m, k = (str(value) for value in lhs.shape)
        rhs_k, n = (str(value) for value in rhs.shape)
        result_m, result_n = (str(value) for value in result.shape)
        batch = None
    else:
        batch, m, k = (str(value) for value in lhs.shape)
        rhs_batch, rhs_k, n = (str(value) for value in rhs.shape)
        result_batch, result_m, result_n = (str(value) for value in result.shape)
        if batch != rhs_batch or batch != result_batch:
            raise ValueError(
                "Ascend batched matmul requires matching batch dimensions."
            )

    if (m, n) != (result_m, result_n) or k != rhs_k:
        raise ValueError("Ascend matmul requires lhs[M,K] @ rhs[K,N] -> output[M,N].")

    return {
        "mode": "tiled-matmul",
        "lhs": operation.operands[0],
        "rhs": operation.operands[1],
        "output": program.outputs[0].name,
        "output_shape": (m, n) if rank == 2 else (batch, m, n),
        "reduction_extent": k,
        "rank": rank,
        "batch": batch,
        "tile": {"m": 16, "n": 16, "k": 64},
    }


def _walk_operations(blocks):
    for block in blocks:
        for operation in block.operations:
            yield operation
            yield from _walk_operations(operation.regions)


def _bind_ascend_matmul_dimensions(program: ssa.Program) -> ssa.Program:
    contract = _ascend_linalg_contract(program)

    if contract is None:
        return program

    output_shape = tuple(contract["output_shape"])
    dimensions = {
        "m": output_shape[-2],
        "n": output_shape[-1],
        "k": contract["reduction_extent"],
    }
    existing = {
        value.name for value in (*program.inputs, *program.outputs) for _ in (0,)
    }

    for operation in _walk_operations(program.blocks):
        existing.update(result.name for result in operation.results)

    bindings = {}
    constants = []

    for dimension, value in dimensions.items():
        if str(value).isidentifier():
            continue

        extent = _static_reduction_extent(value)

        if extent is None:
            raise ValueError(
                "Ascend matmul dimensions must be static integers or existing "
                "shape symbols."
            )

        name = f"ascend_matmul_{dimension}"

        if name in existing:
            raise ValueError(f"Ascend matmul reserved SSA value `{name}` is in use.")

        existing.add(name)
        bindings[dimension] = name
        constants.append(
            ssa.Operation(
                opcode="arith.constant",
                results=(ssa.Value(name=name, type=ssa.Type(kind="index")),),
                attrs={"value": extent, "ascend_linalg": True},
            )
        )

    if not constants:
        return program

    def rewrite(block):
        operations = []

        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)

            if operation.opcode == "linalg.matmul":
                attrs = dict(operation.attrs) | bindings
                operations.extend(constants)
                operations.append(
                    ssa.Operation(
                        opcode=operation.opcode,
                        operands=operation.operands,
                        results=operation.results,
                        attrs=attrs,
                        regions=regions,
                    )
                )
                continue

            operations.append(
                ssa.Operation(
                    opcode=operation.opcode,
                    operands=operation.operands,
                    results=operation.results,
                    attrs=operation.attrs,
                    regions=regions,
                )
            )

        return ssa.Block(name=block.name, args=block.args, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def _rewrite_ascend_batched_matmul_access(
    program: ssa.Program, contract: Mapping[str, Any]
) -> ssa.Program:
    """Restore rank-3 operands lost by generic scalar matmul decomposition.

    The shared decomposition intentionally represents a matrix result with two
    ``index.offset`` values.  For a batched public tile it therefore emits
    ``lhs[batch, k]`` and ``rhs[k, row]``.  This Ascend-only post-decomposition
    rewrite restores the public logical coordinates before source emission:
    ``lhs[batch, row, k]`` and ``rhs[batch, k, col]``.
    """
    if contract.get("rank") != 3:
        return program

    lhs = str(contract["lhs"])
    rhs = str(contract["rhs"])
    operations = tuple(_walk_operations(program.blocks))
    existing = {
        value.name for operation in operations for value in operation.results
    } | {value.name for value in (*program.inputs, *program.outputs)}
    col = "%ascend_matmul_col"

    if col in existing:
        raise ValueError(
            "Ascend matmul reserved SSA value `%ascend_matmul_col` is in use."
        )

    decomposed_offsets = tuple(
        operation
        for operation in operations
        if operation.opcode == "index.offset"
        and len(operation.operands) == 1
        and len(operation.results) == 1
        and operation.attrs.get("decomposition") == "matmul"
    )
    offset_outputs = {operation.operands[0] for operation in decomposed_offsets}

    if len(offset_outputs) != 1:
        raise ValueError(
            "Ascend rank-3 matmul rewrite requires one decomposed output access template."
        )

    output = offset_outputs.pop()
    offsets = {
        int(operation.attrs.get("dim", -1)): operation.results[0].name
        for operation in decomposed_offsets
    }
    batch = offsets.get(0)
    row = offsets.get(1)

    if batch is None or row is None:
        raise ValueError(
            "Ascend rank-3 matmul rewrite requires decomposed output batch and row offsets."
        )

    inserted = False

    def rewrite(block: ssa.Block) -> ssa.Block:
        nonlocal inserted
        rewritten = []

        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            attrs = dict(operation.attrs)
            operands = operation.operands

            if (
                operation.opcode == "index.offset"
                and operation.operands == (output,)
                and attrs.get("dim") == 1
                and attrs.get("decomposition") == "matmul"
                and not inserted
            ):
                rewritten.append(
                    ssa.Operation(
                        opcode=operation.opcode,
                        operands=operands,
                        results=operation.results,
                        attrs=attrs,
                        regions=regions,
                    )
                )
                rewritten.append(
                    ssa.Operation(
                        opcode="index.offset",
                        operands=(output,),
                        results=(ssa.Value(name=col, type=ssa.Type(kind="index")),),
                        attrs={
                            "dim": 2,
                            "decomposition": "matmul",
                            "ascend_access_template": "batch-row-col",
                        },
                    )
                )
                inserted = True
                continue

            if (
                operation.opcode == "tensor.extract"
                and attrs.get("decomposition") == "matmul"
                and len(operands) == 3
            ):
                operand_role = attrs.get("operand")
                induction = operands[-1] if operand_role == "lhs" else operands[1]

                if operand_role == "lhs" and operands[0] == lhs:
                    operands = (lhs, batch, row, induction)
                elif operand_role == "rhs" and operands[0] == rhs:
                    operands = (rhs, batch, induction, col)
                else:
                    raise ValueError(
                        "Ascend rank-3 matmul rewrite encountered an unexpected "
                        "decomposed tensor.extract operand."
                    )

                attrs["ascend_access_template"] = (
                    "batch-row-k" if operand_role == "lhs" else "batch-k-col"
                )

            rewritten.append(
                ssa.Operation(
                    opcode=operation.opcode,
                    operands=operands,
                    results=operation.results,
                    attrs=attrs,
                    regions=regions,
                )
            )

        return ssa.Block(name=block.name, args=block.args, operations=tuple(rewritten))

    rewritten = replace(
        program, blocks=tuple(rewrite(block) for block in program.blocks)
    )

    if not inserted:
        raise ValueError(
            "Ascend rank-3 matmul rewrite could not locate the decomposed output column offset."
        )

    schedule = dict(rewritten.metadata.get("schedule", {}))
    schedule["ascend_batched_access_rewrite"] = {
        "version": 1,
        "mode": "public-access-template",
        "coordinates": {"lhs": ("batch", "row", "k"), "rhs": ("batch", "k", "col")},
        "output": ("batch", "row", "col"),
    }
    metadata = dict(rewritten.metadata)
    metadata["schedule"] = schedule
    metadata["pass_trace"] = tuple(metadata.get("pass_trace", ())) + (
        "ssa.ascend.rewrite_batched_matmul_access",
    )

    return replace(rewritten, metadata=metadata)


def _granularity_for_analysis(analysis: Mapping[str, Any]) -> str:
    if analysis.get("layout_transfer") is not None:
        return "layout-transfer"

    if analysis.get("has_exp_reduction_dot_pattern"):
        return "exp-reduction-dot-region"

    if analysis.get("has_dot"):
        return "blocked-linalg"

    if analysis.get("reduction_count"):
        return "parallel-reduction"

    return "elementwise-grid"


def _logical_view(tensor) -> Mapping[str, str]:
    attrs = dict(tensor.attrs)

    if tensor.ndim == 0:
        return {"domain": "1", "offset": "0", "mask": "True"}

    dimensions = (
        tuple(value.render() for value in tensor.layout.application_shape)
        if tensor.layout and tensor.layout.application_shape
        else tuple(IndexExpr.parse(value).render() for value in tensor.shape)
    )
    offsets = tuple(str(value) for value in attrs.get("view_offsets", ()))

    return {
        "domain": " * ".join(f"({value})" for value in dimensions) or "1",
        "offset": offsets[0] if len(offsets) == 1 else "0",
        "mask": str(attrs.get("view_mask", True)),
    }


def _is_supported_logical_view(tensor, *, allow_rank4_dot_loop: bool = False) -> bool:
    # Recognized dot-loop contracts consume the public access template rather
    # than the ordinary rank-preserving logical-view path.
    if allow_rank4_dot_loop:
        return True

    if tensor.ndim not in {0, 1, 2, 3, 4}:
        return False

    if tensor.layout is None:
        return True

    return len(tensor.layout.application_shape) == tensor.ndim


def register_ssa_passes(registry: "Registry") -> None:
    """Register the Ascend schedule pass."""
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.ASCEND,
        optimize_schedule=AscendOptimizeSchedule,
    )
