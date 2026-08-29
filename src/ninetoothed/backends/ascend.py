"""Ascend backend policy, contracts, and private launch metadata."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.compiler.passes import (
    Context,
    OptimizeSchedule,
    ScheduleCandidate,
)
from ninetoothed.ir import IndexExpr, Kernel, LaunchABI, LaunchBinding, ssa

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


ASCEND_ELEMENTWISE_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_SIDECAR_SCHEMA = 1


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
) -> tuple[str, ...]:
    """Return dtype names outside the verified Ascend capability tier."""
    return tuple(
        sorted(
            {
                "unspecified" if dtype is None else normalize_ascend_dtype(dtype)
                for dtype in dtypes
                if normalize_ascend_dtype(dtype) not in ASCEND_ELEMENTWISE_DTYPES
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


def ascend_cache_key(base_key: str, metadata: Mapping[str, Any]) -> str:
    """Namespace source cache entries by the selected Ascend runtime target."""
    identity = {
        "base": base_key,
        "soc_version": dict(metadata.get("ssa_schedule", {})).get("soc_version", ""),
        "triton_ascend_arch": os.environ.get("TRITON_ASCEND_ARCH", ""),
        "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES", ""),
    }

    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def private_launch_abi(abi: LaunchABI, specs, outputs: tuple[str, ...]) -> LaunchABI:
    """Adapt the public ABI to Ascend's pointer-output overlap contract."""
    spec_by_name = {spec.name: spec for spec in specs}
    bindings = []

    for binding in abi.kernel_args:
        kind = binding.kind

        if (
            binding.source in outputs
            and getattr(spec_by_name.get(binding.source), "ndim", None) == 0
            and kind == "scalar"
        ):
            kind = "tensor"

        access = (
            "write"
            if binding.source in outputs
            else "read"
            if kind in {"tensor", "jagged_values"}
            else binding.access
        )
        bindings.append(
            LaunchBinding(
                name=binding.name,
                source=binding.source,
                kind=kind,
                dim=binding.dim,
                value=binding.value,
                access=access,
            )
        )

    return LaunchABI(
        public_args=abi.public_args,
        kernel_args=tuple(bindings),
        outputs=abi.outputs,
        shape_params=abi.shape_params,
    )


def ascend_logical_domain(specs, outputs: tuple[str, ...]) -> str:
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

    if len(dimensions) != len(source_shape):
        raise ValueError(
            "Ascend logical-domain requires matching output and source ranks."
        )

    offsets = tuple(output.attrs.get("view_offsets", ()))
    offset = static_forward_view_offset(offsets[0]) if len(offsets) == 1 else 0

    if len(offsets) > 1 and any(
        static_forward_view_offset(value) != 0 for value in offsets
    ):
        raise ValueError("Ascend multidimensional logical views require zero offsets.")

    if len(dimensions) == 1:
        return f"min({IndexExpr.parse(dimensions[0]).render()}, ({source_shape[0]} - {offset}))"

    def product(values) -> str:
        return (
            " * ".join(f"({IndexExpr.parse(value).render()})" for value in values)
            or "1"
        )

    return f"min({product(dimensions)}, {product(source_shape)})"


def write_ascend_sidecar(
    source_path: Path,
    *,
    abi: LaunchABI,
    specs,
    outputs,
    metadata: Mapping[str, Any],
) -> Path:
    """Persist private launch data without extending the common artifact schema."""
    path = _sidecar_path(source_path)
    payload = {
        "schema": _SIDECAR_SCHEMA,
        "abi": _ascend_abi_dict(abi),
        "logical_domain": ascend_logical_domain(specs, tuple(outputs)),
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
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

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
    """Restore a private launch ABI from its sidecar representation."""
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


def _is_index(expression: IndexExpr) -> bool:
    return expression.op == "symbol" and expression.value == "index"


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


class AscendOptimizeSchedule(OptimizeSchedule):
    """Select conservative schedules for the initial Ascend support tier."""

    name = "ssa.ascend.optimize_schedule"
    supported_backends = (Target.ASCEND,)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        """Reject SSA features whose Ascend semantics are not verified yet."""
        self._validate_options(context)
        self._validate_supported_program(program, context)
        program = _bind_ascend_matmul_dimensions(program)

        # Keep Ascend-only analysis adjacent to the backend schedule.  The
        # shared pass registry intentionally has no target-injected analysis
        # hook.
        program = _attach_private_alias_contract(program, context)
        lowered = super().run(program, context)
        linalg = _ascend_linalg_contract(program)

        if linalg is None:
            return lowered

        schedule = dict(lowered.metadata.get("schedule", {})) | {
            "ascend_linalg": linalg
        }

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

        if granularity == "parallel-reduction":
            reduction = schedule.get("reduction", {})

            if reduction.get("mode") != "row-vector":
                return ()

            extent = _static_reduction_extent(reduction.get("extent"))

            if extent is not None and not 0 <= extent <= 256:
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
                    },
                    tags=("reduction", "row-vector", "tail-safe"),
                ),
            )

        if granularity == "blocked-linalg" and analysis.get("has_dot"):
            return (
                ScheduleCandidate(
                    name="ascend-matmul-scalar-loop-256",
                    schedule={
                        "tile": {"elements": 256},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                    },
                    constraints={
                        "dtypes": tuple(sorted(ASCEND_ELEMENTWISE_DTYPES)),
                        "layout": "contiguous",
                    },
                    tags=("linalg", "matmul", "scalar-loop", "tail-safe"),
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

            extent = _static_reduction_extent(reduction.get("extent"))

            if extent is not None and not 0 <= extent <= 256:
                raise ValueError(
                    "Ascend row-vector reduction extent "
                    f"{extent} exceeds BLOCK=256; hierarchical partial reduction "
                    "is not implemented."
                )

        tensor_dtypes = tuple(
            tensor.dtype for tensor in context.tensors if not tensor.constexpr
        ) or tuple(
            value.type.dtype for value in program.inputs if value.type.kind == "tensor"
        )
        unsupported_dtypes = unsupported_ascend_elementwise_dtypes(tensor_dtypes)

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

        unsupported_views = [
            tensor.name
            for tensor in context.tensors
            if not tensor.constexpr and not _is_supported_logical_view(tensor)
        ]

        if unsupported_views:
            names = ", ".join(unsupported_views)
            raise ValueError(
                "Ascend backend supports only contiguous base tensors, one-dimensional "
                "static forward views, and zero-offset multidimensional views; "
                "unsupported tensors: "
                f"{names}."
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
        or len(lhs.shape) != 2
        or len(rhs.shape) != 2
        or len(result.shape) != 2
    ):
        raise ValueError(
            "Ascend matmul supports only rank-2 contiguous matrix operands and output."
        )

    m, k = (str(value) for value in lhs.shape)
    rhs_k, n = (str(value) for value in rhs.shape)
    result_m, result_n = (str(value) for value in result.shape)

    if (m, n) != (result_m, result_n) or k != rhs_k:
        raise ValueError("Ascend matmul requires lhs[M,K] @ rhs[K,N] -> output[M,N].")

    extent = _static_reduction_extent(k)

    if extent is not None and not 0 <= extent <= 256:
        raise ValueError(
            "Ascend matmul reduction extent "
            f"{extent} exceeds BLOCK=256; hierarchical partial reduction is not "
            "implemented."
        )

    return {
        "mode": "matrix-scalar-loop",
        "lhs": operation.operands[0],
        "rhs": operation.operands[1],
        "output_shape": (m, n),
        "reduction_extent": k,
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

    dimensions = {
        "m": contract["output_shape"][0],
        "n": contract["output_shape"][1],
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


def _is_supported_logical_view(tensor) -> bool:
    if tensor.ndim == 0:
        return True

    if tensor.layout is None:
        return True

    rank = len(tensor.layout.application_shape)

    if rank != tensor.ndim or rank == 0:
        return False

    attrs = dict(tensor.attrs)
    offsets = tuple(attrs.get("view_offsets", ()))

    if not offsets:
        return True

    if len(offsets) != rank:
        return False

    if rank == 1:
        return is_static_forward_view_offset(offsets[0])

    return all(
        is_static_forward_view_offset(offset) and str(offset) == "0"
        for offset in offsets
    )


def register_ssa_passes(registry: "Registry") -> None:
    """Register the Ascend schedule pass."""
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.ASCEND,
        optimize_schedule=AscendOptimizeSchedule,
    )
