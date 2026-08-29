"""Ascend backend registration and SSA scheduling contracts."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import (
    Artifact,
    Backend,
    Capability,
    Target,
)
from ninetoothed.backends.emitters.ascend import emit
from ninetoothed.compiler.ascend_contracts import (
    ASCEND_ELEMENTWISE_DTYPES,
    is_static_forward_view_offset,
    unsupported_ascend_elementwise_dtypes,
)
from ninetoothed.compiler.effects import tensor_access_modes
from ninetoothed.compiler.passes import (
    BACKEND_SPECIFIC,
    Context,
    OptimizeSchedule,
    Pass,
    ScheduleCandidate,
)
from ninetoothed.ir import IndexExpr, Kernel, ssa

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
        return emit(kernel)


class AscendOptimizeSchedule(OptimizeSchedule):
    """Select conservative schedules for the initial Ascend support tier."""

    name = "ssa.ascend.optimize_schedule"
    supported_backends = (Target.ASCEND,)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        """Reject SSA features whose Ascend semantics are not verified yet."""
        self._validate_options(context)
        self._validate_supported_program(program, context)
        program = _bind_ascend_matmul_dimensions(program)

        lowered = super().run(program, context)
        linalg = _ascend_linalg_contract(program)

        if linalg is None:
            return lowered

        schedule = dict(lowered.metadata.get("schedule", {})) | {
            "ascend_linalg": linalg
        }

        return replace(lowered, metadata=dict(lowered.metadata) | {"schedule": schedule})

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


class AscendAnalyzeAlias(Pass):
    """Record the conservative runtime alias contract for Ascend launches."""

    name = "ssa.ascend.analyze_alias"
    category = BACKEND_SPECIFIC
    phase = "analysis"
    supported_backends = (Target.ASCEND,)

    def run(self, program: ssa.Program, context: Context) -> ssa.Program:
        views = {
            tensor.name: _logical_view(tensor)
            for tensor in context.tensors
            if not tensor.constexpr
        }
        scalar_outputs = {
            value.name
            for value in program.outputs
            if any(
                tensor.name == value.name and not tensor.constexpr and tensor.ndim == 0
                for tensor in context.tensors
            )
        }
        access_modes = tensor_access_modes(
            program, pointer_names=frozenset(scalar_outputs)
        )

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
        raise ValueError(
            "Ascend matmul requires lhs[M,K] @ rhs[K,N] -> output[M,N]."
        )

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
        value.name
        for value in (*program.inputs, *program.outputs)
        for _ in (0,)
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
        analysis_passes=(AscendAnalyzeAlias,),
    )
