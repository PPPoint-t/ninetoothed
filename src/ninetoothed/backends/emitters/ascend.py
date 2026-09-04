"""Ascend Triton syntax hooks for the initial SSA emitter tier."""

import re
from dataclasses import dataclass, replace
from typing import Iterable

from ninetoothed.backends.ascend import (
    ASCEND_ELEMENTWISE_DTYPES,
    UnsupportedBackendOpError,
    normalize_ascend_dtype,
    unsupported_ascend_elementwise_dtypes,
)
from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops
from ninetoothed.backends.emitters.base import ModuleRenderContext
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.ir import Kernel, ssa

_view_base_coords = common.view_base_coords
_linearized_index = common.linearized_index
_source_index_for_value = common.source_index_for_value

_ASCEND_MATH_INTRINSICS = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "cosh",
        "erf",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "log",
        "log1p",
        "log2",
        "log10",
        "maximum",
        "minimum",
        "pow",
        "rand",
        "rsqrt",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)

_ASCEND_FP32_UNARY_INTRINSICS = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "cos",
        "cosh",
        "erf",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "rsqrt",
        "sin",
        "sinh",
        "sqrt",
        "tanh",
    }
)

_GENERIC_SSA_PREFIXES = (
    "arith.",
    "cmp.",
    "index.",
    "shape.",
    "tensor.",
    "mem.",
    "reduce.",
    "scf.",
    "linalg.",
    "symbol.",
    "tuple.",
    "call.",
)


@dataclass(frozen=True, kw_only=True)
class AscendTarget(TritonTarget):
    """Triton Python source accepted by the installed Ascend toolchain."""

    backend: Target = Target.ASCEND
    suffix: str = "ascend.py"
    source_route: str = "ssa-unified-ascend-triton-emitter"
    default_load_mask: bool = False

    def cast(self, dtype: str, value: str) -> str:
        return f"{value}.to(tl.{_triton_dtype(dtype)})"

    def load(self, tensor, index, *, mask=None, other=0.0):
        if mask is None and self.default_load_mask:
            mask = "mask"

        if mask is not None and ("padding_" in index or "padding_" in str(mask)):
            # Triton-Ascend must not receive a negative physical pointer even
            # for a masked lane.  Padding views express invalid logical window
            # positions with the same predicate used by the load, so clamp only
            # those lanes to the tensor base and retain ``other`` as their value.
            # This is the target spelling of the public access-template contract;
            # it does not change its coordinates or introduce a copied im2col.
            safe_index = f"tl.where(({mask}), ({index}), 0)"
            return super().load(tensor, safe_index, mask=f"({mask})", other=other)

        return super().load(tensor, index, mask=mask, other=other)

    def call(self, name, args):
        if name in {"cumsum", "prefix_sum", "scan.cumsum", "scan.prefix_sum"}:
            if len(args) != 1:
                raise ValueError("Ascend scan lowering requires one input value.")
            return f"tl.cumsum({args[0]})"
        if name == "rand":
            if len(args) != 2:
                raise ValueError(
                    "Ascend RNG lowering requires seed and offset operands."
                )
            return f"tl.rand({args[0]}, {args[1]})"

        if name in _ASCEND_FP32_UNARY_INTRINSICS and args:
            # Ascend Vector math is most stable when low-precision operands are
            # promoted in UB.  The result is still written through the public
            # tensor dtype at the store boundary.
            args = tuple(self._upcast_fp32(arg) for arg in args)

        return super().call(name, args)

    @staticmethod
    def _upcast_fp32(value: str) -> str:
        text = str(value)
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
            return text
        if ".to(tl.float32)" in text:
            return text
        return f"({text}).to(tl.float32)"

    def coerce_binary_args(self, operation, args, context):
        if operation.opcode.startswith(("arith.", "math.")):
            low_precision = any(
                getattr(context.value_types.get(operand), "dtype", None)
                in {"float16", "bfloat16", "fp16", "bf16"}
                for operand in operation.operands
            )
            if low_precision:
                return tuple(
                    self._upcast_fp32(arg)
                    if getattr(context.value_types.get(operand), "dtype", None)
                    in {"float16", "bfloat16", "fp16", "bf16"}
                    else arg
                    for operand, arg in zip(operation.operands, args)
                )
        return args

    def atomic_add(self, operands: tuple[str, ...], dtype: str) -> str:
        """Broadcast a scalar destination pointer across the current block."""
        del dtype
        if len(operands) != 2:
            raise ValueError(
                "Ascend atomic_add requires destination and value operands."
            )

        return f"tl.atomic_add({operands[0]} + (index * 0), {operands[1]})"

    def render_module(self, context: ModuleRenderContext) -> str:
        if context.block_program or context.vector_program or context.scalar_program:
            return super().render_module(context)

        kernel = context.kernel
        body = common.rewrite_index_math(context.body, c_style=False)
        runtime_params = set(kernel.metadata.get("runtime_shape_params", ()))
        params = ",\n    ".join(
            (
                *context.variables,
                *context.outputs,
                *[
                    axis if axis in runtime_params else f"{axis}: tl.constexpr"
                    for axis in context.shape_params
                ],
                "BLOCK: tl.constexpr",
            )
        )
        kernel_args = ",\n        ".join(
            (*context.variables, *context.outputs, *context.shape_params)
        )
        total = context.total
        mask_total = total
        launch_total = total
        launch_grid = f"triton.cdiv({launch_total}, block)"
        offsets_expression = "tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)"
        schedule = kernel.ssa.metadata.get("schedule", {})
        reduction = schedule.get("reduction", {})
        scalar_output = any(
            tensor.name in context.outputs and tensor.ndim == 0
            for tensor in kernel.tensors
        )

        if scalar_output:
            total = "1"
            mask_total = total
            launch_total = total
            launch_grid = f"triton.cdiv({launch_total}, block)"

        if (
            schedule.get("granularity") == "parallel-reduction"
            and reduction.get("mode") == "row-vector"
        ):
            mask_total = str(reduction.get("extent", total))
            launch_total = context.grid_total
            launch_grid = launch_total
            offsets_expression = "tl.arange(0, BLOCK)"

        result = _output_result(context.outputs)

        return f"""\"\"\"Ascend Triton lowering generated by NineToothed from ssa.Program.

Kernel: {kernel.kernel_name}
Lowering IR: ssa.Program
\"\"\"

import triton
import triton.language as tl
from triton.language.extra.cann import libdevice


@triton.jit
def {kernel.kernel_name}_kernel(
    {params},
):
    offsets = {offsets_expression}
    {self.index_name} = offsets
    mask = offsets < ({mask_total})
{common.indent_block(body, "    ")}

def launch_{kernel.kernel_name}({", ".join((*context.variables, *context.outputs, *context.shape_params))}):
    block = 256
    grid = ({launch_grid},)
    {kernel.kernel_name}_kernel[grid](
        {kernel_args},
        BLOCK=block,
    )
    return {result}
"""


TARGET = AscendTarget()


def emit(kernel: Kernel):
    """Emit only the explicitly verified first Ascend operation tier."""
    _validate_program(kernel)
    schedule = kernel.ssa.metadata.get("schedule", {})
    target = (
        replace(TARGET, default_load_mask=True)
        if schedule.get("granularity") == "blocked-linalg"
        else TARGET
    )

    kernel = _rewrite_private_scan_ops(kernel)
    artifact = common.emit(kernel, target)

    # TritonTarget.schedule_context consumes compiler/layout.py's LayoutTransfer
    # access maps. Keep the public coordinate helpers visible at this boundary so
    # Ascend-specific rewrites never need a duplicate flat-index implementation.
    if _layout_transfer_present(kernel):
        _validate_layout_transfer_surface(kernel)

    artifact = _rewrite_stride_predicates(artifact)
    artifact = _rewrite_unary_positive(artifact)

    return _rewrite_singleton_broadcast_loads(artifact, kernel)


def _rewrite_private_scan_ops(kernel: Kernel) -> Kernel:
    """Route Ascend scan dialect spellings through generic call emission."""
    if kernel.ssa is None:
        return kernel

    def rewrite(block):
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            opcode = operation.opcode
            if opcode in {"scan.cumsum", "scan.prefix_sum"}:
                opcode = "call.cumsum"
            operations.append(
                ssa.Operation(
                    opcode=opcode,
                    operands=operation.operands,
                    results=operation.results,
                    attrs=operation.attrs,
                    regions=regions,
                )
            )
        return ssa.Block(name=block.name, args=block.args, operations=tuple(operations))

    return replace(
        kernel,
        ssa=replace(
            kernel.ssa, blocks=tuple(rewrite(block) for block in kernel.ssa.blocks)
        ),
    )


def _layout_transfer_present(kernel: Kernel) -> bool:
    schedule = kernel.ssa.metadata.get("schedule", {}) if kernel.ssa else {}
    return (
        schedule.get("granularity") == "layout-transfer"
        and schedule.get("layout_transfer") is not None
    )


def _validate_layout_transfer_surface(kernel: Kernel) -> None:
    transfer = kernel.ssa.metadata["schedule"].get("layout_transfer")
    if not hasattr(transfer, "source") or not hasattr(transfer.source, "access_map"):
        raise ValueError(
            "Ascend layout-transfer emission requires the public LayoutTransfer "
            "coordinate maps."
        )


def _rewrite_stride_predicates(artifact):
    """Translate shared stride guards into Triton-Ascend legal predicates."""
    lines = []

    for line in artifact.primary_source.splitlines(keepends=True):
        if (
            line.startswith("    if ")
            and "_stride_" in line
            and " and " in line
            and line.rstrip().endswith(":")
        ):
            condition = line.removeprefix("    if ").rstrip()[:-1]
            line = (
                "    if "
                + " & ".join(f"({term})" for term in condition.split(" and "))
                + ":\n"
            )
        lines.append(line)

    source = "".join(lines)

    if source == artifact.primary_source:
        return artifact

    return replace(artifact, sources={artifact.primary_source_name: source})


def _rewrite_unary_positive(artifact):
    """Avoid an unnecessary unary plus rejected by the Ascend Triton frontend."""
    source = re.sub(r"\(\+tl\.load\(", "(tl.load(", artifact.primary_source)

    if source == artifact.primary_source:
        return artifact

    return replace(artifact, sources={artifact.primary_source_name: source})


def _rewrite_singleton_broadcast_loads(artifact, kernel: Kernel):
    """Keep the verified rank-1 singleton broadcast contract target-private."""
    source = artifact.primary_source

    for tensor in kernel.tensors:
        if tensor.constexpr or tensor.ndim != 1 or tuple(tensor.shape) != ("1",):
            continue

        source = re.sub(
            rf"tl\.load\({re.escape(tensor.name)} \+ .*?, other=0\.0\)",
            f"tl.load({tensor.name} + 0)",
            source,
        )

    if source == artifact.primary_source:
        return artifact

    return replace(artifact, sources={artifact.primary_source_name: source})


def diagnose_opcode_coverage(kernel: Kernel) -> dict[str, tuple[str, ...]]:
    """Return development-only SSA opcode coverage details for an Ascend kernel."""
    if kernel.ssa is None:
        raise ValueError("Ascend opcode diagnostics require ssa.Program.")

    observed = tuple(sorted({op.opcode for op in _operations(kernel.ssa)}))

    return {
        "observed": observed,
        "supported": tuple(
            opcode for opcode in observed if _is_admitted_opcode(opcode)
        ),
        "unsupported": tuple(
            opcode for opcode in observed if not _is_admitted_opcode(opcode)
        ),
    }


def _output_result(outputs: tuple[str, ...]) -> str:
    if not outputs:
        return "None"

    if len(outputs) == 1:
        return outputs[0]

    return f"({', '.join(outputs)})"


def _validate_program(kernel: Kernel) -> None:
    if kernel.ssa is None:
        raise ValueError("Ascend source emission requires ssa.Program.")

    unsupported = sorted(
        {
            op.opcode
            for op in _operations(kernel.ssa)
            if not _is_admitted_opcode(op.opcode)
        }
    )

    if unsupported:
        names = ", ".join(f"`{opcode}`" for opcode in unsupported)
        raise ValueError(
            "Ascend emitter does not implement the required Triton-Ascend "
            f"intrinsic or SSA family: {names}."
        )

    advanced = dict(kernel.ssa.metadata.get("schedule", {})).get("ascend_advanced", {})
    unsupported_dtypes = unsupported_ascend_elementwise_dtypes(
        tuple(tensor.dtype for tensor in kernel.tensors if not tensor.constexpr),
        allow_rng_auxiliary=bool(advanced.get("rng")),
        allow_atomic=bool(advanced.get("atomic")),
        allow_unspecified=True,
    )

    if unsupported_dtypes:
        raise ValueError(
            "Ascend emitter supports only verified FP16, BF16, and FP32 elementwise "
            "dtypes; received tensor dtypes: "
            f"{', '.join(unsupported_dtypes)}."
        )

    schedule = kernel.ssa.metadata.get("schedule", {})

    reduction = schedule.get("reduction", {})
    axis = reduction.get("axis")
    if isinstance(axis, (tuple, list)) or (isinstance(axis, str) and "," in axis):
        raise UnsupportedBackendOpError(
            "Ascend reduction does not support multi-axis reduction.",
            reason="the verified Ascend reduction schedule has one-axis row-vector semantics and no multi-axis workspace contract.",
            suggestion="reduce one axis at a time or lower the operation to a supported single-axis schedule.",
        )

    if schedule.get("granularity") not in {
        "elementwise-grid",
        "parallel-reduction",
        "layout-transfer",
        "blocked-linalg",
        "scan",
        "exp-reduction-dot-region",
    }:
        raise ValueError(
            "Ascend emitter requires a verified Ascend schedule; received "
            f"granularity `{schedule.get('granularity')}`."
        )

    if schedule.get("granularity") == "parallel-reduction":
        reduction = schedule.get("reduction", {})

        if reduction.get("mode") != "row-vector":
            raise ValueError(
                "Ascend emitter requires the verified row-vector reduction "
                f"contract; received mode `{reduction.get('mode')}`."
            )

    if schedule.get("granularity") == "blocked-linalg":
        tile = schedule.get("tile", {})
        for field, alignment in (("block_m", 16), ("block_n", 16), ("block_k", 16)):
            value = tile.get(field)
            if value is not None:
                try:
                    aligned = int(value) % alignment == 0
                except (TypeError, ValueError):
                    aligned = False
                if not aligned:
                    raise UnsupportedBackendOpError(
                        f"Ascend tile `{field}`={value!r} is not aligned.",
                        reason=f"Ascend Cube tile dimension `{field}` must be a multiple of {alignment}.",
                        suggestion=f"choose a {alignment}-aligned tile or let the Ascend schedule select the verified 16x16x64 tile.",
                    )

    if schedule.get("granularity") == "scan":
        scan = schedule.get("scan", {})
        if scan.get("mode") != "inclusive" or scan.get("axis", 0) != 0:
            raise ValueError(
                "Ascend emitter supports only inclusive axis-0 prefix scans."
            )
        try:
            extent = int(str(scan.get("extent")))
        except (TypeError, ValueError):
            extent = None
        if extent is None or not 0 <= extent <= 256:
            raise ValueError(
                "Ascend emitter supports prefix scans only when the axis extent "
                "is statically bounded by BLOCK=256; cross-block continuation is "
                "not implemented."
            )

    if schedule.get("granularity") == "blocked-linalg":
        linalg = schedule.get("ascend_linalg", {})
        dot_loop = schedule.get("ascend_dot_loop", {})

        if linalg.get("mode") == "tiled-matmul":
            return
        if dot_loop.get("mode") == "generic-dot-loop":
            return
        if linalg.get("mode") != "tiled-matmul":
            raise ValueError(
                "Ascend emitter requires the verified matrix-scalar-loop linalg "
                "contract."
            )

    if (
        schedule.get("granularity") != "layout-transfer"
        and schedule.get("tile", {}).get("elements") != 256
    ):
        raise ValueError(
            "Ascend emitter requires the `fp16-bf16-fp32-elementwise-256` "
            "schedule candidate."
        )


def _operations(program: ssa.Program) -> Iterable[ssa.Operation]:
    for block in program.blocks:
        yield from walk_ops(block.operations)


def _is_admitted_opcode(opcode: str) -> bool:
    if opcode == "select.where":
        return True

    if opcode.startswith("math."):
        return opcode.removeprefix("math.") in _ASCEND_MATH_INTRINSICS

    if opcode in {"scan.cumsum", "scan.prefix_sum", "call.cumsum"}:
        return True

    return opcode.startswith(_GENERIC_SSA_PREFIXES)


def _triton_dtype(dtype: str) -> str:
    normalized = normalize_ascend_dtype(dtype)

    if normalized not in ASCEND_ELEMENTWISE_DTYPES:
        raise ValueError(f"Unsupported Ascend Triton dtype: {dtype!r}.")

    return normalized


__all__ = ["AscendTarget", "TARGET", "diagnose_opcode_coverage", "emit"]
