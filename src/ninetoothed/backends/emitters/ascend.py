"""Ascend Triton syntax hooks for the initial SSA emitter tier."""

import ast
import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from ninetoothed.backends.ascend import (
    ASCEND_ELEMENTWISE_DTYPES,
    ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE,
    UnsupportedBackendOpError,
    ascend_attention_source_contract,
    _validate_ascend_attention_emission_contract,
    _verify_ascend_batched_matmul_contract,
    _verify_ascend_decomposed_matmul_contract,
    normalize_ascend_dtype,
    unsupported_ascend_elementwise_dtypes,
)
from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.analysis import walk_ops
from ninetoothed.backends.emitters.base import ModuleRenderContext, StoreAddressPlan
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.ir import Kernel, ssa

_access_template = common.access_template
_access_axes = common.access_axes
_combined_mask = common.combined_mask
_current_coords = common.current_coords
_dtype_level = common.dtype_level
_emit_element = common.emit_element
_emit_index_value = common.emit_index_value
_indent_lines = common.indent_lines
_load_other = common.load_other
_local_symbol = common.local_symbol
_product = common.product
_source_index_for_value = common.source_index_for_value
_target_index_expr = common.target_index_expr
_value_axes = common.value_axes


class _Conv2dIndexSimplifier(ast.NodeTransformer):
    """Fold only algebraic identities in generated Conv2d pointer math."""

    def visit_BinOp(self, node):  # noqa: N802 - ast visitor API
        node = self.generic_visit(node)
        left, right = node.left, node.right
        if isinstance(right, ast.Constant) and right.value == 0:
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left
            if isinstance(node.op, ast.Mult):
                return ast.Constant(value=0)
        if isinstance(right, ast.Constant) and right.value == 1:
            if isinstance(node.op, (ast.Mult, ast.FloorDiv)):
                return left
            if isinstance(node.op, ast.Mod):
                return ast.Constant(value=0)
        if isinstance(left, ast.Constant) and left.value == 0:
            if isinstance(node.op, ast.Add):
                return right
            if isinstance(node.op, ast.Mult):
                return ast.Constant(value=0)
        if isinstance(left, ast.Constant) and left.value == 1:
            if isinstance(node.op, ast.Mult):
                return right
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            try:
                if isinstance(node.op, ast.Add):
                    return ast.Constant(value=left.value + right.value)
                if isinstance(node.op, ast.Sub):
                    return ast.Constant(value=left.value - right.value)
                if isinstance(node.op, ast.Mult):
                    return ast.Constant(value=left.value * right.value)
                if isinstance(node.op, ast.FloorDiv):
                    return ast.Constant(value=left.value // right.value)
                if isinstance(node.op, ast.Mod):
                    return ast.Constant(value=left.value % right.value)
            except (ArithmeticError, TypeError):
                pass
        return node


def _simplify_conv2d_index_expr(expression: str) -> str:
    """Remove broadcast-neutral terms before Triton infers matrix layouts.

    Access templates intentionally retain generic rank-4 provenance.  Their
    algebra is correct, but expressions such as ``lane % 1`` and ``0 * row``
    introduce extra broadcast dimensions in the Ascend AST.  Folding these
    identities preserves the address while producing the same scalar/vector
    form as the verified native Conv2d kernel.
    """
    try:
        tree = ast.parse(str(expression), mode="eval")
        tree = _Conv2dIndexSimplifier().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree.body)
    except (SyntaxError, ValueError):
        return str(expression)


def _unsupported_attention_emission(reason: str) -> UnsupportedBackendOpError:
    return UnsupportedBackendOpError(
        "Ascend Attention structured emission contract is incomplete.",
        reason=reason,
        suggestion="rebuild the verified retile and resource plan before emission.",
    )

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
    native_block_matmul: bool = False

    def native_program_domain(self, kernel, axes, outer_axes):
        schedule = kernel.ssa.metadata.get("schedule", {}) if kernel.ssa else {}
        access = schedule.get("ascend_access_template_resources", {})
        is_conv2d = (
            isinstance(access, Mapping)
            and access.get("operator") == "conv2d-im2col"
        )
        retile = schedule.get("ascend_attention_retile")
        is_attention = isinstance(retile, Mapping) and bool(retile)
        if not (is_conv2d or is_attention):
            return None
        tile = access.get("tile", {}) if isinstance(access, Mapping) else {}
        scheduled_tile = schedule.get("tile", {})
        m = int(tile.get("m", scheduled_tile.get("block_m", 16)))
        n = int(tile.get("n", scheduled_tile.get("block_n", 16)))
        rows = f"triton.cdiv({axes[0]}, {m})"
        cols = f"triton.cdiv({axes[1]}, {n})"
        grid = retile.get("grid") if isinstance(retile, Mapping) else None
        if not grid:
            grid = f"({_product(outer_axes)}) * ({rows}) * ({cols})"
        return str(grid), "nt_outer_index", f"(nt_matrix_row) * ({axes[1]}) + nt_matrix_col"

    def initialize_emit_context(self, context) -> None:
        """Install Ascend native tile coordinates before SSA traversal."""
        if not context.native_block_program or len(context.output_axes) != 2:
            return
        schedule = context.kernel.ssa.metadata.get("schedule", {})
        access = schedule.get("ascend_access_template_resources", {})
        tile = access.get("tile", {}) if isinstance(access, Mapping) else {}
        scheduled_tile = schedule.get("tile", {})
        m = int(tile.get("m", scheduled_tile.get("block_m", 16)))
        n = int(tile.get("n", scheduled_tile.get("block_n", 16)))
        rows = f"triton.cdiv({context.output_axes[0]}, {m})"
        cols = f"triton.cdiv({context.output_axes[1]}, {n})"
        context.lines.extend(
            [
                "nt_native_program = tl.program_id(0)",
                f"nt_tile_rows = {rows}",
                f"nt_tile_cols = {cols}",
                "nt_tiles_per_outer = nt_tile_rows * nt_tile_cols",
                "nt_outer_index = nt_native_program // nt_tiles_per_outer",
                "nt_tile_index = nt_native_program % nt_tiles_per_outer",
                f"nt_tile_row = (nt_tile_index // nt_tile_cols) * {m}",
                f"nt_tile_col = (nt_tile_index % nt_tile_cols) * {n}",
                f"nt_matrix_row = nt_tile_row + tl.arange(0, {m})[:, None]",
                f"nt_matrix_col = nt_tile_col + tl.arange(0, {n})[None, :]",
                f"nt_matrix_active = (nt_matrix_row < {context.output_axes[0]}) & (nt_matrix_col < {context.output_axes[1]})",
            ]
        )
        context.outer_index_expr = "nt_outer_index"
        context.inner_index_expr = f"(nt_matrix_row) * ({context.output_axes[1]}) + nt_matrix_col"
        context.index_expr = context.inner_index_expr
        context.coordinate_exprs = ("nt_matrix_row", "nt_matrix_col")
        context.row_expr = "nt_matrix_row"
        context.col_expr = "nt_matrix_col"
        context.mask_expr = "nt_matrix_active"

    def store_address_plan(
        self,
        *,
        value_name: str,
        tensor_info,
        level: int,
        context,
    ) -> StoreAddressPlan:
        """Provide coordinates for complete multidimensional Ascend stores."""
        # Conv2d is emitted as a grid of physical 16x16 native matrix
        # programs.  The SSA result still carries the logical outer
        # application shape (for example 64x64), so the generic store path
        # would flatten ``nt_matrix_row/col`` back through a 16-wide view and
        # scramble the NCHW output mapping.  Keep the source access template
        # responsible for NCHW strides, but pass the native matrix coordinates
        # through unchanged.
        schedule = context.kernel.ssa.metadata.get("schedule", {})
        access = schedule.get("ascend_access_template_resources", {})
        if (
            isinstance(access, Mapping)
            and access.get("operator") == "conv2d-im2col"
            and str(getattr(tensor_info, "name", "")).lower() in {"output", "out"}
        ):
            return StoreAddressPlan(
                value_coords=("nt_matrix_row", "nt_matrix_col"),
                mask_coords=("nt_matrix_row", "nt_matrix_col"),
                source="ascend-conv2d-native-tile",
            )
        template = _access_template(tensor_info, level)
        if template is None:
            return StoreAddressPlan()

        template_shape = tuple(str(dim) for dim in template.get("shape", ()))
        value_axes = tuple(str(axis) for axis in _value_axes(value_name, context))
        if not template_shape or value_axes != template_shape:
            return StoreAddressPlan()

        coords = _current_coords(value_axes, context)
        return StoreAddressPlan(
            value_coords=coords,
            mask_coords=coords,
            source="target",
        )

    def loop_state_initializer(self, value, initializer, dtype, context):
        """Materialize Conv2d dot accumulators in the planned physical tile."""
        schedule = context.kernel.ssa.metadata.get("schedule", {})
        access = schedule.get("ascend_access_template_resources", {})
        if (
            not context.native_block_program
            or not isinstance(access, Mapping)
            or access.get("operator") != "conv2d-im2col"
            or value.type.kind != "tensor"
        ):
            return None

        producer = context.operations.get(value.name)
        if producer is None or producer.opcode != "scf.for":
            return None
        if not any(
            operation.opcode in {"linalg.dot", "linalg.matmul"}
            for region in producer.regions
            for operation in walk_ops(region.operations)
        ):
            return None

        tile = access.get("tile", {})
        m = int(tile.get("m", 16))
        n = int(tile.get("n", 16))
        return f"tl.full(({m}, {n}), {initializer}, tl.{common.normalize_dtype(dtype)})"

    def emit_block_dot(self, operation, context, coords=None):
        """Render the verified Conv2d dot as one physical M/N/K tile."""
        schedule = context.kernel.ssa.metadata.get("schedule", {})
        access = schedule.get("ascend_access_template_resources", {})
        if not isinstance(access, dict) and not hasattr(access, "get"):
            return None
        if access.get("operator") != "conv2d-im2col":
            return None
        tile = access.get("tile", {})
        m = int(tile.get("m", 16))
        n = int(tile.get("n", 16))
        k = int(tile.get("k", 16))
        if len(operation.operands) < 2:
            return None
        row = "nt_matrix_row"
        col = "nt_matrix_col"
        lhs_type = context.value_types.get(operation.operands[0])
        true_k = k
        if lhs_type is not None and len(lhs_type.shape) == 2:
            try:
                true_k = int(str(lhs_type.shape[-1]))
            except (TypeError, ValueError):
                true_k = k
        # Keep K tiling as a real loop, matching the verified raw Ascend
        # kernel.  Statically unrolling the K chunks into the surrounding
        # Conv2d reduction loop changes CANN's fragment-layout propagation and
        # yields a transposed logical matrix even though each isolated dot is
        # numerically correct.
        local = f"{_local_symbol(operation.results[0].name, context)}_native"
        context.lines.append(
            context.target.local_decl(
                ssa.Type(kind="tensor", shape=(str(m), str(n)), dtype="float32"),
                local,
                f"tl.zeros(({m}, {n}), tl.float32)",
            )
        )
        k_loop = "nt_conv_k"
        inner_lines = []
        inner_context = context.child(
            lines=inner_lines,
            memo=dict(context.memo),
            local_suffix=f"{context.local_suffix}_conv_k",
        )
        lhs_lane = f"({k_loop} + tl.arange(0, {k}))[None, :]"
        rhs_lane = f"({k_loop} + tl.arange(0, {k}))[:, None]"
        # Preserve physical tile bounds as well as the source access-template
        # bounds (padding and final M/K tails) for both dot operands.
        lhs = self._emit_conv2d_dot_load(
            operation.operands[0],
            (row, lhs_lane),
            inner_context,
            role="lhs",
            row=row,
            col=col,
            true_k=true_k,
        )
        rhs = self._emit_conv2d_dot_load(
            operation.operands[1],
            (rhs_lane, col),
            inner_context,
            role="rhs",
            row=row,
            col=col,
            true_k=true_k,
        )
        inner_lines.append(f"{local} = {local} + tl.dot({lhs}, {rhs})")
        context.lines.append(f"for {k_loop} in range(0, {true_k}, {k}):")
        context.lines.extend(_indent_lines(inner_lines, context.target))
        return local

    def _emit_conv2d_dot_load(
        self,
        name,
        coords,
        context,
        *,
        role,
        row,
        col,
        true_k,
    ):
        """Load one native Conv2d dot operand with a physical tile mask."""
        operation = context.operations.get(name)
        base = name
        extract_indices = ()
        if operation is not None and operation.opcode == "tensor.extract":
            base = operation.operands[0]
            extract_indices = tuple(
                _emit_index_value(operand, context)
                for operand in operation.operands[1:]
            )
        info = context.tensor_infos.get(base)
        if info is None:
            return common.emit_element(name, coords, context)
        level = _dtype_level(base, context)
        if operation is not None and operation.results:
            level = int(
                operation.results[0].type.attrs.get("dtype_level", level)
            )
        axes = _access_axes(
            info, context, level, fallback=_value_axes(base, context)
        )
        view_index = common.linearized_index(coords, axes) if coords else "0"
        source_index = _target_index_expr(
            context.target,
            _source_index_for_value(
                info,
                view_index,
                context,
                level=level,
                extract_indices=extract_indices,
                value_coords=coords,
            ),
        )
        source_index = _simplify_conv2d_index_expr(source_index)
        if role == "lhs":
            mask = f"(({row}) < ({context.output_axes[0]})) & (({coords[1]}) < ({true_k}))"
        else:
            mask = f"(({col}) < ({context.output_axes[1]})) & (({coords[0]}) < ({true_k}))"
        # Specialization replaces padding symbols with integer constants.
        # Symbol presence therefore cannot decide whether a source bound is
        # required: even an unpadded final M/K tile needs its source bounds.
        mask = _combined_mask(
            context.target,
            mask,
            info,
            view_index,
            ctx=context,
            level=level,
            extract_indices=extract_indices,
            value_coords=coords,
        ) or mask
        # Clamp invalid addresses even after padding symbols have been
        # specialized away.  Retain the same predicate for the zero fill.
        safe_index = f"tl.where(({mask}), ({source_index}), 0)"
        return super().load(
            base,
            safe_index,
            mask=mask,
            other=_load_other(info),
        )

    def emit_operation_expression(self, operation, coords, context):
        """Handle Ascend-only physical-domain expressions at one generic hook."""
        if operation.opcode.startswith("cmp."):
            return self._emit_score_mask(operation, coords, context)
        if operation.opcode == "select.where":
            return self._emit_score_where(operation, coords, context)
        return None

    def can_reuse_element(self, name, coords, context):
        # A tagged score predicate must be rebuilt for its requested M/N
        # coordinates; reusing a producer cached in the Q/K head-dimension
        # domain recreates the original (M,64) versus (M,N) mismatch.
        return not self._is_score_mask_value(name, context)

    def _attention_score_coords(self, context, score_shape, coords):
        schedule = context.kernel.ssa.metadata.get("schedule", {}) if getattr(context, "kernel", None) is not None else {}
        retile = schedule.get("ascend_attention_retile", {})
        sequence = str(retile.get("sequence", "1024"))
        m, n = score_shape
        query_tiles = f"(({sequence} + {int(m) - 1}) // {int(m)})" if str(m).isdigit() else f"triton.cdiv({sequence}, {m})"
        query = f"((tl.program_id(0) % {query_tiles}) * {m} + tl.arange(0, {m}))"
        reduce_index = str(getattr(context, "reduce_index", "") or "0")
        if reduce_index == "0":
            suffix = str(getattr(context, "local_suffix", ""))
            if "_body" in suffix:
                reduce_index = suffix.lstrip("_").split("_body", 1)[0] + "_i"
        key = f"(({reduce_index}) * {n} + tl.arange(0, {n}))"
        return query, key

    def _emit_score_mask(self, operation, coords, context):
        attrs = operation.attrs
        score_domain = attrs.get("ascend_attention_mask") == "score-key-bounds"
        if not score_domain and operation.opcode == "cmp.ge":
            schedule = context.kernel.ssa.metadata.get("schedule", {}) if getattr(context, "kernel", None) is not None else {}
            score_domain = bool(schedule.get("ascend_attention_retile"))
        if not score_domain and attrs.get("predicate_source") != "K.access-template.bounds":
            # bounds_valid is an identity select around the original K
            # predicate.  The producer carries the score contract even when
            # the comparison itself was not reached by the private retile.
            for candidate in context.operations.values():
                if candidate.opcode == "select.where" and candidate.results and operation.results:
                    if operation.results[0].name in candidate.operands and candidate.attrs.get("ascend_attention_mask") == "bounds_valid":
                        score_domain = True
                        attrs = candidate.attrs
                        break
        if not score_domain:
            return None
        score_shape = tuple(str(dim) for dim in attrs.get("score_shape", ()))
        if not score_shape and getattr(context, "kernel", None) is not None:
            plan = context.kernel.ssa.metadata.get("schedule", {}).get("ascend_attention_plan", {})
            score_shape = tuple(str(dim) for dim in plan.get("dot_tiles", {}).get("qk", {}).get("tile_shapes", {}).get("result", ()))
        if operation.opcode not in {"cmp.lt", "cmp.ge"} or len(score_shape) != 2 or not coords:
            raise UnsupportedBackendOpError(
                "Ascend Attention score mask contract is incomplete.",
                reason=f"opcode={operation.opcode!r}, score_shape={score_shape!r}, coords={coords!r}.",
                suggestion="preserve score mask provenance from the canonical resource plan.",
            )
        sequence = str(attrs.get("sequence", ""))
        if not sequence and getattr(context, "kernel", None) is not None:
            sequence = str(context.kernel.ssa.metadata.get("schedule", {}).get("ascend_attention_retile", {}).get("sequence", ""))
        if not sequence:
            raise UnsupportedBackendOpError(
                "Ascend Attention score mask has no source sequence extent.",
                reason="the score bounds operation omitted its source dimension.",
                suggestion="attach K source sequence extent during structured retile.",
            )
        key_coord = coords[-1]
        if operation.opcode == "cmp.ge" and len(coords) >= 2:
            query_coord, key_coord = self._attention_score_coords(context, score_shape, coords)
            return f"(({query_coord})[:, None] >= ({key_coord})[None, :])"
        if operation.opcode == "cmp.lt":
            _, key_coord = self._attention_score_coords(context, score_shape, coords)
        return f"(({key_coord}) < ({sequence}))"

    def _is_score_mask_value(self, name, context):
        """Recognize only values on an explicitly tagged score-mask chain."""
        seen = set()
        pending = [name]
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            operation = context.operations.get(value)
            if operation is None:
                continue
            attrs = operation.attrs
            if attrs.get("score_mask_role") == "qk-score-bounds":
                return True
            if attrs.get("ascend_attention_mask") in {
                "bounds_valid",
                "score-key-bounds",
                "key_valid",
            }:
                return True
            pending.extend(operation.operands)
        return False

    def _emit_score_where(self, operation, coords, context):
        is_bounds = operation.attrs.get("ascend_attention_mask") == "bounds_valid" or operation.attrs.get("score_mask_role") == "qk-score-bounds"
        if not is_bounds and operation.operands:
            producer = context.operations.get(operation.operands[0])
            producer_is_bounds = producer is not None and (
                producer.attrs.get("ascend_attention_mask") == "bounds_valid"
                or producer.attrs.get("score_mask_role") == "qk-score-bounds"
            )
            is_identity = (
                len(operation.operands) == 3
                and operation.operands[1] == operation.operands[2]
            )
            is_bounds = producer_is_bounds
            if is_identity and is_bounds:
                operation = producer
        if is_bounds:
            score_shape = tuple(str(dim) for dim in operation.attrs.get("score_shape", ()))
            sequence = str(operation.attrs.get("sequence", ""))
            if len(score_shape) == 2 and sequence:
                key_coord = coords[-1] if coords else "0"
                _, key_coord = self._attention_score_coords(context, score_shape, coords)
                predicate = f"((({key_coord}) < ({sequence}))[None, :])"
                # The bounds-valid identity select is itself a predicate
                # carrier.  A score select has distinct true/false operands
                # and must retain the score value and -inf replacement;
                # returning only the predicate would silently erase QK.
                identity = (
                    len(operation.operands) == 3
                    and operation.operands[1] == operation.operands[2]
                )
                if not identity:
                    values = tuple(
                        common.emit_element(operand, coords, context)
                        for operand in operation.operands[1:]
                    )
                    if len(values) == 2:
                        return self.where(predicate, values[0], values[1])
                if len(coords) == 1:
                    return f"(({key_coord}) < ({sequence}))"
                suffix = "" if "[None, :]" in key_coord or "[:, None]" in key_coord else "[None, :]"
                return predicate if suffix == "[None, :]" else f"((({key_coord}) < ({sequence})){suffix})"
        result_axes = tuple(str(dim) for dim in operation.results[0].type.shape) if operation.results else ()
        condition_axes = _value_axes(operation.operands[0], context) if operation.operands else ()
        score_contract = context.kernel.ssa.metadata.get("schedule", {}).get("ascend_attention_retile", {}) if getattr(context, "kernel", None) is not None else {}
        score_value = operation.operands[1] if len(operation.operands) > 1 else None
        score_op = context.operations.get(score_value) if score_value else None
        score_like = score_op is not None and score_op.opcode == "linalg.dot"
        score_shape = tuple(str(dim) for dim in operation.results[0].type.shape) if operation.results else ()
        if score_value in getattr(context, "value_types", {}):
            score_value_shape = tuple(str(dim) for dim in context.value_types[score_value].shape)
            plan = score_contract.get("dot_tiles", {}).get("qk", {}) if isinstance(score_contract, dict) else {}
            planned = tuple(str(dim) for dim in plan.get("tile_shapes", {}).get("result", ()))
            score_like = score_like or (planned and score_value_shape == planned and score_shape == planned)
        if not score_like and score_value:
            pending = [score_value]
            seen = set()
            while pending and not score_like:
                value = pending.pop()
                if value in seen:
                    continue
                seen.add(value)
                producer = context.operations.get(value)
                if producer is None:
                    continue
                score_like = producer.opcode == "linalg.dot"
                pending.extend(producer.operands)
        if not is_bounds and not (score_like and len(result_axes) == 2 and len(condition_axes) == 2 and result_axes != condition_axes and score_contract):
            return None
        score_shape = tuple(str(dim) for dim in operation.attrs.get("score_shape", ()))
        if not score_shape:
            score_shape = result_axes
        if len(score_shape) != 2:
            return None
        sequence = str(operation.attrs.get("sequence", ""))
        if not sequence:
            return None
        _, key_coord = self._attention_score_coords(context, score_shape, coords)
        predicate = f"((({key_coord}) < ({sequence}))[None, :])"
        if is_bounds:
            return predicate
        values = tuple(common.emit_element(operand, coords, context) for operand in operation.operands[1:])
        if len(values) == 2:
            return self.where(predicate, values[0], values[1])
        return predicate

    def vector_reduce(self, operator: str, operand: str, axis: int | None) -> str:
        if operator == "all":
            # Triton Ascend does not expose tl.all.  Attention stores boolean
            # lanes as 0/1 int32 values; their minimum is nonzero exactly when
            # every lane is true. axis=None produces the scalar predicate
            # required by the all-masked state-preserving branch.
            axis_expr = "None" if axis is None else str(axis)
            return f"(tl.min(({operand}).to(tl.int32), axis={axis_expr}) != 0)"
        return super().vector_reduce(operator, operand, axis)

    def vector_splat(self, shape: str, value: str, dtype: str) -> str:
        # This Triton Ascend build has no tl.bool.  Integer lanes preserve the
        # predicate's 0/1 semantics and remain valid operands for comparisons,
        # where, and the int32-backed all reduction above.
        if normalize_ascend_dtype(dtype) == "bool":
            dtype = "int32"
        return super().vector_splat(shape, value, dtype)

    def unary(self, operator: str, operand: str) -> str:
        # Shared SSA uses C's logical-not token, while generated Ascend source
        # is Python/Triton and must use the Python spelling.
        if operator == "not":
            return f"(not {operand})"
        return super().unary(operator, operand)

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

        # The installed Triton-Ascend language module does not expose
        # ``tl.tanh``, and its libdevice symbol is not accepted by the
        # AST-to-TTIR path for vector values.  Express tanh through the
        # available exponential intrinsic.  Keep this mapping private to
        # Ascend so the CUDA/Triton emitters retain their native spelling.
        if name == "tanh" and args:
            args = tuple(self._upcast_fp32(arg) for arg in args)
            value = args[0]
            return f"(2.0 / (1.0 + tl.exp(-2.0 * ({value}))) - 1.0)"

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

    def schedule_context(self, context: ModuleRenderContext) -> ModuleRenderContext:
        """Expose the private Attention grid contract to module rendering."""
        program = context.kernel.ssa
        schedule = program.metadata.get("schedule", {}) if program is not None else {}
        retile = schedule.get("ascend_attention_retile")
        if not isinstance(retile, dict) and not hasattr(retile, "get"):
            return super().schedule_context(context)
        grid = retile.get("grid")
        if not grid:
            return super().schedule_context(context)
        return replace(context, grid_total=str(grid))

    def render_module(self, context: ModuleRenderContext) -> str:
        if context.block_program or context.vector_program or context.scalar_program:
            return super().render_module(context)

        kernel = context.kernel
        body = common.rewrite_index_math(context.body, c_style=False)
        schedule = kernel.ssa.metadata.get("schedule", {})
        access = schedule.get("ascend_access_template_resources", {})
        native_conv = (
            kernel.ssa.metadata.get("optimization", {}).get("preserve_linalg")
            and isinstance(access, Mapping)
            and access.get("operator") == "conv2d-im2col"
        )
        runtime_params = set(kernel.metadata.get("runtime_shape_params", ()))
        params = ",\n    ".join(
            (
                *context.variables,
                *context.outputs,
                *[
                    axis if axis in runtime_params else f"{axis}: tl.constexpr"
                    for axis in context.shape_params
                ],
                *(("BLOCK: tl.constexpr",) if not native_conv else ()),
            )
        )
        kernel_args = ",\n        ".join(
            (*context.variables, *context.outputs, *context.shape_params)
        )
        total = context.total
        mask_total = total
        launch_total = total
        launch_grid = (
            context.grid_total
            if native_conv
            else f"triton.cdiv({launch_total}, block)"
        )
        offsets_expression = (
            "0"
            if native_conv
            else "tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)"
        )
        tile = dict(schedule.get("ascend_tile_override", schedule.get("tile", {})))
        block_value = tile.get(
            "BLOCK_SIZE_M", tile.get("elements", tile.get("block_m", 256))
        )
        try:
            block_value = int(block_value)
        except (TypeError, ValueError):
            block_value = 256
        compiler_flags = ""
        if schedule.get("granularity") == "blocked-linalg":
            compiler_flags = (
                "\n            multibuffer=False,\n            num_stages=1,"
            )
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
        launch_block_arg = "" if native_conv else "BLOCK=block,"

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
    block = {block_value}
    grid = ({launch_grid},)
    {kernel.kernel_name}_kernel[grid](
            {kernel_args},
            {launch_block_arg}
            {compiler_flags}
    )
    return {result}
"""


TARGET = AscendTarget()


def emit(kernel: Kernel):
    """Emit only the explicitly verified first Ascend operation tier."""
    _validate_program(kernel)
    schedule = kernel.ssa.metadata.get("schedule", {})
    attention_contract = schedule.get("ascend_attention_loop")
    if attention_contract:
        _validate_ascend_attention_emission_contract(kernel)
        retile = schedule.get("ascend_attention_retile")
        plan = schedule.get("ascend_attention_plan")
        if not isinstance(retile, dict) and not hasattr(retile, "get"):
            raise _unsupported_attention_emission(
                "structured retile metadata is absent from the source boundary."
            )
        if not isinstance(plan, dict) and not hasattr(plan, "get"):
            raise _unsupported_attention_emission(
                "verified Attention planner metadata is absent from the source boundary."
            )
        resource_plan = plan.get("resource_plan", {})
        selected_tile = dict(resource_plan.get("selected_tile", {}))
        if (
            selected_tile != dict(plan.get("tile", {}))
            or tuple(retile.get(f"block_{axis}") for axis in ("m", "n", "k"))
            != tuple(selected_tile.get(axis) for axis in ("m", "n", "k"))
        ):
            raise _unsupported_attention_emission(
                "structured retile and planner tile contracts disagree."
            )
        if plan.get("sequence") is not None and retile.get("sequence") != plan.get("sequence"):
            raise _unsupported_attention_emission(
                "structured retile sequence differs from the resource plan."
            )
        attention_source_contract = ascend_attention_source_contract(plan, retile)
    access = schedule.get("ascend_access_template_resources", {})
    native_conv = (
        isinstance(access, Mapping) and access.get("operator") == "conv2d-im2col"
    )
    retile = schedule.get("ascend_attention_retile")
    native_attention = isinstance(retile, Mapping) and bool(retile)
    target = replace(
        TARGET,
        default_load_mask=schedule.get("granularity") == "blocked-linalg",
        native_block_matmul=native_conv or native_attention,
    )

    kernel = _rewrite_private_scan_ops(kernel)
    artifact = common.emit(kernel, target)

    if attention_contract:
        forbidden_shapes = ("128x64", "tl.full((128, 64))", "memref<128x64>")
        if any(shape in artifact.primary_source for shape in forbidden_shapes):
            raise UnsupportedBackendOpError(
                "Ascend Attention source contains an unverified fixed physical shape.",
                reason="the emitted source retained a legacy 128x64 shape.",
                suggestion="emit shapes from the verified Attention tile contract.",
            )
        plan = schedule.get("ascend_attention_plan", {})
        retile = schedule.get("ascend_attention_retile", {})
        if plan.get("status") == "verified-static-single-block":
            expected_grid = int(plan["grid"])
            batch = int(plan["batch"])
            heads = int(plan["heads"])
            sequence = int(plan["sequence"])
            m, n = int(selected_tile["m"]), int(selected_tile["n"])
            query_tiles = (sequence + m - 1) // m
            key_tiles = (sequence + n - 1) // n
            expected_retile_grid = batch * heads * query_tiles
            if expected_grid != expected_retile_grid:
                raise _unsupported_attention_emission(
                    "planner grid is inconsistent with concrete Q shape."
                )
            if (
                f"{batch} * {heads} * triton.cdiv({sequence}, {m})"
                != retile.get("grid")
            ):
                raise _unsupported_attention_emission(
                    "emitted launch grid expression does not consume concrete batch/head/sequence."
                )
            if f"(({sequence} + {n - 1}) // {n})" != retile.get("loop_upper"):
                raise _unsupported_attention_emission(
                    "retiled scf.for upper bound does not match the planned sequence tiles."
                )
            if f"(({sequence} + {m - 1}) // {m})" != retile.get("query_loop_upper"):
                raise _unsupported_attention_emission(
                    "retiled query grid does not match the planned query tiles."
                )
            if key_tiles != plan.get("key_tiles") or query_tiles != plan.get("query_tiles"):
                raise _unsupported_attention_emission(
                    "planner query/key tile counts disagree with selected M/N."
                )

    # Publish solved matrix dimensions through the artifact schedule.  The
    # driver uses this metadata when constructing launch ABI defaults, so the
    # values supplied to Triton's constexpr BLOCK_SIZE_* parameters cannot
    # silently fall back to the 256 candidate defaults.
    override = dict(schedule.get("ascend_tile_override", {}))
    if override:
        artifact_schedule = dict(artifact.metadata.get("ssa_schedule", {}))
        artifact_tile = dict(artifact_schedule.get("tile", {}))
        for source, alias in (
            ("BLOCK_SIZE_M", "block_m"),
            ("BLOCK_SIZE_N", "block_n"),
            ("BLOCK_SIZE_K", "block_k"),
        ):
            if alias in override:
                # ``scheduled_meta_defaults`` matches canonical block_m/n/k
                # names (including prefixed constexpr symbols).  Publish both
                # spellings so source emission and ABI construction observe
                # exactly the same retiled value.
                artifact_tile[alias] = int(override[alias])
                artifact_tile[source] = int(override[alias])
        artifact_schedule["tile"] = artifact_tile
        metadata = dict(artifact.metadata) | {
            "ssa_schedule": artifact_schedule,
            "ascend_tile_override": override,
        }
        artifact = replace(artifact, metadata=metadata)

    # TritonTarget.schedule_context consumes compiler/layout.py's LayoutTransfer
    # access maps. Keep the public coordinate helpers visible at this boundary so
    # Ascend-specific rewrites never need a duplicate flat-index implementation.
    if _layout_transfer_present(kernel):
        _validate_layout_transfer_surface(kernel)

    artifact = _rewrite_stride_predicates(artifact)
    artifact = _rewrite_unary_positive(artifact)
    artifact = _rewrite_singleton_broadcast_loads(artifact, kernel)
    if attention_contract:
        source_metadata = (
            f"\n\n{ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE} = "
            f"{attention_source_contract!r}\n"
        )
        source = artifact.primary_source.rstrip() + source_metadata
        artifact = replace(
            artifact,
            sources={artifact.primary_source_name: source},
            metadata=dict(artifact.metadata)
            | {"ascend_attention_source_contract": attention_source_contract},
        )
    return artifact


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
            if linalg.get("rank") == 2 and not any(
                operation.opcode in {"linalg.matmul", "linalg.dot"}
                for operation in _operations(kernel.ssa)
            ):
                # Public linalg has already been lowered to a scalar K loop.
                # Consume the private structured contract at this boundary so
                # source emission is tied to the proven row/k and k/col map.
                _verify_ascend_decomposed_matmul_contract(kernel.ssa)
            elif linalg.get("rank") == 3 and not any(
                operation.opcode in {"linalg.matmul", "linalg.dot"}
                for operation in _operations(kernel.ssa)
            ):
                _verify_ascend_batched_matmul_contract(kernel.ssa)
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
