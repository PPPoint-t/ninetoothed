"""Ascend backend policy, contracts, and private launch metadata."""

import ast
import hashlib
import json
import os
import re
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
from ninetoothed.naming import remove_prefixes


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

        if kernel.ssa is None:
            return kernel

        # The public scheduler normally resolves tile constexprs immediately.
        # Online softmax needs a stricter target-private choice, though: once
        # those symbols have become literal 128x64 layout dimensions there is
        # no remaining symbol for the Ascend retile pass to substitute.  Apply
        # the Ascend choice first, then let the regular schedule materializer
        # resolve every other meta parameter.
        schedule = dict(kernel.ssa.metadata.get("schedule", {}))
        # Preserve tile provenance before any generic specialization can turn
        # symbolic BLOCK parameters into indistinguishable integer literals.
        # This target-private record is carried through the artifact/sidecar
        # and is the input contract for a later structured retile pass.
        provenance = _capture_ascend_tile_provenance(kernel, schedule)
        if provenance:
            schedule["ascend_tile_provenance"] = provenance
            kernel = _restore_ascend_tile_provenance(kernel, provenance, schedule)
        kernel = _retile_ascend_online_softmax(kernel, schedule)
        schedule = dict(kernel.ssa.metadata.get("schedule", {}))
        if schedule.get("ascend_attention_loop"):
            # The structured pass consumes only the recognized online-softmax
            # contract.  It specializes SSA shapes and records loop/grid
            # bounds before the generic backend specialization runs.
            provenance = _capture_ascend_tile_provenance(kernel, schedule)
            if not provenance:
                raise UnsupportedBackendOpError(
                    "Ascend attention structured retile requires tile provenance.",
                    reason="the online-softmax contract has no explicit tile roles.",
                    suggestion="provide block_m, block_n, and block_k provenance before emission.",
                )
            kernel = _restore_ascend_tile_provenance(kernel, provenance, schedule)
            kernel = _structured_retile_ascend_attention(kernel, provenance)
        # The private attention planner specializes the SSA program itself.  Its
        # backend-neutral specialization walk also rewrites metadata strings,
        # so restore the target-private record before the generic tile pass.
        kernel = _restore_ascend_tile_provenance(kernel, provenance)
        kernel = specialize_schedule_tiles(kernel)
        # Generic specialization has the same metadata behavior.  Keep the
        # original names and operation paths available while resolving tiles.
        kernel = _restore_ascend_tile_provenance(kernel, provenance)
        schedule = dict(kernel.ssa.metadata.get("schedule", {}))
        tile = dict(schedule.get("tile", {}))
        matrix_tile = dict(schedule.get("ascend_matrix_tile", {}))
        dot_loop = dict(schedule.get("ascend_dot_loop", {}))
        is_attention = bool(schedule.get("ascend_attention_loop"))
        attention_resource_plan = None
        if is_attention:
            selected = _selected_attention_tile(schedule)
            attention_resource_plan = schedule["ascend_attention_plan"][
                "resource_plan"
            ]
        # Attention owns a two-dot online-softmax loop. The generic
        # access-template validator is for one-dot Conv2d loops and must not
        # reinterpret this already verified Attention contract as Conv2d.
        access_resources = None
        if not is_attention:
            access_resources = _validate_ascend_access_template_contract(
                kernel.ssa, schedule, kernel.tensors
            )
        if access_resources is not None:
            schedule["ascend_access_template_resources"] = access_resources
            dot_tile = dict(access_resources["tile"])
            # Replace the generic contract's conservative default with the
            # dimensions proven by the actual SSA linalg.dot operands.  This
            # record is exported through the sidecar and consumed by the
            # runtime UB validator.
            schedule["ascend_dot_loop"] = dict(dot_loop) | {"tile": dot_tile}
            solve_input = {
                "block_m": dot_tile["m"],
                "block_n": dot_tile["n"],
                "block_k": dot_tile["k"],
            }
            if access_resources.get("operator") == "conv2d-im2col":
                conv2d_plan = _plan_ascend_conv2d_resources(
                    kernel.ssa, solve_input
                )
                if conv2d_plan["selected_tile"] is None:
                    raise UnsupportedBackendOpError(
                        "Ascend Conv2d has no UB-feasible verified tile.",
                        reason=str(conv2d_plan["rejection_reason"]),
                        suggestion=(
                            "use an Ascend Conv2d lowering with a smaller verified "
                            "im2col tile or select another backend."
                        ),
                    )
                selected_conv2d = dict(conv2d_plan["selected_tile"])
                solve_input = {
                    "block_m": int(selected_conv2d["m"]),
                    "block_n": int(selected_conv2d["n"]),
                    "block_k": int(selected_conv2d["k"]),
                }
                schedule["ascend_conv2d_plan"] = conv2d_plan
        else:
            solve_input = matrix_tile or tile
        if is_attention:
            selected = _attention_tile_dict(
                attention_resource_plan["selected_tile"]
            )
            solved = {
                "block_m": selected["m"],
                "block_n": selected["n"],
                "block_k": selected["k"],
                "BLOCK_SIZE_M": selected["m"],
                "BLOCK_SIZE_N": selected["n"],
                "BLOCK_SIZE_K": selected["k"],
            }
            budget = int(attention_resource_plan["ub_budget_bytes"])
            estimated_peak = int(
                attention_resource_plan["ub_estimated_peak_bytes"]
            )
            ub_plan = AscendUBPlan(
                safe_tile=solved,
                estimated_peak_bytes=estimated_peak,
                workspace_bytes=selected["m"] * selected["n"] * 4,
                safety_margin_bytes=max(0, budget - estimated_peak),
            )
        elif solve_input:
            ub_plan = plan_ascend_ub(kernel.ssa, solve_input)
            if ub_plan.rejection_reason:
                raise UnsupportedBackendOpError(
                    "Ascend matmul tile exceeds the private UB contract.",
                    reason=ub_plan.rejection_reason,
                    suggestion="reduce M/N/K tile dimensions or split the reduction loop.",
                )
            solved = solve_ascend_tile_config(kernel.ssa, solve_input)
        if is_attention or solve_input:
            # Publish the solved dimensions through the canonical schedule
            # namespace consumed by Launch-ABI specialization.  Keep the
            # matrix-specific copy as well, but never leave ``tile`` at the
            # candidate's original (typically 256) value.
            canonical_tile = dict(schedule.get("tile", {}))
            for source, alias in (
                ("BLOCK_SIZE_M", "block_m"),
                ("BLOCK_SIZE_N", "block_n"),
                ("BLOCK_SIZE_K", "block_k"),
            ):
                value = solved.get(source, solved.get(alias))
                if value is not None:
                    canonical_tile[alias] = int(value)
                    canonical_tile[source] = int(value)
            schedule["tile"] = canonical_tile
            if matrix_tile:
                schedule["ascend_matrix_tile"] = dict(solved)
            # This immutable private field is consumed by all Ascend-side
            # emitters/materializers and takes precedence over candidate defaults.
            schedule["ascend_tile_override"] = dict(solved)
            schedule["ascend_ub_plan"] = {
                "estimated_peak_bytes": int(ub_plan.estimated_peak_bytes),
                "workspace_bytes": int(ub_plan.workspace_bytes),
                "safety_margin_bytes": int(ub_plan.safety_margin_bytes),
                "budget_bytes": int(
                    ub_plan.estimated_peak_bytes + ub_plan.safety_margin_bytes
                ),
            }
            if access_resources is not None:
                access_resources = dict(access_resources)
                access_resources["tile"] = {
                    "m": int(solved.get("block_m", solved.get("BLOCK_SIZE_M"))),
                    "n": int(solved.get("block_n", solved.get("BLOCK_SIZE_N"))),
                    "k": int(solved.get("block_k", solved.get("BLOCK_SIZE_K"))),
                }
                access_resources["workspace_bytes"] = int(
                    ub_plan.workspace_bytes
                )
                access_resources["ub_peak_bytes"] = int(
                    ub_plan.estimated_peak_bytes
                )
                schedule["ascend_access_template_resources"] = access_resources
                if access_resources.get("operator") == "conv2d-im2col":
                    schedule["ascend_dot_loop"] = dict(
                        schedule.get("ascend_dot_loop", {})
                    ) | {
                        "tile": dict(access_resources["tile"]),
                    }
                    # Access templates are built from the public static
                    # arrangement.  Retile them only after the Conv2d
                    # resource plan has selected its UB-feasible tile so the
                    # generated source and sidecar consume the same matrix
                    # domain.
                    kernel = replace(
                        kernel,
                        ssa=replace(
                            kernel.ssa,
                            metadata=dict(kernel.ssa.metadata)
                            | {"schedule": schedule},
                        ),
                    )
                    kernel = _retile_ascend_conv2d_access_templates(kernel)
                    schedule = dict(kernel.ssa.metadata.get("schedule", schedule))
            schedule["ascend_tile_provenance"] = _resolve_ascend_tile_provenance(
                provenance or schedule.get("ascend_tile_provenance", {}), solved
            )
            # Triton-Ascend enables ping-pong buffering by default.  Keep the
            # private NineToothed contract within the documented 96 KiB
            # double-buffer budget by explicitly selecting a single stage.
            schedule["multibuffer"] = False
            schedule["num_stages"] = 1
            linalg = dict(schedule.get("ascend_linalg", {}))
            if linalg:
                linalg["tile"] = dict(solved)
                linalg["workspace_tile"] = dict(solved)
                linalg["loop_tile"] = dict(solved)
                schedule["ascend_linalg"] = linalg
            updated_ssa = replace(
                kernel.ssa,
                metadata=dict(kernel.ssa.metadata) | {"schedule": schedule},
            )
            # Attach the same contract directly to matrix operations.  This is
            # intentionally target-private metadata; the shared IR semantics are
            # unchanged while downstream linalg lowering can read exact bounds.
            updated_ssa = _annotate_ascend_linalg_tiles(updated_ssa, solved)
            updated_ssa = _annotate_ascend_decomposed_matmul_contract(
                updated_ssa, solved
            )
            updated_ssa = _resolve_ascend_matmul_operand_dtypes(updated_ssa)
            updated_ssa = _normalize_ascend_decomposed_matmul_accumulators(
                updated_ssa
            )
            updated_ssa = _normalize_ascend_matmul_accumulators(updated_ssa)
            if dict(schedule.get("ascend_linalg", {})).get("rank") == 3:
                updated_ssa = _rewrite_ascend_batched_matmul_access(
                    updated_ssa, dict(schedule["ascend_linalg"])
                )
                schedule = dict(updated_ssa.metadata.get("schedule", schedule))
            matmul_contract = _verify_ascend_matmul_contract(updated_ssa)
            tile_contract = _verify_ascend_tile_consumption(updated_ssa)
            schedule["ascend_matmul_contract"] = matmul_contract
            if tile_contract is not None:
                schedule["ascend_tile_consumption"] = tile_contract
            updated_ssa = replace(
                updated_ssa,
                metadata=dict(updated_ssa.metadata) | {"schedule": schedule},
            )
            metadata = dict(kernel.metadata)
            defaults = dict(metadata.get("meta_defaults", {}))
            for name in defaults:
                normalized = str(name).lower().replace("-", "_")
                for suffix, alias in (
                    ("block_size_m", "block_m"),
                    ("block_size_n", "block_n"),
                    ("block_size_k", "block_k"),
                ):
                    if normalized.endswith(suffix) and alias in solved:
                        defaults[name] = int(solved[alias])
            metadata["meta_defaults"] = defaults
            kernel = replace(kernel, ssa=updated_ssa, metadata=metadata)

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


_ASCEND_TILE_ROLES = {
    "elements": "vector-elements",
    "block_m": "matmul-m",
    "block_n": "matmul-n",
    "block_k": "reduction-k",
}


def _capture_ascend_tile_provenance(
    kernel: Kernel, schedule: Mapping[str, Any]
) -> dict[str, Any]:
    """Capture symbolic tile ownership before any target or generic rewrite.

    The record deliberately lives in the private backend metadata.  SSA itself
    remains unchanged; operation paths are descriptive references that let a
    later Ascend lowering identify the operation whose tile contract it must
    consume.
    """
    if kernel.ssa is None:
        return {}

    existing = schedule.get("ascend_tile_provenance")
    attention_contract = schedule.get("ascend_attention_loop")
    attention_plan = schedule.get("ascend_attention_plan")
    resource_plan = (
        attention_plan.get("resource_plan")
        if isinstance(attention_plan, Mapping)
        else None
    )
    selected_attention_tile = (
        resource_plan.get("selected_tile")
        if isinstance(resource_plan, Mapping)
        else None
    )
    if (
        isinstance(attention_contract, Mapping)
        and isinstance(selected_attention_tile, Mapping)
        and all(key in selected_attention_tile for key in ("m", "n", "k"))
    ):
        selected_attention_tile = _attention_tile_dict(selected_attention_tile)
    else:
        selected_attention_tile = None
    tile_maps = (
        ("tile", dict(schedule.get("tile", {}))),
        ("ascend_matrix_tile", dict(schedule.get("ascend_matrix_tile", {}))),
    )
    tracked: dict[str, int] = {}
    for _, values in tile_maps:
        for key, value in values.items():
            if key in _ASCEND_TILE_ROLES:
                try:
                    tracked[key] = int(value)
                except (TypeError, ValueError):
                    continue

    if not tracked and isinstance(existing, Mapping):
        return dict(existing)
    if not tracked and selected_attention_tile is None:
        return {}

    defaults = dict(kernel.metadata.get("meta_defaults", {}))
    operations = tuple(_ascend_operation_references(kernel.ssa))
    entries = []
    if selected_attention_tile is not None:
        # Attention M/N/K are owned by the canonical resource plan.  Do not
        # infer them from generic schedule fields after planning.
        tracked = {
            "block_m": selected_attention_tile["m"],
            "block_n": selected_attention_tile["n"],
            "block_k": selected_attention_tile["k"],
        }
    for key, candidate in tracked.items():
        parameter = _ascend_original_tile_parameter(key, defaults)
        role = _ASCEND_TILE_ROLES[key]
        candidate_values = _ascend_candidate_values(kernel.ssa, key, candidate)
        references = tuple(
            reference
            for reference in operations
            if _ascend_operation_matches_tile(reference, key, parameter)
        )
        if selected_attention_tile is not None:
            references = tuple(
                reference
                for reference in operations
                if reference["opcode"] == "linalg.dot"
            )
        entries.append(
            {
                "parameter": parameter,
                "tile_parameter": key,
                "role": role,
                "candidate_values": candidate_values,
                "resolved": _ascend_tile_value(
                    dict(schedule.get("ascend_tile_override", {}))
                    or dict(schedule.get("ascend_attention_tile", {})),
                    key,
                ),
                "ssa_operations": references,
            }
        )

    if selected_attention_tile is not None:
        for entry in entries:
            key = str(entry["tile_parameter"])
            entry["candidate_values"] = (int(tracked[key]),)
            entry["resolved"] = int(tracked[key])

    # Keep the original compact fields for sidecar readers while adding a
    # per-parameter record with the information needed for structured lowering.
    return {
        "version": 2,
        "source": "schedule.tile",
        "parameters": tuple(entry["parameter"] for entry in entries),
        "candidate": dict(tracked),
        "entries": tuple(entries),
    }


def _restore_ascend_tile_provenance(
    kernel: Kernel,
    provenance: Mapping[str, Any],
    schedule: Mapping[str, Any] | None = None,
) -> Kernel:
    if kernel.ssa is None or not provenance:
        return kernel

    updated_schedule = dict(
        schedule if schedule is not None else kernel.ssa.metadata.get("schedule", {})
    )
    updated_schedule["ascend_tile_provenance"] = dict(provenance)
    return replace(
        kernel,
        ssa=replace(
            kernel.ssa,
            metadata=dict(kernel.ssa.metadata) | {"schedule": updated_schedule},
        ),
    )


def _resolve_ascend_tile_provenance(
    provenance: Mapping[str, Any], solved: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(provenance)
    resolved = {
        key: int(value)
        for key, value in solved.items()
        if key in _ASCEND_TILE_ROLES
        or key in {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"}
    }
    result["resolved"] = resolved
    entries = []
    for entry in tuple(provenance.get("entries", ())):
        updated = dict(entry)
        key = str(entry.get("tile_parameter", ""))
        value = _ascend_tile_value(solved, key)
        updated["resolved"] = int(value) if value is not None else None
        entries.append(updated)
    if entries:
        result["entries"] = tuple(entries)
    return result


def _ascend_tile_value(values: Mapping[str, Any], key: str) -> int | None:
    aliases = {
        "block_m": ("block_m", "BLOCK_SIZE_M"),
        "block_n": ("block_n", "BLOCK_SIZE_N"),
        "block_k": ("block_k", "BLOCK_SIZE_K"),
        "elements": ("elements",),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in values:
            try:
                return int(values[candidate])
            except (TypeError, ValueError):
                return None
    return None


def _ascend_original_tile_parameter(key: str, defaults: Mapping[str, Any]) -> str:
    expected = {
        "block_m": "block_m",
        "block_n": "block_n",
        "block_k": "block_k",
        "elements": "elements",
    }[key]
    matches = []
    for name in defaults:
        normalized = remove_prefixes(str(name)).lower().replace("-", "_")
        normalized = normalized.replace("block_size_", "block_")
        if normalized == expected or (
            key == "elements" and normalized in {"block", "elements", "tile"}
        ):
            matches.append(str(name))
    return sorted(matches)[0] if matches else key


def _ascend_candidate_values(
    program: ssa.Program, key: str, selected: int
) -> tuple[int, ...]:
    values = [selected]
    for candidate in tuple(program.metadata.get("schedule_candidates", ())):
        candidate_schedule = candidate.get("schedule", {})
        for field in ("tile", "ascend_matrix_tile"):
            value = _ascend_tile_value(candidate_schedule.get(field, {}), key)
            if value is not None:
                values.append(value)
    return tuple(dict.fromkeys(int(value) for value in values))


def _ascend_operation_references(program: ssa.Program):
    def walk(block: ssa.Block, prefix: str):
        for index, operation in enumerate(block.operations):
            path = f"{prefix}/{index}"
            yield {
                "path": path,
                "opcode": operation.opcode,
                "results": tuple(result.name for result in operation.results),
                "operands": tuple(operation.operands),
                "operation": path,
            }
            for region_index, region in enumerate(operation.regions):
                yield from walk(region, f"{path}/region{region_index}")

    for block in program.blocks:
        yield from walk(block, block.name)


def _ascend_operation_matches_tile(
    reference: Mapping[str, Any], key: str, parameter: str
) -> bool:
    if key in {"block_m", "block_n", "block_k"} and reference["opcode"] in {
        "linalg.matmul",
        "linalg.dot",
    }:
        return True
    token = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(parameter)}(?![A-Za-z0-9_])"
    )
    return bool(
        token.search(str(reference.get("operation", "")))
        or token.search(str(reference.get("results", "")))
    )


ASCEND_UB_LIMIT_BYTES = 96 * 1024
ASCEND_UB_NORMAL_FRACTION = 0.90
ASCEND_UB_ATTENTION_FRACTION = 0.55
# These tiles share one structured online-softmax lowering. M=16 is the
# smallest candidate returned by the current UB model; M=32 remains available
# when its resource estimate fits. Every dimension is consumed from the plan.
_ASCEND_ATTENTION_TILE_CANDIDATES = ((16, 32, 32), (32, 32, 32))
# Conv2d is lowered to a padded rank-4 access-template plus one loop-carried
# dot.  The dynamic JIT path has been verified with this smallest matrix tile;
# the CANN planner rejects the larger public build candidates even when the
# generic SSA byte estimate says they fit.  Keep the verified tile explicit so
# static AOT builds cannot select a different CANN resource shape.
_ASCEND_CONV2D_TILE_CANDIDATES = ((16, 16, 16),)


@dataclass(frozen=True)
class AscendUBPlan:
    """Target-private UB planning result consumed by Ascend lowering."""

    safe_tile: Mapping[str, int]
    estimated_peak_bytes: int
    workspace_bytes: int
    safety_margin_bytes: int
    rejection_reason: str | None = None


SUPPORTED_DTYPES = frozenset(
    {"float32", "float16", "bfloat16", "int32", "bool"}
)
UNSUPPORTED_DTYPES = frozenset(
    {"float8_e5m2", "float8_e4m3fn", "float64", "int8"}
)
ASCEND_ELEMENTWISE_DTYPES = SUPPORTED_DTYPES
ASCEND_RNG_DTYPES = frozenset({"float16", "bfloat16", "float32"})
ASCEND_ATOMIC_DTYPES = frozenset({"float16", "bfloat16", "float32", "int32"})
ASCEND_ATTENTION_INPUT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
ASCEND_ATTENTION_INTERNAL_DTYPE = "float32"
ASCEND_ATTENTION_OUTPUT_DTYPES = ASCEND_ATTENTION_INPUT_DTYPES
ASCEND_ATTENTION_ERROR_TOLERANCES = {
    "float16": {"rtol": 0.01, "atol": 0.01},
    # BF16 output quantization is materially wider at sequence=1024 than
    # FP16/FP32.  Keep this threshold independent and explicit in the sidecar
    # contract; FP16/FP32 thresholds remain unchanged.
    "bfloat16": {"rtol": 0.05, "atol": 0.1},
    "float32": {"rtol": 0.025, "atol": 0.025},
}
ASCEND_ATTENTION_DTYPE_CONTRACT_VERSION = 1
ASCEND_ATTENTION_DTYPE_REGISTRY = {
    dtype: {
        "version": ASCEND_ATTENTION_DTYPE_CONTRACT_VERSION,
        "input": {"q": dtype, "k": dtype, "v": dtype, "o": dtype},
        "storage": {"q": dtype, "k": dtype, "v": dtype, "o": dtype},
        "output": dtype,
        "internal": {
            "score": "float32",
            "softmax": "float32",
            "acc": "float32",
            "m_i": "float32",
            "l_i": "float32",
        },
        "intrinsics": {
            "dot": "fp32-accumulate",
            "reduction": "fp32",
            "cast": f"{dtype}->float32 and float32->{dtype}",
        },
        "dot": {
            "operand": "native-fp32" if dtype == "float32" else f"{dtype}-cast-to-float32",
            "accumulator": "float32",
            "score": "float32",
            "value": "float32",
        },
        "load_cast": "none" if dtype == "float32" else f"{dtype}->float32 for dot",
        "store_cast": "none" if dtype == "float32" else f"float32->{dtype}",
        "ub": {"input_bytes": 2 if dtype in {"float16", "bfloat16"} else 4,
               "internal_bytes": 4, "workspace_bytes_per_element": 4},
        "resource_policy": {
            "internal_dtype": "float32",
            "workspace_dtype": "float32",
            "workspace_bytes_per_element": 4,
        },
        "runtime": {
            "device": "Ascend910B4",
            "cann": "9.0.0",
            "torch_npu": "2.7.1",
            "triton": "3.2.0",
            "triton_ascend": "3.2.2",
        },
        "error": dict(ASCEND_ATTENTION_ERROR_TOLERANCES[dtype]),
        "tolerance": dict(ASCEND_ATTENTION_ERROR_TOLERANCES[dtype]),
        "status": (
            "verified-jit-aot-npu"
            if dtype in {"float16", "bfloat16", "float32"}
            else "unsupported-capability"
        ),
    }
    for dtype in sorted(ASCEND_ATTENTION_INPUT_DTYPES)
}
_SIDECAR_SCHEMA = 4
ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE = "__ninetoothed_ascend_attention_contract__"
ASCEND_ATTENTION_SOURCE_CONTRACT_VERSION = 1


def ascend_dtype_legality(dtypes):
    """Validate hardware dtype names without importing torch in the backend."""
    for dtype in dtypes:
        normalized = normalize_ascend_dtype(dtype)
        if normalized in UNSUPPORTED_DTYPES:
            if normalized.startswith("float8"):
                raise UnsupportedBackendOpError(
                    "Ascend 910B3 backend does not support float8 execution."
                )
            raise UnsupportedBackendOpError(
                f"Ascend 910B3 hardware does not support dtype {normalized}; "
                "only FP16, BF16, and FP32 elementwise SSA is verified",
                reason="the dtype has no verified 910B3/CANN execution path.",
                suggestion="cast to float16, bfloat16, float32, or int32 before lowering.",
            )
    return True


def _ssa_tile_values(ssa_graph):
    for block in getattr(ssa_graph, "blocks", ()):
        for operation in getattr(block, "operations", ()):
            for result in getattr(operation, "results", ()):
                yield result


def calculate_ssa_ub_bytes(ssa_graph, tile_config, pipeline_factor=2):
    """Estimate UB bytes from all tiled SSA values and their element dtypes."""
    sizes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "int32": 4,
        "int8": 1,
        "bool": 1,
    }
    m = int(
        tile_config.get(
            "BLOCK_SIZE_M", tile_config.get("block_m", tile_config.get("m", 16))
        )
    )
    n = int(
        tile_config.get(
            "BLOCK_SIZE_N", tile_config.get("block_n", tile_config.get("n", 16))
        )
    )
    k = int(
        tile_config.get(
            "BLOCK_SIZE_K", tile_config.get("block_k", tile_config.get("k", 16))
        )
    )
    tile_values = tuple(_ssa_tile_values(ssa_graph))
    # Model operand traffic (not the already-promoted result): low precision
    # matmul inputs remain FP16/BF16 while the accumulator is always FP32.
    # Prefer the smallest floating tensor dtype in the SSA tile; this avoids
    # charging an FP32 output twice while retaining a conservative 4-byte
    # model for all-FP32 kernels.
    floating = [
        sizes[normalize_ascend_dtype(getattr(value.type, "dtype", None))]
        for value in tile_values
        if normalize_ascend_dtype(getattr(value.type, "dtype", None))
        in {"float16", "bfloat16", "float32"}
    ]
    dtype_bytes = min(floating) if floating else 2
    # A and B input tiles plus the output accumulator, all resident for the
    # duration of a pipelined dot operation.
    input_output = (m * k + k * n) * dtype_bytes + (m * n) * 4
    schedule = dict(getattr(ssa_graph, "metadata", {}).get("schedule", {}))
    if schedule.get("ascend_attention_loop"):
        # Online softmax retains an FP32 score/accumulator tile plus an
        # explicit boolean causal predicate.  Model both allocations rather
        # than treating the dot as an isolated GEMM.
        input_output += m * n * (4 + 1)
        # Online softmax keeps score, probability, value accumulation and
        # reduction temporaries live across the structured branch.  Include
        # those FP32 tiles so the planner cannot admit a source tile that
        # BiShengIR later rejects for UB overflow.
        input_output += 3 * (m * n * 4)
    return input_output * max(1, int(pipeline_factor))


def _ascend_workspace_bytes(tile_config: Mapping[str, Any]) -> int:
    """Return the FP32 accumulator workspace represented by an M/N tile."""
    m = int(tile_config.get("BLOCK_SIZE_M", tile_config.get("block_m", 16)))
    n = int(tile_config.get("BLOCK_SIZE_N", tile_config.get("block_n", 16)))
    return m * n * 4


def _retile_ascend_online_softmax(
    kernel: Kernel, schedule: Mapping[str, Any]
) -> Kernel:
    """Specialize public online-softmax loop blocks to an NPU-safe tile.

    This is keyed by the target-private schedule contract, rather than an
    application/kernel name.  The public IR remains unchanged; only the two
    existing dynamic tile meta symbols are substituted before Ascend emission.
    """
    if not schedule.get("ascend_attention_loop") or kernel.ssa is None:
        return kernel

    attention_plan = schedule["ascend_attention_plan"]
    resource_plan = attention_plan["resource_plan"]
    selected_tile = _selected_attention_tile(schedule)

    provenance = dict(schedule.get("ascend_tile_provenance", {}))
    solved = {
        "block_m": selected_tile["m"],
        "block_n": selected_tile["n"],
        "block_k": selected_tile["k"],
    }
    defaults = dict(kernel.metadata.get("meta_defaults", {}))
    values = {}
    block_values = (solved["block_m"], solved["block_n"])
    for name in defaults:
        normalized = str(name).lower()
        for index, value in enumerate(block_values):
            if normalized.endswith(f"block_size_{index}"):
                defaults[name] = int(value)
                values[name] = int(value)

    if not values:
        return kernel

    from ninetoothed.compiler.specialization import (
        specialize_program,
        specialize_tensor_specs,
    )

    updated_schedule = dict(schedule) | {
        "ascend_tile_override": dict(solved),
        "ascend_attention_tile": dict(solved),
        "tile": dict(schedule.get("tile", {}))
        | {
            "block_m": solved["block_m"],
            "block_n": solved["block_n"],
            "block_k": solved["block_k"],
        },
        "ascend_ub_plan": {
            "estimated_peak_bytes": resource_plan["ub_estimated_peak_bytes"],
            "safety_margin_bytes": max(
                0,
                int(resource_plan["ub_budget_bytes"])
                - int(resource_plan["ub_estimated_peak_bytes"]),
            ),
            "budget_bytes": resource_plan["ub_budget_bytes"],
            "budget_fraction": ASCEND_UB_ATTENTION_FRACTION,
        },
    }
    program = specialize_program(
        replace(
            kernel.ssa,
            metadata=dict(kernel.ssa.metadata) | {"schedule": updated_schedule},
        ),
        values,
    )
    return replace(
        kernel,
        ssa=program,
        metadata=dict(kernel.metadata) | {"meta_defaults": defaults},
    )


def _structured_retile_ascend_attention(
    kernel: Kernel, provenance: Mapping[str, Any]
) -> Kernel:
    """Apply the conservative, contract-driven Attention SSA retile.

    Only the recognized ``ascend_attention_loop`` shape is admitted.  All
    changed dimensions come from provenance roles, so fixed dimensions such as
    ``head_dim=64`` cannot be mistaken for a tile dimension.
    """
    if kernel.ssa is None:
        return kernel
    schedule = dict(kernel.ssa.metadata.get("schedule", {}))
    contract = schedule.get("ascend_attention_loop")
    if not isinstance(contract, Mapping) or contract.get("kind") != "generic-online-softmax-loop":
        raise UnsupportedBackendOpError(
            "Ascend structured retile received an unrecognized attention contract.",
            reason="only generic-online-softmax-loop SSA contracts are supported.",
            suggestion="lower the operation through the verified online-softmax contract.",
        )

    roles = {
        str(entry.get("role")): entry
        for entry in tuple(provenance.get("entries", ()))
        if isinstance(entry, Mapping)
    }
    required = ("matmul-m", "matmul-n", "reduction-k")
    if any(role not in roles for role in required):
        raise UnsupportedBackendOpError(
            "Ascend attention structured retile requires M/N/K tile roles.",
            reason="the SSA contract does not identify all three retile dimensions.",
            suggestion="record matmul-m, matmul-n, and reduction-k provenance.",
        )

    resolved = {
        role: int(roles[role].get("resolved"))
        for role in required
        if roles[role].get("resolved") is not None
    }
    selected_tile = _selected_attention_tile(schedule)
    selected_roles = {
        "matmul-m": selected_tile["m"],
        "matmul-n": selected_tile["n"],
        "reduction-k": selected_tile["k"],
    }
    if resolved != selected_roles:
        raise UnsupportedBackendOpError(
            "Ascend Attention structured retile does not match the resource plan.",
            reason=f"provenance tile roles={resolved!r}, resource tile={selected_roles!r}.",
            suggestion="derive every structured retile dimension from the canonical resource plan.",
        )
    supported = tuple(_ASCEND_ATTENTION_TILE_CANDIDATES)
    if (selected_tile["m"], selected_tile["n"], selected_tile["k"]) not in supported:
        raise UnsupportedBackendOpError(
            "Ascend Attention selected a tile without a verified structured lowering.",
            reason=f"selected tile={selected_tile!r}, verified candidates={supported!r}.",
            suggestion="add and verify the complete SSA retile before admitting this tile.",
        )

    source_shapes = {
        spec.name: tuple(spec.attrs.get("source_shape", spec.shape))
        for spec in kernel.tensors
        if getattr(spec, "ndim", 0) >= 2
    }
    sequence, head_dim = _attention_sequence_and_head_dim(source_shapes)
    if head_dim is not None and head_dim != 64:
        raise UnsupportedBackendOpError(
            "Ascend structured retile supports head_dim=64 and sequence<=1024 only.",
            reason=f"received head_dim={head_dim!r}, sequence={sequence!r}.",
            suggestion="use the verified small-shape Attention contract.",
        )
    if sequence is not None and sequence > 1024:
        raise UnsupportedBackendOpError(
            "Ascend structured retile supports sequence<=1024 only.",
            reason=f"received sequence={sequence!r}.",
            suggestion="use the bounded sequence-tiled Attention contract.",
        )

    q_source_shape = source_shapes.get("q")
    if q_source_shape is None or len(q_source_shape) != 4:
        raise UnsupportedBackendOpError(
            "Ascend Attention retile requires rank-4 Q source provenance.",
            reason=f"source shapes={source_shapes!r}.",
            suggestion="preserve Q batch/head/sequence/head_dim metadata through lowering.",
        )
    batch_expr, heads_expr = str(q_source_shape[0]), str(q_source_shape[1])
    sequence_expr, head_dim_expr = str(q_source_shape[-2]), str(q_source_shape[-1])

    loop_paths = []
    values = {}
    for entry in roles.values():
        parameter = str(entry.get("parameter", ""))
        if parameter:
            role = str(entry.get("role"))
            tile_axis = {
                "matmul-m": "m",
                "matmul-n": "n",
                "reduction-k": "k",
            }.get(role)
            if tile_axis is not None:
                values[parameter] = selected_tile[tile_axis]
    for name in _attention_tile_symbols(kernel.ssa) + _attention_tensor_tile_symbols(
        kernel.tensors
    ):
        for symbol in re.findall(r"[A-Za-z_][A-Za-z0-9_]*BLOCK_SIZE_[012]", str(name)):
            tile_index = int(symbol.rsplit("BLOCK_SIZE_", 1)[1])
            values[symbol] = selected_tile[("m", "n", "k")[tile_index]]

    if not values:
        raise UnsupportedBackendOpError(
            "Ascend attention structured retile found no symbolic tile dimensions.",
            reason="the recognized contract has no SSA tile symbols to rewrite.",
            suggestion="preserve tile symbols until the Ascend private retile hook.",
        )

    for block in kernel.ssa.blocks:
        for operation in _walk_operations((block,)):
            if operation.opcode == "scf.for":
                loop_paths.append(operation)
    if len(loop_paths) != 1 or len(loop_paths[0].regions) != 1:
        raise UnsupportedBackendOpError(
            "Ascend attention structured retile requires one structured scf.for.",
            reason=f"found {len(loop_paths)} candidate loops.",
            suggestion="split unsupported control flow before Ascend lowering.",
        )

    from ninetoothed.compiler.specialization import (
        specialize_program,
        specialize_tensor_specs,
    )

    program = specialize_program(kernel.ssa, values)
    attention_sequence = sequence if sequence is not None else sequence_expr
    attention_head_dim = head_dim if head_dim is not None else head_dim_expr
    dot_tiles = _attention_dot_tile_contracts(
        selected_tile,
        sequence=attention_sequence,
        head_dim=attention_head_dim,
    )
    program = _annotate_ascend_attention_dot_tiles(program, dot_tiles)
    # Retile predicates that feed the QK score select to the score tile shape.
    # Q/K access tensors retain head_dim=64, while the score mask is indexed
    # over query M and key N.  Without this private rewrite Triton sees a
    # (M,64) predicate guarding an (M,N) score and rejects the source during
    # JIT/AOT compilation.
    score_shape = (str(selected_tile["m"]), str(selected_tile["n"]))
    score_results = {
        operation.results[0].name
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "linalg.dot"
        for _ in (0,)
    }
    score_dot_name = next(iter(score_results), None)
    producer_map = {
        result.name: operation
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    }
    score_dot = next(
        operation for operation in _walk_operations(program.blocks)
        if operation.opcode == "linalg.dot"
    )
    score_dot_name = score_dot.results[0].name
    mask_values = set()
    predicate_values = set()
    for operation in _walk_operations(program.blocks):
        if operation.opcode == "select.where" and len(operation.operands) == 3:
            if operation.operands[1] == score_dot_name or operation.operands[1] in mask_values:
                mask_values.add(operation.results[0].name)
                predicate_values.add(operation.operands[0])
    score_predicate_operations = []
    pending_predicates = list(predicate_values)
    seen_predicates = set()
    while pending_predicates:
        predicate = pending_predicates.pop()
        if predicate in seen_predicates:
            continue
        seen_predicates.add(predicate)
        defining = producer_map.get(predicate)
        if defining is None:
            continue
        if defining.opcode == "cmp.lt":
            score_predicate_operations.append(defining)
        pending_predicates.extend(defining.operands)
    # The bounds-valid normalization introduces a boolean select whose value
    # is the predicate itself.  Its producer and the comparison feeding it
    # must share the score (M,N) shape for Triton broadcasting.
    for operation in _walk_operations(program.blocks):
        if operation.results and operation.results[0].name in predicate_values:
            predicate_values.add(operation.results[0].name)
    rewritten_mask = program
    for name in (*mask_values, *predicate_values):
        defining = producer_map.get(name)
        if defining is None or not defining.results:
            continue
        result = defining.results[0]
        if len(tuple(result.type.shape)) != 2:
            continue
        updated = replace(result, type=replace(result.type, shape=score_shape))
        rewritten_mask = _replace_operation(
            rewritten_mask,
            defining,
            replace(defining, results=(updated, *defining.results[1:])),
        )
    program = rewritten_mask
    for operation in score_predicate_operations:
        attrs = dict(operation.attrs)
        attrs.update(
            {
                "ascend_attention_mask": "score-key-bounds",
                "score_shape": score_shape,
                "sequence": str(sequence if sequence is not None else sequence_expr),
            }
        )
        program = _replace_operation(program, operation, replace(operation, attrs=attrs))
    # Mark the original K bounds offset as a score-domain operation.  The
    # private Ascend emitter then uses the score N coordinate for this mask;
    # Q/K loads keep their original head-dimension access templates.
    producer_map = {
        result.name: operation
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    }
    score_mask_names = set(predicate_values)
    marked = program
    for name in tuple(score_mask_names):
        defining = producer_map.get(name)
        if defining is None or defining.opcode != "cmp.lt":
            continue
        cmp_attrs = dict(defining.attrs)
        cmp_attrs.update(
            {
                "ascend_attention_mask": "score-key-bounds",
                "score_shape": score_shape,
                "sequence": str(sequence if sequence is not None else sequence_expr),
            }
        )
        marked = _replace_operation(marked, defining, replace(defining, attrs=cmp_attrs))
        pending = list(defining.operands)
        seen = set()
        while pending:
            operand = pending.pop()
            if operand in seen:
                continue
            seen.add(operand)
            producer = producer_map.get(operand)
            if producer is None:
                continue
            if producer.opcode == "index.offset":
                attrs = dict(producer.attrs)
                attrs.update(
                    {
                        "ascend_attention_mask": "score-key-bounds",
                        "score_shape": score_shape,
                        "sequence": str(sequence if sequence is not None else sequence_expr),
                    }
                )
                marked = _replace_operation(marked, producer, replace(producer, attrs=attrs))
            pending.extend(producer.operands)
    # Preserve the score-mask contract on identity selects as they are copied
    # into nested loop/if regions.  Region-local emit contexts may not contain
    # the original comparison producer, so the select itself must carry the
    # immutable provenance.
    producer_map = {
        result.name: operation
        for operation in _walk_operations(marked.blocks)
        for result in operation.results
    }
    for operation in _walk_operations(marked.blocks):
        if operation.opcode == "cmp.ge":
            attrs = dict(operation.attrs)
            attrs.update({
                "ascend_attention_mask": "score-key-bounds",
                "score_mask_role": "qk-score-bounds",
                "score_shape": score_shape,
                "sequence": str(sequence if sequence is not None else sequence_expr),
            })
            marked = _replace_operation(marked, operation, replace(operation, attrs=attrs))
            continue
        if operation.opcode != "select.where" or len(operation.operands) != 3:
            continue
        condition = producer_map.get(operation.operands[0])
        if condition is None:
            continue
        tagged = condition.attrs.get("score_mask_role") == "qk-score-bounds" or condition.attrs.get("ascend_attention_mask") == "score-key-bounds"
        identity = operation.operands[1] == operation.operands[2]
        chain_tagged = tagged
        pending_chain = [operation.operands[0]]
        seen_chain = set()
        while pending_chain and not chain_tagged:
            value = pending_chain.pop()
            if value in seen_chain:
                continue
            seen_chain.add(value)
            producer = producer_map.get(value)
            if producer is None:
                continue
            attrs = producer.attrs
            chain_tagged = attrs.get("score_mask_role") == "qk-score-bounds" or attrs.get("ascend_attention_mask") == "score-key-bounds"
            pending_chain.extend(producer.operands)
        if not (tagged or identity and chain_tagged):
            continue
        attrs = dict(operation.attrs)
        attrs.update({
            "ascend_attention_mask": "bounds_valid",
            "score_mask_role": "qk-score-bounds",
            "score_shape": score_shape,
            "sequence": str(sequence if sequence is not None else sequence_expr),
        })
        marked = _replace_operation(marked, operation, replace(operation, attrs=attrs))
    # Causal query/key comparisons are score-domain predicates as well.  Keep
    # their provenance on the comparison so nested emission can retile both
    # axes without touching Q/K access templates.
    producer_map = {
        result.name: operation
        for operation in _walk_operations(marked.blocks)
        for result in operation.results
    }
    for operation in _walk_operations(marked.blocks):
        if operation.opcode != "select.where" or len(operation.operands) != 3:
            continue
        predicate = producer_map.get(operation.operands[0])
        if predicate is None or predicate.opcode != "cmp.ge" or operation.operands[1] not in mask_values:
            continue
        attrs = dict(predicate.attrs)
        attrs.update({
            "ascend_attention_mask": "score-key-bounds",
            "score_mask_role": "qk-score-bounds",
            "score_shape": score_shape,
            "sequence": str(sequence if sequence is not None else sequence_expr),
        })
        marked = _replace_operation(marked, predicate, replace(predicate, attrs=attrs))
    program = marked
    loop = next(
        operation
        for block in program.blocks
        for operation in _walk_operations((block,))
        if operation.opcode == "scf.for"
    )
    m, n, k = (selected_tile[axis] for axis in ("m", "n", "k"))
    # Match the remaining loop-local zero/empty tensors to their semantic
    # roles.  These placeholders are created by the generic arrangement and
    # otherwise retain 64-wide physical shapes despite the retiled dots.
    value_width = str(head_dim if head_dim is not None else head_dim_expr)
    def local_shape(result):
        old = tuple(str(dim) for dim in result.type.shape)
        if old == ("64", "64"):
            lowered = result.name.lower()
            return (str(m), str(n)) if "qk" in lowered else (str(m), value_width)
        if old == ("64",):
            lowered = result.name.lower()
            return (str(n),) if "key_valid" in lowered or "bounds" in lowered else (str(m),)
        return old
    for operation in _walk_operations(program.blocks):
        updated = []
        changed = False
        for result in operation.results:
            shape = local_shape(result)
            if shape != tuple(str(dim) for dim in result.type.shape):
                updated.append(replace(result, type=replace(result.type, shape=shape)))
                changed = True
            else:
                updated.append(result)
        if changed:
            program = _replace_operation(program, operation, replace(operation, results=tuple(updated)))
    loop = next(
        operation
        for block in program.blocks
        for operation in _walk_operations((block,))
        if operation.opcode == "scf.for"
    )
    # Retile online-softmax loop-carried tensor values to the physical plan.
    # The public arrangement uses 64-wide placeholders; keeping those types
    # after the QK/PV dot rewrite needlessly allocates a 64x64 UB accumulator.
    state_shapes = {
        0: (str(m), str(head_dim if head_dim is not None else head_dim_expr)),
        1: (str(m),),
        2: (str(m),),
    }
    state_names = set()
    iter_args = tuple(loop.attrs.get("iter_args", ()))
    for index, item in enumerate(iter_args):
        if index >= 3:
            break
        for key in ("initial", "block_arg", "name"):
            value = item.get(key) if isinstance(item, Mapping) else None
            if value:
                state_names.add(str(value))
        if index < len(loop.results):
            state_names.add(loop.results[index].name)
    for operation in _walk_operations(program.blocks):
        if not operation.results:
            continue
        updated_results = list(operation.results)
        changed = False
        for result_index, result in enumerate(operation.results):
            if result.name not in state_names:
                continue
            state_index = next((index for index, item in enumerate(iter_args) if any(result.name == str(item.get(key)) for key in ("initial", "block_arg", "name"))), None)
            if state_index is None:
                state_index = next((index for index, loop_result in enumerate(loop.results) if loop_result.name == result.name), None)
            if state_index is None or state_index not in state_shapes:
                continue
            updated_results[result_index] = replace(result, type=replace(result.type, shape=state_shapes[state_index]))
            changed = True
        if changed:
            program = _replace_operation(program, operation, replace(operation, results=tuple(updated_results)))
    loop = next(
        operation
        for block in program.blocks
        for operation in _walk_operations((block,))
        if operation.opcode == "scf.for"
    )
    loop_attrs = dict(loop.attrs)
    m, n, k = (selected_tile[axis] for axis in ("m", "n", "k"))
    query_tiles = (
        (sequence + m - 1) // m if sequence is not None else f"ceil_div({sequence_expr}, {m})"
    )
    key_tiles = (
        (sequence + n - 1) // n if sequence is not None else f"ceil_div({sequence_expr}, {n})"
    )
    query_upper = (
        f"(({sequence} + {m - 1}) // {m})"
        if sequence is not None
        else f"ceil_div({sequence_expr}, {m})"
    )
    key_upper = (
        f"(({sequence} + {n - 1}) // {n})"
        if sequence is not None
        else f"ceil_div({sequence_expr}, {n})"
    )
    try:
        batch_value, heads_value = int(batch_expr), int(heads_expr)
        grid_expr = (
            f"{batch_value} * {heads_value} * "
            f"triton.cdiv({sequence if sequence is not None else sequence_expr}, {m})"
        )
        grid_value = (
            batch_value * heads_value * query_tiles
            if isinstance(query_tiles, int)
            else f"{batch_value} * {heads_value} * ({query_tiles})"
        )
    except (TypeError, ValueError):
        grid_expr = (
            f"({batch_expr}) * ({heads_expr}) * "
            f"triton.cdiv({sequence if sequence is not None else sequence_expr}, {m})"
        )
        grid_value = f"({batch_expr}) * ({heads_expr}) * ({query_tiles})"
    loop_attrs["ascend_attention_retile"] = {
        "status": "retiled",
        "block_m": m,
        "block_n": n,
        "block_k": k,
        "sequence": sequence if sequence is not None else sequence_expr,
        "head_dim": head_dim if head_dim is not None else head_dim_expr,
        "query_tiles": query_tiles,
        "key_tiles": key_tiles,
        "query_loop_upper": query_upper,
        "key_loop_upper": key_upper,
        "loop_lower": "0",
        "loop_upper": key_upper,
        "loop_step": "1",
        "grid": grid_expr,
        "dot_tiles": dot_tiles,
    }
    preceding = ()
    loop_operands = list(loop.operands)
    existing_names = {value.name for value in (*program.inputs, *program.outputs)}
    existing_names.update(
        argument.name
        for block in program.blocks
        for operation in _walk_operations((block,))
        for region in operation.regions
        for argument in region.args
    )
    existing_names.update(
        result.name
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    )

    def fresh_scalar(prefix: str) -> ssa.Value:
        name = f"%ascend_attention_{prefix}"
        suffix = 0
        while name in existing_names:
            suffix += 1
            name = f"%ascend_attention_{prefix}_{suffix}"
        existing_names.add(name)
        return ssa.Value(
            name=name,
            type=ssa.Type(kind="scalar", dtype="int64"),
        )

    if sequence is not None:
        upper = fresh_scalar("key_tiles")
        loop_operands[1] = upper.name
        preceding = (
            ssa.Operation(
                opcode="arith.constant",
                results=(upper,),
                attrs={"value": key_tiles},
            ),
        )
    else:
        access = contract.get("access_provenance", {})
        key_entry = access.get("k", {}) if isinstance(access, Mapping) else {}
        key_source = key_entry.get("tensor") if isinstance(key_entry, Mapping) else None
        if not key_source or key_source not in {value.name for value in program.inputs}:
            raise UnsupportedBackendOpError(
                "Ascend Attention dynamic key loop has no K source dimension.",
                reason=f"K access provenance={key_entry!r}.",
                suggestion="derive the key loop bound from the verified K source shape.",
            )
        sequence_value = fresh_scalar("key_sequence")
        rounding_value = fresh_scalar("key_sequence_rounded")
        rounding_amount = fresh_scalar("key_tile_rounding")
        divisor = fresh_scalar("key_tile_divisor")
        upper = fresh_scalar("key_tiles")
        preceding = (
            ssa.Operation(
                opcode="shape.dim",
                operands=(str(key_source),),
                results=(sequence_value,),
                attrs={
                    "dim": -2,
                    "source": True,
                    "ascend_attention_retile": "key-sequence",
                },
            ),
            ssa.Operation(
                opcode="arith.constant",
                results=(rounding_amount,),
                attrs={"value": n - 1},
            ),
            ssa.Operation(
                opcode="arith.add",
                operands=(sequence_value.name, rounding_amount.name),
                results=(rounding_value,),
                attrs={"ascend_attention_retile": "ceil-div-numerator"},
            ),
            ssa.Operation(
                opcode="arith.constant",
                results=(divisor,),
                attrs={"value": n},
            ),
            ssa.Operation(
                opcode="arith.floordiv",
                operands=(rounding_value.name, divisor.name),
                results=(upper,),
                attrs={"ascend_attention_retile": "key-loop-upper"},
            ),
        )
        loop_operands[1] = upper.name
    rewritten_loop = replace(
        loop,
        operands=tuple(loop_operands),
        attrs=loop_attrs,
    )
    program = _replace_operation(program, loop, rewritten_loop, preceding=preceding)
    updated_schedule = dict(program.metadata.get("schedule", {}))
    updated_schedule["ascend_attention_retile"] = dict(loop_attrs["ascend_attention_retile"])
    updated_schedule["ascend_tile_provenance"] = dict(provenance)
    attention_plan = dict(updated_schedule.get("ascend_attention_plan", {}))
    attention_plan["tile"] = dict(selected_tile)
    attention_plan["query_tiles"] = query_tiles
    attention_plan["key_tiles"] = key_tiles
    attention_plan["grid"] = grid_value
    attention_plan["dot_tiles"] = dot_tiles
    attention_plan["loop"] = {"lower": "0", "upper": key_upper, "step": 1}
    updated_schedule["ascend_attention_plan"] = attention_plan
    program = replace(program, metadata=dict(program.metadata) | {"schedule": updated_schedule})
    tensors = specialize_tensor_specs(kernel.tensors, values)
    # Keep the private access-template metadata synchronized with the physical
    # dot tile.  Source/logical rank-4 shapes remain unchanged; only the
    # matrix tile used for Q/O versus K/V is retiled.
    retiled_specs = []
    for spec in tensors:
        role = str(spec.name)
        leading = m if role in {"q", "o"} else n if role in {"k", "v"} else None
        if leading is None:
            retiled_specs.append(spec)
            continue
        attrs = dict(spec.attrs)
        dtype_shapes = list(attrs.get("dtype_shapes", ()))
        if dtype_shapes and len(tuple(dtype_shapes[-1])) == 2:
            old_shape = tuple(dtype_shapes[-1])
            dtype_shapes[-1] = (str(leading), str(old_shape[-1]))
            attrs["dtype_shapes"] = tuple(dtype_shapes)
        templates = []
        for template in tuple(attrs.get("access_templates", ())):
            if not isinstance(template, Mapping) or len(tuple(template.get("shape", ()))) != 2:
                templates.append(template)
                continue
            updated = dict(template)
            old_shape = tuple(updated.get("shape", ()))
            updated["shape"] = (str(leading), str(old_shape[-1]))
            # The access-template frontend also bakes the number of matrix
            # tiles into the batch/head/query coordinate decoder.  Changing
            # only the per-tile stride leaves the decoder on the old 64-wide
            # tile count, so program ids after the first old tile are decoded
            # as a different head.  Rewrite that count together with the
            # selected physical M/N tile.
            if sequence is not None and old_shape and str(old_shape[0]).isdigit():
                old_leading = int(old_shape[0])
                old_count = (
                    f"(({sequence} - {old_leading - 1} - 1 + "
                    f"{old_leading} - 1) // {old_leading} + 1)"
                )
                old_count_core = (
                    f"({sequence} - {old_leading - 1} - 1 + "
                    f"{old_leading} - 1) // {old_leading} + 1"
                )
                new_count = (
                    f"(({sequence} - {int(leading) - 1} - 1 + "
                    f"{int(leading)} - 1) // {int(leading)} + 1)"
                )
                new_count_core = (
                    f"({sequence} - {int(leading) - 1} - 1 + "
                    f"{int(leading)} - 1) // {int(leading)} + 1"
                )
                query_count = (
                    f"(({sequence} - {int(m) - 1} - 1 + "
                    f"{int(m)} - 1) // {int(m)} + 1)"
                )
                query_count_core = (
                    f"({sequence} - {int(m) - 1} - 1 + "
                    f"{int(m)} - 1) // {int(m)} + 1"
                )
                key_count = (
                    f"(({sequence} - {int(n) - 1} - 1 + "
                    f"{int(n)} - 1) // {int(n)} + 1)"
                )
                key_count_core = (
                    f"({sequence} - {int(n) - 1} - 1 + "
                    f"{int(n)} - 1) // {int(n)} + 1"
                )
                is_key_template = role in {"k", "v"}

                def rewrite_tile_count(value):
                    if isinstance(value, str):
                        replacement = query_count if is_key_template else new_count
                        replacement_core = (
                            query_count_core if is_key_template else new_count_core
                        )
                        return value.replace(old_count, replacement).replace(
                            old_count_core, replacement_core
                        )
                    if isinstance(value, (tuple, list)):
                        return type(value)(rewrite_tile_count(item) for item in value)
                    return value

                for field, value in tuple(updated.items()):
                    if field in {"linear_offset", "offsets", "mask", "batch_offset"}:
                        updated[field] = rewrite_tile_count(value)
                if is_key_template:
                    # The third source dimension is the looped key axis.  Its
                    # bound uses N tiles, while the first two dimensions still
                    # decode launch ids with the query M tile count.
                    offsets = updated.get("offsets")
                    if isinstance(offsets, (tuple, list)) and len(offsets) > 2:
                        rewritten_offsets = list(offsets)
                        rewritten_offsets[2] = str(rewritten_offsets[2]).replace(
                            query_count, key_count
                        ).replace(query_count_core, key_count_core)
                        updated["offsets"] = type(offsets)(rewritten_offsets)
                    mask = updated.get("mask")
                    if isinstance(mask, str):
                        updated["mask"] = re.sub(
                            rf"(\(\s*\(?[A-Za-z_][A-Za-z0-9_]*\)?\s*<\s*)"
                            rf"{re.escape(query_count_core)}",
                            rf"\g<1>{key_count_core}",
                            mask,
                        )
            # The frontend linearizes the original fixed arrangement before
            # the resource plan selects its physical M/N tile.  Rewrite only
            # the first matrix-axis tile stride; the feature-axis stride
            # remains head_dim and must stay 64.
            for field in ("linear_offset", "offsets", "mask"):
                value = updated.get(field)
                if isinstance(value, str):
                    updated[field] = re.sub(
                        r"\*\s*64\s*\+\s*value_0\b",
                        f"* {leading} + value_0",
                        value,
                    )
                elif isinstance(value, (tuple, list)):
                    updated[field] = tuple(
                        re.sub(
                            r"\*\s*64\s*\+\s*value_0\b",
                            f"* {leading} + value_0",
                            str(item),
                        )
                        for item in value
                    )
            mask = str(updated.get("mask", "True"))
            if mask != "True":
                # Keep source-dimension bounds intact.  The emitter derives
                # the local tile bounds from the retiled template shape; a
                # blanket replacement here would turn a global query bound
                # into ``< M`` and mask every query tile after the first.
                updated["mask"] = mask
            updated["retiled_tile_mask"] = f"(value_0 < {int(leading)})"
            templates.append(updated)
        if templates:
            attrs["access_templates"] = tuple(templates)
        retiled_specs.append(replace(spec, attrs=attrs))
    tensors = tuple(retiled_specs)
    retiled_kernel = replace(
        kernel,
        tensors=tensors,
        ssa=program,
    )
    _verify_ascend_attention_retile_plan(retiled_kernel, selected_tile)

    retiled_schedule = dict(program.metadata.get("schedule", {}))
    retiled_contract = _ascend_attention_loop_contract(program)
    if not isinstance(retiled_contract, Mapping) or retiled_contract.get("kind") != "generic-online-softmax-loop":
        raise UnsupportedBackendOpError(
            "Ascend Attention retile did not preserve its structured SSA contract.",
            reason=f"reconstructed contract={retiled_contract!r}.",
            suggestion="preserve both dots, masks, causal predicate, and loop state through retile.",
        )
    retiled_schedule["ascend_attention_loop"] = retiled_contract
    verified_program = replace(
        program,
        metadata=dict(program.metadata) | {"schedule": retiled_schedule},
    )
    ssa.verify_program(verified_program)
    semantics = verify_ascend_attention_mask_semantics(verified_program)
    if not isinstance(semantics, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention retile could not reverify mask semantics.",
            reason="the post-retile verifier returned no semantic proof.",
            suggestion="keep the verified Attention contract attached to the retiled SSA.",
        )
    retiled_schedule["ascend_attention_mask_semantics"] = semantics
    verified_program = replace(
        verified_program,
        metadata=dict(verified_program.metadata) | {"schedule": retiled_schedule},
    )
    return replace(retiled_kernel, ssa=verified_program)


def _attention_dot_tile_contracts(
    selected_tile: Mapping[str, int], *, sequence: Any, head_dim: Any
) -> Mapping[str, Mapping[str, Any]]:
    """Describe each Attention dot using the planner's M/N/K meanings.

    Attention uses N for the key tile and K for the QK head-dimension
    reduction tile.  The PV dot therefore contracts over N and produces the
    full head dimension, while QK contracts over head_dim in K-sized pieces.
    The logical tensor extents remain intact so the complete head dimension
    is always covered.
    """
    m, n, k = (int(selected_tile[axis]) for axis in ("m", "n", "k"))
    m_text, n_text, k_text = str(m), str(n), str(k)
    sequence_text, head_dim_text = str(sequence), str(head_dim)
    try:
        head_dim_value = int(head_dim)
        qk_reduction_tiles: int | str = (head_dim_value + k - 1) // k
    except (TypeError, ValueError):
        qk_reduction_tiles = f"ceil_div({head_dim_text}, {k})"

    query_tail = f"query_tile_index * {m} + query_lane < {sequence_text}"
    key_tail = f"key_tile_index * {n} + key_lane < {sequence_text}"
    feature_tail = f"feature_lane < {head_dim_text}"
    reduction_tail = (
        f"reduction_tile_index * {k} + reduction_lane < {head_dim_text}"
    )
    return {
        "qk": {
            "role": "qk",
            "resource_tile": {"m": m, "n": n, "k": k},
            "tile_shapes": {
                "lhs": (m_text, k_text),
                "rhs": (k_text, n_text),
                "result": (m_text, n_text),
            },
            "logical_shapes": {
                "lhs": (m_text, head_dim_text),
                "rhs": (head_dim_text, n_text),
                "result": (m_text, n_text),
            },
            "reduction": {
                "extent": head_dim_value if isinstance(head_dim, int) else head_dim_text,
                "tile_extent": k,
                "tile_count": qk_reduction_tiles,
                "tail_mask": reduction_tail,
            },
            "tail_masks": {
                "query": query_tail,
                "key": key_tail,
                "reduction": reduction_tail,
            },
        },
        "pv": {
            "role": "pv",
            "resource_tile": {"m": m, "n": n, "k": k},
            "tile_shapes": {
                "lhs": (m_text, n_text),
                "rhs": (n_text, head_dim_text),
                "result": (m_text, head_dim_text),
            },
            "logical_shapes": {
                "lhs": (m_text, n_text),
                "rhs": (n_text, head_dim_text),
                "result": (m_text, head_dim_text),
            },
            "reduction": {
                "extent": n,
                "tile_extent": n,
                "tile_count": 1,
                "tail_mask": key_tail,
            },
            "tail_masks": {
                "query": query_tail,
                "key": key_tail,
                "feature": feature_tail,
            },
        },
    }


def _annotate_ascend_attention_dot_tiles(
    program: ssa.Program,
    dot_tiles: Mapping[str, Mapping[str, Any]],
) -> ssa.Program:
    """Attach role-specific planned tile contracts to the QK and PV dots."""
    dots = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "linalg.dot"
    )
    if len(dots) != 2:
        raise UnsupportedBackendOpError(
            "Ascend Attention retile requires both planned dot operations.",
            reason=f"found {len(dots)} linalg.dot operations before dot retile.",
            suggestion="preserve the QK and PV dots in the verified Attention loop.",
        )
    rewritten = program
    for operation, role in zip(dots, ("qk", "pv"), strict=True):
        attrs = dict(operation.attrs)
        dot_tile = dict(dot_tiles[role])
        resource_tile = dict(dot_tile["resource_tile"])
        attrs["ascend_attention_dot_tile"] = dot_tile
        attrs["ascend_attention_resource_tile"] = resource_tile
        attrs.update(
            {
                "block_m": resource_tile["m"],
                "block_n": resource_tile["n"],
                "block_k": resource_tile["k"],
            }
        )
        rewritten = _replace_operation(
            rewritten,
            operation,
            ssa.Operation(
                opcode=operation.opcode,
                operands=operation.operands,
                results=operation.results,
                attrs=attrs,
                regions=operation.regions,
            ),
        )
    # The generic lowering keeps logical 64x64 tensor types after tile-symbol
    # specialization.  Ascend's private retile must make the physical QK/PV
    # dot tiles explicit in SSA so the verifier and emitter consume the same
    # M/N/K plan.  Rewrite only the two dot result types and their direct
    # operands; masks and logical extents remain in operation metadata.
    dots = tuple(
        operation
        for operation in _walk_operations(rewritten.blocks)
        if operation.opcode == "linalg.dot"
    )
    for operation, role in zip(dots, ("qk", "pv"), strict=True):
        tile = dot_tiles[role]["tile_shapes"]
        logical = dot_tiles[role]["logical_shapes"]
        result_shape = tuple(tile["result"])
        result = operation.results[0]
        updated_result = replace(result, type=replace(result.type, shape=result_shape))
        updated_operands = list(operation.operands)
        # The resource K role is the QK reduction *chunk*, not the complete
        # head-dimension axis of the loaded Q/K tiles.  Keep the physical
        # operand shapes compatible with the logical Attention tensors; the
        # planned K chunk remains recorded in the dot contract metadata and
        # is consumed by the target lowering.  Rewriting Q/K to K here would
        # make a (M, K) x (K, N) source tile while the access templates still
        # load (M, head_dim) x (head_dim, N), producing an invalid broadcast
        # in the subsequent softmax/value path.
        if role == "qk":
            operand_shapes = (
                (str(logical["lhs"][0]), str(logical["lhs"][1])),
                (str(logical["rhs"][0]), str(logical["rhs"][1])),
            )
        else:
            operand_shapes = (tuple(tile["lhs"]), tuple(tile["rhs"]))
        # Rewrite the defining tensor values for the two dot operands.  The
        # values remain the same SSA names; only their physical tile type is
        # changed, while logical extents stay in the dot contract metadata.
        for operand_index, operand_name in enumerate(operation.operands[:2]):
            for defining in _walk_operations(rewritten.blocks):
                for result_index, defining_result in enumerate(defining.results):
                    if defining_result.name != operand_name:
                        continue
                    new_result = replace(
                        defining_result,
                        type=replace(
                            defining_result.type,
                            shape=operand_shapes[operand_index],
                        ),
                    )
                    rewritten = _replace_operation(
                        rewritten,
                        defining,
                        replace(
                            defining,
                            results=tuple(
                                new_result if index == result_index else value
                                for index, value in enumerate(defining.results)
                            ),
                        ),
                    )
                    break
        rewritten = _replace_operation(
            rewritten,
            operation,
            replace(
                operation,
                operands=tuple(updated_operands),
                results=(updated_result,),
            ),
        )
    return rewritten


def _normalize_attention_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_attention_contract_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_attention_contract_value(item) for item in value)
    return value


def _verify_ascend_attention_retile_plan(
    kernel: Kernel, selected_tile: Mapping[str, int]
) -> None:
    """Prove that the retiled SSA shapes, masks, and bounds consume the plan."""
    if kernel.ssa is None:
        raise UnsupportedBackendOpError(
            "Ascend Attention retile verification requires SSA.",
            reason="the retiled kernel has no SSA program.",
            suggestion="preserve the structured Attention program through retile.",
        )
    program = kernel.ssa
    schedule = dict(program.metadata.get("schedule", {}))
    attention_plan = schedule.get("ascend_attention_plan", {})
    resource_plan = attention_plan.get("resource_plan", {})
    plan_tile = _attention_tile_dict(attention_plan.get("tile", {}))
    resource_tile = _attention_tile_dict(resource_plan.get("selected_tile", {}))
    selected = _attention_tile_dict(selected_tile)
    retile = schedule.get("ascend_attention_retile", {})
    if plan_tile != selected or resource_tile != selected:
        raise UnsupportedBackendOpError(
            "Ascend Attention SSA retile disagrees with the canonical resource plan.",
            reason=(
                f"resource={resource_tile!r}, plan={plan_tile!r}, "
                f"retile_input={selected!r}."
            ),
            suggestion="derive M/N/K only from resource_plan.selected_tile.",
        )
    if tuple(retile.get(f"block_{axis}") for axis in ("m", "n", "k")) != (
        selected["m"],
        selected["n"],
        selected["k"],
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention retile metadata disagrees with the resource plan.",
            reason=f"retile={retile!r}, selected_tile={selected!r}.",
            suggestion="write loop and grid metadata from the selected M/N/K values.",
        )

    types = {value.name: value.type for value in (*program.inputs, *program.outputs)}

    def collect(block):
        for argument in block.args:
            types[argument.name] = argument.type
        for operation in block.operations:
            for result in operation.results:
                types[result.name] = result.type
            for region in operation.regions:
                collect(region)

    for block in program.blocks:
        collect(block)

    def shape(value: str) -> tuple[str, ...]:
        value_type = types.get(value)
        return () if value_type is None else tuple(str(dim) for dim in value_type.shape)

    def compatible_dimension(left: str, right: str) -> bool:
        if left == right:
            return True
        try:
            return int(left) == int(right)
        except (TypeError, ValueError):
            # Distinct symbolic names can denote the same dimension; the
            # structured Q/K/V provenance verifier checks that relationship.
            return True

    dots = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "linalg.dot"
    )
    if len(dots) != 2 or any(len(operation.operands) < 2 or not operation.results for operation in dots):
        raise UnsupportedBackendOpError(
            "Ascend Attention retile requires both QK and PV dots.",
            reason=f"found {len(dots)} linalg.dot operations.",
            suggestion="preserve the two online-softmax dot operations through retile.",
        )
    score_dot, value_dot = dots
    score_lhs, score_rhs = (shape(value) for value in score_dot.operands[:2])
    score_result = tuple(str(dim) for dim in score_dot.results[0].type.shape)
    value_lhs, value_rhs = (shape(value) for value in value_dot.operands[:2])
    value_result = tuple(str(dim) for dim in value_dot.results[0].type.shape)
    m, n = str(selected["m"]), str(selected["n"])
    if not (
        len(score_lhs) == 2
        and len(score_rhs) == 2
        and score_lhs[0] == m
        and compatible_dimension(score_lhs[1], score_rhs[0])
        and score_rhs[1] == n
        and score_result == (m, n)
        and value_lhs == (m, n)
        and len(value_rhs) == 2
        and value_rhs[0] == n
        and value_result == (m, value_rhs[1])
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention QK/PV tensor shapes do not consume the resource tile.",
            reason=(
                f"QK={score_lhs}x{score_rhs}->{score_result}; "
                f"PV={value_lhs}x{value_rhs}->{value_result}; "
                f"selected M/N/K={selected!r}."
            ),
            suggestion="retile QK and PV operand/result shapes from selected M/N/K.",
        )

    expected_dot_tiles = _attention_dot_tile_contracts(
        selected,
        sequence=retile.get("sequence"),
        head_dim=retile.get("head_dim"),
    )
    recorded_dot_tiles = _normalize_attention_contract_value(
        retile.get("dot_tiles", {})
    )
    planned_dot_tiles = _normalize_attention_contract_value(
        attention_plan.get("dot_tiles", {})
    )
    if recorded_dot_tiles != expected_dot_tiles or planned_dot_tiles != expected_dot_tiles:
        raise UnsupportedBackendOpError(
            "Ascend Attention dot tiles disagree with the resource plan.",
            reason=(
                f"retile dot tiles={recorded_dot_tiles!r}, "
                f"plan dot tiles={planned_dot_tiles!r}, "
                f"expected={expected_dot_tiles!r}."
            ),
            suggestion="derive QK/PV tile shapes and reduction masks from resource_plan.selected_tile.",
        )
    for operation, role in ((score_dot, "qk"), (value_dot, "pv")):
        actual_dot_tile = _normalize_attention_contract_value(
            operation.attrs.get("ascend_attention_dot_tile", {})
        )
        resource_tile = operation.attrs.get("ascend_attention_resource_tile", {})
        try:
            actual_resource_tile = _attention_tile_dict(resource_tile)
        except (TypeError, ValueError):
            actual_resource_tile = {}
        try:
            actual_block_tile = {
                axis: int(operation.attrs.get(f"block_{axis}", -1))
                for axis in ("m", "n", "k")
            }
        except (TypeError, ValueError):
            actual_block_tile = {}
        if (
            actual_dot_tile != expected_dot_tiles[role]
            or actual_resource_tile != selected
            or actual_block_tile != selected
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention SSA dot tile was not retiled from the resource plan.",
                reason=(
                    f"{role} dot tile={actual_dot_tile!r}, "
                    f"resource tile={actual_resource_tile!r}, "
                    f"block tile={actual_block_tile!r}, "
                    f"expected={expected_dot_tiles[role]!r}."
                ),
                suggestion="attach the planned role-specific M/N/K tile to each Attention dot.",
            )

    for role, expected_axis in (("q", m), ("o", m), ("k", n), ("v", n)):
        spec = next((item for item in kernel.tensors if item.name == role), None)
        attrs = {} if spec is None else dict(spec.attrs)
        dtype_shapes = tuple(attrs.get("dtype_shapes", ()))
        if not dtype_shapes or len(tuple(dtype_shapes[-1])) != 2:
            raise UnsupportedBackendOpError(
                "Ascend Attention access tile shape is missing after retile.",
                reason=f"{role} dtype_shapes={dtype_shapes!r}.",
                suggestion="specialize Q/O with M and K/V with N from the resource plan.",
            )
        tile_shape = tuple(str(dim) for dim in dtype_shapes[-1])
        templates = tuple(attrs.get("access_templates", ()))
        matrix_templates = tuple(
            item
            for item in templates
            if isinstance(item, Mapping) and len(tuple(item.get("shape", ()))) == 2
        )
        if tile_shape[0] != expected_axis or not matrix_templates:
            raise UnsupportedBackendOpError(
                "Ascend Attention tail-mask tile does not match the resource plan.",
                reason=f"{role} tile_shape={tile_shape!r}, expected leading tile={expected_axis}.",
                suggestion="specialize each access template and tail mask using its planned tile axis.",
            )
        for template in matrix_templates:
            mask = str(template.get("mask", "True"))
            retiled_tile_mask = str(template.get("retiled_tile_mask", ""))
            template_shape = tuple(str(dim) for dim in template.get("shape", ()))
            if (
                template_shape != tile_shape
                or (
                    f"value_0 < {expected_axis}" not in mask
                    and f"value_0 < {expected_axis}" not in retiled_tile_mask
                )
                or mask == "True"
            ):
                raise UnsupportedBackendOpError(
                    "Ascend Attention access mask is stale after retile.",
                    reason=(
                        f"{role} template_shape={template_shape!r}, tile_shape={tile_shape!r}, "
                        f"expected leading tile={expected_axis}."
                    ),
                    suggestion="rebuild query/key tail masks from the selected resource tile.",
                )

    loop = next(
        (operation for operation in _walk_operations(program.blocks) if operation.opcode == "scf.for"),
        None,
    )
    if loop is None or len(loop.operands) < 3:
        raise UnsupportedBackendOpError(
            "Ascend Attention retile lost its key sequence loop.",
            reason="no scf.for bounds remain in the retiled SSA.",
            suggestion="retain the online-softmax key loop while changing tile dimensions.",
        )
    loop_contract = loop.attrs.get("ascend_attention_retile", {})
    if (
        loop_contract.get("loop_upper") != retile.get("key_loop_upper")
        or loop_contract.get("query_loop_upper") != retile.get("query_loop_upper")
        or loop_contract.get("grid") != retile.get("grid")
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention loop bounds or grid disagree with the resource plan.",
            reason=f"loop={dict(loop_contract)!r}, schedule={retile!r}.",
            suggestion="derive query/key bounds and launch grid from selected M/N.",
        )
    sequence = attention_plan.get("sequence")
    if sequence is not None:
        expected_query_tiles = (int(sequence) + selected["m"] - 1) // selected["m"]
        expected_key_tiles = (int(sequence) + selected["n"] - 1) // selected["n"]
        upper_value = types.get(loop.operands[1])
        upper_producer = next(
            (
                operation
                for operation in _walk_operations(program.blocks)
                if operation.results
                and operation.results[0].name == loop.operands[1]
            ),
            None,
        )
        if (
            upper_value is None
            or upper_producer is None
            or upper_producer.opcode != "arith.constant"
            or int(upper_producer.attrs.get("value", -1)) != expected_key_tiles
            or attention_plan.get("query_tiles") != expected_query_tiles
            or attention_plan.get("key_tiles") != expected_key_tiles
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention concrete loop bounds disagree with the selected tile.",
                reason=(
                    f"query_tiles={attention_plan.get('query_tiles')!r}, "
                    f"key_tiles={attention_plan.get('key_tiles')!r}, "
                    f"expected=({expected_query_tiles},{expected_key_tiles})."
                ),
                suggestion="rebuild the query grid and key loop upper bound from M/N.",
            )
    else:
        producers = {
            result.name: operation
            for operation in _walk_operations(program.blocks)
            for result in operation.results
        }
        upper_producer = producers.get(loop.operands[1])
        if (
            upper_producer is None
            or upper_producer.opcode != "arith.floordiv"
            or len(upper_producer.operands) != 2
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention symbolic key loop does not use the planned N tile.",
                reason=f"upper operand producer={upper_producer!r}.",
                suggestion="compute ceil_div(K.sequence, selected N) in SSA.",
            )
        numerator = producers.get(upper_producer.operands[0])
        divisor = producers.get(upper_producer.operands[1])
        access = schedule.get("ascend_attention_loop", {}).get("access_provenance", {})
        key_entry = access.get("k", {}) if isinstance(access, Mapping) else {}
        key_source = key_entry.get("tensor") if isinstance(key_entry, Mapping) else None
        sequence_dim = (
            producers.get(numerator.operands[0])
            if numerator is not None
            and numerator.opcode == "arith.add"
            and numerator.operands
            else None
        )
        rounding = (
            producers.get(numerator.operands[1])
            if numerator is not None
            and numerator.opcode == "arith.add"
            and len(numerator.operands) == 2
            else None
        )
        if (
            numerator is None
            or numerator.opcode != "arith.add"
            or sequence_dim is None
            or sequence_dim.opcode != "shape.dim"
            or sequence_dim.operands != (key_source,)
            or sequence_dim.attrs.get("dim") != -2
            or not sequence_dim.attrs.get("source")
            or divisor is None
            or divisor.opcode != "arith.constant"
            or int(divisor.attrs.get("value", -1)) != selected["n"]
            or rounding is None
            or rounding.opcode != "arith.constant"
            or int(rounding.attrs.get("value", -1)) != selected["n"] - 1
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention symbolic key loop upper bound is stale.",
                reason=(
                    f"N={selected['n']}, sequence_dim={sequence_dim!r}, "
                    f"rounding={rounding!r}, divisor={divisor!r}."
                ),
                suggestion="derive ceil_div(K.sequence, N) from resource_plan.selected_tile.",
            )


def _attention_sequence_and_head_dim(source_shapes):
    if not source_shapes:
        raise UnsupportedBackendOpError(
            "Ascend attention structured retile requires tensor source shapes.",
            reason="the source shape contract is unavailable.",
            suggestion="provide static Q/K/V source shapes before emission.",
        )
    shape = next(iter(source_shapes.values()))
    if len(shape) < 2:
        return None, None
    def static_int(value):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    return static_int(shape[-2]), static_int(shape[-1])


def _attention_tile_symbols(program: ssa.Program):
    symbols = set()
    for block in program.blocks:
        for operation in _walk_operations((block,)):
            for result in operation.results:
                symbols.update(str(dim) for dim in result.type.shape)
    return tuple(symbol for symbol in symbols if "BLOCK_SIZE_" in symbol)


def _attention_tensor_tile_symbols(tensors):
    symbols = set()
    for tensor in tensors:
        for value in tuple(tensor.shape) + tuple(tensor.attrs.get("dtype_shapes", ())):
            text = str(value)
            if "BLOCK_SIZE_" in text:
                symbols.add(text)
    return tuple(symbols)


def _replace_operation(program: ssa.Program, target, replacement, preceding=()):
    def rewrite(block):
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            if operation == target:
                operations.extend(preceding)
                operation = replacement
            else:
                operation = replace(operation, regions=regions)
            operations.append(operation)
        return replace(block, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def solve_ascend_tile_config(ssa_graph, initial_tile_config, max_ub_bytes=None):
    """Downsample the largest BLOCK_SIZE dimension until the UB budget fits."""
    if max_ub_bytes is None:
        schedule = dict(getattr(ssa_graph, "metadata", {}).get("schedule", {}))
        fraction = (
            ASCEND_UB_ATTENTION_FRACTION
            if schedule.get("ascend_attention_loop")
            else ASCEND_UB_NORMAL_FRACTION
        )
        max_ub_bytes = int(ASCEND_UB_LIMIT_BYTES * fraction)
    config = dict(initial_tile_config)
    while calculate_ssa_ub_bytes(ssa_graph, config) > max_ub_bytes:
        candidates = [
            key
            for key, value in config.items()
            if (key.startswith("BLOCK_SIZE_") or key.startswith("block_"))
            and int(value) > 16
        ]
        if not candidates:
            raise UnsupportedBackendOpError(
                "Ascend tile memory footprint exceeds 910B3 UB limit even at minimum tile size.",
                reason="the SSA tile working set exceeds the configured UB budget.",
                suggestion="Operator tile memory footprint exceeds 910B3 UB limit even at minimum tile size. Consider split/loop partitioning.",
            )
        key = max(candidates, key=lambda item: int(config[item]))
        config[key] = max(16, int(config[key]) // 2)
    return config


def plan_ascend_ub(ssa_graph, initial_tile_config) -> AscendUBPlan:
    """Select a safe tile and expose peak usage and rejection diagnostics."""
    schedule = dict(getattr(ssa_graph, "metadata", {}).get("schedule", {}))
    fraction = (
        ASCEND_UB_ATTENTION_FRACTION
        if schedule.get("ascend_attention_loop")
        else ASCEND_UB_NORMAL_FRACTION
    )
    budget = int(ASCEND_UB_LIMIT_BYTES * fraction)
    try:
        safe = solve_ascend_tile_config(ssa_graph, initial_tile_config, budget)
    except UnsupportedBackendOpError as exc:
        return AscendUBPlan(
            safe_tile=dict(initial_tile_config),
            estimated_peak_bytes=calculate_ssa_ub_bytes(ssa_graph, initial_tile_config),
            workspace_bytes=_ascend_workspace_bytes(initial_tile_config),
            safety_margin_bytes=0,
            rejection_reason=str(exc),
        )
    peak = calculate_ssa_ub_bytes(ssa_graph, safe)
    return AscendUBPlan(
        safe_tile={key: int(value) for key, value in safe.items()},
        estimated_peak_bytes=peak,
        workspace_bytes=_ascend_workspace_bytes(safe),
        safety_margin_bytes=max(0, budget - peak),
    )


def _plan_ascend_conv2d_resources(ssa_graph, initial_tile_config):
    """Plan the verified rank-4 im2col Conv2d tile for JIT and AOT.

    CANN's Conv2d lowering retains additional address, mask, and loop
    temporaries that are not represented by the generic SSA byte estimate.
    Consequently a 64x64x64 public candidate can pass that estimate and still
    fail BiShengIR PlanMemory.  The 16x16x16 tile is the smallest candidate
    already exercised by the dynamic Ascend JIT path; making it the sole
    candidate gives static AOT builds the same resource contract.
    """
    schedule = dict(getattr(ssa_graph, "metadata", {}).get("schedule", {}))
    budget = int(ASCEND_UB_LIMIT_BYTES * ASCEND_UB_NORMAL_FRACTION)
    candidates = []
    for m, n, k in _ASCEND_CONV2D_TILE_CANDIDATES:
        tile = {
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "block_m": int(m),
            "block_n": int(n),
            "block_k": int(k),
            "BLOCK_SIZE_M": int(m),
            "BLOCK_SIZE_N": int(n),
            "BLOCK_SIZE_K": int(k),
        }
        estimate = int(calculate_ssa_ub_bytes(ssa_graph, tile))
        candidates.append(
            {
                "tile": {"m": int(m), "n": int(n), "k": int(k)},
                "estimated_ub_bytes": estimate,
                "ub_budget_bytes": budget,
                "workspace_bytes": _ascend_workspace_bytes(tile),
                "accepted": estimate <= budget,
            }
        )

    selected = next((item for item in candidates if item["accepted"]), None)
    return {
        "version": 1,
        "kind": "conv2d-im2col",
        "provenance": "ascend-conv2d-verified-jit-tile",
        "candidate_tiles": tuple(candidates),
        "selected_tile": None if selected is None else dict(selected["tile"]),
        "ub_estimated_peak_bytes": (
            None if selected is None else int(selected["estimated_ub_bytes"])
        ),
        "ub_budget_bytes": budget,
        "workspace_bytes": (
            None if selected is None else int(selected["workspace_bytes"])
        ),
        "rejection_reason": (
            None
            if selected is not None
            else "no verified Conv2d tile fits the Ascend UB budget"
        ),
        "initial_tile": {
            key: int(value)
            for key, value in dict(initial_tile_config).items()
            if key in {"block_m", "block_n", "block_k", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"}
        },
        "schedule_granularity": schedule.get("granularity"),
    }


def _attention_tile_dict(tile: Mapping[str, Any]) -> dict[str, int]:
    """Normalize public/private tile aliases to one M/N/K representation."""
    aliases = {
        "m": ("m", "block_m", "BLOCK_SIZE_M"),
        "n": ("n", "block_n", "BLOCK_SIZE_N"),
        "k": ("k", "block_k", "BLOCK_SIZE_K"),
    }
    result = {}
    for axis, names in aliases.items():
        value = next((tile[name] for name in names if name in tile), None)
        if value is None:
            raise ValueError(f"Attention tile is missing {axis.upper()} dimension: {tile!r}.")
        result[axis] = int(value)
    return result


def ascend_attention_source_contract(
    attention_plan: Mapping[str, Any], attention_retile: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and freeze the complete plan metadata embedded in source/AOT."""
    def mismatch(reason: str):
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=reason,
            suggestion="rebuild the source and AOT sidecar from one verified Attention resource plan.",
        )

    if not isinstance(attention_plan, Mapping) or not isinstance(attention_retile, Mapping):
        mismatch("the resource plan or structured retile metadata is missing.")
    resource_plan = attention_plan.get("resource_plan")
    if not isinstance(resource_plan, Mapping):
        mismatch("attention_plan.resource_plan is missing.")
    try:
        selected = _attention_tile_dict(resource_plan.get("selected_tile", {}))
        plan_tile = _attention_tile_dict(attention_plan.get("tile", {}))
        retile_tile = {
            axis: int(attention_retile[f"block_{axis}"])
            for axis in ("m", "n", "k")
        }
    except (KeyError, TypeError, ValueError) as exc:
        mismatch(f"selected M/N/K is missing or malformed: {exc}.")
    if selected != plan_tile or selected != retile_tile:
        mismatch(
            f"resource tile={selected!r}, attention tile={plan_tile!r}, "
            f"retile tile={retile_tile!r}."
        )
    if min(selected.values()) <= 0:
        mismatch(f"selected tile dimensions must be positive: {selected!r}.")

    candidates = tuple(resource_plan.get("candidate_tiles", ()))
    selected_record = next(
        (
            record
            for record in candidates
            if isinstance(record, Mapping)
            and _normalize_attention_contract_value(record.get("tile", {})) == selected
        ),
        None,
    )
    if selected_record is None or selected_record.get("feasible") is not True:
        mismatch(f"selected tile {selected!r} has no feasible candidate record.")
    try:
        solver_tile = _attention_tile_dict(selected_record.get("solver_tile", {}))
        estimate = int(resource_plan["ub_estimated_peak_bytes"])
        budget = int(resource_plan["ub_budget_bytes"])
        candidate_estimate = int(selected_record["estimated_ub_bytes"])
        candidate_budget = int(selected_record["ub_budget_bytes"])
        candidate_workspace = selected_record["workspace_bytes"]
    except (KeyError, TypeError, ValueError) as exc:
        mismatch(f"selected candidate UB record is incomplete: {exc}.")
    if (
        solver_tile != selected
        or estimate != candidate_estimate
        or budget != candidate_budget
        or estimate > budget
    ):
        mismatch(
            f"selected tile={selected!r}, solver tile={solver_tile!r}, "
            f"estimated UB={estimate}/{candidate_estimate}, budget={budget}/{candidate_budget}."
        )
    if resource_plan.get("rejection_reason") is not None:
        mismatch(
            f"selected resource plan still has rejection reason "
            f"{resource_plan.get('rejection_reason')!r}."
        )

    resource_workspace = resource_plan.get("workspace_bytes")
    planned_workspace = attention_plan.get("workspace_bytes")
    if (
        resource_workspace is None
        or planned_workspace != resource_workspace
        or candidate_workspace != resource_workspace
    ):
        mismatch(
            f"resource workspace={resource_workspace!r}, "
            f"attention workspace={planned_workspace!r}, "
            f"selected candidate workspace={candidate_workspace!r}."
        )
    if resource_plan.get("internal_dtype") != ASCEND_ATTENTION_INTERNAL_DTYPE:
        mismatch(
            f"resource internal dtype={resource_plan.get('internal_dtype')!r}, "
            f"expected {ASCEND_ATTENTION_INTERNAL_DTYPE!r}."
        )
    try:
        head_dim = int(attention_plan.get("head_dim"))
    except (TypeError, ValueError):
        head_dim = None
    if head_dim is not None:
        expected_workspace = selected["m"] * head_dim * 4
        try:
            actual_workspace = int(resource_workspace)
        except (TypeError, ValueError):
            mismatch(f"workspace {resource_workspace!r} is not concrete for head_dim={head_dim}.")
        if actual_workspace != expected_workspace:
            mismatch(
                f"resource workspace={actual_workspace}, expected={expected_workspace} "
                f"for M={selected['m']} and head_dim={head_dim}."
            )
    else:
        head_dim_expression = attention_retile.get("head_dim")
        expected_workspace_expression = (
            f"{selected['m']} * ({head_dim_expression}) * 4"
        )
        actual_compact = "".join(str(resource_workspace).split())
        expected_compact = "".join(expected_workspace_expression.split())
        if actual_compact != expected_compact:
            mismatch(
                f"symbolic resource workspace={resource_workspace!r}, expected "
                f"{expected_workspace_expression!r} from the retiled head_dim."
            )

    for key in ("query_tiles", "key_tiles"):
        if attention_plan.get(key) != attention_retile.get(key):
            mismatch(
                f"{key} differs between planner={attention_plan.get(key)!r} "
                f"and retile={attention_retile.get(key)!r}."
            )
    if attention_plan.get("dot_tiles") != attention_retile.get("dot_tiles"):
        mismatch("QK/PV dot tile metadata differs between planner and retile.")

    try:
        batch = int(attention_plan.get("batch"))
        heads = int(attention_plan.get("heads"))
        sequence = int(attention_plan.get("sequence"))
    except (TypeError, ValueError):
        batch = heads = sequence = None
    if batch is not None and heads is not None and sequence is not None:
        query_tiles = (sequence + selected["m"] - 1) // selected["m"]
        key_tiles = (sequence + selected["n"] - 1) // selected["n"]
        grid = batch * heads * query_tiles
        expected_grid_expr = (
            f"{batch} * {heads} * triton.cdiv({sequence}, {selected['m']})"
        )
        expected_key_upper = f"(({sequence} + {selected['n'] - 1}) // {selected['n']})"
        expected_query_upper = f"(({sequence} + {selected['m'] - 1}) // {selected['m']})"
        if (
            attention_plan.get("query_tiles") != query_tiles
            or attention_plan.get("key_tiles") != key_tiles
            or attention_plan.get("grid") != grid
            or attention_retile.get("query_loop_upper") != expected_query_upper
            or attention_retile.get("key_loop_upper") != expected_key_upper
            or attention_retile.get("loop_upper") != expected_key_upper
            or attention_retile.get("grid") != expected_grid_expr
        ):
            mismatch(
                f"recomputed query/key/grid={(query_tiles, key_tiles, grid)!r} "
                f"does not match planner/retile metadata."
            )

    return {
        "version": ASCEND_ATTENTION_SOURCE_CONTRACT_VERSION,
        "attention_plan": _json_value(attention_plan),
        "attention_retile": _json_value(attention_retile),
    }


def validate_ascend_attention_source_contract(
    attention_plan: Mapping[str, Any],
    attention_retile: Mapping[str, Any],
    source_contract: Mapping[str, Any] | None = None,
    *,
    require_source: bool = False,
) -> dict[str, Any]:
    """Check source metadata against the canonical resource plan and retile."""
    expected = ascend_attention_source_contract(attention_plan, attention_retile)
    if source_contract is None:
        if require_source:
            raise UnsupportedBackendOpError(
                "Ascend Attention plan/source contract mismatch.",
                reason="generated source has no embedded Attention plan metadata.",
                suggestion="regenerate the artifact with the matching Ascend emitter.",
            )
        return expected
    actual = _normalize_attention_contract_value(source_contract)
    normalized_expected = _normalize_attention_contract_value(expected)
    if actual != normalized_expected:
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=(
                f"embedded source contract={actual!r} differs from "
                f"resource plan/retile={normalized_expected!r}."
            ),
            suggestion="regenerate source and sidecar together from the verified Attention schedule.",
        )
    return expected


def _attention_ub_budget_bytes() -> int:
    return int(ASCEND_UB_LIMIT_BYTES * ASCEND_UB_ATTENTION_FRACTION)


def _plan_ascend_attention_resources(
    program: ssa.Program,
    *,
    head_dim: int | None,
    head_dim_expression: Any = None,
    candidates: tuple[Mapping[str, int], ...] | None = None,
) -> Mapping[str, Any]:
    """Choose only an Attention tile that survives the generic UB solver unchanged.

    Candidate order is the schedule's preference order.  The solver may shrink
    its input tile to explain how it could fit, but that smaller result is not
    silently accepted unless it is itself a declared, verified candidate.
    """
    if candidates is None:
        candidates = tuple(
            {"m": m, "n": n, "k": k}
            for m, n, k in _ASCEND_ATTENTION_TILE_CANDIDATES
        )
    budget = _attention_ub_budget_bytes()
    records = []
    selected = None
    for raw_candidate in candidates:
        candidate = _attention_tile_dict(raw_candidate)
        ub_estimate = calculate_ssa_ub_bytes(
            program,
            {
                "block_m": candidate["m"],
                "block_n": candidate["n"],
                "block_k": candidate["k"],
            },
        )
        solver_plan = plan_ascend_ub(
            program,
            {
                "block_m": candidate["m"],
                "block_n": candidate["n"],
                "block_k": candidate["k"],
            },
        )
        solver_tile = _attention_tile_dict(solver_plan.safe_tile)
        reasons = []
        if solver_plan.rejection_reason:
            reasons.append(solver_plan.rejection_reason)
        if ub_estimate > budget:
            reasons.append(
                f"estimated UB {ub_estimate} bytes exceeds budget {budget} bytes"
            )
        if solver_tile != candidate:
            reasons.append(
                f"UB solver would change candidate tile to {solver_tile!r}; "
                "implicit tile changes are not verified for Attention"
            )
        feasible = not reasons
        if head_dim is not None:
            workspace = candidate["m"] * head_dim * 4
        elif head_dim_expression is not None:
            workspace = f"{candidate['m']} * ({head_dim_expression}) * 4"
        else:
            workspace = None
        records.append(
            {
                "tile": candidate,
                "estimated_ub_bytes": int(ub_estimate),
                "solver_tile": solver_tile,
                "solver_estimated_ub_bytes": int(solver_plan.estimated_peak_bytes),
                "ub_budget_bytes": budget,
                "workspace_bytes": workspace,
                "feasible": feasible,
                "rejection_reason": "; ".join(reasons) if reasons else None,
            }
        )
        if feasible and selected is None:
            selected = candidate

    selected_record = next(
        (record for record in records if record["tile"] == selected), None
    )
    rejection_reason = None
    if selected is None:
        if not records:
            rejection_reason = "no verified Attention tile candidates were provided"
        else:
            rejection_reason = "no UB-feasible verified Attention tile candidate"

    return {
        "version": 1,
        "candidate_tiles": tuple(records),
        "selected_tile": selected,
        "ub_estimated_peak_bytes": (
            selected_record["estimated_ub_bytes"] if selected_record else None
        ),
        "ub_budget_bytes": budget,
        "internal_dtype": ASCEND_ATTENTION_INTERNAL_DTYPE,
        "workspace_bytes": (
            selected_record["workspace_bytes"] if selected_record else None
        ),
        "rejection_reason": rejection_reason,
    }


def _attention_resource_plan_error(
    resource_plan: Mapping[str, Any],
) -> UnsupportedBackendOpError:
    candidates = tuple(resource_plan.get("candidate_tiles", ()))
    details = "; ".join(
        (
            f"tile={record.get('tile')!r}, "
            f"estimated_ub={record.get('estimated_ub_bytes')} bytes, "
            f"budget={record.get('ub_budget_bytes')} bytes, "
            f"solver_tile={record.get('solver_tile')!r}"
        )
        for record in candidates
    ) or "no candidate tiles were provided"
    return UnsupportedBackendOpError(
        "Ascend Attention has no UB-feasible verified tile.",
        reason=f"{resource_plan.get('rejection_reason')}; candidates: {details}.",
        suggestion=(
            "provide a verified tile/lowering that fits the UB budget or revise "
            "the resource model using compiler evidence."
        ),
    )


def _selected_attention_tile(schedule: Mapping[str, Any]) -> dict[str, int]:
    """Read and cross-check the sole selected tile from the canonical plan."""
    attention_plan = schedule.get("ascend_attention_plan")
    resource_plan = (
        attention_plan.get("resource_plan")
        if isinstance(attention_plan, Mapping)
        else None
    )
    if not isinstance(resource_plan, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention requires the canonical resource plan.",
            reason="the Attention schedule has no planner-owned tile/UB record.",
            suggestion="run Attention semantic verification and resource planning before lowering.",
        )
    selected = resource_plan.get("selected_tile")
    if not isinstance(selected, Mapping):
        raise _attention_resource_plan_error(resource_plan)
    selected = _attention_tile_dict(selected)
    planned_tile = _attention_tile_dict(attention_plan.get("tile", {}))
    if planned_tile != selected:
        raise UnsupportedBackendOpError(
            "Ascend Attention plan contains inconsistent selected tiles.",
            reason=f"resource selected_tile={selected!r}, attention tile={planned_tile!r}.",
            suggestion="use the resource plan as the sole selected-tile source.",
        )
    return selected


def _annotate_ascend_linalg_tiles(program: ssa.Program, tile: Mapping[str, Any]):
    """Propagate solved M/N/K bounds to Ascend-private linalg operations."""
    if not tile:
        return program

    def rewrite(block):
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            if operation.opcode in {"linalg.matmul", "linalg.dot"}:
                attrs = dict(operation.attrs)
                for source, target in (
                    ("BLOCK_SIZE_M", "block_m"),
                    ("BLOCK_SIZE_N", "block_n"),
                    ("BLOCK_SIZE_K", "block_k"),
                ):
                    value = tile.get(source, tile.get(target))
                    if value is not None:
                        attrs[target] = int(value)
                attrs["ascend_workspace_tile"] = dict(tile)
                attrs["ascend_loop_tile"] = dict(tile)
                operation = ssa.Operation(
                    opcode=operation.opcode,
                    operands=operation.operands,
                    results=operation.results,
                    attrs=attrs,
                    regions=regions,
                )
            else:
                operation = ssa.Operation(
                    opcode=operation.opcode,
                    operands=operation.operands,
                    results=operation.results,
                    attrs=operation.attrs,
                    regions=regions,
                )
            operations.append(operation)
        return ssa.Block(name=block.name, args=block.args, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def _annotate_ascend_decomposed_matmul_contract(
    program: ssa.Program, tile: Mapping[str, Any]
) -> ssa.Program:
    """Attach the resolved tile contract to the private scalar matmul loop.

    ``DecomposeLinalg`` intentionally removes the high-level linalg operation.
    The Ascend emitter therefore receives the structured contract on the
    resulting ``scf.for`` and its two source extracts, rather than relying on
    a schedule flag after the operation has disappeared.
    """
    if not tile:
        return program

    def rewrite(block):
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            attrs = dict(operation.attrs)
            if operation.opcode == "scf.for" and attrs.get("decomposition") == "matmul":
                attrs.update(
                    {
                        "ascend_tile_contract": {
                            "block_m": int(tile.get("block_m", tile.get("BLOCK_SIZE_M", 16))),
                            "block_n": int(tile.get("block_n", tile.get("BLOCK_SIZE_N", 16))),
                            "block_k": int(tile.get("block_k", tile.get("BLOCK_SIZE_K", 16))),
                        },
                        "ascend_workspace_tile": dict(tile),
                        "ascend_loop_tile": dict(tile),
                        "ascend_access_contract": {
                            "lhs": ("row", "k"),
                            "rhs": ("k", "col"),
                            "output": ("row", "col"),
                        },
                    }
                )

                rewritten_regions = []
                for region in regions:
                    region_ops = []
                    for inner in region.operations:
                        inner_attrs = dict(inner.attrs)
                        if inner.opcode == "tensor.extract" and inner_attrs.get(
                            "decomposition"
                        ) == "matmul":
                            role = inner_attrs.get("operand")
                            if role == "lhs":
                                inner_attrs.update(
                                    {
                                        "ascend_access_role": "lhs",
                                        "ascend_access_axes": ("row", "k"),
                                    }
                                )
                            elif role == "rhs":
                                inner_attrs.update(
                                    {
                                        "ascend_access_role": "rhs",
                                        "ascend_access_axes": ("k", "col"),
                                    }
                                )
                        region_ops.append(replace(inner, attrs=inner_attrs))
                    rewritten_regions.append(replace(region, operations=tuple(region_ops)))
                regions = tuple(rewritten_regions)

            operations.append(replace(operation, attrs=attrs, regions=regions))
        return replace(block, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def _ssa_value_types(program: ssa.Program) -> dict[str, ssa.Type]:
    value_types = {
        value.name: value.type
        for value in (*program.inputs, *program.outputs)
    }

    def visit(block):
        value_types.update({value.name: value.type for value in block.args})
        for operation in block.operations:
            value_types.update({value.name: value.type for value in operation.results})
            for region in operation.regions:
                visit(region)

    for block in program.blocks:
        visit(block)
    return value_types


def _ascend_matmul_expected_acc_dtype(lhs: ssa.Type | None, rhs: ssa.Type | None) -> str:
    dtypes = {
        normalize_ascend_dtype(type_.dtype)
        for type_ in (lhs, rhs)
        if type_ is not None
    }
    if dtypes & {"float16", "bfloat16"}:
        return "float32"
    if "float32" in dtypes:
        return "float32"
    if any(_is_ascend_runtime_dtype(dtype) for dtype in dtypes):
        return "float32"
    raise UnsupportedBackendOpError(
        "Ascend matmul has no verified accumulator dtype contract.",
        reason=f"input dtypes are {sorted(dtypes)!r}.",
        suggestion="use FP16/BF16 inputs with FP32 accumulation or FP32 inputs.",
    )


_ASCEND_RUNTIME_DTYPE_PREFIX = "runtime_dtype:"
_ASCEND_MATMUL_INPUT_DTYPES = ("float16", "bfloat16", "float32")


def _is_ascend_runtime_dtype(dtype: str | None) -> bool:
    return isinstance(dtype, str) and dtype.startswith(_ASCEND_RUNTIME_DTYPE_PREFIX)


def _ascend_runtime_dtype(source: str) -> str:
    return f"{_ASCEND_RUNTIME_DTYPE_PREFIX}{source}"


def _ascend_matmul_value_producers(program: ssa.Program) -> dict[str, ssa.Operation]:
    return {
        result.name: operation
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    }


def _resolve_ascend_value_dtype(
    name: str,
    value_types: Mapping[str, ssa.Type],
    producers: Mapping[str, ssa.Operation],
    inputs: Mapping[str, ssa.Type],
    seen: set[str] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve a tiled/view value to its named source tensor dtype.

    Arrangement views may lose the dtype on their result type.  The defining
    access chain still identifies the source tensor, so an untyped source is
    represented as an explicit runtime specialization symbol rather than
    silently accepted as ``None``.
    """
    seen = set() if seen is None else seen
    if name in seen:
        return None, None
    seen.add(name)

    type_ = value_types.get(name) or inputs.get(name)
    producer = producers.get(name)
    if type_ is not None and type_.dtype is not None and not producer:
        return str(type_.dtype), name if name in inputs else None

    if producer is not None:
        if producer.opcode == "tensor.cast":
            dtype = producer.results[0].type.dtype if producer.results else None
            if dtype is not None:
                return str(dtype), None
        if producer.operands:
            resolved, source = _resolve_ascend_value_dtype(
                producer.operands[0], value_types, producers, inputs, seen
            )
            if resolved is not None:
                return resolved, source

    if name in inputs:
        source_dtype = inputs[name].dtype
        if source_dtype is not None:
            return str(source_dtype), name
        return _ascend_runtime_dtype(name), name
    return None, None


def _resolve_ascend_matmul_operand_dtypes(program: ssa.Program) -> ssa.Program:
    """Preserve dtype provenance through arrangement views before verification."""
    value_types = _ssa_value_types(program)
    inputs = {value.name: value.type for value in program.inputs}
    producers = _ascend_matmul_value_producers(program)

    def rewrite(block: ssa.Block) -> ssa.Block:
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            attrs = dict(operation.attrs)
            results = list(operation.results)
            for index, result in enumerate(results):
                # Dot/matmul results are accumulator values, not input
                # operands. Their dtype is established by the accumulator
                # normalization pass below.
                if operation.opcode in {"linalg.dot", "linalg.matmul"}:
                    continue
                if result.type.kind != "tensor" or result.type.dtype is not None:
                    continue
                resolved, source = _resolve_ascend_value_dtype(
                    result.name, value_types, producers, inputs
                )
                if resolved is None:
                    continue
                results[index] = replace(
                    result, type=replace(result.type, dtype=resolved)
                )
                value_types[result.name] = results[index].type
                attrs.setdefault("ascend_dtype_provenance", {})
                attrs["ascend_dtype_provenance"] = dict(
                    attrs["ascend_dtype_provenance"]
                ) | {
                    result.name: {
                        "source_tensor": source,
                        "dtype": resolved,
                        "allowed": _ASCEND_MATMUL_INPUT_DTYPES,
                    }
                }

            if operation.opcode in {"linalg.dot", "linalg.matmul"}:
                provenance = {}
                for operand in operation.operands[:2]:
                    resolved, source = _resolve_ascend_value_dtype(
                        operand, value_types, producers, inputs
                    )
                    if resolved is not None:
                        provenance[operand] = {
                            "source_tensor": source,
                            "dtype": resolved,
                            "allowed": _ASCEND_MATMUL_INPUT_DTYPES,
                        }
                if provenance:
                    attrs["ascend_dot_operand_dtype_provenance"] = provenance

            operations.append(
                replace(
                    operation,
                    results=tuple(results),
                    attrs=attrs,
                    regions=regions,
                )
            )
        return replace(block, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def _normalize_ascend_matmul_accumulators(program: ssa.Program) -> ssa.Program:
    """Promote private dot result values to the declared FP32 accumulator."""
    value_types = _ssa_value_types(program)

    def rewrite(block):
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            if operation.opcode not in {"linalg.dot", "linalg.matmul"} or len(operation.operands) < 2:
                operations.append(replace(operation, regions=regions))
                continue
            lhs = value_types.get(operation.operands[0])
            rhs = value_types.get(operation.operands[1])
            expected = _ascend_matmul_expected_acc_dtype(lhs, rhs)
            attrs = dict(operation.attrs)
            attrs["ascend_accumulator_dtype"] = expected
            results = tuple(operation.results)
            if operation.opcode == "linalg.dot" and results:
                result = results[0]
                if normalize_ascend_dtype(result.type.dtype) != expected:
                    results = (replace(result, type=replace(result.type, dtype=expected)), *results[1:])
            promoted = replace(operation, results=results, attrs=attrs, regions=regions)
            operations.append(promoted)
            value_types.update({value.name: value.type for value in results})
        return replace(block, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def _normalize_ascend_decomposed_matmul_accumulators(
    program: ssa.Program,
) -> ssa.Program:
    """Make the scalar K-loop's carried accumulator explicitly FP32.

    ``DecomposeLinalg`` materializes the initial zero using the output dtype.
    Ascend's dot contract instead requires FP32 state for FP16/BF16 inputs, so
    update the private decomposed loop and its arithmetic values together.
    """

    def rewrite(block: ssa.Block) -> ssa.Block:
        initial_names = {
            str(item.get("initial"))
            for operation in block.operations
            if operation.opcode == "scf.for"
            and operation.attrs.get("decomposition") == "matmul"
            for item in tuple(operation.attrs.get("iter_args", ()))
        }
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            attrs = dict(operation.attrs)
            results = tuple(operation.results)

            if operation.opcode == "arith.constant" and any(
                result.name in initial_names for result in results
            ):
                results = tuple(
                    replace(
                        result,
                        type=replace(result.type, kind="scalar", dtype="float32"),
                    )
                    for result in results
                )

            if (
                operation.opcode == "scf.for"
                and operation.attrs.get("decomposition") == "matmul"
                and regions
            ):
                attrs["ascend_accumulator_dtype"] = "float32"
                results = tuple(
                    replace(result, type=replace(result.type, dtype="float32"))
                    for result in results
                )
                iter_args = tuple(attrs.get("iter_args", ()))
                block_arg_names = {
                    str(item.get("block_arg"))
                    for item in iter_args
                }
                rewritten_regions = []
                for region in regions:
                    region_args = tuple(
                        replace(
                            arg,
                            type=replace(arg.type, kind="scalar", dtype="float32"),
                        )
                        if arg.name in block_arg_names
                        else arg
                        for arg in region.args
                    )
                    region_ops = []
                    for inner in region.operations:
                        inner_results = tuple(inner.results)
                        if inner.opcode in {"arith.mul", "arith.add"}:
                            inner_results = tuple(
                                replace(
                                    result,
                                    type=replace(
                                        result.type, kind="scalar", dtype="float32"
                                    ),
                                )
                                for result in inner_results
                            )
                        region_ops.append(replace(inner, results=inner_results))
                    rewritten_regions.append(
                        replace(
                            region,
                            args=region_args,
                            operations=tuple(region_ops),
                        )
                    )
                regions = tuple(rewritten_regions)

            rewritten = replace(
                operation,
                results=results,
                attrs=attrs,
                regions=regions,
            )
            operations.append(rewritten)
        return replace(block, operations=tuple(operations))

    return replace(program, blocks=tuple(rewrite(block) for block in program.blocks))


def _verify_ascend_matmul_contract(program: ssa.Program) -> Mapping[str, Any]:
    """Check FP32 accumulation and output-cast boundaries in Ascend SSA."""
    value_types = _ssa_value_types(program)
    dot_count = 0
    for operation in _walk_operations(program.blocks):
        if operation.opcode not in {"linalg.dot", "linalg.matmul"}:
            continue
        dot_count += 1
        if len(operation.operands) < 2 or not operation.results:
            raise UnsupportedBackendOpError(
                "Ascend matmul dtype contract is incomplete.",
                reason="the linalg operation lacks two operands or a result.",
                suggestion="lower matmul/dot with explicit tensor operands and result.",
            )
        operand_provenance = dict(
            operation.attrs.get("ascend_dot_operand_dtype_provenance", {})
        )
        for operand in operation.operands[:2]:
            operand_type = value_types.get(operand)
            dtype = normalize_ascend_dtype(
                operand_type.dtype if operand_type is not None else None
            )
            if dtype is None:
                raise UnsupportedBackendOpError(
                    "Ascend matmul dot operand dtype provenance is incomplete.",
                    reason=f"operand `{operand}` still has dtype None after private view tracing.",
                    suggestion="trace tensor.extract/view operands back to a named TensorSpec or runtime dtype contract.",
                )
            if _is_ascend_runtime_dtype(dtype):
                record = dict(operand_provenance.get(operand, {}))
                source = record.get("source_tensor")
                allowed = tuple(record.get("allowed", ()))
                if not source or allowed != _ASCEND_MATMUL_INPUT_DTYPES:
                    raise UnsupportedBackendOpError(
                        "Ascend matmul runtime dtype provenance is incomplete.",
                        reason=f"operand `{operand}` has symbolic dtype `{dtype}` without an explicit source/allow-list.",
                        suggestion="publish source_tensor and the FP16/BF16/FP32 runtime allow-list.",
                    )
        expected = _ascend_matmul_expected_acc_dtype(
            value_types.get(operation.operands[0]), value_types.get(operation.operands[1])
        )
        actual = normalize_ascend_dtype(operation.results[0].type.dtype)
        if actual != expected:
            raise UnsupportedBackendOpError(
                "Ascend matmul dot result violates the accumulator dtype contract.",
                reason=f"expected {expected}, received {actual}.",
                suggestion="promote the dot result and accumulator state to FP32.",
            )
        if operation.attrs.get("ascend_accumulator_dtype") != expected:
            raise UnsupportedBackendOpError(
                "Ascend matmul lowering did not publish accumulator dtype metadata.",
                reason="the downstream dot lowering cannot prove its accumulator type.",
                suggestion="attach ascend_accumulator_dtype to every linalg dot/matmul.",
            )

    for loop in (op for op in _walk_operations(program.blocks) if op.opcode == "scf.for"):
        body_ops = tuple(_walk_operations(loop.regions))
        if not any(op.opcode in {"linalg.dot", "linalg.matmul"} for op in body_ops):
            continue
        iter_args = tuple(dict(loop.attrs).get("iter_args", ()))
        if not iter_args or len(loop.operands) < 3 + len(iter_args):
            raise UnsupportedBackendOpError(
                "Ascend loop-carried accumulator contract is incomplete.",
                reason="the dot loop has no explicit initial accumulator state.",
                suggestion="carry an FP32 accumulator through scf.for iter_args.",
            )
        region = loop.regions[0] if loop.regions else None
        block_args = {arg.name: arg.type for arg in region.args} if region else {}
        yields = next((op for op in reversed(body_ops) if op.opcode == "scf.yield"), None)
        for index, item in enumerate(iter_args):
            name = str(item.get("name"))
            initial = value_types.get(loop.operands[3 + index])
            carried = block_args.get(str(item.get("block_arg")), value_types.get(str(item.get("block_arg"))))
            updated = value_types.get(yields.operands[index]) if yields and index < len(yields.operands) else None
            for label, type_ in (("initial", initial), ("block argument", carried), ("updated", updated)):
                if type_ is not None and normalize_ascend_dtype(type_.dtype) != "float32":
                    raise UnsupportedBackendOpError(
                        "Ascend loop-carried accumulator violates FP32 contract.",
                        reason=f"{label} `{name}` has dtype {type_.dtype!r}.",
                        suggestion="initialize, carry, update, and yield the accumulator as FP32.",
                    )

    return {"dot_count": dot_count, "accumulator_dtype": "float32"}


def _verify_ascend_tile_consumption(program: ssa.Program) -> Mapping[str, Any] | None:
    schedule = dict(program.metadata.get("schedule", {}))
    linalg_ops = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode in {"linalg.dot", "linalg.matmul"}
    )
    if not linalg_ops:
        linalg = dict(schedule.get("ascend_linalg", {}))
        if linalg.get("mode") == "tiled-matmul" and linalg.get("rank") == 3:
            return _verify_ascend_batched_matmul_contract(program)
        if linalg.get("mode") == "tiled-matmul":
            return _verify_ascend_decomposed_matmul_contract(program)
        return None
    tile = schedule.get("ascend_tile_override") or schedule.get("ascend_matrix_tile")
    if not tile:
        raise UnsupportedBackendOpError(
            "Ascend matmul tile contract was not consumed by lowering.",
            reason="no ascend_matrix_tile or ascend_tile_override reaches linalg lowering.",
            suggestion="resolve the private tile before handing the operation to BiShengIR.",
        )
    for operation in linalg_ops:
        attrs = dict(operation.attrs)
        workspace = attrs.get("ascend_workspace_tile")
        loop_tile = attrs.get("ascend_loop_tile")
        if not workspace or not loop_tile:
            raise UnsupportedBackendOpError(
                "Ascend matmul tile metadata was not consumed by linalg lowering.",
                reason="linalg.dot/matmul is missing ascend workspace or loop tile metadata.",
                suggestion="propagate the resolved private tile onto every linalg operation.",
            )
        if any(
            int(workspace.get(axis, workspace.get(alias, -1))) == 256
            for axis, alias in (("BLOCK_SIZE_M", "block_m"), ("BLOCK_SIZE_N", "block_n"))
        ):
            raise UnsupportedBackendOpError(
                "Ascend matmul retains a fixed 256x256 workspace contract.",
                reason=f"workspace tile is {workspace!r}.",
                suggestion="use the UB-solved tile instead of the schedule candidate default.",
            )
    return {"workspace_tile_consumed": True, "operation_count": len(linalg_ops)}


def _verify_ascend_decomposed_matmul_contract(
    program: ssa.Program,
) -> Mapping[str, Any]:
    """Verify that a decomposed rank-2 matmul carries a consumed contract."""
    loops = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
        and operation.attrs.get("decomposition") == "matmul"
    )
    if len(loops) != 1:
        raise UnsupportedBackendOpError(
            "Ascend decomposed matmul requires one structured rank-2 loop.",
            reason=f"found {len(loops)} matmul loops after linalg decomposition.",
            suggestion="lower one [M,K] @ [K,N] operation with one K reduction loop.",
        )

    loop = loops[0]
    attrs = dict(loop.attrs)
    tile = attrs.get("ascend_tile_contract")
    workspace = attrs.get("ascend_workspace_tile")
    loop_tile = attrs.get("ascend_loop_tile")
    access = attrs.get("ascend_access_contract")
    if not isinstance(tile, Mapping) or not isinstance(workspace, Mapping) or not isinstance(loop_tile, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend decomposed matmul tile metadata was not consumed.",
            reason="the scalar matmul loop has no resolved workspace and loop tile contract.",
            suggestion="attach the resolved Ascend tile to the decomposed scf.for before emission.",
        )
    if access != {"lhs": ("row", "k"), "rhs": ("k", "col"), "output": ("row", "col")}:
        raise UnsupportedBackendOpError(
            "Ascend decomposed matmul access contract is incomplete.",
            reason=f"received access contract {access!r}.",
            suggestion="prove lhs[row,k], rhs[k,col], and output[row,col] coordinates.",
        )
    if any(int(workspace.get(axis, workspace.get(alias, -1))) == 256 for axis, alias in (
        ("BLOCK_SIZE_M", "block_m"), ("BLOCK_SIZE_N", "block_n")
    )):
        raise UnsupportedBackendOpError(
            "Ascend matmul retains a fixed 256x256 workspace contract.",
            reason=f"workspace tile is {workspace!r}.",
            suggestion="use the UB-solved tile on the decomposed loop.",
        )

    induction = str(attrs.get("induction", "%kk"))
    region = loop.regions[0] if loop.regions else None
    body = tuple(region.operations) if region is not None else ()
    extracts = {
        str(operation.attrs.get("operand")): operation
        for operation in body
        if operation.opcode == "tensor.extract"
        and operation.attrs.get("decomposition") == "matmul"
    }
    for role, expected_axes in (("lhs", ("row", "k")), ("rhs", ("k", "col"))):
        operation = extracts.get(role)
        if operation is None or operation.attrs.get("ascend_access_axes") != expected_axes:
            raise UnsupportedBackendOpError(
                "Ascend decomposed matmul source access contract is incomplete.",
                reason=f"missing {role} access provenance on the K loop.",
                suggestion="preserve the structured row/k and k/col tensor.extract operations.",
            )
        indices = operation.operands[1:]
        if role == "lhs" and len(indices) != 2:
            raise UnsupportedBackendOpError(
                "Ascend rank-2 lhs access contract is invalid.",
                reason=f"lhs extract indices are {indices!r}.",
                suggestion="use lhs[row, induction].",
            )
        if role == "rhs" and len(indices) != 2:
            raise UnsupportedBackendOpError(
                "Ascend rank-2 rhs access contract is invalid.",
                reason=f"rhs extract indices are {indices!r}.",
                suggestion="use rhs[induction, col].",
            )
    if extracts["lhs"].operands[-1] != induction or extracts["rhs"].operands[1] != induction:
        raise UnsupportedBackendOpError(
            "Ascend decomposed matmul reduction coordinate is invalid.",
            reason="lhs/rhs extracts do not use the scf.for induction variable as K.",
            suggestion="use one shared K induction value for lhs[row,k] and rhs[k,col].",
        )
    return {
        "workspace_tile_consumed": True,
        "lowering": "ascend-decomposed-rank2-matmul",
        "access_contract": access,
        "accumulator_dtype": "float32",
    }


def _verify_ascend_batched_matmul_contract(
    program: ssa.Program,
) -> Mapping[str, Any]:
    """Verify the explicit rank-3 batch/row/column access contract."""
    schedule = dict(program.metadata.get("schedule", {}))
    contract = dict(schedule.get("ascend_linalg", {}))
    rewrite = dict(schedule.get("ascend_batched_access_rewrite", {}))
    expected_access = {
        "lhs": ("batch", "row", "k"),
        "rhs": ("batch", "k", "col"),
        "output": ("batch", "row", "col"),
    }
    if contract.get("rank") != 3 or contract.get("batch") is None:
        raise UnsupportedBackendOpError(
            "Ascend batched matmul requires an explicit rank-3 batch contract.",
            reason=f"received linalg contract rank={contract.get('rank')!r}, batch={contract.get('batch')!r}.",
            suggestion="provide [B,M,K] @ [B,K,N] -> [B,M,N] metadata.",
        )
    if rewrite.get("access_contract") != expected_access or rewrite.get("batch_broadcast"):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul access provenance is incomplete.",
            reason=f"received coordinates={rewrite.get('coordinates')!r}, batch_broadcast={rewrite.get('batch_broadcast')!r}.",
            suggestion="prove batch-row-k, batch-k-col, and batch-row-col without broadcasting.",
        )

    loops = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
        and operation.attrs.get("decomposition") == "matmul"
    )
    if len(loops) != 1:
        raise UnsupportedBackendOpError(
            "Ascend batched matmul requires one structured K loop.",
            reason=f"found {len(loops)} decomposed matmul loops.",
            suggestion="lower one batch-aware K reduction loop.",
        )
    loop = loops[0]
    attrs = dict(loop.attrs)
    if attrs.get("ascend_access_contract") != expected_access:
        raise UnsupportedBackendOpError(
            "Ascend batched matmul loop access contract is incomplete.",
            reason=f"received {attrs.get('ascend_access_contract')!r}.",
            suggestion="attach the rank-3 contract to the decomposed scf.for.",
        )
    if attrs.get("ascend_mask_contract") != dict(rewrite.get("mask", {})):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul mask contract is incomplete.",
            reason="loop mask bounds are not the same contract as the source access rewrite.",
            suggestion="preserve batch, row, col, and K bounds for every load/store.",
        )
    workspace = attrs.get("ascend_workspace_tile")
    loop_tile = attrs.get("ascend_loop_tile")
    if not isinstance(workspace, Mapping) or not isinstance(loop_tile, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul tile metadata was not consumed.",
            reason="the rank-3 K loop has no resolved workspace/loop tile.",
            suggestion="propagate the UB-solved tile before source emission.",
        )
    if any(
        int(workspace.get(axis, workspace.get(alias, -1))) == 256
        for axis, alias in (("BLOCK_SIZE_M", "block_m"), ("BLOCK_SIZE_N", "block_n"))
    ):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul retains a fixed 256x256 workspace contract.",
            reason=f"workspace tile is {workspace!r}.",
            suggestion="use the resolved private tile.",
        )

    values = {
        value.name: value.type
        for value in (*program.inputs, *program.outputs)
    }
    lhs_type = values.get(str(contract.get("lhs")))
    rhs_type = values.get(str(contract.get("rhs")))
    out_type = values.get(str(contract.get("output")))
    if any(
        type_ is None or type_.kind != "tensor" or len(type_.shape) != 3
        for type_ in (lhs_type, rhs_type, out_type)
    ):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul requires three rank-3 tensor bindings.",
            reason="lhs, rhs, or output shape provenance is missing.",
            suggestion="preserve rank-3 TensorSpec shapes through lowering.",
        )
    if not (lhs_type.shape[0] == rhs_type.shape[0] == out_type.shape[0] == contract["batch"]):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul batch dimensions are not equal.",
            reason=f"lhs={lhs_type.shape}, rhs={rhs_type.shape}, output={out_type.shape}.",
            suggestion="use equal B on all three rank-3 tensors; broadcasting is unsupported.",
        )

    induction = str(attrs.get("induction", "%kk"))
    body = tuple(loop.regions[0].operations) if loop.regions else ()
    extracts = {
        str(operation.attrs.get("operand")): operation
        for operation in body
        if operation.opcode == "tensor.extract"
        and operation.attrs.get("decomposition") == "matmul"
    }
    lhs_extract = extracts.get("lhs")
    rhs_extract = extracts.get("rhs")
    if lhs_extract is None or rhs_extract is None:
        raise UnsupportedBackendOpError(
            "Ascend batched matmul is missing lhs/rhs access operations.",
            reason="the decomposed K loop has no two tensor.extract operations.",
            suggestion="preserve explicit batch-row-k and batch-k-col extracts.",
        )
    if len(lhs_extract.operands) != 4 or len(rhs_extract.operands) != 4:
        raise UnsupportedBackendOpError(
            "Ascend batched rhs access must have batch,k,col coordinates.",
            reason=f"lhs={lhs_extract.operands!r}, rhs={rhs_extract.operands!r}.",
            suggestion="use lhs[batch,row,k] and rhs[batch,k,col].",
        )
    batch_name = lhs_extract.operands[1]
    row_name = lhs_extract.operands[2]
    col_name = rhs_extract.operands[3]
    if lhs_extract.operands != (str(contract["lhs"]), batch_name, row_name, induction):
        raise UnsupportedBackendOpError(
            "Ascend batched lhs coordinates are not proven.",
            reason=f"lhs extract operands are {lhs_extract.operands!r}.",
            suggestion="use lhs[batch,row,k] with the loop induction as k.",
        )
    if rhs_extract.operands != (str(contract["rhs"]), batch_name, induction, col_name):
        raise UnsupportedBackendOpError(
            "Ascend batched rhs coordinates are not proven.",
            reason=f"rhs extract operands are {rhs_extract.operands!r}.",
            suggestion="use rhs[batch,k,col] with the same batch and K induction.",
        )
    stores = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "mem.store"
        and len(operation.operands) == 2
        and operation.operands[1] == str(contract["output"])
    )
    if len(stores) != 1 or dict(stores[0].attrs).get("ascend_access_axes") != expected_access["output"]:
        raise UnsupportedBackendOpError(
            "Ascend batched output coordinate contract is incomplete.",
            reason="the output store is not annotated as out[batch,row,col].",
            suggestion="attach batch-row-col provenance to the output store.",
        )
    value_types = _ssa_value_types(program)
    iter_args = tuple(attrs.get("iter_args", ()))
    if len(iter_args) != 1:
        raise UnsupportedBackendOpError(
            "Ascend batched matmul requires one FP32 loop-carried accumulator.",
            reason=f"iter_args={iter_args!r}.",
            suggestion="carry one FP32 accumulator through the K loop.",
        )
    initial = value_types.get(str(iter_args[0].get("initial")))
    block_arg = next(
        (arg.type for arg in loop.regions[0].args if arg.name == str(iter_args[0].get("block_arg"))),
        None,
    )
    yielded = next((operation for operation in body if operation.opcode == "scf.yield"), None)
    updated = value_types.get(yielded.operands[0]) if yielded and yielded.operands else None
    if any(type_ is None or normalize_ascend_dtype(type_.dtype) != "float32" for type_ in (initial, block_arg, updated, loop.results[0].type)):
        raise UnsupportedBackendOpError(
            "Ascend batched matmul accumulator is not uniformly FP32.",
            reason="initial, block argument, update, and loop result must all be FP32.",
            suggestion="normalize the private batched K-loop accumulator before emission.",
        )
    ub_plan = dict(schedule.get("ascend_ub_plan", {}))
    expected_workspace = int(workspace.get("block_m", workspace.get("BLOCK_SIZE_M", 0))) * int(workspace.get("block_n", workspace.get("BLOCK_SIZE_N", 0))) * 4
    if int(ub_plan.get("workspace_bytes", -1)) != expected_workspace:
        raise UnsupportedBackendOpError(
            "Ascend batched matmul workspace metadata is inconsistent.",
            reason=f"UB plan reports {ub_plan.get('workspace_bytes')}, tile requires {expected_workspace}.",
            suggestion="publish the same resolved workspace tile to UB planning and emission.",
        )
    return {
        "workspace_tile_consumed": True,
        "lowering": "ascend-decomposed-rank3-batched-matmul",
        "access_contract": expected_access,
        "batch_broadcast": False,
        "grid_contract": rewrite.get("grid"),
        "mask_contract": rewrite.get("mask"),
        "accumulator_dtype": "float32",
    }


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
    ),
    # The 910B4 device identity is confirmed by the migration runner, while
    # its per-core resource capacities have not yet been independently
    # verified.  Keep those fields unknown so scheduling does not infer B3
    # capacities or silently reuse its profile.
    "Ascend910B4": AscendSocProfile(
        name="Ascend910B4",
        cube_cores=None,
        vector_cores=None,
        l2_cache_bytes=None,
    ),
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
        "devices": tuple(_ASCEND_SOC_PROFILES),
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
            "attention": {
                "q": tuple(sorted(ASCEND_ATTENTION_INPUT_DTYPES)),
                "k": tuple(sorted(ASCEND_ATTENTION_INPUT_DTYPES)),
                "v": tuple(sorted(ASCEND_ATTENTION_INPUT_DTYPES)),
                "o": tuple(sorted(ASCEND_ATTENTION_OUTPUT_DTYPES)),
                "score": ASCEND_ATTENTION_INTERNAL_DTYPE,
                "softmax": ASCEND_ATTENTION_INTERNAL_DTYPE,
                "m_i": ASCEND_ATTENTION_INTERNAL_DTYPE,
                "l_i": ASCEND_ATTENTION_INTERNAL_DTYPE,
                "accumulator": ASCEND_ATTENTION_INTERNAL_DTYPE,
                "casts": "Q/K/V to FP32 for score/softmax/dot accumulation; O cast to requested output dtype",
                "error_tolerance": {
                    dtype: dict(values)
                    for dtype, values in ASCEND_ATTENTION_ERROR_TOLERANCES.items()
                },
                "registry": {dtype: dict(contract) for dtype, contract in ASCEND_ATTENTION_DTYPE_REGISTRY.items()},
            },
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
            "attention": "verified-static-fp16-bf16-fp32-inputs-fp32-internal-static-head64-seq1024-causal-and-noncausal",
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
        "ascend_source_contract_schema": 10,
        "ascend_attention_retile_contract": "score-mask-mn-v10",
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


def ascend_logical_domain_is_abi_bound(expression, abi) -> bool:
    """Return whether a logical-domain expression has only public ABI symbols."""
    from ninetoothed.ir import IndexExpr

    available = {
        binding.name
        for binding in abi.kernel_args
        if binding.kind in {"shape", "stride", "meta", "constexpr"}
    }

    def bound(node) -> bool:
        if node.op == "symbol":
            return str(node.value) in available
        return all(bound(operand) for operand in node.operands)

    return bound(IndexExpr.parse(expression))


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
    schedule = dict(metadata.get("ssa_metadata", {})).get("schedule", {})
    attention_plan = dict(schedule).get("ascend_attention_plan")
    attention_retile = dict(schedule).get("ascend_attention_retile")
    attention_loop = dict(schedule).get("ascend_attention_loop")
    attention_source_contract = None
    if attention_loop:
        if not isinstance(attention_plan, Mapping) or not isinstance(
            attention_retile, Mapping
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention plan/source contract mismatch.",
                reason="sidecar metadata is missing the resource plan or structured retile.",
                suggestion="materialize only an artifact emitted from the verified Attention schedule.",
            )
        attention_source_contract = ascend_attention_source_contract(
            attention_plan, attention_retile
        )
        # The sidecar is tied to the exact generated source.  Catch a stale or
        # hand-edited source before publishing AOT metadata instead of waiting
        # until a later reload to discover the split contract.
        source_contract = _read_ascend_attention_source_contract(source_path)
        validate_ascend_attention_source_contract(
            attention_plan,
            attention_retile,
            source_contract,
            require_source=True,
        )
    logical_domain = ascend_logical_domain(
        specs,
        tuple(outputs),
        allow_access_template=ascend_uses_access_template(metadata),
    )
    if not ascend_logical_domain_is_abi_bound(logical_domain, abi):
        # Match JIT materialization: when a pre-specialization tile symbol is
        # absent from the launch ABI, the concrete runtime output extent is the
        # authoritative logical domain.
        logical_domain = None

    payload = {
        "schema": _SIDECAR_SCHEMA,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "backend": "ascend",
        "runtime_device": "npu",
        # This is the exact public LaunchABI from Compilation.  A sidecar must
        # never contain a second ABI with backend-specific launch semantics.
        "launch_abi": _ascend_abi_dict(abi),
        "logical_domain": logical_domain,
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
        "tile_provenance": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_tile_provenance"),
        "ub_plan": dict(metadata.get("ssa_schedule", {})).get("ascend_ub_plan"),
        "advanced_contract": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_advanced"),
        "dot_loop": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_dot_loop"),
        "access_template_resources": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_access_template_resources"),
        "conv2d_plan": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_conv2d_plan"),
        "attention_loop": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_loop"),
        "attention_mask_semantics": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_mask_semantics"),
        "attention_key_valid": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_key_valid"),
        "attention_loop_state": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_loop_state"),
        "attention_plan": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_plan"),
        "attention_retile": dict(metadata.get("ssa_metadata", {}))
        .get("schedule", {})
        .get("ascend_attention_retile"),
        "attention_source_contract": attention_source_contract,
        "attention_dtype_registry": {
            dtype: dict(contract)
            for dtype, contract in ASCEND_ATTENTION_DTYPE_REGISTRY.items()
        },
        "toolchain": {"cann_version": ascend_toolchain_version()},
    }
    payload = _json_value(payload)
    if attention_loop:
        _validate_attention_sidecar_fields(payload)
    from ninetoothed.compiler.cache import atomic_write_text
    atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True),
    )

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
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=(
                f"sidecar schema {payload.get('schema')!r} does not contain the "
                f"required plan/source contract schema {_SIDECAR_SCHEMA}."
            ),
            suggestion="rebuild the source and sidecar with the current Ascend Attention emitter.",
        )

    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if payload.get("backend") is not None and payload.get("backend") != "ascend":
        raise UnsupportedBackendOpError(
            "Ascend AOT reload contract mismatch.",
            reason=f"sidecar backend={payload.get('backend')!r}, expected 'ascend'.",
            suggestion="reload an artifact produced by the Ascend materializer.",
        )
    if payload.get("source_sha256") is not None and payload.get("source_sha256") != actual_hash:
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=(
                f"source_sha256={payload.get('source_sha256')!r} does not match "
                f"actual source hash {actual_hash!r}."
            ),
            suggestion="restore the source paired with this sidecar or rebuild the artifact.",
        )

    if payload.get("attention_loop"):
        _validate_attention_sidecar_fields(payload)

    source_contract = _read_ascend_attention_source_contract(source_path)
    sidecar_plan = payload.get("attention_plan")
    sidecar_retile = payload.get("attention_retile")
    sidecar_contract = payload.get("attention_source_contract")
    sidecar_attention_loop = payload.get("attention_loop")
    if (
        sidecar_attention_loop
        or any(value is not None for value in (sidecar_plan, sidecar_retile, sidecar_contract))
        or source_contract is not None
    ):
        if not isinstance(sidecar_plan, Mapping) or not isinstance(sidecar_retile, Mapping):
            raise UnsupportedBackendOpError(
                "Ascend Attention plan/source contract mismatch.",
                reason="AOT sidecar is missing the complete resource plan or retile record.",
                suggestion="rebuild the sidecar from the verified Attention schedule.",
            )
        validate_ascend_attention_source_contract(
            sidecar_plan,
            sidecar_retile,
            sidecar_contract,
            require_source=True,
        )
        validate_ascend_attention_source_contract(
            sidecar_plan,
            sidecar_retile,
            source_contract,
            require_source=True,
        )

    return payload


def _validate_attention_sidecar_fields(payload: Mapping[str, Any]) -> None:
    """Require the complete immutable Attention publication contract."""
    plan = payload.get("attention_plan")
    resource = plan.get("resource_plan") if isinstance(plan, Mapping) else None
    required = {
        "attention_plan": plan,
        "attention_retile": payload.get("attention_retile"),
        "attention_mask_semantics": payload.get("attention_mask_semantics"),
        "attention_source_contract": payload.get("attention_source_contract"),
        "attention_dtype_registry": payload.get("attention_dtype_registry"),
        "resource_plan_version": resource.get("version") if isinstance(resource, Mapping) else None,
        "selected_tile": resource.get("selected_tile") if isinstance(resource, Mapping) else None,
        "ub_budget_bytes": resource.get("ub_budget_bytes") if isinstance(resource, Mapping) else None,
        "ub_estimated_peak_bytes": resource.get("ub_estimated_peak_bytes") if isinstance(resource, Mapping) else None,
        "workspace_bytes": resource.get("workspace_bytes") if isinstance(resource, Mapping) else None,
        "query_tiles": plan.get("query_tiles") if isinstance(plan, Mapping) else None,
        "key_tiles": plan.get("key_tiles") if isinstance(plan, Mapping) else None,
        "grid": plan.get("grid") if isinstance(plan, Mapping) else None,
        "dot_tiles": plan.get("dot_tiles") if isinstance(plan, Mapping) else None,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=f"sidecar is missing required fields: {', '.join(missing)}.",
            suggestion="rebuild the artifact and sidecar from the canonical resource plan.",
        )
    registry = payload["attention_dtype_registry"]
    expected_registry = {
        dtype: dict(contract)
        for dtype, contract in ASCEND_ATTENTION_DTYPE_REGISTRY.items()
    }

    def canonicalize(value):
        if isinstance(value, Mapping):
            return {str(key): canonicalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonicalize(item) for item in value]
        return value

    if canonicalize(registry) != canonicalize(expected_registry):
        raise UnsupportedBackendOpError(
            "Ascend Attention dtype registry contract mismatch.",
            reason="sidecar registry differs from the canonical JIT registry.",
            suggestion="materialize JIT and AOT artifacts with one immutable dtype registry.",
        )
    selected = resource["selected_tile"]
    if not isinstance(selected, Mapping) or any(axis not in selected for axis in ("m", "n", "k")):
        raise UnsupportedBackendOpError(
            "Ascend Attention AOT contract mismatch.",
            reason=f"resource plan selected_tile is incomplete: {selected!r}.",
            suggestion="persist selected M/N/K from the verified resource plan.",
        )
    plan_tile = plan.get("tile")
    if _json_value(plan_tile) != _json_value(selected):
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=f"attention_plan.tile={plan_tile!r} differs from resource selected_tile={selected!r}.",
            suggestion="reload an artifact whose sidecar was generated from one canonical plan.",
        )
    retile = payload.get("attention_retile")
    if isinstance(retile, Mapping):
        retile_tile = {axis: retile.get(f"block_{axis}") for axis in ("m", "n", "k")}
        if any(value is None for value in retile_tile.values()) or _json_value(retile_tile) != _json_value(selected):
            raise UnsupportedBackendOpError(
                "Ascend Attention plan/source contract mismatch.",
                reason=f"attention_retile tile={retile_tile!r} differs from selected_tile={selected!r}.",
                suggestion="rebuild the sidecar from the same retiled resource plan.",
            )


def _read_ascend_attention_source_contract(source_path: Path) -> Mapping[str, Any] | None:
    """Read the emitter-owned literal plan metadata without importing Triton."""
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise UnsupportedBackendOpError(
            "Ascend Attention plan/source contract mismatch.",
            reason=f"generated source metadata cannot be read or parsed: {exc}.",
            suggestion="regenerate the source artifact before AOT reload.",
        ) from exc
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE
            for target in statement.targets
        ):
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as exc:
                raise UnsupportedBackendOpError(
                    "Ascend Attention plan/source contract mismatch.",
                    reason="embedded source Attention metadata is not a literal contract.",
                    suggestion="regenerate source through the verified Ascend emitter.",
                ) from exc
            if not isinstance(value, Mapping):
                raise UnsupportedBackendOpError(
                    "Ascend Attention plan/source contract mismatch.",
                    reason="embedded source Attention metadata is malformed.",
                    suggestion="regenerate source through the verified Ascend emitter.",
                )
            return value
    return None


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
        "logical_sequence_axis": -2,
        "key_mask_value": "-inf-before-softmax",
        "value_mask_value": 0.0,
    }


def _validate_ascend_access_template_contract(
    program: ssa.Program,
    schedule: Mapping[str, Any],
    tensor_specs=(),
) -> Mapping[str, Any] | None:
    """Validate and size the private padding/flatten/dot-loop path.

    Conv2d is admitted only when the frontend has already lowered it to the
    public access-template contract.  This verifier does not introduce a
    ``call.conv2d`` operation; it checks that the existing SSA loop and its
    templates describe one coherent tile before CANN sees the source.
    """
    contract = schedule.get("ascend_dot_loop")
    if not contract:
        return None
    if contract.get("mode") != "generic-dot-loop" or contract.get("layout") != "public-access-template":
        raise UnsupportedBackendOpError(
            "Ascend access-template Conv2d contract is unsupported.",
            reason="only the public generic dot-loop layout is modeled.",
            suggestion="lower Conv2d through padding, flatten, and one loop-carried dot.",
        )

    loops = [op for op in _walk_operations(program.blocks) if op.opcode == "scf.for"]
    dots = [op for op in _walk_operations(program.blocks) if op.opcode == "linalg.dot"]
    if len(loops) != 1 or len(dots) != 1:
        raise UnsupportedBackendOpError(
            "Ascend access-template Conv2d requires one dot loop.",
            reason=f"found {len(loops)} loops and {len(dots)} linalg.dot operations.",
            suggestion="keep one loop-carried accumulator around one tiled dot.",
        )
    loop = loops[0]
    if not loop.regions or len(tuple(dict(loop.attrs).get("iter_args", ()))) != 1:
        raise UnsupportedBackendOpError(
            "Ascend access-template Conv2d requires a loop-carried accumulator.",
            reason="the SSA loop has no single accumulator state.",
            suggestion="carry the FP32 output tile through scf.for.",
        )

    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }
    for operation in _walk_operations(program.blocks):
        value_types.update({value.name: value.type for value in operation.results})
    if len(dots[0].operands) != 2:
        raise UnsupportedBackendOpError(
            "Ascend access-template dot requires two extracted tiles.",
            reason=f"dot operands are {dots[0].operands!r}.",
            suggestion="preserve lhs/rhs tensor.extract operations before linalg.dot.",
        )
    lhs = value_types.get(dots[0].operands[0])
    rhs = value_types.get(dots[0].operands[1])
    result = dots[0].results[0].type if dots[0].results else None
    if (
        not lhs
        or not rhs
        or not result
        or len(lhs.shape) != 2
        or len(rhs.shape) != 2
        or len(result.shape) != 2
        or lhs.shape[1] != rhs.shape[0]
        or result.shape != (lhs.shape[0], rhs.shape[1])
    ):
        raise UnsupportedBackendOpError(
            "Ascend access-template dot tile shapes are inconsistent.",
            reason=f"lhs={getattr(lhs, 'shape', None)!r}, rhs={getattr(rhs, 'shape', None)!r}, result={getattr(result, 'shape', None)!r}.",
            suggestion="use equal M/N/K tile shapes in the flattened dot contract.",
        )
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise UnsupportedBackendOpError(
            "Ascend access-template dot requires rank-2 tiles.",
            reason="flattening did not produce matrix tiles.",
            suggestion="flatten the convolution windows before linalg.dot.",
        )

    template_tile_values = []
    for tensor in tensor_specs:
        for shape in tuple(getattr(tensor, "attrs", {}).get("dtype_shapes", ())):
            try:
                candidate = tuple(int(str(item)) for item in shape)
            except (TypeError, ValueError):
                continue
            if len(candidate) == 2 and all(item > 0 for item in candidate):
                template_tile_values.append(candidate)

    def static_int(value):
        try:
            parsed = int(str(value))
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    m, k_lhs = (static_int(value) for value in lhs.shape)
    k_rhs, n = (static_int(value) for value in rhs.shape)
    symbolic_tile = any("BLOCK_SIZE_" in str(value) for value in (*lhs.shape, *rhs.shape))
    if symbolic_tile:
        # The generic access-template frontend specializes its first verified
        # Conv2d tile to 16 lanes per matrix dimension before Triton rendering.
        # Keep this private bound explicit until symbolic tile bindings are
        # exported by the frontend.
        m = m or 16
        n = n or 16
        k_lhs = k_lhs or 16
        k_rhs = k_rhs or 16
    if None in {m, n, k_lhs, k_rhs} and template_tile_values:
        # The public arrangement retains concrete dtype tile shapes even when
        # SSA dimensions remain symbolic BLOCK_SIZE_* names.
        m, n = template_tile_values[-1]
        k_lhs = k_rhs = template_tile_values[0][1]
    if None in {m, n, k_lhs, k_rhs} or k_lhs != k_rhs:
        raise UnsupportedBackendOpError(
            "Ascend access-template dot tile dimensions must be static.",
            reason=f"lhs={lhs.shape!r}, rhs={rhs.shape!r}.",
            suggestion="specialize the tile dimensions before source emission.",
        )

    # Every public access template must carry both coordinates and the exact
    # predicate used by the load.  Padding is represented in the coordinate
    # expression; its mask must remain present so the target load can clamp the
    # physical pointer without changing the logical convolution.
    templates = []
    for tensor in tensor_specs:
        templates.extend(tuple(getattr(tensor, "attrs", {}).get("access_templates", ())))
    # Tensor specs are normally held on Kernel, so inspect operation attrs when
    # the program does not carry that optional side table.
    template_attrs = tuple(
        attr
        for operation in _walk_operations(program.blocks)
        for attr in tuple(operation.attrs.get("access_templates", ()))
    )
    templates.extend(template_attrs)
    padding_symbols = any(
        "padding_" in str(item.get("linear_offset", ""))
        or "padding_" in str(item.get("offsets", ""))
        for item in templates
        if isinstance(item, Mapping)
    )
    has_masks = all(
        isinstance(item, Mapping) and item.get("mask")
        for item in templates
    ) if templates else True
    if templates and not has_masks:
        raise UnsupportedBackendOpError(
            "Ascend access-template Conv2d requires load masks.",
            reason="one or more flattened tiles have coordinates without a mask.",
            suggestion="carry the same padding/stride/dilation predicate into tl.load.",
        )

    core_limit = schedule.get("core_dim_limit", 65535)
    try:
        core_limit = int(core_limit)
    except (TypeError, ValueError) as exc:
        raise UnsupportedBackendOpError(
            "Ascend access-template core limit is not an integer.",
            reason=f"received core_dim_limit={core_limit!r}.",
            suggestion="set max_core_dim to a positive integer.",
        ) from exc
    if core_limit <= 0 or core_limit > 65535:
        raise UnsupportedBackendOpError(
            "Ascend access-template core limit is outside the NPU range.",
            reason=f"received core_dim_limit={core_limit}.",
            suggestion="use a core/grid limit between 1 and 65535.",
        )

    output_spec = next(
        (tensor for tensor in tensor_specs if getattr(tensor, "name", "") in {"output", "out"}),
        None,
    )
    source_ranks = tuple(
        len(tuple(getattr(tensor, "attrs", {}).get("source_shape", ())))
        for tensor in tensor_specs
        if getattr(tensor, "attrs", {}).get("source_shape") is not None
    )
    # The public generic dot-loop is also used by ordinary matrix tests.  A
    # rank-4 input/filter/output trio identifies the im2col Conv2d lowering
    # without depending on a kernel or test name.
    is_conv2d = len(source_ranks) >= 3 and all(rank == 4 for rank in source_ranks[:3])
    grid_estimate = None
    if output_spec is not None:
        try:
            elements = 1
            for dimension in tuple(output_spec.attrs.get("source_shape", output_spec.shape)):
                elements *= int(str(dimension))
            grid_estimate = (elements + m * n - 1) // (m * n)
        except (AttributeError, TypeError, ValueError):
            grid_estimate = None
    if grid_estimate is not None and grid_estimate > core_limit:
        raise UnsupportedBackendOpError(
            "Ascend access-template Conv2d grid exceeds the core limit.",
            reason=f"estimated grid {grid_estimate} exceeds limit {core_limit}.",
            suggestion="split the launch domain or lower max_core_dim before CANN compilation.",
        )

    return {
        "mode": "generic-dot-loop",
        "operator": "conv2d-im2col" if is_conv2d else "generic-dot-loop",
        "source_ranks": source_ranks,
        "tile": {"m": m, "n": n, "k": k_lhs},
        "workspace_bytes": m * n * 4,
        "padding_coordinates": padding_symbols,
        "mask": has_masks,
        "stride_dilation": "encoded-in-access-template",
        "core_grid_limit": core_limit,
        "grid_estimate": grid_estimate,
    }


def _retile_ascend_conv2d_access_templates(kernel: Kernel) -> Kernel:
    """Retile rank-4 im2col templates before generic source emission.

    Conv2d's access expressions contain the original arrangement tile even
    after the private UB planner has selected a smaller matrix tile.  Update
    only the matrix axes and tile-count decoders; source feature strides and
    padding predicates remain tied to the original rank-4 tensors.
    """
    schedule = dict(kernel.ssa.metadata.get("schedule", {}))
    access = schedule.get("ascend_access_template_resources")
    if not isinstance(access, Mapping) or access.get("operator") != "conv2d-im2col":
        return kernel
    tile = access.get("tile")
    if not isinstance(tile, Mapping):
        return kernel
    m, n, k = (int(tile[axis]) for axis in ("m", "n", "k"))
    tensors = []
    changed = False
    for spec in kernel.tensors:
        attrs = dict(spec.attrs)
        templates = []
        for template in tuple(attrs.get("access_templates", ())):
            if not isinstance(template, Mapping):
                templates.append(template)
                continue
            updated = dict(template)
            shape = tuple(str(dim) for dim in updated.get("shape", ()))
            if len(shape) != 2:
                templates.append(template)
                continue
            # ``source_name`` is the generated tensor provenance identifier
            # (for example ``ninetoothed_tensor_0``), not the semantic
            # im2col role.  The TensorSpec name remains the stable role
            # contract (lhs/rhs/output) used by the access-template builder.
            role = str(spec.name).lower()
            if role in {"input", "lhs"}:
                updated["shape"] = (str(m), str(k))
            elif role in {"filter", "rhs"}:
                updated["shape"] = (str(k), str(n))
            elif role in {"output", "out"}:
                updated["shape"] = (str(m), str(n))
            else:
                updated["shape"] = shape
            updated["ascend_conv2d_tile"] = {"m": m, "n": n, "k": k}
            templates.append(updated)
            changed = True
        if templates:
            attrs["access_templates"] = tuple(templates)
        dtype_shapes = list(attrs.get("dtype_shapes", ()))
        if dtype_shapes and len(tuple(dtype_shapes[-1])) == 2:
            role = str(spec.name).lower()
            if role in {"input", "lhs"}:
                dtype_shapes[-1] = (str(m), str(k))
            elif role in {"filter", "rhs"}:
                dtype_shapes[-1] = (str(k), str(n))
            elif role in {"output", "out"}:
                dtype_shapes[-1] = (str(m), str(n))
            attrs["dtype_shapes"] = tuple(dtype_shapes)
            changed = True
        tensors.append(replace(spec, attrs=attrs) if changed else spec)
    if not changed:
        return kernel
    return replace(kernel, tensors=tuple(tensors))


def _ascend_attention_loop_contract(
    program: ssa.Program, *, require_value_mask: bool = True
) -> Mapping[str, Any] | None:
    """Build a structured contract for one online-softmax SSA candidate."""

    loops = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    )
    candidates = []
    for loop in loops:
        body = tuple(_walk_operations(loop.regions))
        dots = tuple(operation for operation in body if operation.opcode == "linalg.dot")
        if (
            len(dots) == 2
            and any(operation.opcode == "reduce.max" for operation in body)
            and any(operation.opcode == "reduce.sum" for operation in body)
            and any(operation.opcode == "math.exp2" for operation in body)
        ):
            candidates.append(loop)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise UnsupportedBackendOpError(
            "Ascend attention requires exactly one online-softmax candidate.",
            reason=f"found {len(candidates)} candidate scf.for loops.",
            suggestion="lower one structured Attention loop before Ascend scheduling.",
        )

    loop = candidates[0]
    body = tuple(_walk_operations(loop.regions))
    iter_args = tuple(dict(loop.attrs).get("iter_args", ()))
    if len(iter_args) != 3:
        raise UnsupportedBackendOpError(
            "Ascend Attention candidate requires three loop-carried states.",
            reason=f"found {len(iter_args)} loop-carried values.",
            suggestion="carry acc, m_i, and l_i through the online-softmax loop.",
        )
    if not any(operation.opcode == "scf.if" for operation in body):
        raise UnsupportedBackendOpError(
            "Ascend Attention candidate requires structured control flow.",
            reason="the online-softmax loop has no scf.if branch.",
            suggestion="preserve causal or all-masked structured control flow.",
        )

    dots = tuple(operation for operation in body if operation.opcode == "linalg.dot")
    score_dot, value_dot = dots
    producers = {
        result.name: operation
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    }
    input_names = {
        value.name
        for value in (*program.inputs, *program.outputs)
        if value.type.kind == "tensor"
    }

    def source_root(name: str, seen: set[str] | None = None) -> str | None:
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen.add(name)
        if name in input_names:
            return name
        operation = producers.get(name)
        if operation is None:
            return None
        for operand in operation.operands:
            root = source_root(operand, seen)
            if root is not None:
                return root
        return None

    def operand_shape(name: str) -> tuple[str, ...]:
        operation = producers.get(name)
        if operation is not None and operation.opcode in {"tensor.extract", "linalg.transpose"}:
            source = source_root(operation.operands[0]) if operation.operands else None
            if source is not None:
                for value in (*program.inputs, *program.outputs):
                    if value.name == source:
                        attrs = dict(value.type.attrs)
                        shapes = tuple(attrs.get("dtype_shapes", ()))
                        if shapes:
                            return tuple(str(dim) for dim in shapes[-1])
        if operation is not None and operation.results:
            return tuple(str(dim) for dim in operation.results[0].type.shape)
        for value in (*program.inputs, *program.outputs):
            if value.name == name:
                return tuple(str(dim) for dim in value.type.shape)
        return ()

    score_lhs, score_rhs = score_dot.operands[:2]
    value_lhs, value_rhs = value_dot.operands[:2]
    q_source = source_root(score_lhs)
    k_source = source_root(score_rhs)
    v_source = source_root(value_rhs)
    if q_source is None or k_source is None or v_source is None:
        raise UnsupportedBackendOpError(
            "Ascend Attention dot provenance is incomplete.",
            reason=(
                f"score sources q={q_source!r}, k={k_source!r}; "
                f"value source v={v_source!r}."
            ),
            suggestion="preserve Q/K/V tensor.extract provenance through dot lowering.",
        )

    score_lhs_shape = operand_shape(score_lhs)
    score_rhs_shape = operand_shape(score_rhs)
    score_result_shape = tuple(str(dim) for dim in score_dot.results[0].type.shape)
    value_lhs_shape = operand_shape(value_lhs)
    value_rhs_shape = operand_shape(value_rhs)
    value_result_shape = tuple(str(dim) for dim in value_dot.results[0].type.shape)
    if (
        len(score_lhs_shape) != 2
        or len(score_rhs_shape) != 2
        or len(score_result_shape) != 2
        or len(value_lhs_shape) != 2
        or len(value_rhs_shape) != 2
        or len(value_result_shape) != 2
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention dot shapes do not match score/value roles.",
            reason=(
                f"score=({score_lhs_shape},{score_rhs_shape})->{score_result_shape}; "
                f"value=({value_lhs_shape},{value_rhs_shape})->{value_result_shape}."
            ),
            suggestion="preserve QK=[query,key] and PV=[query,value] dot shapes.",
        )

    negative_inf = {
        result.name
        for operation in body
        if operation.opcode == "arith.constant"
        and str(operation.attrs.get("value")) in {"-inf", "-Infinity", "-float('inf')"}
        for result in operation.results
    }
    score_masks = tuple(
        operation
        for operation in body
        if operation.opcode == "select.where"
        and len(operation.operands) == 3
        and operation.operands[1] == score_dot.results[0].name
        and operation.operands[2] in negative_inf
    )
    causal_masks = tuple(
        operation
        for operation in body
        if operation.opcode == "cmp.ge"
        and len(operation.operands) == 2
    )
    value_mask = next(
        (
            operation
            for operation in body
            if operation.opcode == "select.where"
            and operation.attrs.get("ascend_attention_mask") == "value"
            and operation.results
            and operation.results[0].name in value_dot.operands
        ),
        None,
    )
    if not score_masks:
        raise UnsupportedBackendOpError(
            "Ascend Attention contract is missing the K score mask.",
            reason="no select.where(score, -inf) feeds the score path.",
            suggestion="materialize the K bounds mask before reduce.max.",
        )
    if value_mask is None and require_value_mask:
        raise UnsupportedBackendOpError(
            "Ascend Attention contract is missing the V value mask.",
            reason="the second dot has no explicit zero-valued V select.",
            suggestion="run Ascend V value-mask normalization before contract construction.",
        )

    state_names = tuple(str(item.get("name")) for item in iter_args)
    state_roles = {
        role: name
        for role, name in zip(("acc", "m_i", "l_i"), state_names, strict=True)
    }
    yields = (
        loop.regions[0].operations[-1]
        if loop.regions[0].operations
        and loop.regions[0].operations[-1].opcode == "scf.yield"
        else None
    )
    if yields is None or len(yields.operands) != 3:
        raise UnsupportedBackendOpError(
            "Ascend Attention contract has no three-value loop yield.",
            reason="loop region does not end with acc/m_i/l_i yield.",
            suggestion="preserve the three online-softmax loop-carried values.",
        )

    specs = {value.name: value for value in (*program.inputs, *program.outputs)}

    def rank4_access(name: str, role: str) -> Mapping[str, Any]:
        value = specs.get(name)
        attrs = {} if value is None else dict(value.type.attrs)
        source_shape = tuple(str(dim) for dim in attrs.get("source_shape", ()))
        if len(source_shape) != 4:
            raise UnsupportedBackendOpError(
                "Ascend Attention requires rank-4 access provenance.",
                reason=f"{role} source `{name}` has shape {source_shape!r}.",
                suggestion="preserve [batch, head, position, dim] source metadata.",
            )
        position = "query_position" if role in {"q", "o"} else "key_position"
        dimension = "head_dim" if role in {"q", "k"} else "value_dim"
        return {
            "tensor": name,
            "coordinates": ("batch", "head", position, dimension),
            "source_shape": source_shape,
            "access_templates": len(tuple(attrs.get("access_templates", ()))),
        }

    output_source = source_root(program.outputs[0].name) if program.outputs else None
    if output_source is None:
        output_source = program.outputs[0].name if program.outputs else None
    if output_source is None:
        raise UnsupportedBackendOpError(
            "Ascend Attention contract cannot identify output provenance.",
            reason="the program has no tensor output.",
            suggestion="preserve the O tensor binding through SSA lowering.",
        )

    access = {
        "q": rank4_access(q_source, "q"),
        "k": rank4_access(k_source, "k"),
        "v": rank4_access(v_source, "v"),
        "o": rank4_access(output_source, "o"),
    }
    source_shapes = {
        role: tuple(item["source_shape"])
        for role, item in access.items()
    }
    def dimensions_compatible(*values: str) -> bool:
        concrete = set()
        for value in values:
            try:
                concrete.add(int(value))
            except (TypeError, ValueError):
                continue
        return len(concrete) <= 1

    if not (
        dimensions_compatible(*(source_shapes[role][0] for role in ("q", "k", "v", "o")))
        and dimensions_compatible(*(source_shapes[role][1] for role in ("q", "k", "v", "o")))
        and dimensions_compatible(source_shapes["q"][2], source_shapes["o"][2])
        and dimensions_compatible(source_shapes["k"][2], source_shapes["v"][2])
        and dimensions_compatible(source_shapes["q"][3], source_shapes["k"][3])
        and dimensions_compatible(source_shapes["v"][3], source_shapes["o"][3])
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention batch/head provenance is inconsistent.",
            reason=f"source shapes={source_shapes!r}.",
            suggestion="require equal batch and head dimensions for Q/K/V/O.",
        )

    def static_int(value: Any) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    sequence = static_int(source_shapes["q"][-2])
    head_dim = static_int(source_shapes["q"][-1])
    schedule = dict(program.metadata.get("schedule", {}))
    tile = dict(schedule.get("ascend_attention_tile", {}))
    if not tile:
        tile = dict(schedule.get("ascend_dot_loop", {}).get("tile", {}))
    key_valid = schedule.get("ascend_attention_key_valid")
    state_normalization = schedule.get("ascend_attention_loop_state")
    if not require_value_mask and (
        not isinstance(key_valid, Mapping) or not isinstance(state_normalization, Mapping)
    ):
        return {
            "version": 1,
            "kind": "generic-online-softmax-loop-candidate",
            "mode": "candidate",
            "status": "recognized-before-normalization",
            "tile": tile,
        }
    if not isinstance(key_valid, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention key-valid contract is missing.",
            reason="normalization metadata was not preserved before contract construction.",
            suggestion="run key-valid normalization before building the Attention contract.",
        )
    if not isinstance(state_normalization, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention loop-state contract is missing.",
            reason="state-preserving branch metadata was not preserved before contract construction.",
            suggestion="run online-softmax loop-state normalization before contract construction.",
        )
    all_masked_name = key_valid.get("all_masked")
    state_if = next(
        (
            operation
            for operation in body
            if operation.opcode == "scf.if"
            and operation.attrs.get("ascend_attention_state_normalization")
        ),
        None,
    )
    if state_if is None or state_if.operands[0] != all_masked_name:
        raise UnsupportedBackendOpError(
            "Ascend Attention state-preserving branch is not connected.",
            reason=f"expected all_masked={all_masked_name!r} in a result-producing scf.if.",
            suggestion="connect all_masked to the normalized acc/m_i/l_i branch.",
        )
    causal_mode = "causal" if causal_masks else "non-causal"

    return {
        "version": 2,
        "causal": causal_mode,
        "mode": causal_mode,
        "layout": "public-access-template",
        "status": "verified-static-public-online-softmax",
        "kind": "generic-online-softmax-loop",
        "score_dot": {
            "operation": score_dot.results[0].name,
            "lhs": q_source,
            "rhs": k_source,
            "reduction_axis": "head_dim",
            "lhs_shape": score_lhs_shape,
            "rhs_shape": score_rhs_shape,
            "result_shape": score_result_shape,
        },
        "value_dot": {
            "operation": value_dot.results[0].name,
            "lhs": value_lhs,
            "rhs": v_source,
            "reduction_axis": "key_position",
            "lhs_shape": value_lhs_shape,
            "rhs_shape": value_rhs_shape,
            "result_shape": value_result_shape,
        },
        "k_score_mask": {
            "operations": tuple(operation.results[0].name for operation in score_masks),
            "invalid_value": "-inf",
            "before": "reduce.max",
        },
        "key_bounds_mask": {
            "operation": score_masks[0].results[0].name,
            "predicate": score_masks[0].operands[0],
            "invalid_value": "-inf",
            "before": "reduce.max",
            "source": "K.access-template.bounds",
        },
        "combined_key_valid": {
            "operation": key_valid.get("predicate"),
            "bounds_predicate": key_valid.get("bounds_predicate"),
            "causal_predicate": key_valid.get("causal_predicate"),
            "source": key_valid.get("source"),
        },
        "all_masked_predicate": {
            "operation": all_masked_name,
            "input": key_valid.get("not_predicate"),
            "axis": key_valid.get("axis", "key_position"),
            "source": key_valid.get("source"),
        },
        "v_value_mask": (
            {
                "operation": value_mask.results[0].name,
                "predicate": value_mask.operands[0],
                "load": value_mask.operands[1],
                "zero": value_mask.operands[2],
                "invalid_value": 0.0,
            }
            if value_mask is not None
            else None
        ),
        "causal_mask": {
            "operations": tuple(operation.results[0].name for operation in causal_masks),
            "predicate": "query_position >= key_position",
        },
        "loop_carried_state": {
            "roles": state_roles,
            "yield": tuple(yields.operands),
            "dtype": "float32",
        },
        "state_preserving_branch": {
            "operation": state_if.results[0].name,
            "predicate": state_if.operands[0],
            "old_state": tuple(state_if.attrs.get("old_state", ())),
            "new_state": tuple(state_if.attrs.get("new_state", ())),
            "roles": tuple(state_if.attrs.get("state_roles", ())),
            "dtype": "float32",
            "provenance": state_if.attrs.get("provenance"),
        },
        "access_provenance": access,
        "tile": tile,
        "sequence": sequence,
        "head_dim": head_dim,
    }


def _normalize_ascend_attention_value_mask(program: ssa.Program) -> ssa.Program:
    """Materialize the V bounds predicate for the recognized value dot.

    Public layout lowering retains V's source bounds in its access-template
    metadata.  CUDA can apply that predicate while emitting a masked load, but
    Ascend's private verifier needs the zero value to be visible in SSA before
    the second dot.  This pass only rewrites the V operand of the recognized
    online-softmax value dot; score dots and unrelated operations are untouched.
    """
    loops = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    )
    if len(loops) != 1 or len(loops[0].regions) != 1:
        raise UnsupportedBackendOpError(
            "Ascend attention value-mask normalization requires one structured loop.",
            reason=f"found {len(loops)} candidate loops.",
            suggestion="preserve one online-softmax scf.for before Ascend lowering.",
        )

    loop = loops[0]
    body = tuple(_walk_operations(loop.regions))
    dots = tuple(operation for operation in body if operation.opcode == "linalg.dot")
    if len(dots) != 2:
        raise UnsupportedBackendOpError(
            "Ascend attention value-mask normalization requires two dots.",
            reason=f"found {len(dots)} linalg.dot operations.",
            suggestion="preserve score and value dots in the online-softmax contract.",
        )

    value_dot = dots[1]
    if len(value_dot.operands) < 2:
        raise UnsupportedBackendOpError(
            "Ascend attention value dot has no V operand.",
            reason="the second linalg.dot does not have two operands.",
            suggestion="preserve P and V operands through SSA lowering.",
        )

    v_name = value_dot.operands[1]
    producers = {
        result.name: operation
        for operation in body
        for result in operation.results
    }
    v_load = producers.get(v_name)
    if v_load is None or v_load.opcode != "tensor.extract" or not v_load.operands:
        raise UnsupportedBackendOpError(
            "Ascend attention value-mask normalization cannot identify V load.",
            reason=f"value dot operand `{v_name}` is not a tensor.extract.",
            suggestion="preserve tensor.extract(V, key_tile) before the value dot.",
        )

    source_tensor = v_load.operands[0]
    existing = {
        value.name
        for value in (*program.inputs, *program.outputs)
    }
    existing.update(
        result.name
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    )

    def fresh(prefix: str, type_: ssa.Type) -> ssa.Value:
        index = 0
        name = f"%ascend_{prefix}"
        while name in existing:
            index += 1
            name = f"%ascend_{prefix}_{index}"
        existing.add(name)
        return ssa.Value(name=name, type=type_)

    index_type = ssa.Type(kind="index")
    value_type = v_load.results[0].type
    key_position = fresh("v_key_position", index_type)
    sequence = fresh("v_sequence", index_type)
    predicate = fresh("v_valid", ssa.Type(kind="scalar", dtype="bool"))
    zero = fresh("v_zero", value_type)
    masked = fresh("v_masked", value_type)

    inserted = (
        ssa.Operation(
            opcode="index.offset",
            operands=(v_name,),
            results=(key_position,),
            attrs={
                "dim": -2,
                "ascend_attention_mask": "value",
                "source": "access-template",
            },
        ),
        ssa.Operation(
            opcode="shape.dim",
            operands=(source_tensor,),
            results=(sequence,),
            attrs={
                "dim": -2,
                "source": True,
                "ascend_attention_mask": "value",
            },
        ),
        ssa.Operation(
            opcode="cmp.lt",
            operands=(key_position.name, sequence.name),
            results=(predicate,),
            attrs={
                "ascend_attention_mask": "value",
                "predicate_source": "V.access-template",
            },
        ),
        ssa.Operation(
            opcode="arith.constant",
            results=(zero,),
            attrs={
                "value": 0.0,
                "dtype": value_type.dtype,
                "ascend_attention_mask": "value",
            },
        ),
        ssa.Operation(
            opcode="select.where",
            operands=(predicate.name, v_name, zero.name),
            results=(masked,),
            attrs={
                "ascend_attention_mask": "value",
                "predicate_source": "V.access-template",
                "consumer": "value-dot",
                "zero_value": 0.0,
            },
        ),
    )

    def rewrite_block(block: ssa.Block) -> ssa.Block:
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite_block(region) for region in operation.regions)
            current = replace(operation, regions=regions)
            if operation is value_dot:
                current = replace(
                    current,
                    operands=(current.operands[0], masked.name, *current.operands[2:]),
                )
                operations.extend(inserted)
            operations.append(current)
        return replace(block, operations=tuple(operations))

    rewritten = replace(program, blocks=tuple(rewrite_block(block) for block in program.blocks))
    schedule = dict(rewritten.metadata.get("schedule", {}))
    schedule["ascend_attention_value_mask"] = {
        "source": "V.access-template",
        "predicate": predicate.name,
        "load": v_name,
        "zero": zero.name,
        "masked": masked.name,
        "consumer": "value-dot",
        "dtype": value_type.dtype,
    }
    normalized = replace(
        rewritten,
        metadata=dict(rewritten.metadata) | {"schedule": schedule},
    )
    ssa.verify_program(normalized)
    return normalized


def _normalize_ascend_attention_key_valid(program: ssa.Program) -> ssa.Program:
    """Materialize a unified key-valid and all-masked predicate.

    The public lowering may expose bounds and causal predicates as separate
    score selects.  Ascend keeps their provenance separate, then records one
    key-valid predicate for the score path and an explicit ``reduce.all``
    reduction for all-masked tile detection.  This pass only handles the
    recognized two-dot online-softmax candidate and does not rewrite tensor
    coordinates or value loads.
    """
    loops = tuple(
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    )
    if len(loops) != 1 or len(loops[0].regions) != 1:
        raise UnsupportedBackendOpError(
            "Ascend attention key-valid normalization requires one structured loop.",
            reason=f"found {len(loops)} candidate loops.",
            suggestion="preserve one online-softmax scf.for before Ascend lowering.",
        )
    loop = loops[0]
    body = tuple(_walk_operations(loop.regions))
    dots = tuple(operation for operation in body if operation.opcode == "linalg.dot")
    if len(dots) != 2:
        raise UnsupportedBackendOpError(
            "Ascend attention key-valid normalization requires two dots.",
            reason=f"found {len(dots)} linalg.dot operations.",
            suggestion="preserve score and value dots in the online-softmax contract.",
        )
    score_dot = dots[0]
    producers = {
        result.name: operation
        for operation in body
        for result in operation.results
    }

    def source_root(name: str, seen: set[str] | None = None) -> str | None:
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen.add(name)
        operation = producers.get(name)
        if operation is None:
            return name
        for operand in operation.operands:
            root = source_root(operand, seen.copy())
            if root is not None:
                return root
        return None

    score_masks = tuple(
        operation
        for operation in body
        if operation.opcode == "select.where"
        and len(operation.operands) == 3
        and operation.operands[1] == score_dot.results[0].name
    )
    if not score_masks:
        raise UnsupportedBackendOpError(
            "Ascend attention key-valid normalization cannot find score mask.",
            reason="the score dot has no explicit select.where predicate.",
            suggestion="preserve the K bounds score mask before reduce.max.",
        )

    # The bounds predicate is defined after the score dot.  A causal predicate
    # may be defined inside the following scf.if, so it cannot be referenced
    # in the parent region without being yielded as an if result.
    bounds_mask = score_masks[0]
    bounds_predicate = bounds_mask.operands[0]
    causal_masks = tuple(
        operation
        for operation in body
        if operation.opcode == "select.where"
        and len(operation.operands) == 3
        and operation.operands[1] == score_masks[-1].results[0].name
        and operation is not bounds_mask
    )
    causal_predicate = causal_masks[0].operands[0] if causal_masks else None
    causal_if = next(
        (
            operation for operation in loop.regions[0].operations
            if operation.opcode == "scf.if"
            and causal_masks
            and any(mask is nested for region in operation.regions
                    for nested in region.operations for mask in causal_masks)
        ),
        None,
    )
    if causal_predicate is not None and (
        causal_if is None or len(causal_if.regions) != 2
        or len(causal_if.results) != 1
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention cannot propagate causal key validity.",
            reason="the causal predicate is not in a one-result, two-region scf.if.",
            suggestion="preserve the causal predicate and score as structured if results.",
        )

    existing = {
        value.name for value in (*program.inputs, *program.outputs)
    }
    existing.update(
        result.name
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    )

    def fresh(prefix: str, type_: ssa.Type) -> ssa.Value:
        index = 0
        name = f"%ascend_{prefix}"
        while name in existing:
            index += 1
            name = f"%ascend_{prefix}_{index}"
        existing.add(name)
        return ssa.Value(name=name, type=type_)

    bounds_producer = producers.get(bounds_predicate)
    if bounds_producer is None or not bounds_producer.results:
        raise UnsupportedBackendOpError(
            "Ascend Attention cannot trace the K bounds predicate.",
            reason=f"predicate {bounds_predicate!r} has no SSA producer.",
            suggestion="preserve K source bounds as an explicit cmp operation.",
        )
    score_type = score_dot.results[0].type
    score_shape = tuple(str(dim) for dim in score_type.shape)
    if len(score_shape) != 2:
        raise UnsupportedBackendOpError(
            "Ascend Attention score mask requires a rank-2 score tile.",
            reason=f"score result shape={score_shape!r}.",
            suggestion="preserve the planned M/N score tile before mask normalization.",
        )
    bool_type = bounds_producer.results[0].type
    bounds_valid = fresh("bounds_valid", bool_type)
    key_valid = fresh("key_valid", bool_type) if causal_if is not None else bounds_valid
    not_key_valid = fresh("not_key_valid", bool_type)
    false_value = fresh("false", bool_type)
    all_masked = fresh("all_masked", ssa.Type(kind="scalar", dtype="bool"))
    bounds_identity = ssa.Operation(
        opcode="select.where",
        operands=(bounds_predicate, bounds_predicate, bounds_predicate),
        results=(bounds_valid,),
        attrs={
            "ascend_attention_mask": "bounds_valid",
            "score_mask_role": "qk-score-bounds",
            "predicate_source": "K.access-template.bounds",
            "bounds_predicate": bounds_predicate,
            "causal_predicate": causal_predicate,
            "preserves_coordinates": True,
            "score_shape": score_shape,
            "sequence": str(
                next(
                    (
                        tuple(value.type.attrs.get("source_shape", value.type.shape))[-2]
                        for value in program.inputs
                        if value.name == "k"
                    ),
                    score_shape[1],
                )
            ),
        },
    )
    after_score = (
        ssa.Operation(
            opcode="arith.constant",
            results=(false_value,),
            attrs={"value": False, "dtype": "bool"},
        ),
        ssa.Operation(
            opcode="cmp.eq",
            operands=(key_valid.name, false_value.name),
            results=(not_key_valid,),
            attrs={
                "ascend_attention_mask": "key_valid",
                "predicate_source": "K.access-template.bounds",
            },
        ),
        ssa.Operation(
            opcode="reduce.all",
            operands=(not_key_valid.name,),
            results=(all_masked,),
            attrs={
                # This scalar predicate means that every query/key pair in
                # the current tile is invalid.  In causal mode key validity
                # has rank two, so reducing only one axis would leave a
                # vector and make it invalid as the scalar condition of the
                # state-preserving scf.if.
                "axis": None,
                "axis_role": "all_key_valid_elements",
                "ascend_attention_mask": "all_masked",
                "predicate_source": "K.access-template.bounds",
                "key_valid": key_valid.name,
                "causal_predicate": causal_predicate,
            },
        ),
    )
    combined = fresh("causal_key_valid", bool_type) if causal_if is not None else None

    def rewrite_block(block: ssa.Block) -> ssa.Block:
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite_block(region) for region in operation.regions)
            current = replace(operation, regions=regions)
            if operation is bounds_mask:
                current = replace(
                    current,
                    operands=(bounds_valid.name, *current.operands[1:]),
                    attrs=dict(current.attrs)
                    | {
                        "predicate_source": "Ascend.bounds_valid",
                        "bounds_valid": bounds_valid.name,
                    },
                )
                operations.append(bounds_identity)
            if operation is causal_if:
                then_region, else_region = current.regions
                then_yield = then_region.operations[-1]
                else_yield = else_region.operations[-1]
                if (
                    then_yield.opcode != "scf.yield"
                    or else_yield.opcode != "scf.yield"
                ):
                    raise UnsupportedBackendOpError(
                        "Ascend Attention causal branch has no structured yields.",
                        reason="the causal score branch cannot yield key validity.",
                        suggestion="preserve scf.yield in both causal branches.",
                    )
                then_ops = then_region.operations[:-1] + (
                    ssa.Operation(
                        opcode="arith.and",
                        operands=(bounds_valid.name, causal_predicate),
                        results=(combined,),
                        attrs={"ascend_attention_mask": "key_valid"},
                    ),
                    replace(then_yield, operands=(*then_yield.operands, combined.name)),
                )
                else_ops = else_region.operations[:-1] + (
                    replace(
                        else_yield,
                        operands=(*else_yield.operands, bounds_valid.name),
                    ),
                )
                current = replace(
                    current,
                    results=(*current.results, key_valid),
                    regions=(
                        replace(then_region, operations=then_ops),
                        replace(else_region, operations=else_ops),
                    ),
                )
            operations.append(current)
            if operation is causal_if or (causal_if is None and operation is bounds_mask):
                operations.extend(after_score)
        return replace(block, operations=tuple(operations))

    rewritten = replace(program, blocks=tuple(rewrite_block(block) for block in program.blocks))
    schedule = dict(rewritten.metadata.get("schedule", {}))
    schedule["ascend_attention_key_valid"] = {
        "predicate": key_valid.name,
        "bounds_predicate": bounds_predicate,
        "causal_predicate": causal_predicate,
        "not_predicate": not_key_valid.name,
        "all_masked": all_masked.name,
        "axis": "all_key_valid_elements",
        "source": "K.bounds+causal" if causal_if is not None else "K.access-template.bounds",
        "coordinates_preserved": True,
    }
    normalized = replace(
        rewritten,
        metadata=dict(rewritten.metadata) | {"schedule": schedule},
    )
    ssa.verify_program(normalized)
    return normalized


def _normalize_ascend_attention_loop_state(program: ssa.Program) -> ssa.Program:
    """Guard online-softmax state updates with the normalized all-masked flag."""
    loops = tuple(
        operation for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    )
    if len(loops) != 1 or len(loops[0].regions) != 1:
        raise UnsupportedBackendOpError(
            "Ascend attention state normalization requires one structured loop.",
            reason=f"found {len(loops)} candidate loops.",
            suggestion="preserve one online-softmax scf.for before Ascend lowering.",
        )
    loop = loops[0]
    iter_args = tuple(dict(loop.attrs).get("iter_args", ()))
    region = loop.regions[0]
    if len(iter_args) != 3 or len(region.args) != 4:
        raise UnsupportedBackendOpError(
            "Ascend attention state normalization requires acc, m_i, and l_i.",
            reason=f"iter_args={len(iter_args)}, block_args={len(region.args)}.",
            suggestion="preserve three FP32 loop-carried states.",
        )
    schedule = dict(program.metadata.get("schedule", {}))
    key_contract = schedule.get("ascend_attention_key_valid")
    all_masked = key_contract.get("all_masked") if isinstance(key_contract, Mapping) else None
    if not all_masked:
        raise UnsupportedBackendOpError(
            "Ascend attention state normalization requires all_masked provenance.",
            reason="the key-valid normalization did not publish an all-masked predicate.",
            suggestion="run key-valid normalization before loop-state normalization.",
        )
    # Only move the loop's direct operations into the normal branch.  Using
    # `_walk_operations` here would flatten nested regions and duplicate the
    # score/value computations when rebuilding the scf.if.
    body = list(region.operations)
    yields = body[-1] if body and body[-1].opcode == "scf.yield" else None
    if yields is None or len(yields.operands) != 3:
        raise UnsupportedBackendOpError(
            "Ascend attention state normalization requires a three-value yield.",
            reason="online-softmax loop has no acc/m_i/l_i yield.",
            suggestion="preserve the three state updates through scf.yield.",
        )
    state_args = tuple(argument.name for argument in region.args[1:])
    updates = tuple(yields.operands)
    value_types = {
        value.name: value.type for value in (*program.inputs, *program.outputs)
    }
    value_types.update(
        {
            result.name: result.type
            for operation in _walk_operations(program.blocks)
            for result in operation.results
        }
    )
    value_types.update(
        {
            argument.name: argument.type
            for operation in _walk_operations(program.blocks)
            for region_ in operation.regions
            for argument in region_.args
        }
    )
    for name in (*state_args, *updates):
        type_ = value_types.get(name)
        if type_ is None or normalize_ascend_dtype(type_.dtype) != "float32":
            raise UnsupportedBackendOpError(
                "Ascend attention loop state must remain FP32.",
                reason=f"state `{name}` has type {None if type_ is None else type_.dtype!r}.",
                suggestion="normalize acc, m_i, and l_i states to FP32 before emission.",
            )
    if any(name == state for name, state in zip(updates, state_args, strict=True)):
        raise UnsupportedBackendOpError(
            "Ascend attention state normalization found no distinct update path.",
            reason=f"updates={updates!r}, state_args={state_args!r}.",
            suggestion="preserve explicit normal-path state updates before normalization.",
        )

    # The branch controls the complete score/value/reduction/state update path.
    # All-masked yields the old block arguments without executing normal_body.
    all_masked_index = next(
        index
        for index, operation in enumerate(body[:-1])
        if any(result.name == all_masked for result in operation.results)
    )
    normal_body = tuple(body[all_masked_index + 1 : -1]) + (
        ssa.Operation(opcode="scf.yield", operands=updates),
    )
    masked_region = ssa.Block(
        name="all_masked",
        operations=(ssa.Operation(opcode="scf.yield", operands=state_args),),
    )
    normal_region = ssa.Block(name="normal", operations=normal_body)
    state_if_results = tuple(
        ssa.Value(name=f"%ascend_state_{index}", type=region.args[index + 1].type)
        for index in range(3)
    )
    existing = {
        value.name for value in (*program.inputs, *program.outputs)
    }
    existing.update(
        result.name for operation in _walk_operations(program.blocks)
        for result in operation.results
    )
    state_if_results = tuple(
        ssa.Value(
            name=(
                result.name
                if result.name not in existing
                else f"{result.name}_{index}"
            ),
            type=result.type,
        )
        for index, result in enumerate(state_if_results)
    )
    state_if = ssa.Operation(
        opcode="scf.if",
        operands=(all_masked,),
        results=state_if_results,
        attrs={
            "ascend_attention_state_normalization": True,
            "all_masked": all_masked,
            "state_roles": ("acc", "m_i", "l_i"),
            "old_state": state_args,
            "new_state": updates,
            "provenance": "Ascend.online-softmax.all-masked",
        },
        regions=(masked_region, normal_region),
    )
    prefix = tuple(body[: all_masked_index + 1])
    new_operations = prefix + (
        state_if,
        ssa.Operation(
            opcode="scf.yield",
            operands=tuple(result.name for result in state_if_results),
        ),
    )
    rewritten_region = replace(region, operations=new_operations)
    rewritten_loop = replace(loop, regions=(rewritten_region,))

    def rewrite_block(block: ssa.Block) -> ssa.Block:
        operations = []
        for operation in block.operations:
            regions = tuple(rewrite_block(child) for child in operation.regions)
            current = replace(operation, regions=regions)
            if operation is loop:
                current = rewritten_loop
            operations.append(current)
        return replace(block, operations=tuple(operations))

    rewritten = replace(program, blocks=tuple(rewrite_block(block) for block in program.blocks))
    schedule["ascend_attention_loop_state"] = {
        "all_masked": all_masked,
        "roles": {"acc": state_args[0], "m_i": state_args[1], "l_i": state_args[2]},
        "old_state": state_args,
        "new_state": updates,
        "branch": state_if_results[0].name,
        "dtype": "float32",
        "provenance": "Ascend.online-softmax.all-masked",
    }
    normalized = replace(rewritten, metadata=dict(rewritten.metadata) | {"schedule": schedule})
    ssa.verify_program(normalized)
    return normalized


def _plan_ascend_attention_contract(
    program: ssa.Program, contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Plan the first static, single-block Ascend Attention contract."""
    if not isinstance(contract, Mapping) or contract.get("kind") != "generic-online-softmax-loop":
        raise UnsupportedBackendOpError(
            "Ascend Attention planner received an invalid structured contract.",
            reason="only the generic online-softmax contract is supported.",
            suggestion="construct the verified structured Attention contract first.",
        )
    access = contract.get("access_provenance")
    if isinstance(access, Mapping):
        values_by_name = {value.name: value for value in (*program.inputs, *program.outputs)}
        role_dtypes = {}
        for role in ("q", "k", "v", "o"):
            entry = access.get(role)
            tensor_name = entry.get("tensor") if isinstance(entry, Mapping) else None
            value = values_by_name.get(str(tensor_name)) if tensor_name is not None else None
            dtype = normalize_ascend_dtype(
                getattr(getattr(value, "type", None), "dtype", None)
            )
            if dtype is not None:
                role_dtypes[role] = dtype
        # Access provenance can refer to frontend aliases rather than the
        # final SSA argument names.  Include tensor-valued program arguments
        # so unsupported storage dtypes are rejected before state
        # normalization or source generation.
        argument_dtypes = {
            normalize_ascend_dtype(getattr(getattr(value, "type", None), "dtype", None))
            for value in program.inputs
            if normalize_ascend_dtype(getattr(getattr(value, "type", None), "dtype", None))
            not in {None, "bool"}
        }
        unsupported = sorted(
            {
                dtype
                for dtype in (*role_dtypes.values(), *argument_dtypes)
                if dtype not in ASCEND_ATTENTION_DTYPE_REGISTRY
            }
        )
        if unsupported:
            raise UnsupportedBackendOpError(
                "Ascend Attention dtype capability is unsupported.",
                reason=(
                    f"Q/K/V/O roles={role_dtypes!r}; unsupported={unsupported!r}; "
                    f"verified registry={tuple(sorted(ASCEND_ATTENTION_DTYPE_REGISTRY))}."
                ),
                suggestion="use one uniform FP16, BF16, or FP32 Q/K/V/O dtype.",
            )
    if contract.get("mode") not in {"causal", "non-causal"}:
        raise UnsupportedBackendOpError(
            "Ascend Attention planner requires an explicit causal mode.",
            reason=f"received mode={contract.get('mode')!r}.",
            suggestion="record mode=causal or mode=non-causal in the structured contract.",
        )
    for field in (
        "score_dot", "value_dot", "key_bounds_mask", "combined_key_valid",
        "all_masked_predicate", "v_value_mask", "loop_carried_state",
        "state_preserving_branch", "access_provenance",
    ):
        if not isinstance(contract.get(field), Mapping):
            raise UnsupportedBackendOpError(
                "Ascend Attention planner received an incomplete contract.",
                reason=f"missing field `{field}`.",
                suggestion="run Attention verifier before resource planning.",
            )

    def positive_int(value: Any, field: str) -> int | None:
        """Resolve a dimension when specialized, otherwise keep it symbolic.

        The first ``make`` compilation is built from symbolic Tensor specs.
        Runtime tensor dimensions are supplied by the lazy materializer and
        trigger a concrete specialization before launch.  Rejecting the
        symbolic form here would prevent that specialization from occurring.
        """
        try:
            result = int(str(value))
        except (TypeError, ValueError):
            return None
        if result <= 0:
            raise UnsupportedBackendOpError(
                "Ascend Attention planner requires positive dimensions.",
                reason=f"{field}={result} is not positive.",
                suggestion="use a non-empty static Attention shape.",
            )
        return result

    if not isinstance(access, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention planner requires access provenance.",
            reason="rank-4 Q/K/V/O metadata is absent.",
            suggestion="build the structured access contract before planning.",
        )
    q_shape = tuple(access.get("q", {}).get("source_shape", ()))
    k_shape = tuple(access.get("k", {}).get("source_shape", ()))
    if len(q_shape) != 4 or len(k_shape) != 4:
        raise UnsupportedBackendOpError(
            "Ascend Attention planner requires rank-4 Q/K sources.",
            reason=f"q_shape={q_shape!r}, k_shape={k_shape!r}.",
            suggestion="use [batch,head,sequence,head_dim] source tensors.",
        )
    batch = positive_int(q_shape[0], "batch")
    heads = positive_int(q_shape[1], "heads")
    sequence = positive_int(contract.get("sequence"), "sequence")
    if sequence is None:
        sequence = positive_int(q_shape[2], "sequence")
    head_dim = positive_int(contract.get("head_dim"), "head_dim")
    if head_dim is None:
        head_dim = positive_int(q_shape[3], "head_dim")
    if head_dim is not None and head_dim != 64:
        raise UnsupportedBackendOpError(
            "Ascend Attention planner supports head_dim=64 only.",
            reason=f"received head_dim={head_dim}.",
            suggestion="use the first-version head_dim=64 contract.",
        )
    if sequence is not None and sequence > 1024:
        raise UnsupportedBackendOpError(
            "Ascend Attention planner supports sequence<=1024 only.",
            reason=f"received sequence={sequence}.",
            suggestion="use the bounded sequence-tiled Attention contract.",
        )

    verified_tiles = tuple(
        dict(zip(("m", "n", "k"), tile, strict=True))
        for tile in _ASCEND_ATTENTION_TILE_CANDIDATES
    )
    contract_tile = dict(contract.get("tile", {}))
    # The public arrangement may expose its 16x16x64 candidate before the
    # Ascend private retile hook runs. Validate that input candidate
    # structurally; the resource planner considers only verified private tiles.
    candidate_tile = {key: contract_tile.get(key) for key in ("m", "n", "k")}
    accepted_candidates = ({"m": 16, "n": 16, "k": 64}, *verified_tiles)
    if contract_tile and candidate_tile not in accepted_candidates:
        raise UnsupportedBackendOpError(
            "Ascend Attention planner received an unsupported tile.",
            reason=f"contract tile={contract_tile!r}, verified candidates={verified_tiles!r}.",
            suggestion="use a tile from the verified Attention resource candidates.",
        )
    resource_plan = _plan_ascend_attention_resources(
        program,
        head_dim=head_dim,
        head_dim_expression=(
            contract.get("head_dim")
            if contract.get("head_dim") is not None
            else q_shape[-1]
        ),
    )
    if resource_plan["selected_tile"] is None:
        raise _attention_resource_plan_error(resource_plan)
    resolved_tile = _attention_tile_dict(resource_plan["selected_tile"])
    sequence_expr = str(q_shape[2])
    query_tiles = (
        (sequence + resolved_tile["m"] - 1) // resolved_tile["m"]
        if sequence is not None
        else f"ceil_div({sequence_expr}, {resolved_tile['m']})"
    )
    key_tiles = (
        (sequence + resolved_tile["n"] - 1) // resolved_tile["n"]
        if sequence is not None
        else f"ceil_div({sequence_expr}, {resolved_tile['n']})"
    )
    grid = (
        batch * heads * query_tiles
        if batch is not None and heads is not None and isinstance(query_tiles, int)
        else f"({q_shape[0]}) * ({q_shape[1]}) * ({query_tiles})"
    )
    schedule = dict(program.metadata.get("schedule", {}))
    core_limit = int(schedule.get("core_dim_limit", 65535))
    if core_limit <= 0 or isinstance(grid, int) and grid > core_limit:
        raise UnsupportedBackendOpError(
            "Ascend Attention grid exceeds the private core limit.",
            reason=f"grid={grid}, core_limit={core_limit}.",
            suggestion="reduce batch/head/query tiles or increase the permitted core grid.",
        )

    workspace_bytes = resource_plan["workspace_bytes"]
    dot_iterations = (
        query_tiles * key_tiles
        if isinstance(query_tiles, int) and isinstance(key_tiles, int)
        else f"({query_tiles}) * ({key_tiles})"
    )
    complexity = {
        "query_tiles": query_tiles,
        "key_tiles": key_tiles,
        "key_tile_iterations_per_query": key_tiles,
        "dot_iterations": dot_iterations,
        "estimated_ssa_operations": (
            dot_iterations * 2
            if isinstance(dot_iterations, int)
            else f"2 * ({dot_iterations})"
        ),
        "bounded_by_sequence": 1024,
        "loop_based": True,
    }

    loop = next(
        (operation for operation in _walk_operations(program.blocks) if operation.opcode == "scf.for"),
        None,
    )
    if loop is None or len(loop.operands) < 3:
        raise UnsupportedBackendOpError(
            "Ascend Attention planner cannot prove the sequence loop.",
            reason="the structured contract has no scf.for bounds and step.",
            suggestion="preserve one sequence tile loop with lower=0 and step=1.",
        )
    loop_contract = {
        "lower": "0",
        "upper": (
            f"(({sequence} + {resolved_tile['n'] - 1}) // {resolved_tile['n']})"
            if sequence is not None
            else f"ceil_div({sequence_expr}, {resolved_tile['n']})"
        ),
        "step": 1,
    }
    return {
        "version": 1,
        "status": (
            "verified-static-single-block"
            if all(value is not None for value in (batch, heads, sequence, head_dim))
            else "verified-runtime-guarded-single-block"
        ),
        "batch": batch,
        "heads": heads,
        "sequence": sequence,
        "head_dim": head_dim,
        "tile": resolved_tile,
        "resource_plan": resource_plan,
        "query_tiles": query_tiles,
        "key_tiles": key_tiles,
        "grid": grid,
        "core_grid_limit": core_limit,
        "loop": loop_contract,
        "query_tail_mask": True,
        "key_tail_mask": True,
        "workspace_bytes": workspace_bytes,
        "cross_block_workspace": False,
        "ub_peak_bytes": resource_plan["ub_estimated_peak_bytes"],
        "ub_budget_bytes": resource_plan["ub_budget_bytes"],
        "ub_safety_margin_bytes": max(
            0,
            int(resource_plan["ub_budget_bytes"])
            - int(resource_plan["ub_estimated_peak_bytes"]),
        ),
        "compile_complexity": complexity,
        "mode": contract["mode"],
        "all_masked_predicate": contract["all_masked_predicate"],
        "state_preserving_branch": contract["state_preserving_branch"],
    }


def _validate_ascend_attention_emission_contract(kernel: Kernel) -> None:
    """Require the complete private contract at the emitter boundary."""
    if kernel.ssa is None:
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter requires SSA metadata.",
            reason="the kernel has no structured Attention program.",
            suggestion="run the Ascend verifier and planner before emission.",
        )
    schedule = dict(kernel.ssa.metadata.get("schedule", {}))
    contract = schedule.get("ascend_attention_loop")
    semantics = schedule.get("ascend_attention_mask_semantics")
    plan = schedule.get("ascend_attention_plan")
    required_semantics = {
        "k_score_mask_before_max",
        "v_value_mask_zero_before_dot",
        "causal_query_key_compare",
        "loop_carried_state",
        "all_masked_state_preserving_branch",
    }
    if not isinstance(contract, Mapping) or contract.get("version", 0) < 2:
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter requires a structured contract.",
            reason="the version 2 Attention contract is absent.",
            suggestion="complete Ascend Attention verification before emission.",
        )
    if not isinstance(semantics, Mapping) or not required_semantics.issubset(semantics):
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter requires verified mask semantics.",
            reason="score/value masks or loop-state proof is incomplete.",
            suggestion="run verify_ascend_attention_mask_semantics successfully.",
        )
    if not isinstance(plan, Mapping) or plan.get("status") not in {
        "verified-static-single-block",
        "verified-runtime-guarded-single-block",
    }:
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter requires a verified resource plan.",
            reason="tile, UB, workspace, grid, or sequence planning is absent.",
            suggestion="run the private Attention planner before emission.",
        )
    resource_plan = plan.get("resource_plan")
    selected_tile = (
        resource_plan.get("selected_tile")
        if isinstance(resource_plan, Mapping)
        else None
    )
    if not isinstance(selected_tile, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter requires the canonical resource tile.",
            reason="resource_plan.selected_tile is missing.",
            suggestion="run the unified Attention resource planner before emission.",
        )
    selected_tile = _attention_tile_dict(selected_tile)
    planned_tile = _attention_tile_dict(plan.get("tile", {}))
    if planned_tile != selected_tile:
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter received an inconsistent tile contract.",
            reason=f"resource tile={selected_tile!r}, planner tile={planned_tile!r}.",
            suggestion="carry the canonical resource-plan tile through source emission.",
        )
    for field in (
        "score_dot",
        "value_dot",
        "k_score_mask",
        "key_bounds_mask",
        "combined_key_valid",
        "all_masked_predicate",
        "v_value_mask",
        "causal_mask",
        "loop_carried_state",
        "state_preserving_branch",
        "access_provenance",
    ):
        if not contract.get(field):
            raise UnsupportedBackendOpError(
                "Ascend Attention emitter received an incomplete contract.",
                reason=f"missing structured field `{field}`.",
                suggestion="emit only after all Attention provenance is verified.",
            )
    mode = contract.get("mode")
    if mode not in {"causal", "non-causal"}:
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter requires an explicit causal mode.",
            reason=f"mode={mode!r}.",
            suggestion="emit only a causal or non-causal verified contract.",
        )
    if plan.get("mode") != mode:
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter received a stale planner mode.",
            reason=f"contract mode={mode!r}, plan mode={plan.get('mode')!r}.",
            suggestion="rebuild the planner output from the current contract.",
        )
    if plan.get("all_masked_predicate") != contract.get("all_masked_predicate"):
        raise UnsupportedBackendOpError(
            "Ascend Attention emitter received a stale all_masked plan.",
            reason="planner and structured contract refer to different all_masked metadata.",
            suggestion="rebuild the verified Attention plan.",
        )
    _verify_ascend_attention_retile_plan(kernel, selected_tile)


def verify_ascend_attention_mask_semantics(
    program: ssa.Program,
) -> Mapping[str, Any] | None:
    """Verify mask and online-softmax ordering for an Ascend attention loop.

    This verifier intentionally requires explicit SSA contracts for both score
    and value masking. Access-template load predicates alone are not enough to
    prove that invalid V lanes contribute zero to the second dot, nor that an
    all-masked tile preserves the previous online-softmax state.
    """
    schedule = dict(program.metadata.get("schedule", {}))
    contract = schedule.get("ascend_attention_loop")
    if not contract:
        return None
    if not isinstance(contract, Mapping) or contract.get("version", 0) < 2:
        raise UnsupportedBackendOpError(
            "Ascend Attention requires the structured contract schema.",
            reason="the schedule has no score/value/access provenance contract.",
            suggestion="construct the version 2 structured Attention contract first.",
        )

    loops = [
        operation
        for operation in _walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    ]
    if len(loops) != 1 or len(loops[0].regions) != 1:
        raise UnsupportedBackendOpError(
            "Ascend attention mask verification requires one structured loop.",
            reason=f"found {len(loops)} scf.for operations.",
            suggestion="lower one online-softmax loop before Ascend emission.",
        )

    loop = loops[0]
    body = tuple(_walk_operations(loop.regions))
    positions = {id(operation): index for index, operation in enumerate(body)}
    dots = tuple(
        operation
        for operation in body
        if operation.opcode == "linalg.dot"
    )
    if len(dots) != 2:
        raise UnsupportedBackendOpError(
            "Ascend attention mask verification requires score and value dots.",
            reason=f"found {len(dots)} linalg.dot operations.",
            suggestion="preserve the two-dot online-softmax SSA contract.",
        )

    score_dot, value_dot = dots

    # Consume the structured dot contract instead of inferring roles from dot
    # order alone.  The producer map includes operations outside the loop,
    # such as q preprocessing and tensor casts.
    producers = {
        result.name: operation
        for operation in _walk_operations(program.blocks)
        for result in operation.results
    }
    input_names = {
        value.name
        for value in (*program.inputs, *program.outputs)
        if value.type.kind == "tensor"
    }

    def source_root(name: str, seen: set[str] | None = None) -> str | None:
        seen = set() if seen is None else seen
        if name in seen:
            return None
        if name in input_names:
            return name
        seen.add(name)
        operation = producers.get(name)
        if operation is None:
            return None
        for operand in operation.operands:
            root = source_root(operand, seen.copy())
            if root is not None:
                return root
        return None

    score_contract = contract.get("score_dot")
    value_contract = contract.get("value_dot")
    if not isinstance(score_contract, Mapping) or not isinstance(value_contract, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention dot provenance is missing.",
            reason="score_dot/value_dot are absent from the structured contract.",
            suggestion="record Q/K and P/V operands with reduction axes and shapes.",
        )
    if (
        score_contract.get("operation") != score_dot.results[0].name
        or value_contract.get("operation") != value_dot.results[0].name
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention dot contract does not match SSA operations.",
            reason="contract operation ids are stale or point to different dots.",
            suggestion="rebuild the structured contract after private normalization.",
        )
    actual_q = source_root(score_dot.operands[0])
    actual_k = source_root(score_dot.operands[1])
    actual_v = source_root(value_dot.operands[1])
    if (
        actual_q != score_contract.get("lhs")
        or actual_k != score_contract.get("rhs")
        or actual_v != value_contract.get("rhs")
        or score_contract.get("reduction_axis") != "head_dim"
        or value_contract.get("reduction_axis") != "key_position"
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention dot operand provenance is not proven.",
            reason=(
                f"actual q/k/v=({actual_q!r},{actual_k!r},{actual_v!r}); "
                f"contract q/k/v=({score_contract.get('lhs')!r},"
                f"{score_contract.get('rhs')!r},{value_contract.get('rhs')!r})."
            ),
            suggestion="preserve Q/K/P/V source provenance and the two reduction axes.",
        )
    score_result_shape = tuple(str(dim) for dim in score_dot.results[0].type.shape)
    expected_score_shape = tuple(str(dim) for dim in score_contract.get("result_shape", ()))
    if expected_score_shape and score_result_shape != expected_score_shape:
        raise UnsupportedBackendOpError(
            "Ascend Attention score dot shape contract is stale.",
            reason=f"SSA result={score_result_shape!r}, contract={expected_score_shape!r}.",
            suggestion="record query_tile x key_tile as the score result shape.",
        )
    value_result_shape = tuple(str(dim) for dim in value_dot.results[0].type.shape)
    expected_value_shape = tuple(str(dim) for dim in value_contract.get("result_shape", ()))
    if expected_value_shape and value_result_shape != expected_value_shape:
        raise UnsupportedBackendOpError(
            "Ascend Attention value dot shape contract is stale.",
            reason=f"SSA result={value_result_shape!r}, contract={expected_value_shape!r}.",
            suggestion="record query_tile x value_dim as the value result shape.",
        )

    access = contract.get("access_provenance")
    if not isinstance(access, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention access provenance is missing.",
            reason="the structured contract has no rank-4 Q/K/V/O access map.",
            suggestion="record batch, head, position, and dimension coordinates.",
        )
    expected_coordinates = {
        "q": ("batch", "head", "query_position", "head_dim"),
        "k": ("batch", "head", "key_position", "head_dim"),
        "v": ("batch", "head", "key_position", "value_dim"),
        "o": ("batch", "head", "query_position", "value_dim"),
    }
    shapes = {}
    for role, coordinates in expected_coordinates.items():
        item = access.get(role)
        if not isinstance(item, Mapping) or tuple(item.get("coordinates", ())) != coordinates:
            raise UnsupportedBackendOpError(
                "Ascend Attention rank-4 coordinate contract is incomplete.",
                reason=f"{role} coordinates={None if item is None else item.get('coordinates')!r}.",
                suggestion="use explicit [batch,head,position,dim] provenance for Q/K/V/O.",
            )
        shape = tuple(str(dim) for dim in item.get("source_shape", ()))
        if len(shape) != 4 or not item.get("access_templates"):
            raise UnsupportedBackendOpError(
                "Ascend Attention rank-4 source or access template is missing.",
                reason=f"{role} source_shape={shape!r}, templates={item.get('access_templates') if isinstance(item, Mapping) else None!r}.",
                suggestion="preserve rank-4 source shape and masked access templates.",
            )
        shapes[role] = shape
    def dimensions_compatible(*values: str) -> bool:
        concrete = set()
        for value in values:
            try:
                concrete.add(int(value))
            except (TypeError, ValueError):
                continue
        return len(concrete) <= 1

    if not (
        dimensions_compatible(*(shapes[role][0] for role in ("q", "k", "v", "o")))
        and dimensions_compatible(*(shapes[role][1] for role in ("q", "k", "v", "o")))
        and dimensions_compatible(shapes["q"][2], shapes["o"][2])
        and dimensions_compatible(shapes["k"][2], shapes["v"][2])
        and dimensions_compatible(shapes["q"][3], shapes["k"][3])
        and dimensions_compatible(shapes["v"][3], shapes["o"][3])
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention Q/K/V/O sequence or batch/head contract is inconsistent.",
            reason=f"source shapes={shapes!r}.",
            suggestion="require equal batch/head, Q/O query, K/V key, and compatible dims.",
        )
    specs = {value.name: value for value in (*program.inputs, *program.outputs)}
    for role, item in access.items():
        tensor = item.get("tensor") if isinstance(item, Mapping) else None
        spec = specs.get(tensor)
        templates = () if spec is None else tuple(dict(spec.type.attrs).get("access_templates", ()))
        if not templates or any(
            not isinstance(template, Mapping) or not str(template.get("mask", "True"))
            or str(template.get("mask", "True")) == "True"
            for template in templates
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention access template does not prove tail masking.",
                reason=f"{role} has no non-trivial masked access template.",
                suggestion="retain query/key bounds predicates for every rank-4 tile.",
            )
    negative_inf = {
        result.name
        for operation in body
        if operation.opcode == "arith.constant"
        and str(operation.attrs.get("value")) in {"-inf", "-Infinity", "-float('inf')"}
        for result in operation.results
    }
    score_masks = tuple(
        operation
        for operation in body
        if operation.opcode == "select.where"
        and len(operation.operands) == 3
        and operation.operands[1] == score_dot.results[0].name
        and operation.operands[2] in negative_inf
    )
    reduce_max = next(
        (operation for operation in body if operation.opcode == "reduce.max"),
        None,
    )
    if not score_masks or reduce_max is None or positions[id(score_masks[0])] > positions[id(reduce_max)]:
        raise UnsupportedBackendOpError(
            "Ascend attention requires K score masking before reduce.max.",
            reason="the SSA does not prove select.where(score, -inf) before max reduction.",
            suggestion="materialize the K score mask as an explicit SSA select before reduce.max.",
        )
    score_mask_contract = contract.get("k_score_mask")
    if not isinstance(score_mask_contract, Mapping) or not score_mask_contract.get("operations"):
        raise UnsupportedBackendOpError(
            "Ascend Attention K score mask contract is missing.",
            reason="the structured contract has no key bounds mask operation.",
            suggestion="record the -inf score select and its reduce.max ordering.",
        )
    score_mask_names = set(score_mask_contract.get("operations", ()))
    if not score_mask_names.intersection(
        result.name for operation in score_masks for result in operation.results
    ) or score_mask_contract.get("invalid_value") != "-inf":
        raise UnsupportedBackendOpError(
            "Ascend Attention K score mask contract does not match SSA.",
            reason="score mask operation or invalid value is stale.",
            suggestion="use select.where(score, -inf) before reduce.max.",
        )
    reduce_max_position = positions[id(reduce_max)]
    for operation in body:
        if operation.opcode in {"math.exp2", "reduce.sum"} and positions[id(operation)] < reduce_max_position:
            raise UnsupportedBackendOpError(
                "Ascend Attention softmax operation precedes reduce.max.",
                reason=f"{operation.opcode} appears before the score max reduction.",
                suggestion="apply score masking and reduce.max before exp2/reduce.sum.",
            )

    by_result = {
        result.name: operation
        for operation in body
        for result in operation.results
    }

    def position_expr(name: str) -> bool:
        operation = by_result.get(name)
        if operation is None:
            return False
        if operation.opcode == "index.offset":
            return int(operation.attrs.get("dim", 0)) == -2
        return bool(
            operation.opcode == "tensor.view"
            and operation.operands
            and position_expr(operation.operands[0])
        )

    causal_comparisons = tuple(
        operation
        for operation in body
        if operation.opcode == "cmp.ge"
        and len(operation.operands) == 2
        and all(position_expr(operand) for operand in operation.operands)
    )
    mode = contract.get("mode")
    if mode not in {"causal", "non-causal"}:
        raise UnsupportedBackendOpError(
            "Ascend Attention contract has no explicit causal mode.",
            reason=f"mode={mode!r}.",
            suggestion="record mode=causal or mode=non-causal.",
        )
    if mode == "causal" and not causal_comparisons:
        raise UnsupportedBackendOpError(
            "Ascend causal attention requires query/key position comparison.",
            reason="no cmp.ge over explicit query_position/key_position offsets was found.",
            suggestion="represent causal masking as query_position >= key_position in SSA.",
        )
    if mode == "non-causal" and causal_comparisons:
        raise UnsupportedBackendOpError(
            "Ascend non-causal Attention contains a causal predicate.",
            reason="contract mode is non-causal but SSA contains query/key cmp.ge.",
            suggestion="remove causal masking or mark the contract causal.",
        )

    zero_values = {
        result.name
        for operation in body
        if operation.opcode in {"arith.constant", "tensor.full"}
        and operation.attrs.get("value", operation.attrs.get("fill")) in {0, 0.0, "0", "0.0"}
        for result in operation.results
    }
    value_zero_masks = tuple(
        operation
        for operation in body
        if operation.opcode == "select.where"
        and len(operation.operands) == 3
        and operation.operands[2] in zero_values
        and operation.results
    )

    # A value mask may be consumed directly by the V dot or through a private
    # tensor.cast.  Walk that short SSA chain instead of comparing the mask
    # result to the dot result itself.
    value_dot_inputs = set(value_dot.operands)
    changed = True
    while changed:
        changed = False
        for operation in body:
            if operation.opcode == "tensor.cast" and operation.results:
                if operation.results[0].name in value_dot_inputs:
                    for operand in operation.operands:
                        changed |= operand not in value_dot_inputs
                        value_dot_inputs.add(operand)
    if not value_zero_masks or not any(
        operation.results[0].name in value_dot_inputs for operation in value_zero_masks
    ):
        raise UnsupportedBackendOpError(
            "Ascend attention requires an explicit zero value mask before V dot.",
            reason="V masking is only implicit in access-template load predicates; SSA has no zero-valued V select.",
            suggestion="add a value-mask SSA select(value_predicate, V_load, 0.0) feeding the second dot.",
        )
    value_mask_contract = contract.get("v_value_mask")
    if not isinstance(value_mask_contract, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention V value mask contract is missing.",
            reason="the structured contract has no V predicate, load, and zero provenance.",
            suggestion="record the explicit V zero select before the value dot.",
        )
    value_mask_operation = next(
        (
            operation
            for operation in value_zero_masks
            if operation.results[0].name == value_mask_contract.get("operation")
        ),
        None,
    )
    if value_mask_operation is None or value_mask_operation.results[0].name not in value_dot_inputs:
        raise UnsupportedBackendOpError(
            "Ascend Attention V value mask is not connected to the value dot.",
            reason="the contract select result or V load is not on the second dot path.",
            suggestion="connect select.where(value_predicate,V_load,zero) to value dot.",
        )
    zero_name = value_mask_operation.operands[2]
    zero_producer = producers.get(zero_name)
    if (
        zero_producer is None
        or zero_producer.opcode != "arith.constant"
        or zero_producer.attrs.get("value") not in {0, 0.0, "0", "0.0"}
        or zero_producer.results[0].type.dtype
        != producers.get(value_mask_operation.operands[1], value_mask_operation).results[0].type.dtype
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention V value mask zero has the wrong dtype or provenance.",
            reason="zero must be a typed zero compatible with the V load.",
            suggestion="create arith.constant 0.0 with the V operand dtype.",
        )

    state_normalization = schedule.get("ascend_attention_loop_state")
    key_valid_contract = schedule.get("ascend_attention_key_valid")
    if not isinstance(key_valid_contract, Mapping) or not key_valid_contract.get("all_masked"):
        raise UnsupportedBackendOpError(
            "Ascend Attention all-masked provenance is missing.",
            reason="the verifier cannot trace all_masked to the K key-valid predicate.",
            suggestion="run Ascend key-valid normalization before state verification.",
        )
    if not isinstance(state_normalization, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention loop-state normalization metadata is missing.",
            reason="acc/m_i/l_i state branch provenance was not recorded.",
            suggestion="run Ascend online-softmax state normalization before verification.",
        )
    if state_normalization.get("all_masked") != key_valid_contract.get("all_masked"):
        raise UnsupportedBackendOpError(
            "Ascend Attention state branch uses a stale all_masked predicate.",
            reason=(
                f"key-valid={key_valid_contract.get('all_masked')!r}, "
                f"state={state_normalization.get('all_masked')!r}."
            ),
            suggestion="rebuild state normalization from the current key-valid contract.",
        )
    contract_fields = {
        "key_bounds_mask": contract.get("key_bounds_mask"),
        "combined_key_valid": contract.get("combined_key_valid"),
        "all_masked_predicate": contract.get("all_masked_predicate"),
        "state_preserving_branch": contract.get("state_preserving_branch"),
    }
    if any(not isinstance(value, Mapping) for value in contract_fields.values()):
        raise UnsupportedBackendOpError(
            "Ascend Attention normalized contract is incomplete.",
            reason=f"missing fields={tuple(name for name, value in contract_fields.items() if not isinstance(value, Mapping))!r}.",
            suggestion="record normalized mask and loop-state operations one-to-one with SSA.",
        )
    if contract_fields["all_masked_predicate"].get("operation") != key_valid_contract.get("all_masked"):
        raise UnsupportedBackendOpError(
            "Ascend Attention all_masked contract is stale.",
            reason="contract operation does not match key-valid normalization metadata.",
            suggestion="rebuild the contract after key-valid normalization.",
        )

    iter_args = tuple(dict(loop.attrs).get("iter_args", ()))
    yields = (
        loop.regions[0].operations[-1]
        if loop.regions[0].operations
        and loop.regions[0].operations[-1].opcode == "scf.yield"
        else None
    )
    if len(iter_args) != 3 or yields is None or len(yields.operands) != 3:
        raise UnsupportedBackendOpError(
            "Ascend attention requires acc/m_i/l_i loop-carried state.",
            reason="the online-softmax loop state is not a three-value yield.",
            suggestion="carry acc, m_i, and l_i through scf.for.",
        )

    state_names = tuple(str(item.get("name")) for item in iter_args)
    state_contract = contract.get("loop_carried_state")
    if not isinstance(state_contract, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention loop state contract is missing.",
            reason="acc/m_i/l_i roles are not recorded.",
            suggestion="record all three loop-carried state roles and dtypes.",
        )
    state_roles = state_contract.get("roles")
    if not isinstance(state_roles, Mapping) or tuple(state_roles) != ("acc", "m_i", "l_i"):
        raise UnsupportedBackendOpError(
            "Ascend Attention loop state roles are incomplete.",
            reason=f"received roles={state_roles!r}.",
            suggestion="record acc, m_i, and l_i in that order.",
        )
    value_types = {
        value.name: value.type
        for value in (*program.inputs, *program.outputs)
    }
    value_types.update(
        {
            result.name: result.type
            for operation in _walk_operations(program.blocks)
            for result in operation.results
        }
    )
    loop_args = loop.regions[0].args[1:]
    if len(loop_args) != 3:
        raise UnsupportedBackendOpError(
            "Ascend Attention loop block arguments are incomplete.",
            reason=f"found {len(loop_args)} carried block arguments.",
            suggestion="preserve acc, m_i, and l_i block arguments.",
        )
    for initial, block_arg, update in zip(
        loop.operands[3:], loop_args, yields.operands, strict=True
    ):
        initial_type = value_types.get(initial)
        update_type = value_types.get(update)
        if (
            initial_type is None
            or normalize_ascend_dtype(block_arg.type.dtype) != "float32"
            or normalize_ascend_dtype(initial_type.dtype) != "float32"
            or update_type is None
            or normalize_ascend_dtype(update_type.dtype) != "float32"
        ):
            raise UnsupportedBackendOpError(
                "Ascend Attention loop state dtype is not uniformly FP32.",
                reason=(
                    f"initial={None if initial_type is None else initial_type.dtype!r}, "
                    f"block={block_arg.type.dtype!r}, "
                    f"update={None if update_type is None else update_type.dtype!r}."
                ),
                suggestion="normalize acc, m_i, and l_i initial/block/update values to FP32.",
            )
    update_branches = tuple(
        operation
        for operation in body
        if operation.opcode == "scf.if"
        and len(operation.regions) == 2
        and any(
            any(state in operand for operand in operation.regions[0].operations[-1].operands)
            for state in state_names
            if operation.regions[0].operations
        )
    )
    if not update_branches:
        raise UnsupportedBackendOpError(
            "Ascend attention requires an all-masked state-preserving branch.",
            reason="no scf.if preserves acc, m_i, and l_i when a tile has no valid K lanes.",
            suggestion="add an all-masked predicate and yield the previous three loop-carried values unchanged.",
        )

    state_arg_names = tuple(argument.name for argument in loop_args)
    preserving_branch = False
    for branch in update_branches:
        for region in branch.regions:
            if not region.operations or region.operations[-1].opcode != "scf.yield":
                continue
            yielded = tuple(region.operations[-1].operands)
            if yielded == state_arg_names:
                preserving_branch = True
    if not preserving_branch:
        raise UnsupportedBackendOpError(
            "Ascend Attention all-masked branch does not preserve loop state.",
            reason="no scf.if branch yields the previous acc, m_i, and l_i values unchanged.",
            suggestion="yield the three loop block arguments in the all-masked branch.",
        )

    normalized_branch = next(
        (
            operation
            for operation in body
            if operation.opcode == "scf.if"
            and operation.attrs.get("ascend_attention_state_normalization")
        ),
        None,
    )
    if normalized_branch is None or normalized_branch.operands[0] != key_valid_contract.get("all_masked"):
        raise UnsupportedBackendOpError(
            "Ascend Attention state branch is not connected to all_masked.",
            reason="the result-producing scf.if does not consume normalized all_masked.",
            suggestion="guard state updates with the key-valid all_masked predicate.",
        )
    if tuple(normalized_branch.attrs.get("old_state", ())) != state_arg_names:
        raise UnsupportedBackendOpError(
            "Ascend Attention state branch old-state provenance is incomplete.",
            reason=f"old_state={normalized_branch.attrs.get('old_state')!r}.",
            suggestion="yield the original acc/m_i/l_i block arguments in all-masked branch.",
        )
    if tuple(normalized_branch.attrs.get("new_state", ())) != tuple(
        normalized_branch.regions[1].operations[-1].operands
        if normalized_branch.regions
        and normalized_branch.regions[1].operations
        and normalized_branch.regions[1].operations[-1].opcode == "scf.yield"
        else ()
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention state branch new-state provenance is stale.",
            reason=f"new_state={normalized_branch.attrs.get('new_state')!r}.",
            suggestion="record the normal-path acc/m_i/l_i updates in the branch contract.",
        )

    return {
        "k_score_mask_before_max": True,
        "v_value_mask_zero_before_dot": True,
        "causal_query_key_compare": True,
        "loop_carried_state": ("acc", "m_i", "l_i"),
        "all_masked_state_preserving_branch": True,
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
        attention_loop = _ascend_attention_loop_contract(
            program, require_value_mask=False
        )
        if attention_loop is not None:
            attention_argument_dtypes = {
                normalize_ascend_dtype(getattr(getattr(value, "type", None), "dtype", None))
                for value in program.inputs
                if normalize_ascend_dtype(getattr(getattr(value, "type", None), "dtype", None))
                not in {None, "bool"}
            }
            unsupported_attention_dtypes = sorted(
                dtype
                for dtype in attention_argument_dtypes
                if dtype not in ASCEND_ATTENTION_DTYPE_REGISTRY
                and not _is_ascend_runtime_dtype(dtype)
            )
            if unsupported_attention_dtypes:
                raise UnsupportedBackendOpError(
                    "Ascend Attention dtype capability is unsupported.",
                    reason=(
                        f"program argument dtypes={sorted(attention_argument_dtypes)!r}; "
                        f"unsupported={unsupported_attention_dtypes!r}; "
                        f"verified registry={tuple(sorted(ASCEND_ATTENTION_DTYPE_REGISTRY))}."
                    ),
                    suggestion="use one uniform FP16, BF16, or FP32 Q/K/V/O dtype.",
                )
            program = _normalize_ascend_attention_value_mask(program)
            program = _normalize_ascend_attention_key_valid(program)
            program = _normalize_ascend_attention_loop_state(program)
            attention_loop = _ascend_attention_loop_contract(program)
            attention_program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {"ascend_attention_loop": attention_loop}
                },
            )
            mask_semantics = verify_ascend_attention_mask_semantics(attention_program)
            attention_plan = _plan_ascend_attention_contract(
                attention_program, attention_loop
            )
            program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {
                        "ascend_attention_loop": attention_loop,
                        "ascend_attention_mask_semantics": mask_semantics,
                        "ascend_attention_plan": attention_plan,
                    }
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
        program = self._validate_supported_program(program, context)
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

        # DecomposeLinalg runs after this backend pass.  Publish the private
        # preserve decision here, after dot provenance is known, so Conv2d's
        # linalg.dot reaches the Ascend native block emitter.
        optimization = dict(lowered.metadata.get("optimization", {}))
        if (
            dot_loop is not None
            and not schedule.get("ascend_attention_loop")
            and any(
                len(tuple(value.type.attrs.get("source_shape", value.type.shape))) == 4
                for value in (*lowered.inputs, *lowered.outputs)
                if value.type.kind == "tensor"
            )
        ):
            optimization["preserve_linalg"] = True

        return replace(
            lowered,
            metadata=dict(lowered.metadata)
            | {"schedule": schedule, "optimization": optimization},
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
                        # Keep explicit matrix dimensions so the private UB
                        # solver can resize the actual linalg working set.
                        "tile": {"elements": 256},
                        "ascend_matrix_tile": {
                            "block_m": 256,
                            "block_n": 256,
                            "block_k": 256,
                        },
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
            # The Attention contract is built and verified before the generic
            # candidate selection pass.  Preserve that concrete mode and
            # provenance when attaching candidate metadata; the old
            # placeholder used ``mode=generic-online-softmax-loop`` and
            # silently invalidated causal/non-causal specialization.
            attention_contract = schedule.get("ascend_attention_loop")
            if not isinstance(attention_contract, Mapping):
                return ()
            return (
                ScheduleCandidate(
                    name="ascend-generic-online-softmax-loop",
                    schedule={
                        "tile": {"elements": 256},
                        "vector_width": 1,
                        "core_dim_limit": _max_core_dim(context),
                        "ascend_attention_loop": dict(attention_contract),
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
        del backend
        if (
            schedule.get("granularity") == "blocked-linalg"
            and analysis.get("has_dot")
            and schedule.get("ascend_dot_loop")
            and not schedule.get("ascend_attention_loop")
        ):
            # Keep rank-4 im2col linalg.dot intact so the Ascend emitter can
            # render the planner-owned matrix tile instead of scalarizing the
            # public 64-wide access domain.
            return {"preserve_linalg": True}
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
    ) -> ssa.Program:
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
            program = _validate_ascend_reduction_contract(program, reduction)

        tensor_dtypes = tuple(
            tensor.dtype for tensor in context.tensors if not tensor.constexpr
        ) or tuple(
            value.type.dtype for value in program.inputs if value.type.kind == "tensor"
        )
        ascend_dtype_legality(tensor_dtypes)
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

        return program


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


def _validate_ascend_reduction_contract(
    program: ssa.Program, reduction: Mapping[str, Any]
) -> ssa.Program:
    """Validate the first Ascend reduction tier before CANN emission.

    The public analysis may describe scalar fallback or a dynamic domain.  The
    Ascend target admits only one axis over one row-vector domain; all other
    forms fail while the structured metadata is still available to callers.
    """
    if reduction.get("mode") != "row-vector":
        raise UnsupportedBackendOpError(
            "Ascend row-vector reduction contract is required for granularity `parallel-reduction`.",
            reason=f"received reduction mode {reduction.get('mode')!r}.",
            suggestion="lower one static axis with a single output row domain.",
        )

    axis = reduction.get("axis")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise UnsupportedBackendOpError(
            "Ascend reduction axis must be one compile-time integer.",
            reason=f"received axis {axis!r}.",
            suggestion="provide one integer reduction axis.",
        )

    value_shape = tuple(str(dim) for dim in reduction.get("value_shape", ()))
    result_shape = tuple(str(dim) for dim in reduction.get("result_shape", ()))
    if not value_shape or axis < 0 or axis >= len(value_shape):
        raise UnsupportedBackendOpError(
            "Ascend reduction requires a ranked single-axis input.",
            reason=f"value_shape={value_shape!r}, axis={axis!r}.",
            suggestion="use a ranked tensor and reduce exactly one axis.",
        )

    expected = value_shape[:axis] + value_shape[axis + 1 :]
    keepdim = bool(reduction.get("keepdim", False))
    expected_output = (
        value_shape[:axis] + ("1",) + value_shape[axis + 1 :]
        if keepdim
        else expected
    )
    if result_shape != expected:
        # Keepdim is represented by the result shape in some frontend paths;
        # accept it only when the singleton axis is explicit and unambiguous.
        if len(result_shape) == len(value_shape) and result_shape[axis] == "1" and tuple(
            result_shape[:axis] + result_shape[axis + 1 :]
        ) == expected:
            keepdim = True
            expected_output = result_shape
        else:
            raise UnsupportedBackendOpError(
                "Ascend reduction output shape does not match its axis contract.",
                reason=f"input={value_shape!r}, axis={axis}, output={result_shape!r}.",
                suggestion="use output_shape=input_shape with the reduced axis removed, or keepdim=True with a singleton axis.",
            )

    extent_text = reduction.get("extent")
    extent = _static_reduction_extent(extent_text)
    if extent is not None and extent < 0:
        raise UnsupportedBackendOpError(
            "Ascend reduction extent cannot be negative.",
            reason=f"received extent {extent}.",
            suggestion="provide a non-negative static extent.",
        )
    if extent is None and not str(extent_text).strip().isidentifier():
        raise UnsupportedBackendOpError(
            "Ascend reduction extent must be static or a named shape symbol.",
            reason=f"received extent {extent_text!r}.",
            suggestion="bind the extent to an integer or identifier before scheduling.",
        )

    if extent == 0 and any(
        operation.opcode in {"reduce.max", "reduce.min"}
        for operation in _walk_operations(program.blocks)
    ):
        raise UnsupportedBackendOpError(
            "Ascend empty max/min reduction has no supported identity contract.",
            reason="an empty row cannot produce a max/min value without a defined runtime identity.",
            suggestion="use sum for empty rows or reject the empty shape before CANN lowering.",
        )

    dtypes = {
        value.type.dtype
        for value in (*program.inputs, *program.outputs)
        if value.type.kind == "tensor" and value.type.dtype
    }
    accumulator_dtype = "float32" if dtypes & {"float16", "bfloat16", "bf16"} else None
    if dtypes & {"float16", "bfloat16", "bf16"} and accumulator_dtype != "float32":
        raise UnsupportedBackendOpError(
            "Ascend reduction accumulator dtype is not provably FP32.",
            reason=f"tensor dtypes are {sorted(dtypes)!r}.",
            suggestion="accumulate FP16/BF16 reductions in FP32 and cast at store.",
        )

    schedule = dict(program.metadata.get("schedule", {}))
    schedule["reduction"] = dict(reduction) | {
        "axis": axis,
        "mode": "row-vector",
        "extent": extent_text,
        "result_shape": result_shape,
        "output_shape": expected_output,
        "keepdim": keepdim,
        "accumulator_dtype": accumulator_dtype or "float32",
        "tail_mask": True,
        "empty_input": "sum_identity_zero" if extent == 0 else "rejected-or-runtime-guarded",
        "workspace": "none",
    }
    return replace(program, metadata=dict(program.metadata) | {"schedule": schedule})


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

    access_contract = {
        "lhs": ("batch", "row", "k"),
        "rhs": ("batch", "k", "col"),
        "output": ("batch", "row", "col"),
    }
    mask_contract = {
        "lhs": ("batch", "row", "k"),
        "rhs": ("batch", "k", "col"),
        "output": ("batch", "row", "col"),
        "bounds": "0 <= batch < B and 0 <= row < M and 0 <= col < N and 0 <= k < K",
    }
    grid_contract = {
        "logical_elements": (
            f"({contract['batch']}) * ({contract['output_shape'][1]}) * "
            f"({contract['output_shape'][2]})"
        ),
        "batch": contract["batch"],
        "m": contract["output_shape"][1],
        "n": contract["output_shape"][2],
        "mode": "flattened-batch-row-col",
    }
    inserted = False

    def rewrite(block: ssa.Block) -> ssa.Block:
        nonlocal inserted
        rewritten = []

        for operation in block.operations:
            regions = tuple(rewrite(region) for region in operation.regions)
            attrs = dict(operation.attrs)
            operands = operation.operands

            if operation.opcode == "scf.for" and attrs.get("decomposition") == "matmul":
                attrs["ascend_access_contract"] = access_contract
                attrs["ascend_mask_contract"] = mask_contract
                attrs["ascend_grid_contract"] = grid_contract

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
                attrs["ascend_access_axes"] = (
                    access_contract["lhs"]
                    if operand_role == "lhs"
                    else access_contract["rhs"]
                )

            if (
                operation.opcode == "mem.store"
                and attrs.get("decomposition") == "matmul"
                and len(operands) == 2
                and operands[1] == output
            ):
                attrs["ascend_access_template"] = "batch-row-col"
                attrs["ascend_access_axes"] = access_contract["output"]

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
        # Preserve the original lhs/rhs coordinate field for sidecar readers;
        # the complete three-way proof lives in access_contract.
        "coordinates": {
            "lhs": access_contract["lhs"],
            "rhs": access_contract["rhs"],
        },
        "access_contract": access_contract,
        "output": access_contract["output"],
        "mask": mask_contract,
        "grid": grid_contract,
        "batch_broadcast": False,
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
