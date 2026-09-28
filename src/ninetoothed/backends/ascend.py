"""Ascend backend policy, contracts, and private launch metadata."""

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
        else:
            solve_input = matrix_tile or tile
        if solve_input:
            ub_plan = plan_ascend_ub(kernel.ssa, solve_input)
            if ub_plan.rejection_reason:
                raise UnsupportedBackendOpError(
                    "Ascend matmul tile exceeds the private UB contract.",
                    reason=ub_plan.rejection_reason,
                    suggestion="reduce M/N/K tile dimensions or split the reduction loop.",
                )
            solved = solve_ascend_tile_config(kernel.ssa, solve_input)
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
    if not tracked:
        return {}

    defaults = dict(kernel.metadata.get("meta_defaults", {}))
    operations = tuple(_ascend_operation_references(kernel.ssa))
    entries = []
    for key, candidate in tracked.items():
        parameter = _ascend_original_tile_parameter(key, defaults)
        role = _ASCEND_TILE_ROLES[key]
        candidate_values = _ascend_candidate_values(kernel.ssa, key, candidate)
        references = tuple(
            reference
            for reference in operations
            if _ascend_operation_matches_tile(reference, key, parameter)
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


@dataclass(frozen=True)
class AscendUBPlan:
    """Target-private UB planning result consumed by Ascend lowering."""

    safe_tile: Mapping[str, int]
    estimated_peak_bytes: int
    workspace_bytes: int
    safety_margin_bytes: int
    rejection_reason: str | None = None


SUPPORTED_DTYPES = frozenset(
    {"float32", "float16", "bfloat16", "int32", "int8", "bool"}
)
UNSUPPORTED_DTYPES = frozenset({"float8_e5m2", "float8_e4m3fn", "float64"})
ASCEND_ELEMENTWISE_DTYPES = SUPPORTED_DTYPES
ASCEND_RNG_DTYPES = frozenset({"float16", "bfloat16", "float32"})
ASCEND_ATOMIC_DTYPES = frozenset({"float16", "bfloat16", "float32", "int32"})
_SIDECAR_SCHEMA = 3


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

    # Start below the 96 KiB budget because causal score masks and the online
    # accumulator have overlapping lifetimes in BiShengIR.
    provenance = dict(schedule.get("ascend_tile_provenance", {}))
    candidate = dict(provenance.get("candidate", {}))
    # The public contract carries a runtime causal branch (``public-scf-if``)
    # rather than a compile-time boolean.  Reserve the causal footprint for
    # both variants so a cached kernel cannot overflow when the flag is true.
    causal_contract = schedule.get("ascend_attention_loop", {}).get("causal")
    causal_tile_m = 32 if causal_contract else 64
    initial = {
        "block_m": min(int(candidate.get("block_m", causal_tile_m)), causal_tile_m),
        "block_n": min(int(candidate.get("block_n", 32)), 32),
        "block_k": min(int(candidate.get("block_k", 32)), 32),
    }
    plan = plan_ascend_ub(kernel.ssa, initial)
    if plan.rejection_reason:
        raise UnsupportedBackendOpError(
            "Ascend attention tile cannot satisfy the private UB budget.",
            reason=plan.rejection_reason,
            suggestion="Reduce the attention tile or split the sequence loop.",
        )
    solved = dict(plan.safe_tile)
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
            "estimated_peak_bytes": plan.estimated_peak_bytes,
            "safety_margin_bytes": plan.safety_margin_bytes,
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
    if not isinstance(contract, Mapping) or contract.get("mode") != "generic-online-softmax-loop":
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
    if resolved != {"matmul-m": 32, "matmul-n": 32, "reduction-k": 32}:
        raise UnsupportedBackendOpError(
            "Ascend attention structured retile requires the 32x32x32 tile.",
            reason=f"resolved tile roles are {resolved!r}.",
            suggestion="use BLOCK_M=32, BLOCK_N=32, and BLOCK_K=32 for this contract.",
        )

    source_shapes = {
        spec.name: tuple(spec.attrs.get("source_shape", spec.shape))
        for spec in kernel.tensors
        if getattr(spec, "ndim", 0) >= 2
    }
    sequence, head_dim = _attention_sequence_and_head_dim(source_shapes)
    if head_dim != 64 or sequence > 64:
        raise UnsupportedBackendOpError(
            "Ascend structured retile supports head_dim=64 and sequence<=64 only.",
            reason=f"received head_dim={head_dim!r}, sequence={sequence!r}.",
            suggestion="use the verified small-shape Attention contract.",
        )

    loop_paths = []
    values = {}
    for entry in roles.values():
        parameter = str(entry.get("parameter", ""))
        if parameter:
            role = str(entry.get("role"))
            values[parameter] = 32 if role in required else values.get(parameter)
    for name in _attention_tile_symbols(kernel.ssa) + _attention_tensor_tile_symbols(
        kernel.tensors
    ):
        for symbol in re.findall(r"[A-Za-z_][A-Za-z0-9_]*BLOCK_SIZE_[01]", str(name)):
            values[symbol] = 32

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
    loop = next(
        operation
        for block in program.blocks
        for operation in _walk_operations((block,))
        if operation.opcode == "scf.for"
    )
    loop_attrs = dict(loop.attrs)
    loop_attrs["ascend_attention_retile"] = {
        "block_m": 32,
        "block_n": 32,
        "block_k": 32,
        "sequence": int(sequence),
        "loop_lower": "0",
        "loop_upper": f"(({int(sequence)} + 31) // 32)",
        "loop_step": "1",
        "grid": f"triton.cdiv({int(sequence)}, 32)",
    }
    upper_name = "%ascend_attention_sequence_tiles"
    loop_operands = list(loop.operands)
    loop_operands[1] = upper_name
    rewritten_loop = replace(
        loop,
        operands=tuple(loop_operands),
        attrs=loop_attrs,
    )
    upper_constant = ssa.Operation(
        opcode="arith.constant",
        results=(
            ssa.Value(
                name=upper_name,
                type=ssa.Type(kind="scalar", dtype="int64"),
            ),
        ),
        attrs={"value": (int(sequence) + 31) // 32},
    )
    program = _replace_operation(
        program,
        loop,
        rewritten_loop,
        preceding=(upper_constant,),
    )
    updated_schedule = dict(program.metadata.get("schedule", {}))
    updated_schedule["ascend_attention_retile"] = dict(loop_attrs["ascend_attention_retile"])
    updated_schedule["ascend_tile_provenance"] = dict(provenance)
    program = replace(program, metadata=dict(program.metadata) | {"schedule": updated_schedule})
    return replace(
        kernel,
        tensors=specialize_tensor_specs(kernel.tensors, values),
        ssa=program,
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
    try:
        return int(shape[-2]), int(shape[-1])
    except (TypeError, ValueError) as exc:
        raise UnsupportedBackendOpError(
            "Ascend attention structured retile requires static sequence and head_dim.",
            reason=f"source shape {shape!r} is symbolic.",
            suggestion="specialize sequence and head_dim before Ascend emission.",
        ) from exc


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
        "tile": {"m": m, "n": n, "k": k_lhs},
        "workspace_bytes": m * n * 4,
        "padding_coordinates": padding_symbols,
        "mask": has_masks,
        "stride_dilation": "encoded-in-access-template",
        "core_grid_limit": core_limit,
        "grid_estimate": grid_estimate,
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
    if not causal_comparisons:
        raise UnsupportedBackendOpError(
            "Ascend causal attention requires query/key position comparison.",
            reason="no cmp.ge over explicit query_position/key_position offsets was found.",
            suggestion="represent causal masking as query_position >= key_position in SSA.",
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

    iter_args = tuple(dict(loop.attrs).get("iter_args", ()))
    yields = next(
        (operation for operation in body if operation.opcode == "scf.yield"), None
    )
    if len(iter_args) != 3 or yields is None or len(yields.operands) != 3:
        raise UnsupportedBackendOpError(
            "Ascend attention requires acc/m_i/l_i loop-carried state.",
            reason="the online-softmax loop state is not a three-value yield.",
            suggestion="carry acc, m_i, and l_i through scf.for.",
        )

    state_names = tuple(str(item.get("name")) for item in iter_args)
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
        attention_loop = _ascend_attention_loop_contract(program)
        if attention_loop is not None:
            attention_program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {"ascend_attention_loop": attention_loop}
                },
            )
            mask_semantics = verify_ascend_attention_mask_semantics(attention_program)
            program = replace(
                program,
                metadata=dict(program.metadata)
                | {
                    "schedule": dict(program.metadata.get("schedule", {}))
                    | {
                        "ascend_attention_loop": attention_loop,
                        "ascend_attention_mask_semantics": mask_semantics,
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
