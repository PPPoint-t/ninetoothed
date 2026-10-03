"""Ascend Triton Python artifact materialization and reload."""

import types
from pathlib import Path
from typing import Any, Mapping

from ninetoothed.backends.ascend import (
    UnsupportedBackendOpError,
    ASCEND_ATTENTION_DTYPE_REGISTRY,
    ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE,
    ascend_abi_from_dict,
    ascend_cache_key,
    ascend_logical_domain,
    ascend_logical_domain_is_abi_bound,
    ascend_uses_access_template,
    normalize_ascend_dtype,
    read_ascend_sidecar,
    static_forward_view_offset,
    unsupported_ascend_elementwise_dtypes,
    validate_build_policy,
    validate_ascend_attention_source_contract,
    write_ascend_sidecar,
)
from ninetoothed.backends.ascend_layout import (
    AscendLayoutCapabilityError,
    admit_tensor_layout,
    reject_write_overlap,
)
from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.compiler.cache import (
    atomic_write_text,
    cache_lock,
    compilation_cache_key,
    write_manifest,
    write_source,
)


class AscendMaterializer(Materializer):
    """Publish and reload source-only Ascend Triton artifacts."""

    target = Target.ASCEND

    def jit_materialize(self, compilation, *, output_dir: str | Path | None = None):
        return _materialize(compilation, output_dir=output_dir)

    def aot_build(self, compilation, *, output_dir: str | Path):
        return _materialize(compilation, output_dir=output_dir)

    def load_built_artifact(self, built: BuiltArtifact):
        if built.source.backend != Target.ASCEND:
            raise ValueError("Built artifact does not belong to the Ascend backend.")

        source_path = Path(built.source_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Ascend built artifact source does not exist: {source_path}."
            )

        from ninetoothed.compiler.runtime import _runtime_specs

        specs = _runtime_specs(built.source)
        sidecar = read_ascend_sidecar(source_path)
        abi = ascend_abi_from_dict(sidecar["launch_abi"])
        tensor_sources = {
            binding.source for binding in abi.kernel_args if binding.kind == "tensor"
        }
        tensor_specs = tuple(spec for spec in specs if spec.name in tensor_sources)
        _validate_ascend_dtype_specs(
            tensor_specs,
            allow_rng_auxiliary=bool(
                (sidecar.get("advanced_contract") or {}).get("rng")
            ),
            allow_atomic=bool((sidecar.get("advanced_contract") or {}).get("atomic")),
        )
        module = _load_source_module(source_path, built.source.kernel_name)

        try:
            launch = getattr(module, built.source.entrypoint)
        except AttributeError as exc:
            raise RuntimeError(
                "Cannot reload Ascend artifact "
                f"`{built.source.kernel_name}` from `{source_path}`: missing "
                f"entrypoint `{built.source.entrypoint}`."
            ) from exc

        return _ascend_wrapper(
            launch,
            abi,
            specs,
            source_path=source_path,
            kernel_name=built.source.kernel_name,
            max_core_dim=sidecar["max_core_dim"],
            module=module,
            dtype_specs=tensor_specs,
            logical_domain=sidecar["logical_domain"],
            reduction_schedule=sidecar.get("reduction_schedule"),
            layout_contract=sidecar.get("layout_contract"),
            advanced_contract=sidecar.get("advanced_contract"),
            dot_loop=sidecar.get("dot_loop"),
            attention_loop=sidecar.get("attention_loop"),
            attention_mask_semantics=sidecar.get("attention_mask_semantics"),
            attention_key_valid=sidecar.get("attention_key_valid"),
            attention_loop_state=sidecar.get("attention_loop_state"),
            attention_plan=sidecar.get("attention_plan"),
            attention_retile=sidecar.get("attention_retile"),
            attention_dtype_registry=sidecar.get("attention_dtype_registry"),
            linalg_contract=sidecar.get("linalg_contract"),
        )


def _materialize(compilation, *, output_dir: str | Path | None):
    from ninetoothed.compiler.runtime import Handle, _built_manifest

    artifact = compilation.artifact
    schedule = artifact.metadata.get("ssa_metadata", {}).get("schedule", {})
    _validate_ascend_dtype_specs(
        compilation.kernel.tensors,
        allow_rng_auxiliary=bool(schedule.get("ascend_advanced", {}).get("rng")),
        allow_atomic=bool(schedule.get("ascend_advanced", {}).get("atomic")),
    )
    validate_build_policy(artifact.metadata, compilation.request)
    cache_key = ascend_cache_key(compilation_cache_key(compilation), artifact.metadata)
    source = write_source(
        artifact.kernel_name,
        artifact.primary_source,
        "ascend.py",
        cache_key=cache_key,
    )

    with cache_lock(source):
        write_manifest(
            source.with_suffix(".manifest.json"),
            _built_manifest(compilation, cache_key, source, None),
        )

    published_source = _publish_source(source, output_dir)
    abi = compilation.launch_abi
    write_ascend_sidecar(
        source,
        abi=abi,
        specs=compilation.kernel.tensors,
        outputs=artifact.metadata.get("outputs", ()),
        metadata=artifact.metadata,
    )
    if published_source != source:
        write_ascend_sidecar(
            published_source,
            abi=abi,
            specs=compilation.kernel.tensors,
            outputs=artifact.metadata.get("outputs", ()),
            metadata=artifact.metadata,
        )
    module = _load_source_module(source, artifact.kernel_name)

    try:
        launch = getattr(module, artifact.entrypoint)
    except AttributeError as exc:
        raise RuntimeError(
            "Cannot materialize Ascend artifact "
            f"`{artifact.kernel_name}` from `{source}`: missing entrypoint "
            f"`{artifact.entrypoint}`."
        ) from exc

    kernel = getattr(module, f"{artifact.kernel_name}_kernel", None)
    access_template = ascend_uses_access_template(artifact.metadata)
    wrapped = _ascend_wrapper(
        launch,
        abi,
        compilation.kernel.tensors,
        source_path=published_source,
        kernel_name=artifact.kernel_name,
        max_core_dim=_max_core_dim(artifact.metadata),
        module=module,
        # Ordinary elementwise launch source already embeds its scheduled meta
        # tile.  Its runtime validation must use the concrete output extent;
        # only access-template schedules retain a symbolic private domain.
        logical_domain=(
            ascend_logical_domain(
                compilation.kernel.tensors,
                tuple(artifact.metadata.get("outputs", ())),
                allow_access_template=True,
            )
            # An access-template artifact may have its tile meta parameters
            # specialized into source before launch ABI construction.  Retain
            # symbolic validation only when every symbol has one, and exactly
            # one, ABI binding; otherwise output.numel() is the authoritative
            # runtime domain.  This prevents stale, doubly-prefixed layout
            # symbols from becoming phantom launch parameters.
            if access_template
            and ascend_logical_domain_is_abi_bound(
                ascend_logical_domain(
                    compilation.kernel.tensors,
                    tuple(artifact.metadata.get("outputs", ())),
                    allow_access_template=True,
                ),
                abi,
            )
            else None
        ),
        reduction_schedule=artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("reduction"),
        layout_contract=_artifact_layout_contract(artifact.metadata),
        advanced_contract=schedule.get("ascend_advanced"),
        dot_loop=schedule.get("ascend_dot_loop"),
        attention_loop=schedule.get("ascend_attention_loop"),
        attention_mask_semantics=schedule.get("ascend_attention_mask_semantics"),
        attention_key_valid=schedule.get("ascend_attention_key_valid"),
        attention_loop_state=schedule.get("ascend_attention_loop_state"),
        attention_plan=schedule.get("ascend_attention_plan"),
        attention_retile=schedule.get("ascend_attention_retile"),
        attention_dtype_registry=ASCEND_ATTENTION_DTYPE_REGISTRY,
        linalg_contract=schedule.get("ascend_linalg"),
    )

    return Handle(compilation, (module, kernel), wrapped, published_source)


def _publish_source(source: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return source

    destination = Path(output_dir) / source.name
    atomic_write_text(destination, source.read_text(encoding="utf-8"))

    return destination


def _load_source_module(source_path: Path, kernel_name: str):
    """Execute generated source without mutating ``sys.modules``."""
    module = types.ModuleType(f"_ninetoothed_ascend_{kernel_name}")
    module.__file__ = str(source_path)
    source = source_path.read_text(encoding="utf-8")

    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Cannot import Ascend artifact "
            f"`{kernel_name}` from `{source_path}` because dependency "
            f"`{exc.name}` is unavailable. Install the matching Triton Ascend "
            "and CANN runtime before materializing or loading this artifact."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Cannot import Ascend artifact `{kernel_name}` from `{source_path}`."
        ) from exc

    return module


def _ascend_wrapper(
    function,
    abi,
    specs,
    *,
    source_path: Path,
    kernel_name: str,
    max_core_dim: int,
    module,
    dtype_specs=None,
    logical_domain=None,
    reduction_schedule=None,
    layout_contract=None,
    advanced_contract=None,
    dot_loop=None,
    attention_loop=None,
    attention_mask_semantics=None,
    attention_key_valid=None,
    attention_loop_state=None,
    attention_plan=None,
    attention_retile=None,
    attention_dtype_registry=None,
    linalg_contract=None,
):
    from ninetoothed.compiler.runtime import (
        _bound_values,
        _empty_launch,
        _public_values,
    )

    def launch(*args, **kwargs):
        public = _public_values(abi, args, kwargs, specs=specs, target=Target.ASCEND)
        attention_source_contract = getattr(
            module, ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE, None
        )
        if attention_loop and attention_source_contract is None:
            raise UnsupportedBackendOpError(
                "Ascend Attention plan/source contract mismatch.",
                reason="loaded source has no embedded resource plan metadata.",
                suggestion="reload an AOT artifact emitted with the matching Attention plan.",
            )

        _validate_ascend_bindings(
            abi,
            public,
            specs,
            max_core_dim,
            dtype_specs=dtype_specs,
            logical_domain=logical_domain,
            reduction_schedule=reduction_schedule,
            layout_contract=layout_contract,
            advanced_contract=advanced_contract,
            dot_loop=dot_loop,
            attention_loop=attention_loop,
            attention_mask_semantics=attention_mask_semantics,
            attention_key_valid=attention_key_valid,
            attention_loop_state=attention_loop_state,
            attention_plan=attention_plan,
            attention_retile=attention_retile,
            attention_source_contract=attention_source_contract,
            attention_dtype_registry=attention_dtype_registry,
            linalg_contract=linalg_contract,
        )

        if _empty_launch(abi, public):
            return _ascend_outputs(abi, public)

        values, keepalive = _bound_values(abi, public, scalar_mode="value")
        # The public ABI correctly records a rank-0 value as ``scalar``.  The
        # Ascend emitter still writes rank-0 outputs through a pointer, so this
        # is the one platform calling-convention adaptation left at submission
        # time.  It does not alter or serialize a second LaunchABI.
        for index, binding in enumerate(abi.kernel_args):
            if binding.kind == "scalar" and binding.source in abi.outputs:
                values[index] = public[binding.source]
                keepalive.append(public[binding.source])
        stream = _current_npu_stream(public)
        keepalive.extend((module, stream))

        try:
            function(*values)
        except Exception as exc:
            raise RuntimeError(
                "Ascend kernel launch failed "
                f"(backend=ascend, kernel={kernel_name}, source={source_path}, "
                f"max_core_dim={max_core_dim})."
            ) from exc
        finally:
            keepalive.clear()

        return _ascend_outputs(abi, public)

    return launch


def _validate_access_template_contract(dot_loop, attention_loop) -> bool:
    """Admit rank-4 storage only for recognized private generic-SSA schedules."""
    if attention_loop:
        contract = attention_loop
    elif dot_loop:
        contract = dot_loop
    else:
        return False

    if not isinstance(contract, Mapping):
        raise ValueError("Ascend access-template launch contract is malformed.")

    # Attention uses ``mode`` for causal/non-causal semantics.  Its structural
    # kind is carried separately, so runtime admission must not interpret a
    # causal mode as the access-template kind.
    kind = contract.get("kind", contract.get("mode"))
    if kind not in {"generic-dot-loop", "generic-online-softmax-loop"}:
        raise ValueError(
            "Ascend rank-4 storage requires a recognized generic dot-loop or "
            "online-softmax access-template contract."
        )

    return True


def _validate_dot_loop_runtime_shapes(spec_by_name, tensors, dot_loop) -> None:
    """Keep generic dot-loop launches within the statically emitted ABI."""
    if not dot_loop:
        return

    if (
        dot_loop.get("mode") != "generic-dot-loop"
        or dot_loop.get("layout") != "public-access-template"
        or not dot_loop.get("loop_carried")
    ):
        raise ValueError("Ascend generic dot-loop contract is malformed.")

    for name, value in tensors.items():
        spec = spec_by_name[name]
        source_shape = tuple(spec.attrs.get("source_shape", ()))

        concrete_shape = []
        for dimension in source_shape:
            try:
                concrete_shape.append(int(dimension))
            except (TypeError, ValueError):
                concrete_shape.append(None)

        if any(
            expected is not None and actual != expected
            for actual, expected in zip(
                tuple(value.shape), concrete_shape, strict=False
            )
        ):
            raise ValueError(
                f"Ascend generic dot-loop argument `{name}` has shape "
                f"{tuple(value.shape)}; expected the compiled access-template "
                f"shape {tuple(concrete_shape)}."
            )

        if value.storage_offset() != 0:
            raise ValueError(
                f"Ascend generic dot-loop argument `{name}` must have storage offset zero."
            )


def _validate_attention_runtime_contract(
    spec_by_name,
    tensors,
    attention_loop,
    attention_mask_semantics,
    attention_key_valid,
    attention_loop_state,
    attention_plan,
    max_core_dim,
    attention_dtype_registry=None,
    *,
    attention_retile=None,
    attention_source_contract=None,
) -> None:
    """Validate runtime bindings against the verified Attention sidecar contract."""
    if not isinstance(attention_loop, Mapping) or attention_loop.get("version", 0) < 2:
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime requires a version 2 structured contract.",
            reason="the JIT/AOT binding metadata is missing the verified contract.",
            suggestion="rebuild the artifact through the Ascend Attention verifier.",
        )
    if not isinstance(attention_plan, Mapping) or attention_plan.get("status") not in {
        "verified-static-single-block",
        "verified-runtime-guarded-single-block",
    }:
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime requires a verified resource plan.",
            reason="tile, workspace, UB, or grid metadata is absent or stale.",
            suggestion="rebuild the artifact with the Ascend Attention planner.",
        )
    # Validate that the runtime's sidecar plan is the same full plan embedded
    # by the emitter.  This check covers selected M/N/K, UB candidate evidence,
    # workspace, static tile counts, and the launch-grid expression.
    validate_ascend_attention_source_contract(
        attention_plan,
        attention_retile,
        attention_source_contract,
        require_source=True,
    )
    required_semantics = {
        "k_score_mask_before_max",
        "v_value_mask_zero_before_dot",
        "causal_query_key_compare",
        "loop_carried_state",
        "all_masked_state_preserving_branch",
    }
    if not isinstance(attention_mask_semantics, Mapping) or not required_semantics.issubset(
        attention_mask_semantics
    ):
        raise UnsupportedBackendOpError(
            "Ascend Attention AOT sidecar is missing verified mask semantics.",
            reason="mask ordering, causal predicate, or loop-state metadata was not persisted.",
            suggestion="rebuild the sidecar from the verified Attention schedule.",
        )
    if not isinstance(attention_key_valid, Mapping) or not isinstance(attention_loop_state, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention AOT sidecar is missing normalization provenance.",
            reason="key-valid or loop-state normalization metadata was not persisted.",
            suggestion="rebuild the artifact with the verified Attention normalization passes.",
        )

    access = attention_loop.get("access_provenance")
    if not isinstance(access, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime has no access provenance.",
            reason="Q/K/V/O bindings cannot be derived from the sidecar.",
            suggestion="preserve rank-4 access provenance through AOT materialization.",
        )

    aliases = {}
    for name, spec in spec_by_name.items():
        aliases[name] = name
        source_name = spec.attrs.get("source_name")
        if source_name:
            aliases[str(source_name)] = name

    def resolve_tensor(role):
        entry = access.get(role)
        name = entry.get("tensor") if isinstance(entry, Mapping) else None
        resolved = aliases.get(str(name)) if name is not None else None
        if resolved is None or resolved not in tensors:
            raise UnsupportedBackendOpError(
                "Ascend Attention runtime binding is incomplete.",
                reason=f"{role} tensor `{name}` is not present in the launch ABI.",
                suggestion="bind all Q/K/V/O tensors recorded by access provenance.",
            )
        return resolved, tensors[resolved]

    resolved = {role: resolve_tensor(role) for role in ("q", "k", "v", "o")}
    registry = attention_dtype_registry or {}
    expected_shapes = {}
    for role, (name, value) in resolved.items():
        shape = tuple(int(dim) for dim in tuple(value.shape))
        if len(shape) != 4:
            raise UnsupportedBackendOpError(
                "Ascend Attention runtime requires rank-4 Q/K/V/O tensors.",
                reason=f"{role} `{name}` has runtime shape {shape!r}.",
                suggestion="use [batch, head, sequence, head_dim] tensors.",
            )
        expected_shapes[role] = shape
        dtype = normalize_ascend_dtype(getattr(value, "dtype", None))
        if dtype not in registry:
            raise UnsupportedBackendOpError(
                "Ascend Attention runtime dtype is not in the verified capability registry.",
                reason=f"{role} `{name}` has dtype {getattr(value, 'dtype', None)!r}; registry keys={tuple(registry)}.",
                suggestion="add a complete compile, NPU, and AOT capability record before enabling this dtype.",
            )

    q_shape = expected_shapes["q"]
    if q_shape[3] != 64:
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime supports head_dim=64 only.",
            reason=f"runtime head_dim={q_shape[3]}.",
            suggestion="specialize an artifact with head_dim=64.",
        )
    if q_shape[2] <= 0 or q_shape[2] > 1024:
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime supports sequence<=1024 only.",
            reason=f"runtime sequence={q_shape[2]}.",
            suggestion="use the bounded key-tiled Attention contract or reject before CANN.",
        )
    planned_values = {}
    for field in ("batch", "heads", "sequence", "head_dim"):
        value = attention_plan.get(field)
        try:
            planned_values[field] = int(value)
        except (TypeError, ValueError):
            planned_values[field] = None
    plan_shape = tuple(planned_values[field] for field in ("batch", "heads", "sequence", "head_dim"))
    if any(expected is not None and actual != expected for actual, expected in zip(q_shape, plan_shape)):
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime shape does not match the compiled plan.",
            reason=f"runtime Q shape={q_shape}, planned shape={plan_shape}.",
            suggestion="specialize a new artifact for the runtime batch/head/sequence/head_dim.",
        )
    if any(shape != q_shape for shape in expected_shapes.values()):
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime Q/K/V/O shapes are inconsistent.",
            reason=f"runtime shapes={expected_shapes!r}.",
            suggestion="require equal rank-4 batch, head, sequence, and head_dim extents.",
        )

    resource_plan = attention_plan.get("resource_plan")
    resource_tile = (
        resource_plan.get("selected_tile")
        if isinstance(resource_plan, Mapping)
        else None
    )
    planned_tile = attention_plan.get("tile")
    if not isinstance(resource_tile, Mapping) or not isinstance(planned_tile, Mapping):
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime requires the canonical resource tile.",
            reason="resource_plan.selected_tile or planner tile is missing.",
            suggestion="preserve the unified Attention resource plan in artifact metadata.",
        )
    selected_tile = {axis: int(resource_tile[axis]) for axis in ("m", "n", "k")}
    plan_tile = {axis: int(planned_tile[axis]) for axis in ("m", "n", "k")}
    if selected_tile != plan_tile:
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime received inconsistent resource tiles.",
            reason=f"resource tile={selected_tile!r}, planner tile={plan_tile!r}.",
            suggestion="use resource_plan.selected_tile as the sole M/N/K source.",
        )

    try:
        core_limit = int(attention_plan["core_grid_limit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedBackendOpError(
            "Ascend Attention resource plan has invalid grid metadata.",
            reason=f"plan={dict(attention_plan)!r}.",
            suggestion="record integer grid and core_grid_limit values.",
        ) from exc
    runtime_query_tiles = (q_shape[2] + selected_tile["m"] - 1) // selected_tile["m"]
    runtime_key_tiles = (q_shape[2] + selected_tile["n"] - 1) // selected_tile["n"]
    grid = q_shape[0] * q_shape[1] * runtime_query_tiles
    planned_grid = attention_plan.get("grid")
    if isinstance(planned_grid, int) and grid != planned_grid:
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime grid differs from the selected resource tile.",
            reason=f"runtime grid={grid}, planned grid={planned_grid}, tile={selected_tile!r}.",
            suggestion="rebuild the launch grid from selected M and runtime Q shape.",
        )
    for field, actual in (
        ("query_tiles", runtime_query_tiles),
        ("key_tiles", runtime_key_tiles),
    ):
        planned = attention_plan.get(field)
        if isinstance(planned, int) and planned != actual:
            raise UnsupportedBackendOpError(
                "Ascend Attention runtime tile count differs from the selected resource tile.",
                reason=(
                    f"runtime {field}={actual}, planned {field}={planned}, "
                    f"tile={selected_tile!r}, sequence={q_shape[2]}."
                ),
                suggestion="recompute query/key tile counts from runtime shape and planned M/N.",
            )
    if grid <= 0 or grid > min(int(max_core_dim), core_limit):
        raise UnsupportedBackendOpError(
            "Ascend Attention runtime grid exceeds the verified core limit.",
            reason=f"grid={grid}, plan_limit={core_limit}, runtime_limit={max_core_dim}.",
            suggestion="specialize a smaller launch grid or update the verified plan.",
        )
    expected_workspace = selected_tile["m"] * q_shape[3] * 4
    planned_workspace_value = attention_plan.get("workspace_bytes")
    try:
        planned_workspace = int(planned_workspace_value)
    except (TypeError, ValueError) as exc:
        if planned_workspace_value == resource_plan.get("workspace_bytes"):
            # Symbolic plans record M * (head_dim expression) * 4.  The shared
            # source contract has already verified that expression against the
            # retile; resolve it from the bound runtime head dimension here.
            planned_workspace = expected_workspace
        else:
            raise UnsupportedBackendOpError(
                "Ascend Attention resource plan has invalid workspace metadata.",
                reason=f"workspace_bytes={planned_workspace_value!r}.",
                suggestion="record a concrete or plan-consistent workspace size.",
            ) from exc
    if planned_workspace != expected_workspace:
        raise UnsupportedBackendOpError(
            "Ascend Attention workspace does not match runtime shape.",
            reason=f"planned={planned_workspace}, expected={expected_workspace}.",
            suggestion="re-specialize the artifact for the concrete head_dim.",
        )
    resource_workspace_value = resource_plan.get("workspace_bytes")
    try:
        resource_workspace = int(resource_workspace_value)
    except (TypeError, ValueError) as exc:
        if resource_workspace_value == planned_workspace_value:
            resource_workspace = expected_workspace
        else:
            raise UnsupportedBackendOpError(
                "Ascend Attention resource plan has invalid workspace metadata.",
                reason=f"resource_plan.workspace_bytes={resource_workspace_value!r}.",
                suggestion="record the concrete workspace size in the selected resource candidate.",
            ) from exc
    if resource_workspace != expected_workspace:
        raise UnsupportedBackendOpError(
            "Ascend Attention resource workspace differs from runtime requirements.",
            reason=(
                f"resource plan={resource_workspace}, runtime={expected_workspace}, "
                f"tile={selected_tile!r}, head_dim={q_shape[3]}."
            ),
            suggestion="recompute workspace from selected M and the bound head dimension.",
        )


def validate_tile_ub_capacity(tile, *, dtype_bytes=2, ub_limit_bytes=192 * 1024):
    """Reject tiles whose input, output, and FP32 accumulator exceed UB budget."""
    try:
        m, n, k = (int(tile[key]) for key in ("m", "n", "k"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Ascend tile contract must provide integer m, n, and k."
        ) from exc
    if min(m, n, k) <= 0 or dtype_bytes <= 0:
        raise ValueError("Ascend tile dimensions and dtype size must be positive.")
    required = (m * k + k * n) * dtype_bytes + (m * n) * 4
    if required > ub_limit_bytes:
        raise UnsupportedBackendOpError(
            f"Ascend tile {m}x{n}x{k} requires {required} UB bytes (limit {ub_limit_bytes}).",
            reason="the tile working set exceeds the verified 910B3 UB capacity.",
            suggestion="reduce BLOCK_SIZE_M/N/K or split the operation into smaller tiles.",
        )
    return required


def _validate_ascend_bindings(
    abi,
    public,
    specs,
    max_core_dim: int,
    *,
    dtype_specs=None,
    logical_domain=None,
    reduction_schedule=None,
    layout_contract=None,
    advanced_contract=None,
    dot_loop=None,
    attention_loop=None,
    attention_mask_semantics=None,
    attention_key_valid=None,
    attention_loop_state=None,
    attention_plan=None,
    attention_retile=None,
    attention_source_contract=None,
    attention_dtype_registry=None,
    linalg_contract=None,
) -> None:
    spec_by_name = {spec.name: spec for spec in specs}
    _validate_ascend_dtype_specs(
        tuple(dtype_specs) if dtype_specs is not None else tuple(spec_by_name.values()),
        allow_rng_auxiliary=bool((advanced_contract or {}).get("rng")),
        allow_atomic=bool((advanced_contract or {}).get("atomic")),
    )
    access_template = _validate_access_template_contract(dot_loop, attention_loop)
    tensors = {}

    for binding in abi.kernel_args:
        scalar_output = binding.kind == "scalar" and binding.source in abi.outputs

        if (
            binding.kind != "tensor"
            and not scalar_output
            or binding.source not in public
        ):
            continue

        value = public[binding.source]

        if binding.source not in spec_by_name:
            continue

        expected_dtype = normalize_ascend_dtype(spec_by_name[binding.source].dtype)
        actual_dtype = normalize_ascend_dtype(getattr(value, "dtype", None))

        if actual_dtype != expected_dtype:
            raise TypeError(
                f"Ascend kernel argument `{binding.source}` has dtype {actual_dtype}; "
                f"expected {expected_dtype}."
            )

        try:
            admit_tensor_layout(
                binding.source,
                value,
                allow_rank4_access_template=True,
                allow_zero_stride_read=binding.access == "read",
                # Loads use the same per-dimension source-stride address
                # expression as CUDA.  An aliased source is consequently safe
                # for read-only clone/gather semantics; any writer remains
                # subject to the non-overlapping layout contract.
                allow_overlapping_read=binding.access == "read",
            )
        except AscendLayoutCapabilityError as exc:
            # Preserve the established diagnostics for malformed metadata and
            # invalid storage spans.  Overlapping views are rejected directly
            # by ``admit_tensor_layout`` as UnsupportedBackendOpError above.
            raise TypeError(str(exc)) from exc

        _validate_storage_span(binding.source, value)
        tensors[binding.source] = value

    if not tensors:
        return

    if attention_loop:
        _validate_attention_runtime_contract(
            spec_by_name,
            tensors,
            attention_loop,
            attention_mask_semantics,
            attention_key_valid,
            attention_loop_state,
            attention_plan,
            max_core_dim,
            attention_dtype_registry,
            attention_retile=attention_retile,
            attention_source_contract=attention_source_contract,
        )

    _validate_dot_loop_runtime_shapes(spec_by_name, tensors, dot_loop)
    if dot_loop:
        tile = dot_loop.get("tile", {})
        sample_dtype = next(iter(tensors.values()), None)
        dtype_bytes = (
            4
            if normalize_ascend_dtype(getattr(sample_dtype, "dtype", None)) == "float32"
            else 2
        )
        validate_tile_ub_capacity(tile, dtype_bytes=dtype_bytes)

    matmul_inputs = _validate_matmul_runtime_shapes(
        spec_by_name, tensors, linalg_contract
    )

    if not abi.outputs:
        raise ValueError("Ascend elementwise launch requires an output tensor.")

    output_names = tuple(abi.outputs)
    output_name = output_names[0]
    output = tensors.get(output_name)

    if output is None:
        raise ValueError(
            f"Ascend elementwise launch requires its output tensor `{output_name}`."
        )

    missing_outputs = tuple(name for name in output_names if name not in tensors)

    if missing_outputs:
        raise ValueError(
            "Ascend elementwise launch requires output tensor arguments: "
            f"{', '.join(missing_outputs)}."
        )

    logical_elements = (
        output.numel()
        if logical_domain is None
        else _resolve_logical_domain(logical_domain, abi, public)
    )
    output_shape = tuple(output.shape)

    reduction_inputs = _row_reduction_inputs(
        tensors,
        output_names,
        output_shape,
        reduction_schedule,
    )
    layout_inputs = _layout_transfer_inputs(
        tensors,
        output_names,
        layout_contract,
    )
    advanced_inputs = frozenset()
    if advanced_contract and advanced_contract.get("kind") == "conv2d-im2col":
        advanced_inputs = frozenset(
            name
            for name in (
                advanced_contract.get("input"),
                advanced_contract.get("weight"),
                advanced_contract.get("output"),
            )
            if name
        )
    elif advanced_contract and advanced_contract.get("atomic"):
        advanced_inputs = frozenset(tensors)

    if not output_shape:
        if logical_elements != 1:
            raise ValueError("Ascend scalar output launch requires logical domain one.")
    elif len(output_shape) not in ({1, 2, 3, 4} if access_template else {1, 2, 3}):
        raise ValueError(
            "Ascend elementwise launch supports only one-, two-, or three-dimensional "
            "contiguous output tensors."
        )

    for name, value in tensors.items():
        shape = tuple(value.shape)

        is_output = name in output_names

        if access_template:
            continue

        if is_output and not shape:
            if output_shape:
                raise ValueError(
                    "Ascend multi-output elementwise launch requires every output "
                    f"shape to match {output_shape}; `{name}` is scalar."
                )

            continue

        if not output_shape:
            if is_output:
                raise ValueError(
                    "Ascend multi-output elementwise launch requires every output "
                    "to be scalar when its primary output is scalar."
                )

            if name in reduction_inputs:
                continue

            if shape != (1,):
                raise ValueError(
                    "Ascend scalar output launch accepts only scalar-compatible "
                    f"(1,) tensor inputs; `{name}` has shape {shape}."
                )

            continue

        if _uses_public_access_template(spec_by_name[name]):
            _validate_binding_storage_span(name, value, spec_by_name[name])
            continue

        if name in reduction_inputs or name in layout_inputs or name in advanced_inputs:
            continue

        if name in matmul_inputs:
            continue

        if not is_output and _is_supported_broadcast_shape(shape, output_shape):
            continue

        if len(shape) != len(output_shape):
            raise ValueError(
                "Ascend elementwise launch requires tensor arguments to match the "
                f"output rank {len(output_shape)}; `{name}` has shape {shape}."
            )

        if is_output and shape != output_shape:
            raise ValueError(
                "Ascend multi-output elementwise launch requires every output "
                f"shape to match {output_shape}; `{name}` has shape {shape}."
            )

        if is_output and value.numel() != output.numel():
            raise ValueError(
                "Ascend multi-output elementwise launch requires every output "
                f"to contain {output.numel()} elements; `{name}` has {value.numel()}."
            )

        required_elements = logical_elements

        if value.numel() < required_elements:
            raise ValueError(
                f"Ascend logical view for '{name}' requires {required_elements} "
                f"elements but its base tensor has {value.numel()}."
            )

        if not is_output and shape != output_shape:
            raise ValueError(
                "Ascend elementwise broadcast requires an input shape equal to the "
                f"output shape {output_shape}, (1,), or (1, N); `{name}` has shape "
                f"{shape}."
            )

    _reject_storage_aliases(abi, tensors)
    core_dim = (logical_elements + 255) // 256

    if core_dim > max_core_dim:
        raise ValueError(
            "Ascend launch grid exceeds `max_core_dim`: "
            f"required {core_dim}, limit {max_core_dim}."
        )


def _validate_matmul_runtime_shapes(spec_by_name, tensors, contract):
    """Validate dynamic matmul dimensions while leaving shape semantics to SSA."""
    if not contract or contract.get("mode") != "tiled-matmul":
        return frozenset()

    names = (contract.get("lhs"), contract.get("rhs"), contract.get("output"))
    if any(name not in tensors or name not in spec_by_name for name in names):
        raise ValueError("Ascend matmul launch is missing a tensor binding.")

    lhs, rhs, output = (tensors[name] for name in names)
    if (
        len(lhs.shape) not in {2, 3}
        or tuple(rhs.shape)[: len(lhs.shape) - 2]
        != tuple(lhs.shape)[: len(lhs.shape) - 2]
    ):
        raise ValueError(
            "Ascend matmul runtime tensors must have matching batch ranks."
        )
    if len(lhs.shape) == 2:
        expected = (lhs.shape[0], rhs.shape[1])
        if tuple(output.shape) != expected or lhs.shape[1] != rhs.shape[0]:
            raise ValueError(
                "Ascend matmul runtime shapes do not satisfy MxK @ KxN -> MxN."
            )
    else:
        expected = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
        if tuple(output.shape) != expected or lhs.shape[2] != rhs.shape[1]:
            raise ValueError("Ascend batched matmul runtime shapes are incompatible.")
    return frozenset(names)


def _is_supported_broadcast_shape(shape, output_shape) -> bool:
    """Accept trailing-aligned singleton broadcasts emitted by shared SSA.

    The emitter maps singleton axes to coordinate zero and maps omitted leading
    axes to zero as well.  Runtime admission mirrors precisely that contract;
    expanded zero-stride tensors remain rejected by ``admit_tensor_layout``.
    """
    if len(shape) > len(output_shape):
        return False

    aligned_shape = (1,) * (len(output_shape) - len(shape)) + tuple(shape)

    return all(
        input_size == 1 or input_size == output_size
        for input_size, output_size in zip(aligned_shape, output_shape)
    )


def _uses_public_access_template(spec) -> bool:
    """Return whether compiler-produced access metadata owns this binding span."""
    return bool(getattr(spec, "attrs", {}).get("access_templates", ()))


def _validate_binding_storage_span(name: str, value: Any, spec) -> None:
    """Validate one binding against its public access-template contract.

    The generated masks bound every template coordinate by the binding's source
    dimensions.  For a contiguous runtime tensor, its own ``numel`` is
    therefore the exact accessible element span; output shape is irrelevant.
    """
    source_ndim = int(spec.attrs.get("source_ndim", spec.ndim))
    templates = tuple(spec.attrs.get("access_templates", ()))

    if len(tuple(value.shape)) != source_ndim:
        raise ValueError(
            f"Ascend access-template binding `{name}` has rank {len(tuple(value.shape))}; "
            f"expected source rank {source_ndim}."
        )

    for template in templates:
        offsets = tuple(template.get("offsets", ()))

        if (
            len(offsets) != source_ndim
            or not template.get("linear_offset")
            or not template.get("mask")
        ):
            raise ValueError(
                f"Ascend access-template binding `{name}` has malformed source offsets."
            )

    _validate_storage_span(name, value)


def _row_reduction_inputs(tensors, output_names, output_shape, reduction_schedule):
    """Identify inputs consumed by a verified row-reduction schedule.

    A direct reduction writes the reduced domain, while fused Softmax and
    RMSNorm consume the same row reduction before returning to the original
    value domain.  Both forms use the same emitter schedule; this admission
    check only recognizes their declared rank/domain relationship.
    """
    if not reduction_schedule or reduction_schedule.get("mode") != "row-vector":
        return frozenset()

    raw_axis = reduction_schedule.get("axis")

    if isinstance(raw_axis, bool) or not isinstance(raw_axis, int):
        raise ValueError("Ascend row-vector reduction requires an integer axis.")

    inputs = set()

    for name, value in tensors.items():
        if name in output_names:
            continue

        shape = tuple(value.shape)

        if len(shape) != len(output_shape) + 1:
            continue

        axis = raw_axis if raw_axis >= 0 else raw_axis + len(shape)

        if not 0 <= axis < len(shape):
            raise ValueError(
                "Ascend row-vector reduction axis "
                f"{raw_axis} is outside input rank {len(shape)}."
            )

        if tuple(shape[:axis] + shape[axis + 1 :]) != output_shape:
            continue

        inputs.add(name)

    value_shape = tuple(reduction_schedule.get("value_shape", ()))
    result_shape = tuple(reduction_schedule.get("result_shape", ()))
    output_shape_symbols = tuple(str(dim) for dim in output_shape)
    fused_value_domain = len(value_shape) == len(output_shape) and (
        tuple(str(dim) for dim in result_shape) == output_shape_symbols
        or len(result_shape) == len(output_shape) - 1
    )

    if not inputs and fused_value_domain:
        inputs.update(
            name
            for name, value in tensors.items()
            if name not in output_names
            and (
                tuple(str(dim) for dim in value.shape)
                == tuple(str(dim) for dim in value_shape)
                or (
                    len(result_shape) == len(output_shape) - 1
                    and tuple(value.shape) == output_shape
                )
            )
        )

    if not inputs:
        raise ValueError(
            "Ascend row-vector reduction requires a contiguous input in its direct "
            "reduced domain or fused value domain."
        )

    return frozenset(inputs)


def _layout_transfer_inputs(tensors, output_names, layout_contract):
    """Validate direct transpose bindings from the private artifact sidecar."""
    if not layout_contract:
        candidates = [name for name in tensors if name not in output_names]
        if len(candidates) == 1 and len(output_names) == 1:
            source_name, destination_name = candidates[0], output_names[0]
            source_shape = tuple(tensors[source_name].shape)
            destination_shape = tuple(tensors[destination_name].shape)
            if len(source_shape) == len(
                destination_shape
            ) == 2 and destination_shape == tuple(reversed(source_shape)):
                return frozenset((source_name, destination_name))
        return frozenset()

    source_name = layout_contract.get("source_binding")
    destination_name = layout_contract.get("destination_binding")
    permutation = tuple(layout_contract.get("permutation", ()))

    if (
        not isinstance(source_name, str)
        or not isinstance(destination_name, str)
        or destination_name not in output_names
        or source_name not in tensors
        or destination_name not in tensors
    ):
        raise ValueError("Ascend layout transfer sidecar has invalid tensor bindings.")

    source_shape = tuple(tensors[source_name].shape)
    destination_shape = tuple(tensors[destination_name].shape)

    if (
        len(source_shape) != 2
        or len(destination_shape) != 2
        or permutation != (1, 0)
        or tuple(destination_shape) != tuple(reversed(source_shape))
    ):
        raise ValueError(
            "Ascend layout transfer supports only a rank-2 direct transpose with "
            "matching physical shapes."
        )

    return frozenset((source_name, destination_name))


def _artifact_layout_contract(metadata):
    """Read the private schedule contract from JIT and AOT artifact shapes."""
    direct = metadata.get("layout_transfer")
    if direct:
        return direct
    for key in ("ssa_schedule", "ssa_metadata"):
        value = metadata.get(key, {})
        if isinstance(value, Mapping):
            schedule = value.get("schedule", value)
            if isinstance(schedule, Mapping) and schedule.get("layout_transfer"):
                return schedule["layout_transfer"]
    return None


def _ascend_outputs(abi, public):
    outputs = tuple(public[name] for name in abi.outputs)

    if not outputs:
        return None

    if len(outputs) == 1:
        return outputs[0]

    return outputs


def _validate_storage_span(name: str, value: Any) -> None:
    storage = value.untyped_storage()
    storage_elements = storage.nbytes() // value.element_size()
    start = value.storage_offset()
    end = start + value.numel()

    if start < 0 or end > storage_elements:
        raise ValueError(
            f"Ascend kernel argument `{name}` exceeds its underlying storage span."
        )


def _validate_ascend_dtype_specs(
    specs, *, allow_rng_auxiliary: bool = False, allow_atomic: bool = False
) -> None:
    unsupported = unsupported_ascend_elementwise_dtypes(
        tuple(spec.dtype for spec in specs if not getattr(spec, "constexpr", False)),
        allow_rng_auxiliary=allow_rng_auxiliary,
        allow_atomic=allow_atomic,
    )

    if unsupported:
        raise ValueError(
            "Ascend materializer supports only verified FP16, BF16, FP32, and INT32 "
            "elementwise dtypes; received tensor dtypes: "
            f"{', '.join(unsupported)}."
        )


def _resolve_logical_domain(expression, abi, public) -> int:
    value = _resolve_expression(expression, abi, public)

    if not isinstance(value, int) or value < 0:
        raise ValueError(
            "Ascend logical-domain launch expression must resolve to a "
            "non-negative integer."
        )

    return value


def _logical_offset(spec, abi, public) -> int:
    offsets = tuple(spec.attrs.get("view_offsets", ()))

    if not offsets:
        return 0

    if len(offsets) == 1:
        del abi, public

        return static_forward_view_offset(offsets[0])

    if any(static_forward_view_offset(offset) != 0 for offset in offsets):
        raise ValueError(
            f"Ascend multidimensional logical view for '{spec.name}' must provide "
            "only zero offsets."
        )

    del abi, public

    return 0


def _resolve_expression(expression, abi, public, *, symbols=None) -> int:
    from ninetoothed.compiler.runtime import _binding_value
    from ninetoothed.ir import IndexExpr

    values = {
        binding.name: _binding_value(binding, public)
        for binding in abi.kernel_args
        if binding.kind in {"shape", "stride", "meta", "constexpr"}
    } | dict(symbols or {})
    root = IndexExpr.parse(expression)

    def resolve(node):
        if node.op == "constant":
            return node.value

        if node.op == "symbol":
            try:
                return values[str(node.value)]
            except KeyError as exc:
                raise ValueError(
                    "Ascend launch expression references unresolved symbol "
                    f"'{node.value}'."
                ) from exc

        if node.op == "add":
            return resolve(node.operands[0]) + resolve(node.operands[1])

        if node.op == "sub":
            return resolve(node.operands[0]) - resolve(node.operands[1])

        if node.op == "mul":
            return resolve(node.operands[0]) * resolve(node.operands[1])

        if node.op == "floordiv":
            return resolve(node.operands[0]) // resolve(node.operands[1])

        if node.op == "call" and node.value == "min" and len(node.operands) == 2:
            return min(resolve(node.operands[0]), resolve(node.operands[1]))

        raise ValueError(
            f"Ascend launch expression contains unsupported operation '{node.op}'."
        )

    value = resolve(root)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Ascend launch expression did not resolve to an integer.")

    return value


def _reject_storage_aliases(abi, tensors) -> None:
    access_by_name = {
        binding.source: (
            binding.access or ("write" if binding.source in abi.outputs else "read")
        )
        for binding in abi.kernel_args
        if binding.kind == "tensor" and binding.source in tensors
    }

    try:
        reject_write_overlap(tensors, access_by_name)
    except AscendLayoutCapabilityError as exc:
        raise ValueError(str(exc)) from exc


def _current_npu_stream(public):
    tensors = tuple(value for value in public.values() if hasattr(value, "device"))

    if not tensors:
        return None

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ascend kernel launch requires PyTorch with torch_npu installed."
        ) from exc

    npu = getattr(torch, "npu", None)

    if npu is None or not hasattr(npu, "current_stream"):
        raise RuntimeError(
            "Ascend kernel launch requires PyTorch with torch_npu installed."
        )

    try:
        return npu.current_stream(tensors[0].device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire the current Ascend NPU stream; verify torch_npu, "
            "CANN, and the selected NPU device."
        ) from exc


def _max_core_dim(metadata) -> int:
    schedule = dict(metadata.get("ssa_schedule", {}))
    value = schedule.get("core_dim_limit", 65535)

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "Ascend artifact has an invalid `core_dim_limit` metadata value."
        )

    return value


__all__ = ["AscendMaterializer"]
