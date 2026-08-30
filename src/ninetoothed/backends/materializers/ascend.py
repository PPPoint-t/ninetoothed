"""Ascend Triton Python artifact materialization and reload."""

import types
from pathlib import Path
from typing import Any, Mapping

import triton
import triton.language as tl

from ninetoothed.backends.ascend import (
    ascend_abi_from_dict,
    ascend_cache_key,
    ascend_logical_domain,
    ascend_uses_access_template,
    normalize_ascend_dtype,
    private_launch_abi,
    read_ascend_sidecar,
    static_forward_view_offset,
    unsupported_ascend_elementwise_dtypes,
    validate_build_policy,
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


@triton.jit
def _ascend_partial_sum_kernel(
    inp, out, input_extent, output_extent, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // output_extent
    chunk = pid % output_extent
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < input_extent
    values = tl.load(inp + row * input_extent + offsets, mask=mask, other=0.0)
    tl.store(
        out + row * output_extent + chunk, tl.sum(tl.where(mask, values, 0.0), axis=0)
    )


@triton.jit
def _ascend_partial_max_kernel(
    inp, out, input_extent, output_extent, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // output_extent
    chunk = pid % output_extent
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < input_extent
    values = tl.load(inp + row * input_extent + offsets, mask=mask, other=-float("inf"))
    tl.store(
        out + row * output_extent + chunk,
        tl.max(tl.where(mask, values, -float("inf")), axis=0),
    )


@triton.jit
def _ascend_partial_min_kernel(
    inp, out, input_extent, output_extent, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // output_extent
    chunk = pid % output_extent
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < input_extent
    values = tl.load(inp + row * input_extent + offsets, mask=mask, other=float("inf"))
    tl.store(
        out + row * output_extent + chunk,
        tl.min(tl.where(mask, values, float("inf")), axis=0),
    )


@triton.jit
def _ascend_tiled_matmul_kernel(
    lhs,
    rhs,
    out,
    m,
    n,
    k,
    batch,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_n = (n + BLOCK_N - 1) // BLOCK_N
    tiles_m = (m + BLOCK_M - 1) // BLOCK_M
    tile = pid % (tiles_m * tiles_n)
    batch_id = pid // (tiles_m * tiles_n)
    tile_m = tile // tiles_n
    tile_n = tile % tiles_n
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        lhs_ptrs = lhs + batch_id * m * k + rows[:, None] * k + kk[None, :]
        rhs_ptrs = rhs + batch_id * k * n + kk[:, None] * n + cols[None, :]
        lhs_vals = tl.load(
            lhs_ptrs, mask=(rows[:, None] < m) & (kk[None, :] < k), other=0.0
        )
        rhs_vals = tl.load(
            rhs_ptrs, mask=(kk[:, None] < k) & (cols[None, :] < n), other=0.0
        )
        acc += tl.dot(lhs_vals, rhs_vals, out_dtype=tl.float32)
    out_ptrs = out + batch_id * m * n + rows[:, None] * n + cols[None, :]
    tl.store(out_ptrs, acc, mask=(rows[:, None] < m) & (cols[None, :] < n))


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
        abi = ascend_abi_from_dict(sidecar["abi"])
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
            tensor_specs,
            source_path=source_path,
            kernel_name=built.source.kernel_name,
            max_core_dim=sidecar["max_core_dim"],
            module=module,
            logical_domain=sidecar["logical_domain"],
            reduction_schedule=sidecar.get("reduction_schedule"),
            partial_reduction=sidecar.get("partial_reduction"),
            linalg_contract=sidecar.get("linalg_contract"),
            layout_contract=sidecar.get("layout_contract"),
            advanced_contract=sidecar.get("advanced_contract"),
            dot_loop=sidecar.get("dot_loop"),
            attention_loop=sidecar.get("attention_loop"),
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
    abi = private_launch_abi(
        compilation.launch_abi,
        compilation.kernel.tensors,
        tuple(artifact.metadata.get("outputs", ())),
    )
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
    wrapped = _ascend_wrapper(
        launch,
        abi,
        compilation.kernel.tensors,
        source_path=published_source,
        kernel_name=artifact.kernel_name,
        max_core_dim=_max_core_dim(artifact.metadata),
        module=module,
        logical_domain=ascend_logical_domain(
            compilation.kernel.tensors,
            tuple(artifact.metadata.get("outputs", ())),
            allow_access_template=ascend_uses_access_template(artifact.metadata),
        ),
        reduction_schedule=artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("reduction"),
        partial_reduction=artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("ascend_partial_reduction"),
        linalg_contract=artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("ascend_linalg"),
        layout_contract=_artifact_layout_contract(artifact.metadata),
        advanced_contract=schedule.get("ascend_advanced"),
        dot_loop=schedule.get("ascend_dot_loop"),
        attention_loop=schedule.get("ascend_attention_loop"),
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
    logical_domain=None,
    reduction_schedule=None,
    partial_reduction=None,
    linalg_contract=None,
    layout_contract=None,
    advanced_contract=None,
    dot_loop=None,
    attention_loop=None,
):
    from ninetoothed.compiler.runtime import (
        _bound_values,
        _empty_launch,
    )

    def launch(*args, **kwargs):
        public = _ascend_public_values(abi, args, kwargs, specs)

        _validate_ascend_bindings(
            abi,
            public,
            specs,
            max_core_dim,
            logical_domain=logical_domain,
            reduction_schedule=reduction_schedule,
            partial_reduction=partial_reduction,
            linalg_contract=linalg_contract,
            layout_contract=layout_contract,
            advanced_contract=advanced_contract,
            dot_loop=dot_loop,
            attention_loop=attention_loop,
        )

        if _empty_launch(abi, public):
            return _ascend_outputs(abi, public)

        if partial_reduction:
            return _launch_partial_reduction(
                abi,
                public,
                specs,
                reduction_schedule,
                partial_reduction,
            )

        if linalg_contract:
            return _launch_tiled_matmul(
                abi,
                public,
                linalg_contract,
            )

        values, keepalive = _bound_values(abi, public, scalar_mode="value")
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


def _ascend_public_values(abi, args, kwargs, specs) -> dict[str, Any]:
    """Bind public arguments with the NPU-only contract kept private."""
    if len(args) > len(abi.public_args):
        raise TypeError(f"Expected at most {len(abi.public_args)} arguments.")
    values = dict(zip(abi.public_args, args))
    unexpected = set(kwargs) - set(abi.public_args)
    if unexpected:
        raise TypeError(
            f"Unexpected kernel arguments: {', '.join(sorted(unexpected))}."
        )
    duplicate = set(values) & set(kwargs)
    if duplicate:
        raise TypeError(
            f"Multiple values for kernel arguments: {', '.join(sorted(duplicate))}."
        )
    values.update(kwargs)
    missing = tuple(name for name in abi.public_args if name not in values)
    if missing:
        raise TypeError(f"Missing kernel arguments: {', '.join(missing)}.")

    expected_device = None
    for spec in specs:
        if getattr(spec, "constexpr", False) or spec.name not in values:
            continue
        value = values[spec.name]
        source_ndim = int(spec.attrs.get("source_ndim", spec.ndim))
        if source_ndim == 0 and getattr(spec, "ndim", 0) == 0:
            # Inputs may be scalar values; scalar outputs were made tensors by
            # The private ABI makes scalar outputs device pointers.
            if hasattr(value, "device"):
                pass
            else:
                continue
        if not hasattr(value, "device") or not hasattr(value, "is_contiguous"):
            raise TypeError(
                f"Ascend kernel argument `{spec.name}` must be a tensor on an NPU device."
            )
        device = value.device
        if getattr(device, "type", str(device).split(":")[0]) != "npu":
            raise TypeError(
                f"Ascend kernel argument `{spec.name}` must be on an NPU device."
            )
        if expected_device is not None and device != expected_device:
            raise TypeError("All Ascend tensor arguments must use the same NPU device.")
        expected_device = device
    return values


def _validate_access_template_contract(dot_loop, attention_loop) -> bool:
    """Admit rank-4 storage only for recognized private generic-SSA schedules."""
    if attention_loop:
        contract = attention_loop
    elif dot_loop:
        raise ValueError(
            "Ascend generic dot-loop runtime is fail-closed: the public access "
            "template source path is not numerically verified on Ascend910B3."
        )
    else:
        return False

    if not isinstance(contract, Mapping):
        raise ValueError("Ascend access-template launch contract is malformed.")

    mode = contract.get("mode")
    if mode not in {"generic-dot-loop", "generic-online-softmax-loop"}:
        raise ValueError(
            "Ascend rank-4 storage requires a recognized generic dot-loop or "
            "online-softmax access-template contract."
        )

    return True


def _validate_ascend_bindings(
    abi,
    public,
    specs,
    max_core_dim: int,
    *,
    logical_domain=None,
    reduction_schedule=None,
    partial_reduction=None,
    linalg_contract=None,
    layout_contract=None,
    advanced_contract=None,
    dot_loop=None,
    attention_loop=None,
) -> None:
    spec_by_name = {spec.name: spec for spec in specs}
    _validate_ascend_dtype_specs(
        tuple(spec_by_name.values()),
        allow_rng_auxiliary=bool((advanced_contract or {}).get("rng")),
        allow_atomic=bool((advanced_contract or {}).get("atomic")),
    )
    access_template = _validate_access_template_contract(dot_loop, attention_loop)
    tensors = {}

    for binding in abi.kernel_args:
        if binding.kind != "tensor" or binding.source not in public:
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
            layout = admit_tensor_layout(
                binding.source,
                value,
                allow_rank4_access_template=access_template,
            )
        except AscendLayoutCapabilityError as exc:
            raise TypeError(str(exc)) from exc

        if not layout.contiguous:
            raise TypeError(
                "Ascend layout lowering has not verified runtime non-contiguous "
                f"stride addressing for `{binding.source}`; pass a contiguous tensor "
                "or use a verified direct layout-transfer kernel."
            )

        tensors[binding.source] = value

    if not tensors:
        return

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
    _validate_partial_reduction(partial_reduction, reduction_schedule)
    linalg_inputs = _matmul_inputs(
        tensors,
        output_names,
        output_shape,
        linalg_contract,
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

            if name in reduction_inputs or name in linalg_inputs:
                continue

            if shape != (1,):
                raise ValueError(
                    "Ascend scalar output launch accepts only scalar-compatible "
                    f"(1,) tensor inputs; `{name}` has shape {shape}."
                )

            continue

        if (
            name in reduction_inputs
            or name in linalg_inputs
            or name in layout_inputs
            or name in advanced_inputs
        ):
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

        if not is_output and _is_supported_broadcast_shape(shape, output_shape):
            continue

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


def _is_supported_broadcast_shape(shape, output_shape) -> bool:
    if shape == output_shape:
        return True

    if len(output_shape) == 1:
        return shape == (1,)

    return len(output_shape) == 2 and shape == (1, output_shape[1])


def _row_reduction_inputs(tensors, output_names, output_shape, reduction_schedule):
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

    if not inputs:
        raise ValueError(
            "Ascend row-vector reduction requires a contiguous input whose shape "
            "equals the output shape with the reduction axis inserted."
        )

    return frozenset(inputs)


def _validate_partial_reduction(contract, reduction_schedule):
    if not contract:
        return
    if (
        not isinstance(contract, Mapping)
        or contract.get("strategy") != "hierarchical-private-stages"
    ):
        raise ValueError("Ascend partial-reduction contract is malformed.")
    if not reduction_schedule or reduction_schedule.get("mode") != "row-vector":
        raise ValueError("Ascend partial reduction requires a row-vector reduction.")
    stages = contract.get("stages", ())
    if not stages or any(int(stage.get("block", 0)) != 256 for stage in stages):
        raise ValueError("Ascend partial reduction stages require BLOCK=256.")


def _launch_partial_reduction(abi, public, specs, reduction_schedule, contract):
    """Launch the private bounded-fan-in reduction tree on one NPU stream."""
    import torch

    _validate_partial_reduction(contract, reduction_schedule)
    if not reduction_schedule:
        raise ValueError("Ascend partial reduction is missing its reduction schedule.")
    input_name = next(
        name
        for name in public
        if name not in abi.outputs and hasattr(public[name], "numel")
    )
    output_name = abi.outputs[0]
    source = public[input_name]
    output = public[output_name]
    if not source.is_contiguous() or not output.is_contiguous():
        raise ValueError("Ascend partial reduction requires contiguous tensors.")
    axis = reduction_schedule.get("axis", source.ndim - 1)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise ValueError("Ascend partial reduction axis must be an integer.")
    if axis < 0:
        axis += source.ndim
    if not 0 <= axis < source.ndim:
        raise ValueError("Ascend partial reduction axis is outside the input rank.")
    extent = int(reduction_schedule.get("extent", source.shape[axis]))
    if extent != int(source.shape[axis]):
        raise ValueError(
            "Ascend partial reduction extent does not match the input shape."
        )
    source_for_reduce = (
        source if axis == source.ndim - 1 else source.movedim(axis, -1).contiguous()
    )
    outer = source_for_reduce.numel() // extent if extent else 0
    stages = tuple(contract["stages"])
    operator = contract.get("operator", "sum")
    if outer == 0:
        return _ascend_outputs(abi, public)

    if operator not in {"sum", "max", "min"}:
        raise ValueError(
            f"Ascend partial reduction operator `{operator}` is unsupported."
        )

    stage_kernel = {
        "sum": _ascend_partial_sum_kernel,
        "max": _ascend_partial_max_kernel,
        "min": _ascend_partial_min_kernel,
    }[operator]

    current = source_for_reduce.reshape((outer, extent))
    for index, stage in enumerate(stages):
        input_extent = int(stage["input_extent"])
        output_extent = int(stage["output_extent"])
        if input_extent != current.shape[1]:
            raise ValueError("Ascend partial reduction stage extents are inconsistent.")
        is_final = index == len(stages) - 1
        target = (
            output.reshape((outer,))
            if is_final
            else torch.empty(
                (outer, output_extent), device=source.device, dtype=source.dtype
            )
        )
        stage_kernel[(outer * output_extent,)](
            current,
            target,
            input_extent,
            output_extent,
            BLOCK=256,
        )
        if not is_final:
            current = target
            stream = getattr(torch.npu, "current_stream", lambda: None)()
            if stream is not None and hasattr(stream, "synchronize"):
                stream.synchronize()

    return _ascend_outputs(abi, public)


def _launch_tiled_matmul(abi, public, contract):
    """Launch the private 16x16x64 tiled matmul kernel."""
    lhs_name = contract.get("lhs")
    rhs_name = contract.get("rhs")
    output_name = abi.outputs[0]
    lhs = public.get(lhs_name)
    rhs = public.get(rhs_name)
    output = public.get(output_name)
    if lhs is None or rhs is None or output is None:
        raise ValueError("Ascend tiled matmul launch is missing tensor arguments.")
    if not lhs.is_contiguous() or not rhs.is_contiguous() or not output.is_contiguous():
        raise ValueError("Ascend tiled matmul requires contiguous tensors.")
    if lhs.ndim not in {2, 3} or rhs.ndim != lhs.ndim or output.ndim != lhs.ndim:
        raise ValueError("Ascend tiled matmul supports only rank-2 or rank-3 tensors.")
    if lhs.ndim == 2:
        m, k = (int(value) for value in lhs.shape)
        rhs_k, n = (int(value) for value in rhs.shape)
        batch = 1
    else:
        batch, m, k = (int(value) for value in lhs.shape)
        rhs_batch, rhs_k, n = (int(value) for value in rhs.shape)
        if batch != rhs_batch or int(output.shape[0]) != batch:
            raise ValueError(
                "Ascend batched matmul requires matching batch dimensions."
            )
    if k != rhs_k or tuple(output.shape[-2:]) != (m, n):
        raise ValueError("Ascend tiled matmul runtime shapes are inconsistent.")
    if lhs.dtype != rhs.dtype or lhs.dtype != output.dtype:
        raise TypeError("Ascend tiled matmul requires matching input/output dtypes.")
    _ascend_tiled_matmul_kernel[(batch * ((m + 15) // 16) * ((n + 15) // 16),)](
        lhs,
        rhs,
        output,
        m,
        n,
        k,
        batch,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=64,
    )
    import torch

    stream = getattr(torch.npu, "current_stream", lambda: None)()
    if stream is not None and hasattr(stream, "synchronize"):
        stream.synchronize()
    return _ascend_outputs(abi, public)


def _matmul_inputs(tensors, output_names, output_shape, linalg_contract):
    if not linalg_contract:
        return frozenset()

    if linalg_contract.get("mode") not in {"matrix-scalar-loop", "tiled-matmul"}:
        raise ValueError("Ascend linalg launch has an unsupported contract mode.")

    if len(output_names) != 1 or len(output_shape) not in {2, 3}:
        raise ValueError(
            "Ascend matmul requires exactly one rank-2 or rank-3 output tensor."
        )

    lhs_name = linalg_contract.get("lhs")
    rhs_name = linalg_contract.get("rhs")

    if not isinstance(lhs_name, str) or not isinstance(rhs_name, str):
        raise ValueError("Ascend matmul launch is missing operand bindings.")

    try:
        lhs = tensors[lhs_name]
        rhs = tensors[rhs_name]
    except KeyError as exc:
        raise ValueError(
            f"Ascend matmul launch is missing operand `{exc.args[0]}`."
        ) from exc

    lhs_shape = tuple(lhs.shape)
    rhs_shape = tuple(rhs.shape)

    if len(lhs_shape) not in {2, 3} or len(rhs_shape) != len(lhs_shape):
        raise ValueError(
            "Ascend matmul requires rank-2 contiguous inputs or matching rank-3 batched inputs."
        )

    if len(lhs_shape) == 2:
        m, k = lhs_shape
        rhs_k, n = rhs_shape
    else:
        batch, m, k = lhs_shape
        rhs_batch, rhs_k, n = rhs_shape
        if batch != rhs_batch or output_shape != (batch, m, n):
            raise ValueError(
                "Ascend batched matmul requires matching batch/output shapes."
            )

    if (m, n) != output_shape[-2:] or k != rhs_k:
        raise ValueError(
            "Ascend matmul requires runtime shapes lhs[M,K] @ rhs[K,N] -> output[M,N]."
        )

    if lhs.dtype != rhs.dtype or lhs.dtype != tensors[output_names[0]].dtype:
        raise TypeError(
            "Ascend matmul currently requires matching input/output dtypes."
        )

    return frozenset((lhs_name, rhs_name))


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
            "Ascend materializer supports only verified FP16, BF16, and FP32 "
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
