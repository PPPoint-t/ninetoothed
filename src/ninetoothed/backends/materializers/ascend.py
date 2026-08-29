"""Ascend Triton Python artifact materialization and reload."""

import types
from pathlib import Path
from typing import Any

from ninetoothed.backends.core import BuiltArtifact, Target
from ninetoothed.backends.materializers.base import Materializer
from ninetoothed.compiler.ascend_contracts import (
    normalize_ascend_dtype,
    static_forward_view_offset,
    unsupported_ascend_elementwise_dtypes,
)
from ninetoothed.compiler.cache import (
    atomic_write_text,
    cache_lock,
    compilation_cache_key,
    write_manifest,
    write_source,
)
from ninetoothed.compiler.layout_runtime import (
    memory_spans_overlap,
    tensor_memory_span,
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

        from ninetoothed.compiler.runtime import (
            _launch_abi_from_dict,
            _launch_plan_from_dict,
            _runtime_specs,
        )

        specs = _runtime_specs(built.source)
        _validate_ascend_dtype_specs(specs)
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
            _launch_abi_from_dict(built.abi),
            specs,
            launch_plan=_launch_plan_from_dict(
                built.source.metadata.get("launch_plan", {})
            ),
            source_path=source_path,
            kernel_name=built.source.kernel_name,
            max_core_dim=_max_core_dim(built.source.metadata),
            module=module,
            reduction_schedule=specs
            and built.source.metadata.get("ssa_metadata", {})
            .get("schedule", {})
            .get("reduction"),
            linalg_contract=built.source.metadata.get("ssa_metadata", {})
            .get("schedule", {})
            .get("ascend_linalg"),
        )


def _materialize(compilation, *, output_dir: str | Path | None):
    from ninetoothed.compiler.runtime import Handle, _built_manifest

    artifact = compilation.artifact
    _validate_ascend_dtype_specs(compilation.kernel.tensors)
    cache_key = compilation_cache_key(compilation)
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
        compilation.launch_abi,
        compilation.kernel.tensors,
        launch_plan=compilation.launch_plan,
        source_path=published_source,
        kernel_name=artifact.kernel_name,
        max_core_dim=_max_core_dim(artifact.metadata),
        module=module,
        reduction_schedule=artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("reduction"),
        linalg_contract=artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("ascend_linalg"),
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
    launch_plan=None,
    source_path: Path,
    kernel_name: str,
    max_core_dim: int,
    module,
    reduction_schedule=None,
    linalg_contract=None,
):
    from ninetoothed.compiler.runtime import (
        _bound_values,
        _empty_launch,
        _public_values,
    )

    def launch(*args, **kwargs):
        public = _public_values(
            abi,
            args,
            kwargs,
            specs=specs,
            expected_device_type="npu",
        )

        _validate_ascend_bindings(
            abi,
            public,
            specs,
            max_core_dim,
            logical_domain=(
                launch_plan.logical_domain if launch_plan is not None else None
            ),
            reduction_schedule=reduction_schedule,
            linalg_contract=linalg_contract,
        )

        if _empty_launch(abi, public):
            return _ascend_outputs(abi, public)

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


def _validate_ascend_bindings(
    abi,
    public,
    specs,
    max_core_dim: int,
    *,
    logical_domain=None,
    reduction_schedule=None,
    linalg_contract=None,
) -> None:
    spec_by_name = {spec.name: spec for spec in specs}
    _validate_ascend_dtype_specs(tuple(spec_by_name.values()))
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

        if not value.is_contiguous():
            raise TypeError(
                f"Ascend kernel argument `{binding.source}` must be contiguous."
            )

        _validate_storage_span(binding.source, value)

        if value.storage_offset() != 0:
            raise TypeError(
                f"Ascend kernel argument '{binding.source}' must be a base tensor "
                "with storage_offset() == 0."
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
    linalg_inputs = _matmul_inputs(
        tensors,
        output_names,
        output_shape,
        linalg_contract,
    )

    if not output_shape:
        if logical_elements != 1:
            raise ValueError("Ascend scalar output launch requires logical domain one.")
    elif len(output_shape) not in {1, 2, 3}:
        raise ValueError(
            "Ascend elementwise launch supports only one-, two-, or three-dimensional "
            "contiguous output tensors."
        )

    for name, value in tensors.items():
        shape = tuple(value.shape)

        is_output = name in output_names

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

        if name in reduction_inputs or name in linalg_inputs:
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

        offset = _logical_offset(spec_by_name[name], abi, public)

        required_elements = offset + logical_elements

        if value.numel() < required_elements:
            raise ValueError(
                f"Ascend logical view for '{name}' requires {required_elements} "
                f"elements but its base tensor has {value.numel()}."
            )

        if not is_output and offset == 0 and shape != output_shape:
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

        extent = int(shape[axis])

        if extent > 256:
            raise ValueError(
                "Ascend row-vector reduction extent "
                f"{extent} exceeds BLOCK=256; hierarchical partial reduction "
                "is not implemented."
            )

        inputs.add(name)

    if not inputs:
        raise ValueError(
            "Ascend row-vector reduction requires a contiguous input whose shape "
            "equals the output shape with the reduction axis inserted."
        )

    return frozenset(inputs)


def _matmul_inputs(tensors, output_names, output_shape, linalg_contract):
    if not linalg_contract:
        return frozenset()

    if linalg_contract.get("mode") != "matrix-scalar-loop":
        raise ValueError("Ascend linalg launch has an unsupported contract mode.")

    if len(output_names) != 1 or len(output_shape) != 2:
        raise ValueError("Ascend matmul requires exactly one rank-2 output tensor.")

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

    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise ValueError("Ascend matmul requires rank-2 contiguous inputs.")

    m, k = lhs_shape
    rhs_k, n = rhs_shape

    if (m, n) != output_shape or k != rhs_k:
        raise ValueError(
            "Ascend matmul requires runtime shapes lhs[M,K] @ rhs[K,N] -> "
            "output[M,N]."
        )

    if k > 256:
        raise ValueError(
            "Ascend matmul reduction extent "
            f"{k} exceeds BLOCK=256; hierarchical partial reduction is not "
            "implemented."
        )

    if lhs.dtype != rhs.dtype or lhs.dtype != tensors[output_names[0]].dtype:
        raise TypeError("Ascend matmul currently requires matching input/output dtypes.")

    return frozenset((lhs_name, rhs_name))


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


def _validate_ascend_dtype_specs(specs) -> None:
    unsupported = unsupported_ascend_elementwise_dtypes(
        tuple(spec.dtype for spec in specs if not getattr(spec, "constexpr", False))
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
    bindings = {
        binding.source: binding
        for binding in abi.kernel_args
        if binding.kind == "tensor" and binding.source in tensors
    }
    writers = tuple(
        (name, tensors[name])
        for name, binding in bindings.items()
        if binding.access in {"write", "read_write"}
    )
    readers = tuple(
        (name, tensors[name])
        for name, binding in bindings.items()
        if binding.access in {"read", "read_write"}
    )

    for writer_name, writer in writers:
        for reader_name, reader in readers:
            if writer is reader or memory_spans_overlap(
                tensor_memory_span(writer),
                tensor_memory_span(reader),
            ):
                raise ValueError(
                    "Ascend launch rejects storage overlap between writer "
                    f"'{writer_name}' and reader '{reader_name}'."
                )

    for index, (first_name, first) in enumerate(writers):
        for second_name, second in writers[index + 1 :]:
            if first is second or memory_spans_overlap(
                tensor_memory_span(first),
                tensor_memory_span(second),
            ):
                raise ValueError(
                    "Ascend launch rejects storage overlap between writers "
                    f"'{first_name}' and '{second_name}'."
                )


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
