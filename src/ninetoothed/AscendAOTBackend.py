import functools
import importlib.util
import json
import pathlib
import sys
import uuid


class AscendAOTBackend:
    """NPU runtime backend for AOT entrypoints.

    This backend intentionally avoids CUDA-only `triton.tools.compile`
    and `nvcc` stages. It reuses generated Python launch wrappers and
    resolves the NPU-specific launch symbol when present.
    """

    def __call__(self, func, *, kernel_name, num_warps, num_stages):
        num_warps, num_stages = _normalize_compile_options(num_warps, num_stages)
        artifact = build_kernel_artifact(
            func, kernel_name=kernel_name, num_warps=num_warps, num_stages=num_stages
        )
        return load_kernel_artifact(artifact)


# -----------------------------------------------------------------------------
# Policy / Routing
# -----------------------------------------------------------------------------


def should_use_ascend_aot_dispatch(caller):
    """Return whether Ascend AOT dispatch should be used for current runtime."""
    if caller not in ("torch", "ascend"):
        return False

    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _normalize_compile_options(num_warps, num_stages):
    """Fill missing compile options using project defaults."""
    from ninetoothed.utils import calculate_default_configs

    default_num_warps, default_num_stages = calculate_default_configs()

    if num_warps is None:
        num_warps = default_num_warps

    if num_stages is None:
        num_stages = default_num_stages

    return int(num_warps), int(num_stages)


# -----------------------------------------------------------------------------
# Artifact Build / Load
# -----------------------------------------------------------------------------


def build_kernel_artifact(func, *, kernel_name, num_warps, num_stages):
    """Build a Python-launch-wrapper artifact manifest for Ascend runtime."""
    # Lazy import prevents cache-dir resolution at module import time
    # inside process-pool workers.
    from ninetoothed.generation import CodeGenerator

    code_generator = CodeGenerator()
    source_file = code_generator(
        func,
        caller="torch",
        kernel_name=kernel_name,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=None,
        prettify=False,
    )

    source_path = pathlib.Path(source_file).resolve()
    launch_name = code_generator.launch_func_name

    artifact = {
        "schema_version": 1,
        "backend": "ascend",
        "kind": "python_launch_wrapper",
        "kernel_name": kernel_name,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "source_file": str(source_path),
        "launch_name": launch_name,
        "preferred_launch_name": f"{launch_name}_npu",
    }
    artifact["artifact_id"] = _make_artifact_id(artifact)

    manifest_path = source_path.with_suffix(f".{kernel_name}.ascend-aot.manifest.json")
    _write_manifest(manifest_path, artifact)
    artifact["manifest_path"] = str(manifest_path)

    return artifact


def load_kernel_artifact(artifact):
    """Load a launch callable from an artifact dict or manifest path."""
    artifact = _normalize_artifact(artifact)

    source_path = pathlib.Path(artifact["source_file"])
    module_name = f"{source_path.stem}_{artifact.get('artifact_id', uuid.uuid4().hex)}"
    module = _import_from_path(module_name, str(source_path))
    module_vars = vars(module)

    launch_name = artifact["launch_name"]
    preferred_launch_name = artifact["preferred_launch_name"]

    if preferred_launch_name in module_vars:
        launch_name = preferred_launch_name

    if launch_name not in module_vars:
        raise KeyError(f"Launch symbol `{launch_name}` not found in `{source_path}`.")

    return module_vars[launch_name]


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _normalize_artifact(artifact):
    if isinstance(artifact, (str, pathlib.Path)):
        return _read_manifest(pathlib.Path(artifact))

    if isinstance(artifact, dict):
        manifest_path = artifact.get("manifest_path")

        if manifest_path:
            merged = _read_manifest(pathlib.Path(manifest_path))
            merged.update(artifact)
            return merged

        return _backfill_legacy_artifact(artifact)

    raise TypeError("Ascend artifact must be a dict or manifest path.")


def _backfill_legacy_artifact(artifact):
    source_file = artifact["source_file"]
    launch_name = artifact["launch_name"]
    normalized = dict(artifact)

    normalized.setdefault("schema_version", 1)
    normalized.setdefault("backend", "ascend")
    normalized.setdefault("kind", "python_launch_wrapper")
    normalized.setdefault("preferred_launch_name", f"{launch_name}_npu")
    normalized.setdefault(
        "artifact_id",
        uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}:{launch_name}").hex,
    )

    return normalized


def _write_manifest(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True))


def _read_manifest(path):
    return _backfill_legacy_artifact(json.loads(path.read_text()))


def _make_artifact_id(artifact):
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex


# -----------------------------------------------------------------------------
# Record Build / Assemble
# -----------------------------------------------------------------------------


def build_record(
    premake,
    config,
    *,
    kernel_name,
    arg_to_int,
    generate_suffix,
    annotate_application,
):
    """Build one precompiled record for a single config tuple."""
    args, kwargs, compilation_configs = config

    arrangement, application, tensors = premake(*args, **kwargs)

    import inspect

    premake_signature = inspect.signature(premake)
    bound_arguments = premake_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    combination = bound_arguments.arguments
    combination = {f"{name}_": value for name, value in combination.items()}
    combination |= compilation_configs

    for name, value in combination.items():
        combination[name] = arg_to_int(value)

    kernel_name_ = f"{kernel_name}_{generate_suffix(combination.values())}"

    annotate_application(arrangement, application, tensors)

    built_kernel = build_kernel_artifact(
        application,
        kernel_name=kernel_name_,
        num_warps=compilation_configs.get("num_warps"),
        num_stages=compilation_configs.get("num_stages"),
    )

    application_signature = inspect.signature(application)
    param_names = tuple(application_signature.parameters.keys())

    return kernel_name_, param_names, combination, config, tensors, built_kernel


def build_from_records(
    records,
    *,
    meta_parameters,
    caller,
    kernel_name,
    output_dir,
    arg_to_int,
    kernel_launch_error_cls,
    auto_tune_fn,
    auto_tuned_kernel_cls,
):
    """Assemble dispatcher and auto-tuned wrapper from prebuilt records."""
    configs = tuple(record[3] for record in records)
    all_tensors = tuple(record[4] for record in records)
    all_param_names = tuple(record[1] for record in records)
    combinations = tuple(record[2] for record in records)
    built_kernels = tuple(load_kernel_artifact(record[5]) for record in records)

    tensor_param_names = tuple(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    non_tensor_param_names = tuple(
        functools.reduce(lambda x, y: x | y, combinations, {})
    )

    kernel_before_auto_tuning = build_dispatch_kernel(
        tensor_param_names=tensor_param_names,
        non_tensor_param_names=non_tensor_param_names,
        all_param_names=all_param_names,
        combinations=combinations,
        built_kernels=built_kernels,
        arg_to_int=arg_to_int,
        kernel_launch_error_cls=kernel_launch_error_cls,
    )

    if meta_parameters is None:
        return kernel_before_auto_tuning
    # TODO


# -----------------------------------------------------------------------------
# Dispatch / Matching
# -----------------------------------------------------------------------------


def _match_combination(non_tensor_args, combination, arg_to_int):
    """Return True when runtime non-tensor args match one prebuilt config."""
    for key, expected in combination.items():
        if key not in non_tensor_args:
            return False

        if arg_to_int(non_tensor_args[key]) != expected:
            return False

    return True


def build_dispatch_kernel(
    *,
    tensor_param_names,
    non_tensor_param_names,
    all_param_names,
    combinations,
    built_kernels,
    arg_to_int,
    kernel_launch_error_cls,
):
    """Build runtime dispatcher that selects a concrete kernel by config."""

    def dispatch_kernel(*args, **kwargs):
        if kwargs:
            raise TypeError("Ascend AOT dispatcher only supports positional arguments.")

        expected_num_args = len(tensor_param_names) + len(non_tensor_param_names)

        if len(args) != expected_num_args:
            raise TypeError(f"Expected {expected_num_args} arguments, got {len(args)}.")

        tensor_args = dict(zip(tensor_param_names, args[: len(tensor_param_names)]))
        non_tensor_args = dict(
            zip(non_tensor_param_names, args[len(tensor_param_names) :])
        )

        for param_names, combination, kernel in zip(
            all_param_names, combinations, built_kernels
        ):
            if not _match_combination(non_tensor_args, combination, arg_to_int):
                continue

            call_args = tuple(tensor_args[name] for name in param_names)
            return kernel(*call_args)

        raise kernel_launch_error_cls(
            "No matching Ascend AOT kernel configuration found."
        )

    return dispatch_kernel
