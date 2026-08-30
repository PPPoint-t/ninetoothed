"""Hardware-independent contracts for the initial Ascend build policy."""

import importlib
from types import SimpleNamespace

import pytest

from ninetoothed import Tensor
from ninetoothed.backends.ascend import ascend_cache_key
from ninetoothed.build import build
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _arrangement(input, other, output):
    return tuple(tensor.tile((64,)) for tensor in (input, other, output))


def _application(input, other, output):
    output = input + other  # noqa: F841


def _request(**options):
    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=(
            Tensor(1, dtype="float32"),
            Tensor(1, dtype="float32"),
            Tensor(1, dtype="float32"),
        ),
        backend="ascend",
        backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        **options,
    )


def test_ascend_compile_has_one_ssa_schedule_and_no_runtime_candidates():
    compilation = DEFAULT_COMPILER.compile(_request(max_num_configs=1))

    assert compilation.artifact.backend.value == "ascend"
    assert compilation.artifact.metadata["source_route"] == (
        "ssa-unified-ascend-triton-emitter"
    )
    schedule = compilation.artifact.metadata["ssa_schedule"]
    assert schedule["granularity"] == "elementwise-grid"
    assert dict(schedule["tile"]) == {"elements": 256}
    assert schedule["vector_width"] == 1
    assert schedule["core_dim_limit"] == 8
    assert compilation.launch_plan.tuning_candidates == ()


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"num_warps": 4}, "does not support `num_warps`"),
        ({"num_stages": 2}, "does not support `num_stages`"),
        ({"max_num_configs": 2}, "auto-tuning is not supported"),
    ),
)
def test_ascend_compile_rejects_runtime_tuning_controls(options, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        DEFAULT_COMPILER.compile(_request(**options))


def test_ascend_cache_key_isolated_by_soc_and_toolchain_target(monkeypatch):
    monkeypatch.setenv("TRITON_ASCEND_ARCH", "Ascend910B3")
    first = ascend_cache_key("base", {"ssa_schedule": {"soc_version": "Ascend910B3"}})

    monkeypatch.setenv("TRITON_ASCEND_ARCH", "Ascend310P3")
    changed_arch = ascend_cache_key(
        "base", {"ssa_schedule": {"soc_version": "Ascend910B3"}}
    )
    changed_soc = ascend_cache_key(
        "base", {"ssa_schedule": {"soc_version": "Ascend910B4"}}
    )

    assert first != changed_arch
    assert changed_arch != changed_soc


def test_ascend_build_uses_generic_multiple_candidate_path(tmp_path, monkeypatch):
    build_module = importlib.import_module("ninetoothed.build")

    materialized = []

    def materialize(compilation, *, output_dir, mode):
        materialized.append((compilation, output_dir, mode))

        class Materialized:
            pass

        result = Materialized()
        result._source = "source"
        result._artifact = compilation.artifact
        result._backend = "ascend"
        result._kernel = None
        result._library = None
        result._ssa = compilation.kernel.ssa
        result._pass_trace = compilation.pass_trace
        result._launch_plan = compilation.launch_plan
        result._built_artifact = SimpleNamespace(
            cache_key=compilation.artifact.kernel_name
        )

        return result

    monkeypatch.setattr(build_module.DEFAULT_COMPILER, "materialize", materialize)

    def premake():
        return (
            _arrangement,
            _application,
            (
                Tensor(1, dtype="float32"),
                Tensor(1, dtype="float32"),
                Tensor(1, dtype="float32"),
            ),
        )

    handle = build(
        premake,
        (((), {}, {}), ((), {}, {})),
        backend="ascend",
        output_dir=tmp_path,
    )

    assert len(materialized) == 2
    assert all(variant.handle._tuner is not None for variant in handle._variants)


def test_ascend_build_compiles_each_runtime_variant_once(tmp_path, monkeypatch):
    build_module = importlib.import_module("ninetoothed.build")
    materialized = []

    def materialize(compilation, *, output_dir, mode):
        materialized.append((compilation, output_dir, mode))

        return SimpleNamespace(
            _source="source",
            _artifact=compilation.artifact,
            _backend="ascend",
            _kernel=None,
            _library=None,
            _ssa=compilation.kernel.ssa,
            _pass_trace=compilation.pass_trace,
            _launch_plan=compilation.launch_plan,
            _built_artifact=SimpleNamespace(cache_key=compilation.artifact.kernel_name),
        )

    monkeypatch.setattr(build_module.DEFAULT_COMPILER, "materialize", materialize)

    def premake(size):
        def arrangement(input, other, output):
            return tuple(tensor.tile((size,)) for tensor in (input, other, output))

        return (
            arrangement,
            _application,
            (
                Tensor(1, dtype="float32"),
                Tensor(1, dtype="float32"),
                Tensor(1, dtype="float32"),
            ),
        )

    handle = build(
        premake,
        (((64,), {}, {}), ((128,), {}, {})),
        backend="ascend",
        output_dir=tmp_path,
    )

    assert len(materialized) == 2
    assert all(
        compilation.launch_plan.tuning_candidates == ()
        for compilation, _, _ in materialized
    )
    assert all(variant.handle._tuner is None for variant in handle._variants)
