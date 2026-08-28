from types import SimpleNamespace

import pytest

from ninetoothed import Tensor
from ninetoothed.backends.core import Target
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.ascend_contracts import (
    is_static_forward_view_offset,
    static_forward_view_offset,
)
from ninetoothed.compiler.driver import (
    _launch_access_modes,
    _launch_plan_dict,
    _logical_view_offset,
)
from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.compiler.runtime import _launch_plan_from_dict
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec


def _program(source: str, tensors: tuple[TensorSpec, ...], kind: str):
    program = from_source(source, tensors, kind=kind)
    assert program is not None

    return program


def _elementwise_program(dtype: str = "float32"):
    return _program(
        "\ndef add(x, y, out):\n    out = x + y\n",
        (
            TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="y"),
            TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="out"),
        ),
        "add",
    )


def _offset_arrangement(input, output):
    return input[1:258], output[1:258]


def _offset_application(input, output):
    output = input + input  # noqa: F841


def _matrix_arrangement(input, other, output):
    return tuple(tensor.tile((17, 31)) for tensor in (input, other, output))


def _volume_arrangement(input, other, output):
    return tuple(tensor.tile((2, 17, 31)) for tensor in (input, other, output))


def _matrix_application(input, other, output):
    output = input + other  # noqa: F841


def _multi_output_matrix_arrangement(input, other, out0, out1):
    return tuple(tensor.tile((17, 31)) for tensor in (input, other, out0, out1))


def _multi_output_volume_arrangement(input, other, out0, out1):
    return tuple(tensor.tile((2, 17, 31)) for tensor in (input, other, out0, out1))


def _mismatched_multi_output_arrangement(input, other, out0, out1):
    return (
        input.tile((17, 31)),
        other.tile((17, 31)),
        out0.tile((17, 31)),
        out1.tile((17, 30)),
    )


def _multi_output_application(input, other, out0, out1):
    out0 = input + other  # noqa: F841
    out1 = input - other  # noqa: F841


def test_ascend_elementwise_schedule_is_conservative_and_deterministic():
    lowered = lower_for_target(
        _elementwise_program(),
        backend=Target.ASCEND,
        compiler_options={"backend_options": {"max_core_dim": 1024}},
    )

    assert tuple(lowered.metadata["pass_trace"]) == (
        "ssa.canonicalize",
        "ssa.analyze_effects",
        "ssa.select_schedule",
        "ssa.ascend.analyze_alias",
        "ssa.ascend.optimize_schedule",
        "ssa.decompose_linalg",
    )
    assert (
        lowered.metadata["selected_schedule_candidate"]
        == "fp16-bf16-fp32-elementwise-256"
    )
    schedule = lowered.metadata["schedule"]
    assert schedule["granularity"] == "elementwise-grid"
    assert schedule["indexing"] == "flat-contiguous"
    assert schedule["parallelism"] == "program-blocks"
    assert schedule["tile"] == {"elements": 256}
    assert schedule["vector_width"] == 1
    assert schedule["core_dim_limit"] == 1024


def test_ascend_launch_plan_carries_static_offset_logical_domain():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_offset_arrangement,
            application=_offset_application,
            tensors=(Tensor(1, dtype="float32"), Tensor(1, dtype="float32")),
            backend=Target.ASCEND,
        )
    )

    assert compilation.launch_plan.logical_domain.render().startswith("min(257,")
    assert "ssa.ascend.analyze_alias" in compilation.pass_trace
    analysis = compilation.artifact.metadata["ssa_metadata"]["ascend_alias_analysis"]
    assert analysis["policy"] == "reject-storage-overlap"
    assert analysis["logical_views"]["input"]["offset"] == "index + 1"
    assert analysis["logical_views"]["output"]["offset"] == "index + 1"
    accesses = {
        binding.source: binding.access
        for binding in compilation.launch_abi.kernel_args
        if binding.kind == "tensor"
    }
    assert accesses == dict(analysis["access_modes"])


@pytest.mark.parametrize(
    ("arrangement", "rank", "domain"),
    (
        (_matrix_arrangement, 2, "min(((1 * 17) * 31),"),
        (_volume_arrangement, 3, "min((((1 * 2) * 17) * 31),"),
    ),
)
def test_ascend_launch_plan_accepts_contiguous_multidimensional_logical_domains(
    arrangement, rank, domain
):
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_matrix_application,
            tensors=tuple(Tensor(rank, dtype="float32") for _ in range(3)),
            backend=Target.ASCEND,
        )
    )

    assert compilation.launch_plan.logical_domain.render().startswith(domain)
    analysis = compilation.artifact.metadata["ssa_metadata"]["ascend_alias_analysis"]
    assert (
        analysis["logical_views"]["input"]["domain"]
        == "(" + ") * (".join("2 17 31".split()[-rank:]) + ")"
    )
    restored = _launch_plan_from_dict(_launch_plan_dict(compilation.launch_plan))
    assert (
        restored.logical_domain.render()
        == compilation.launch_plan.logical_domain.render()
    )


@pytest.mark.parametrize(
    ("arrangement", "rank"),
    (
        (_multi_output_matrix_arrangement, 2),
        (_multi_output_volume_arrangement, 3),
    ),
)
def test_ascend_launch_plan_accepts_matching_multidimensional_outputs(
    arrangement, rank
):
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_multi_output_application,
            tensors=tuple(Tensor(rank, dtype="float32") for _ in range(4)),
            backend=Target.ASCEND,
        )
    )

    assert compilation.launch_abi.outputs == ("out0", "out1")
    access = {
        binding.source: binding.access
        for binding in compilation.launch_abi.kernel_args
        if binding.kind == "tensor"
    }
    assert access["out0"] == access["out1"] == "write"
    restored = _launch_plan_from_dict(_launch_plan_dict(compilation.launch_plan))
    assert (
        restored.logical_domain.render()
        == compilation.launch_plan.logical_domain.render()
    )


def test_ascend_launch_plan_rejects_mismatched_multiple_output_domains():
    with pytest.raises(ValueError, match="output application shapes to match"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_mismatched_multi_output_arrangement,
                application=_multi_output_application,
                tensors=tuple(Tensor(2, dtype="float32") for _ in range(4)),
                backend=Target.ASCEND,
            )
        )


def test_ascend_launch_abi_requires_alias_analysis_metadata():
    artifact = SimpleNamespace(backend=Target.ASCEND, metadata={})

    with pytest.raises(ValueError, match="ssa.ascend.analyze_alias"):
        _launch_access_modes(artifact, _elementwise_program())


@pytest.mark.parametrize(
    ("expression", "offset"),
    (("0", 0), ("3", 3), ("index", 0), ("index + 3", 3)),
)
def test_ascend_static_view_offset_contract_is_shared(expression, offset):
    spec = TensorSpec(
        ndim=1,
        shape=("n",),
        dtype="float32",
        name="out",
        attrs={"view_offsets": (expression,)},
    )

    assert is_static_forward_view_offset(expression)
    assert static_forward_view_offset(expression) == offset
    assert _logical_view_offset(spec) == offset


@pytest.mark.parametrize("expression", ("-1", "index - 1", "index + n"))
def test_ascend_static_view_offset_contract_rejects_non_static_or_backward(expression):
    spec = TensorSpec(
        ndim=1,
        shape=("n",),
        dtype="float32",
        name="out",
        attrs={"view_offsets": (expression,)},
    )

    assert not is_static_forward_view_offset(expression)

    with pytest.raises(ValueError, match="static forward offset"):
        _logical_view_offset(spec)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "fp16", "bf16"))
def test_ascend_accepts_verified_low_precision_dtypes_before_source_emission(dtype):
    lowered = lower_for_target(_elementwise_program(dtype), backend=Target.ASCEND)

    assert (
        lowered.metadata["selected_schedule_candidate"]
        == "fp16-bf16-fp32-elementwise-256"
    )


@pytest.mark.parametrize("dtype", ("float64", "int32"))
def test_ascend_rejects_unverified_dtype_before_source_emission(dtype):
    with pytest.raises(ValueError, match="only FP16, BF16, and FP32 elementwise SSA"):
        lower_for_target(_elementwise_program(dtype), backend=Target.ASCEND)


def test_ascend_rejects_unspecified_runtime_dtype_before_source_emission():
    with pytest.raises(
        ValueError, match="only FP16, BF16, and FP32 elementwise SSA.*unspecified"
    ):
        lower_for_target(
            _elementwise_program(),
            backend=Target.ASCEND,
            tensors=(
                TensorSpec(ndim=1, shape=("n",), dtype=None, name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype=None, name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype=None, name="out"),
            ),
        )


def test_ascend_rejects_layout_transfer_before_source_emission():
    program = _program(
        "\ndef transpose(x, out):\n    out = x.T\n",
        (
            TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
            TensorSpec(ndim=2, shape=("n", "m"), dtype="float32", name="out"),
        ),
        "transpose",
    )

    with pytest.raises(ValueError, match="granularity `layout-transfer`"):
        lower_for_target(program, backend=Target.ASCEND)


def test_ascend_accepts_structured_contiguous_tiled_layouts():
    tensors = tuple(Tensor(1, dtype="float32") for _ in range(3))
    arranged = tuple(tensor.tile((64,)) for tensor in tensors)
    specs = tensor_specs(("x", "y", "out"), arranged)
    program = _program("\ndef add(x, y, out):\n    out = x + y\n", specs, "add")

    lowered = lower_for_target(program, backend=Target.ASCEND, tensors=specs)

    assert (
        lowered.metadata["selected_schedule_candidate"]
        == "fp16-bf16-fp32-elementwise-256"
    )


def test_ascend_rejects_jagged_tensors_before_source_emission():
    tensors = (
        TensorSpec(
            ndim=1,
            shape=("n",),
            dtype="float32",
            jagged_dim=0,
            name="x",
        ),
    )

    with pytest.raises(ValueError, match="does not support jagged tensors"):
        lower_for_target(_elementwise_program(), backend=Target.ASCEND, tensors=tensors)


def test_ascend_accepts_scalar_abi_before_source_emission():
    program = _program(
        "\ndef scale(x, alpha, out):\n    out = x * alpha\n",
        (
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=0, shape=(), dtype="float32", name="alpha"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
        "scale",
    )

    lowered = lower_for_target(program, backend=Target.ASCEND)

    assert lowered.metadata["schedule"]["granularity"] == "elementwise-grid"


def test_ascend_rejects_invalid_core_limit_before_schedule_selection():
    with pytest.raises(ValueError, match="between 1 and 65535"):
        lower_for_target(
            _elementwise_program(),
            backend=Target.ASCEND,
            compiler_options={"backend_options": {"max_core_dim": 0}},
        )


@pytest.mark.parametrize(
    ("source", "tensors", "kind", "granularity"),
    (
        (
            "\ndef reduce(x, out):\n    out = sum(x, axis=0)\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=0, shape=(), dtype="float32", name="out"),
            ),
            "reduce",
            "parallel-reduction",
        ),
        (
            "\ndef matmul(a, b, out):\n    out = a @ b\n",
            (
                TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
                TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
                TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
            ),
            "matmul",
            "blocked-linalg",
        ),
    ),
)
def test_ascend_rejects_unverified_schedule_granularity(
    source, tensors, kind, granularity
):
    program = _program(source, tensors, kind)

    with pytest.raises(ValueError, match=f"granularity `{granularity}`"):
        lower_for_target(program, backend=Target.ASCEND)


def test_ascend_rejects_cuda_schedule_parameters():
    with pytest.raises(ValueError, match="does not support `num_warps`"):
        lower_for_target(
            _elementwise_program(),
            backend=Target.ASCEND,
            compiler_options={"num_warps": 4},
        )

    with pytest.raises(
        ValueError, match="does not support schedule option `num_stages`"
    ):
        lower_for_target(
            _elementwise_program(),
            backend=Target.ASCEND,
            pass_options={
                "ssa.ascend.optimize_schedule": {"schedule": {"num_stages": 2}}
            },
        )
