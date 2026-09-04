import pytest

from ninetoothed import Tensor
from ninetoothed.backends.ascend import (
    _ascend_advanced_contract,
    _ascend_attention_loop_contract,
    ascend_capability_matrix,
    ascend_logical_domain,
    is_static_forward_view_offset,
    static_forward_view_offset,
)
from ninetoothed.backends.core import Target
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec, ssa


def test_ascend_llm_attention_capability_boundaries_are_explicit():
    operations = ascend_capability_matrix()["operations"]
    assert operations["attention"].startswith("verified-static")
    assert operations["paged_attention"].startswith("fail-closed")
    assert operations["varlen_attention"].startswith("fail-closed")
    assert operations["gqa_mqa"].startswith("fail-closed")


def test_ascend_weight_only_and_fp8_capability_boundaries_are_explicit():
    matrix = ascend_capability_matrix()
    assert matrix["operations"]["weight_only_matmul"].startswith("fail-closed")
    assert matrix["dtypes"]["fail_closed"]["int8"].startswith("no-verified")
    assert matrix["dtypes"]["fail_closed"]["int4"].startswith("no-native")
    assert "not-supported" in matrix["dtypes"]["fail_closed"]["float8_e4m3fn"]
    assert "not-supported" in matrix["dtypes"]["fail_closed"]["float8_e5m2"]


def test_ascend_pipeline_and_autotune_boundaries_are_explicit():
    micro = ascend_capability_matrix()["microarchitecture"]
    assert micro["double_buffering"].startswith("not-emittable")
    assert micro["async_hbm_l1_ub_l0"] == "not-verified"
    assert micro["tile_autotuning"].startswith("fail-closed")
    assert micro["verified_gemm_tile"] == {"m": 16, "n": 16, "k": 64}


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


def test_ascend_rng_contract_requires_seed_and_offset():
    tensors = (
        TensorSpec(ndim=1, shape=("128",), dtype="float32", name="out"),
        TensorSpec(ndim=0, shape=(), dtype="int32", name="seed"),
        TensorSpec(ndim=1, shape=("128",), dtype="int32", name="offset"),
    )
    program = _program(
        "\ndef random(out, seed, offset):\n    out = rand(seed, offset)\n",
        tensors,
        "random",
    )
    contract = _ascend_advanced_contract(program)
    assert contract["kind"] == "rng-atomic"
    assert contract["seed_offset_abi"] == "seed,offset"


def test_ascend_advanced_contract_rejects_unlowered_conv():
    operation = ssa.Operation(
        opcode="call.conv2d",
        operands=("x",),
        results=(
            ssa.Value(
                name="%0",
                type=ssa.Type(kind="tensor", shape=("128",), dtype="float32"),
            ),
        ),
    )
    program = ssa.Program(
        kind="conv",
        inputs=(
            ssa.Value(
                name="x", type=ssa.Type(kind="tensor", shape=("128",), dtype="float32")
            ),
        ),
        outputs=(
            ssa.Value(
                name="out",
                type=ssa.Type(kind="tensor", shape=("128",), dtype="float32"),
            ),
        ),
        blocks=(ssa.Block(operations=(operation,)),),
    )
    with pytest.raises(ValueError, match="generic dot-loop"):
        _ascend_advanced_contract(program)


def test_ascend_attention_contract_rejects_incomplete_loop_carried_ssa():
    value = ssa.Value(name="%v", type=ssa.Type(kind="scalar", dtype="float32"))
    loop = ssa.Operation(
        opcode="scf.for",
        operands=("%zero", "%extent", "%one", "%state"),
        results=(value,),
        attrs={"iter_args": ({"name": "state"},)},
        regions=(
            ssa.Block(
                operations=(
                    ssa.Operation(opcode="linalg.dot"),
                    ssa.Operation(opcode="math.exp"),
                    ssa.Operation(opcode="scf.yield", operands=("%state",)),
                )
            ),
        ),
    )
    program = ssa.Program(kind="attention", blocks=(ssa.Block(operations=(loop,)),))
    assert _ascend_attention_loop_contract(program) is None


def test_ascend_attention_contract_rejects_unrelated_exp_dot_loop():
    loop = ssa.Operation(
        opcode="scf.for",
        attrs={"iter_args": ({"name": "state"},)},
        regions=(
            ssa.Block(
                operations=(
                    ssa.Operation(opcode="linalg.dot"),
                    ssa.Operation(opcode="math.exp2"),
                    ssa.Operation(opcode="scf.yield"),
                )
            ),
        ),
    )
    assert (
        _ascend_attention_loop_contract(
            ssa.Program(kind="unrelated", blocks=(ssa.Block(operations=(loop,)),))
        )
        is None
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

    assert ascend_logical_domain(
        compilation.kernel.tensors, compilation.artifact.metadata["outputs"]
    ).startswith("min(257,")
    analysis = compilation.artifact.metadata["ssa_metadata"]["ascend_alias_analysis"]
    assert analysis["policy"] == "reject-storage-overlap"
    assert analysis["logical_views"]["input"]["offset"] == "index + 1"
    assert analysis["logical_views"]["output"]["offset"] == "index + 1"


@pytest.mark.parametrize(
    ("arrangement", "rank", "dimensions"),
    (
        (_matrix_arrangement, 2, ("(17)", "(31)")),
        (_volume_arrangement, 3, ("(2)", "(17)", "(31)")),
    ),
)
def test_ascend_launch_plan_accepts_contiguous_multidimensional_logical_domains(
    arrangement, rank, dimensions
):
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_matrix_application,
            tensors=tuple(Tensor(rank, dtype="float32") for _ in range(3)),
            backend=Target.ASCEND,
        )
    )

    domain = ascend_logical_domain(
        compilation.kernel.tensors, compilation.artifact.metadata["outputs"]
    )
    assert domain.startswith("min(")
    assert all(dimension in domain for dimension in dimensions)
    analysis = compilation.artifact.metadata["ssa_metadata"]["ascend_alias_analysis"]
    assert (
        analysis["logical_views"]["input"]["domain"]
        == "(" + ") * (".join("2 17 31".split()[-rank:]) + ")"
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
    assert ascend_logical_domain(
        compilation.kernel.tensors, compilation.artifact.metadata["outputs"]
    )


def test_ascend_private_logical_domain_handles_multiple_outputs():
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_mismatched_multi_output_arrangement,
            application=_multi_output_application,
            tensors=tuple(Tensor(2, dtype="float32") for _ in range(4)),
            backend=Target.ASCEND,
        )
    )
    assert ascend_logical_domain(
        compilation.kernel.tensors, compilation.artifact.metadata["outputs"]
    )


@pytest.mark.parametrize(
    ("expression", "offset"),
    (("0", 0), ("3", 3), ("index", 0), ("index + 3", 3)),
)
def test_ascend_static_view_offset_contract_is_shared(expression, offset):
    assert is_static_forward_view_offset(expression)
    assert static_forward_view_offset(expression) == offset


@pytest.mark.parametrize("expression", ("-1", "index - 1", "index + n"))
def test_ascend_static_view_offset_contract_rejects_non_static_or_backward(expression):
    assert not is_static_forward_view_offset(expression)


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "fp16", "bf16"))
def test_ascend_accepts_verified_low_precision_dtypes_before_source_emission(dtype):
    lowered = lower_for_target(_elementwise_program(dtype), backend=Target.ASCEND)

    assert (
        lowered.metadata["selected_schedule_candidate"]
        == "fp16-bf16-fp32-elementwise-256"
    )


@pytest.mark.parametrize("dtype", ("float64",))
def test_ascend_rejects_unverified_dtype_before_source_emission(dtype):
    with pytest.raises(ValueError, match="only FP16, BF16, and FP32 elementwise SSA"):
        lower_for_target(_elementwise_program(dtype), backend=Target.ASCEND)


def test_ascend_accepts_unspecified_dtype_for_runtime_specialization():
    lowered = lower_for_target(
        _elementwise_program(),
        backend=Target.ASCEND,
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype=None, name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype=None, name="y"),
            TensorSpec(ndim=1, shape=("n",), dtype=None, name="out"),
        ),
    )

    assert lowered.metadata["target_backend"] == Target.ASCEND.value


def test_ascend_emits_private_layout_transfer_schedule():
    program = _program(
        "\ndef transpose(x, out):\n    out = x.T\n",
        (
            TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
            TensorSpec(ndim=2, shape=("n", "m"), dtype="float32", name="out"),
        ),
        "transpose",
    )

    lowered = lower_for_target(program, backend=Target.ASCEND)

    assert lowered.metadata["schedule"]["granularity"] == "layout-transfer"
    assert lowered.metadata["schedule"]["ascend_block_meta"] == {
        "TILE_M": 16,
        "TILE_N": 16,
    }


def test_ascend_accepts_emittable_row_vector_reduction_schedule():
    program = _program(
        "\ndef reduce(x, out):\n    out = sum(x, axis=1)\n",
        (
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
        "reduce",
    )

    lowered = lower_for_target(
        program,
        backend=Target.ASCEND,
        tensors=(
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
    )

    assert lowered.metadata["schedule"]["granularity"] == "parallel-reduction"
    assert lowered.metadata["schedule"]["reduction"]["mode"] == "row-vector"
    assert lowered.metadata["selected_schedule_candidate"] == "ascend-row-reduction-256"


def test_ascend_rejects_non_emittable_reduction_schedule():
    program = _program(
        "\ndef reduce(x, out):\n    out = sum(x)\n",
        (
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
        "reduce",
    )

    with pytest.raises(ValueError, match="row-vector reduction"):
        lower_for_target(program, backend=Target.ASCEND)


def test_ascend_keeps_large_row_reduction_in_unified_ssa():
    program = _program(
        "\ndef reduce(x, out):\n    out = sum(x, axis=1)\n",
        (
            TensorSpec(ndim=2, shape=("rows", "257"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
        "reduce",
    )

    lowered = lower_for_target(
        program,
        backend=Target.ASCEND,
        tensors=(
            TensorSpec(ndim=2, shape=("rows", "257"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
    )
    assert "ascend_partial_reduction" not in lowered.metadata["schedule"]
    assert lowered.metadata["selected_schedule_candidate"] == "ascend-row-reduction-256"


@pytest.mark.ascend_next_stage
def test_ascend_large_row_reduction_has_no_materializer_contract():
    program = _program(
        "\ndef reduce(x, out):\n    out = sum(x, axis=1)\n",
        (
            TensorSpec(ndim=2, shape=("rows", "1025"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
        "reduce",
    )
    lowered = lower_for_target(
        program,
        backend=Target.ASCEND,
        tensors=(
            TensorSpec(ndim=2, shape=("rows", "1025"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
    )
    assert "ascend_partial_reduction" not in lowered.metadata["schedule"]


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
    ),
)
def test_ascend_rejects_unverified_schedule_granularity(
    source, tensors, kind, granularity
):
    program = _program(source, tensors, kind)

    with pytest.raises(ValueError, match=f"granularity `{granularity}`"):
        lower_for_target(program, backend=Target.ASCEND)


def test_ascend_accepts_decomposed_contiguous_matmul_schedule():
    tensors = (
        TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
        TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )
    lowered = lower_for_target(
        _program("\ndef matmul(a, b, out):\n    out = a @ b\n", tensors, "matmul"),
        backend=Target.ASCEND,
        tensors=tensors,
    )

    assert lowered.metadata["schedule"]["granularity"] == "blocked-linalg"
    assert (
        lowered.metadata["selected_schedule_candidate"]
        == "ascend-tiled-matmul-16x16x64"
    )
    assert lowered.metadata["schedule"]["ascend_linalg"]["reduction_extent"] == "k"


def test_ascend_accepts_matmul_partial_reduction_above_fixed_block():
    tensors = (
        TensorSpec(ndim=2, shape=("m", "257"), dtype="float32", name="a"),
        TensorSpec(ndim=2, shape=("257", "n"), dtype="float32", name="b"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )

    lowered = lower_for_target(
        _program("\ndef matmul(a, b, out):\n    out = a @ b\n", tensors, "matmul"),
        backend=Target.ASCEND,
        tensors=tensors,
    )
    assert lowered.metadata["schedule"]["ascend_linalg"]["reduction_extent"] == "257"


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
