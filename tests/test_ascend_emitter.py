import ast

import pytest

from ninetoothed.backends import emit
from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters.ascend import diagnose_opcode_coverage
from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec, ssa


def _kernel(
    source: str, *, name: str = "add", tensors: tuple[TensorSpec, ...] | None = None
) -> Kernel:
    tensors = tensors or (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
    )
    program = from_source(source, tensors, kind=name)
    assert program is not None

    return Kernel(
        kernel_name=name,
        source=source,
        source_language="ninetoothed-python",
        entrypoint=name,
        tensors=tensors,
        ssa=program,
    )


def test_ascend_emits_stable_elementwise_triton_source():
    kernel = _kernel("\ndef add(x, y, out):\n    out = x + y\n")

    first = emit(kernel, Target.ASCEND)
    second = emit(kernel, Target.ASCEND)

    assert first.sources == second.sources
    assert first.primary_source_name == "add.ascend.py"
    assert first.language == "python/triton"
    assert first.entrypoint == "launch_add"
    assert first.metadata["ssa_schedule"]["tile"] == {"elements": 256}
    assert first.metadata["ssa_schedule"]["core_dim_limit"] == 65535
    assert "from triton.language.extra.cann import libdevice" in first.primary_source
    assert (
        "offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)"
        in first.primary_source
    )
    assert "block = 256" in first.primary_source
    assert "num_warps=" not in first.primary_source
    assert "num_stages=" not in first.primary_source
    assert "tl.load(x + index, mask=mask, other=0.0)" in first.primary_source
    assert "tl.store(out + index, v0, mask=mask)" in first.primary_source
    ast.parse(first.primary_source)


def test_ascend_emits_pow_from_generic_ssa():
    kernel = _kernel("\ndef add(x, y, out):\n    out = x ** y\n")

    source = emit(kernel, Target.ASCEND).primary_source

    assert "pow(" in source
    ast.parse(source)


def test_ascend_emits_rand_with_seed_offset_abi():
    tensors = (
        TensorSpec(ndim=1, shape=("128",), dtype="float32", name="out"),
        TensorSpec(ndim=0, shape=(), dtype="int32", name="seed"),
        TensorSpec(ndim=1, shape=("128",), dtype="int32", name="offset"),
    )
    program = from_source(
        "\ndef random(out, seed, offset):\n    out = rand(seed, offset)\n",
        tensors,
        kind="random",
    )
    kernel = Kernel(
        kernel_name="random",
        source="",
        source_language="test",
        entrypoint="random",
        tensors=tensors,
        ssa=program,
    )
    source = emit(kernel, Target.ASCEND).primary_source
    assert "tl.rand(seed, tl.load(offset + index" in source
    ast.parse(source)


def test_ascend_emits_explicit_prefix_scan_ssa():
    tensors = (
        TensorSpec(ndim=1, shape=("128",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("128",), dtype="float32", name="out"),
    )
    value = ssa.Value(
        name="%scan", type=ssa.Type(kind="tensor", shape=("128",), dtype="float32")
    )
    program = ssa.Program(
        kind="scan",
        inputs=(
            ssa.Value(
                name="x", type=ssa.Type(kind="tensor", shape=("128",), dtype="float32")
            ),
            ssa.Value(
                name="out",
                type=ssa.Type(kind="tensor", shape=("128",), dtype="float32"),
            ),
        ),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="call.cumsum", operands=("x",), results=(value,)
                    ),
                    ssa.Operation(opcode="mem.store", operands=("%scan", "out")),
                )
            ),
        ),
        metadata={
            "schedule": {
                "granularity": "scan",
                "tile": {"elements": 256},
                "scan": {"mode": "inclusive", "axis": 0, "extent": "128"},
            }
        },
    )
    kernel = Kernel(
        kernel_name="scan",
        source="",
        source_language="test",
        entrypoint="scan",
        tensors=tensors,
        ssa=program,
    )
    source = emit(kernel, Target.ASCEND).primary_source
    assert "tl.cumsum" in source
    ast.parse(source)


@pytest.mark.parametrize("intrinsic", ("exp", "exp2", "log", "sqrt", "tanh"))
def test_ascend_emits_math_intrinsics_from_generic_ssa(intrinsic):
    kernel = _kernel(
        f"\ndef math_op(x, out):\n    out = {intrinsic}(x)\n",
        name="math_op",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
    )

    source = emit(kernel, Target.ASCEND).primary_source

    assert f"tl.{intrinsic}(" in source
    ast.parse(source)


def test_ascend_emits_fill_from_tensor_full():
    kernel = _kernel(
        "\ndef fill(out):\n    out = full((n,), 2.5)\n",
        name="fill",
        tensors=(TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),),
    )

    source = emit(kernel, Target.ASCEND).primary_source

    assert "tl.store(out + index" in source
    assert "2.5" in source
    ast.parse(source)


def test_ascend_emits_contiguous_view_and_index_offset():
    kernel = _kernel(
        "\ndef view_copy(x, out):\n    i = x.offsets(0)\n    out[i] = x\n",
        name="view_copy",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
    )

    source = emit(kernel, Target.ASCEND).primary_source

    assert "tl.load(x + index" in source
    assert "tl.store(out + v0" in source
    ast.parse(source)


def test_ascend_consumes_public_layout_transfer_coordinate_maps():
    tensors = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("n", "m"), dtype="float32", name="out"),
    )
    source = "\ndef transpose(x, out):\n    out = x.T\n"
    program = from_source(source, tensors, kind="transpose")
    assert program is not None
    lowered = lower_for_target(program, backend=Target.ASCEND, tensors=tensors)

    assert lowered.metadata["schedule"]["layout_transfer"] is not None
    kernel = Kernel(
        kernel_name="transpose",
        source=source,
        source_language="ninetoothed-python",
        entrypoint="transpose",
        tensors=tensors,
        ssa=lowered,
    )
    artifact = emit(kernel, Target.ASCEND)

    assert "tl.trans(" in artifact.primary_source
    assert "source_value_0" in artifact.primary_source
    assert "destination_value_0" in artifact.primary_source
    ast.parse(artifact.primary_source)


def test_ascend_emits_structured_loop_and_if():
    kernel = _kernel(
        "\ndef control(x, out):\n    for i in range(n):\n        out[i] = x[i] + 1.0\n",
        name="control",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
    )

    source = emit(kernel, Target.ASCEND).primary_source

    assert "for loop_i in range(0, n, 1):" in source
    assert "tl.store(out + loop_i" in source
    ast.parse(source)


def test_ascend_emits_nested_if_and_for_from_structured_ssa():
    kernel = _kernel(
        """
def nested_control(x, out):
    for i in range(n):
        if x[i] > 0.0:
            out[i] = exp(x[i])
        else:
            out[i] = log(x[i] + 1.0)
""",
        name="nested_control",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
    )

    source = emit(kernel, Target.ASCEND).primary_source

    assert "for loop_i in range(0, n, 1):" in source
    assert "if v4_loop_body:" in source
    assert "tl.exp(" in source
    assert "tl.log(" in source
    ast.parse(source)


def test_ascend_emits_row_vector_reduction():
    kernel = _kernel(
        "\ndef reduce(x, out):\n    out = sum(x, axis=1)\n",
        name="reduce",
        tensors=(
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
        ),
    )
    lowered = lower_for_target(
        kernel.ssa, backend=Target.ASCEND, tensors=kernel.tensors
    )
    kernel = Kernel(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        tensors=kernel.tensors,
        ssa=lowered,
    )
    source = emit(kernel, Target.ASCEND).primary_source

    assert "tl.sum(" in source
    assert "offsets = tl.arange(0, BLOCK)" in source
    ast.parse(source)


def test_ascend_emits_decomposed_matmul_with_tail_safe_loads():
    tensors = (
        TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
        TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )
    kernel = _kernel(
        "\ndef matmul(a, b, out):\n    out = a @ b\n",
        name="matmul",
        tensors=tensors,
    )
    lowered = lower_for_target(kernel.ssa, backend=Target.ASCEND, tensors=tensors)
    kernel = Kernel(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        tensors=tensors,
        ssa=lowered,
    )
    source = emit(kernel, Target.ASCEND).primary_source

    assert "linalg.matmul" not in source
    assert "for v10_i in range(0, k, 1):" in source
    assert source.count("mask=mask, other=0.0") >= 2
    assert diagnose_opcode_coverage(kernel)["unsupported"] == ()
    ast.parse(source)


def test_ascend_rewrites_batched_matmul_access_templates_before_emission():
    tensors = (
        TensorSpec(ndim=3, shape=("2", "3", "257"), dtype="float32", name="a"),
        TensorSpec(ndim=3, shape=("2", "257", "5"), dtype="float32", name="b"),
        TensorSpec(ndim=3, shape=("2", "3", "5"), dtype="float32", name="out"),
    )
    kernel = _kernel(
        "\ndef batched_matmul(a, b, out):\n    out = a @ b\n",
        name="batched_matmul",
        tensors=tensors,
    )
    artifact = emit(kernel, Target.ASCEND)
    rewrite = artifact.metadata["ssa_metadata"]["schedule"][
        "ascend_batched_access_rewrite"
    ]
    source = artifact.primary_source

    assert rewrite["coordinates"] == {
        "lhs": ("batch", "row", "k"),
        "rhs": ("batch", "k", "col"),
    }
    assert "tl.load(a + (v1_v10_body) * (3 * 257)" in source
    assert "(v2_v10_body) * (257) + (v10_i)" in source
    assert "tl.load(b + (v1_v10_body) * (257 * 5)" in source
    assert "(v10_i) * (5) + (vascend_matmul_col_v10_body)" in source
    ast.parse(source)


@pytest.mark.parametrize(
    ("source", "tensors", "opcode"),
    (
        (
            "\ndef reduce(x, out):\n    out = sum(x)\n",
            (
                TensorSpec(ndim=1, shape=("cols",), dtype="float32", name="x"),
                TensorSpec(ndim=0, shape=(), dtype="float32", name="out"),
            ),
            "tl.sum(",
        ),
        (
            "\ndef reduce(x, out):\n    out = max(x, axis=0)\n",
            (
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("cols",), dtype="float32", name="out"),
            ),
            "tl.max(",
        ),
        (
            "\ndef reduce(x, out):\n    out = min(x, axis=1)\n",
            (
                TensorSpec(
                    ndim=3,
                    shape=("depth", "rows", "cols"),
                    dtype="float32",
                    name="x",
                ),
                TensorSpec(
                    ndim=2, shape=("depth", "cols"), dtype="float32", name="out"
                ),
            ),
            "tl.min(",
        ),
    ),
)
def test_ascend_emits_ranked_row_vector_reductions(source, tensors, opcode):
    kernel = _kernel(source, name="reduce", tensors=tensors)
    lowered = lower_for_target(kernel.ssa, backend=Target.ASCEND, tensors=tensors)
    kernel = Kernel(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        tensors=tensors,
        ssa=lowered,
    )
    artifact = emit(kernel, Target.ASCEND)

    assert opcode in artifact.primary_source
    assert diagnose_opcode_coverage(kernel)["unsupported"] == ()
    ast.parse(artifact.primary_source)


def test_ascend_canonicalizes_unary_positive_to_its_operand():
    kernel = _kernel("\ndef pos(x, y, out):\n    out = +x\n", name="pos")

    source = emit(kernel, Target.ASCEND).primary_source

    assert "v0 = (tl.load(x + index, mask=mask, other=0.0)" in source
    assert "+tl.load(" not in source
    ast.parse(source)


def test_ascend_emits_scalar_abi_input_by_value():
    source = emit(
        _kernel(
            "\ndef scale(x, alpha, out):\n    out = x * alpha\n",
            name="scale",
            tensors=(
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=0, shape=(), dtype="float32", name="alpha"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        ),
        Target.ASCEND,
    ).primary_source

    assert "* alpha" in source
    assert "tl.load(alpha" not in source
    ast.parse(source)


def test_ascend_emits_mixed_scalar_vector_type_promotion():
    source = emit(
        _kernel(
            "\ndef scale(x, alpha, out):\n    out = x + alpha\n",
            name="scale",
            tensors=(
                TensorSpec(ndim=1, shape=("n",), dtype="float16", name="x"),
                TensorSpec(ndim=0, shape=(), dtype="float32", name="alpha"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        ),
        Target.ASCEND,
    ).primary_source

    assert " + alpha" in source
    ast.parse(source)


def test_ascend_emits_singleton_broadcast_coordinates():
    tensors = (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("1",), dtype="float32", name="bias"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
    )
    program = from_source(
        "\ndef add(x, bias, out):\n    out = x + bias\n", tensors, kind="add"
    )
    assert program is not None
    kernel = Kernel(
        kernel_name="add",
        source="\ndef add(x, bias, out):\n    out = x + bias\n",
        source_language="ninetoothed-python",
        entrypoint="add",
        tensors=tensors,
        ssa=program,
    )

    source = emit(kernel, Target.ASCEND).primary_source

    assert "tl.load(bias + 0)" in source
    ast.parse(source)


def test_ascend_emits_multiple_output_stores_and_tuple_return():
    tensors = tuple(
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name=name)
        for name in ("x", "y", "out0", "out1")
    )
    source = emit(
        _kernel(
            "\ndef pair(x, y, out0, out1):\n    out0 = x + y\n    out1 = x - y\n",
            name="pair",
            tensors=tensors,
        ),
        Target.ASCEND,
    ).primary_source

    assert source.count("tl.store(out0 +") == 1
    assert source.count("tl.store(out1 +") == 1
    assert "return (out0, out1)" in source
    ast.parse(source)


@pytest.mark.parametrize("shape", (("m", "n"), ("b", "m", "n")))
def test_ascend_emits_flat_contiguous_multidimensional_accesses(shape):
    tensors = tuple(
        TensorSpec(ndim=len(shape), shape=shape, dtype="float32", name=name)
        for name in ("x", "y", "out")
    )
    source = emit(
        _kernel("\ndef add(x, y, out):\n    out = x + y\n", tensors=tensors),
        Target.ASCEND,
    ).primary_source

    assert f"triton.cdiv({' * '.join(shape)}, block)" in source
    assert "tl.load(x +" in source
    assert "tl.store(out + index" in source
    ast.parse(source)


@pytest.mark.parametrize(
    ("dtype", "triton_dtype"),
    (("float16", "float16"), ("bfloat16", "bfloat16")),
)
def test_ascend_emits_verified_low_precision_casts(dtype, triton_dtype):
    source = emit(
        _kernel(
            "\ndef cast(x, out):\n    out = x.to(" + dtype + ")\n",
            name="cast",
            tensors=(
                TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="out"),
            ),
        ),
        Target.ASCEND,
    ).primary_source

    assert f".to(tl.{triton_dtype})" in source
    ast.parse(source)


def test_ascend_emitter_rejects_unverified_dtype_when_called_directly():
    kernel = _kernel(
        "\ndef add(x, y, out):\n    out = x + y\n",
        tensors=(
            TensorSpec(ndim=1, shape=("n",), dtype="float64", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float64", name="y"),
            TensorSpec(ndim=1, shape=("n",), dtype="float64", name="out"),
        ),
    )

    with pytest.raises(ValueError, match="only FP16, BF16, and FP32 elementwise SSA"):
        emit(kernel, Target.ASCEND)
