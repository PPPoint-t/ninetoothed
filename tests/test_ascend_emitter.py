import ast

import pytest

from ninetoothed.backends import emit
from ninetoothed.backends.core import Target
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec


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


def test_ascend_rejects_unverified_elementwise_opcode_at_emission():
    kernel = _kernel("\ndef add(x, y, out):\n    out = x ** y\n")

    with pytest.raises(ValueError, match=r"unsupported SSA opcode\(s\): `arith.pow`"):
        emit(kernel, Target.ASCEND)


def test_ascend_canonicalizes_unary_positive_to_its_operand():
    kernel = _kernel("\ndef pos(x, y, out):\n    out = +x\n", name="pos")

    source = emit(kernel, Target.ASCEND).primary_source

    assert "v0 = tl.load(x + index, mask=mask, other=0.0)" in source
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
