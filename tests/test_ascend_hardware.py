"""Capability-gated integration tests for the first Ascend execution tier."""

import os

import pytest

from ninetoothed import Tensor, bfloat16, float16, float32
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact

pytestmark = pytest.mark.skipif(
    os.environ.get("NINETOOTHED_RUN_ASCEND_TESTS") != "1",
    reason="set NINETOOTHED_RUN_ASCEND_TESTS=1 on an Ascend runner",
)


def _arrangement(input, other, output):
    return tuple(tensor.tile((513,)) for tensor in (input, other, output))


def _unary_arrangement(input, output):
    return tuple(tensor.tile((513,)) for tensor in (input, output))


def _scalar_extract_arrangement(input, output):
    return input.tile((1,)), output


def _scalar_input_arrangement(input, alpha, output):
    return input.tile((513,)), alpha, output.tile((513,))


def _application(input, other, output):
    output = input + other  # noqa: F841


def _broadcast_arrangement(input, bias, output):
    return input.tile((513,)), bias.tile((1,)), output.tile((513,))


def _broadcast_application(input, bias, output):
    output = input + bias  # noqa: F841


def _matrix_arrangement(input, other, output):
    return tuple(tensor.tile((17, 31)) for tensor in (input, other, output))


def _volume_arrangement(input, other, output):
    return tuple(tensor.tile((2, 17, 31)) for tensor in (input, other, output))


def _matrix_broadcast_arrangement(input, bias, output):
    return input.tile((17, 31)), bias.tile((1, 31)), output.tile((17, 31))


def _matrix_scalar_arrangement(input, alpha, output):
    return input.tile((17, 31)), alpha, output.tile((17, 31))


def _multi_output_matrix_arrangement(input, other, out0, out1):
    return tuple(tensor.tile((17, 31)) for tensor in (input, other, out0, out1))


def _multi_output_volume_arrangement(input, other, out0, out1):
    return tuple(tensor.tile((2, 17, 31)) for tensor in (input, other, out0, out1))


def _multi_output_scalar_arrangement(input, alpha, out0, out1):
    return input.tile((17, 31)), alpha, out0.tile((17, 31)), out1.tile((17, 31))


def _offset_arrangement(input, output):
    return input[1:258], output[1:258]


def _offset_application(input, output):
    output = input + input  # noqa: F841


def _sub_application(input, other, output):
    output = input - other  # noqa: F841


def _mul_application(input, other, output):
    output = input * other  # noqa: F841


def _div_application(input, other, output):
    output = input / other  # noqa: F841


def _neg_application(input, output):
    output = -input  # noqa: F841


def _pos_application(input, output):
    output = +input  # noqa: F841


def _constant_application(input, output):
    output = input + 1.25  # noqa: F841


def _cast_application(input, output):
    output = input.to(float32)  # noqa: F841


def _scalar_extract_application(input, output):
    output = input[0]  # noqa: F841


def _scalar_input_mul_application(input, alpha, output):
    output = input * alpha  # noqa: F841


def _matrix_scalar_mul_application(input, alpha, output):
    output = input * alpha  # noqa: F841


def _multi_output_application(input, other, out0, out1):
    out0 = input + other  # noqa: F841
    out1 = input - other  # noqa: F841


def _multi_output_scalar_application(input, alpha, out0, out1):
    out0 = input * alpha  # noqa: F841
    out1 = input + alpha  # noqa: F841


def _select_eq_application(input, other, output):
    output = input if input == other else other  # noqa: F841


def _select_ne_application(input, other, output):
    output = input if input != other else other  # noqa: F841


def _select_lt_application(input, other, output):
    output = input if input < other else other  # noqa: F841


def _select_le_application(input, other, output):
    output = input if input <= other else other  # noqa: F841


def _select_gt_application(input, other, output):
    output = input if input > other else other  # noqa: F841


def _select_ge_application(input, other, output):
    output = input if input >= other else other  # noqa: F841


def _compile(application, arity, tmp_path, arrangement=_arrangement, tensors=None):
    return DEFAULT_COMPILER.materialize(
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=arrangement,
                application=application,
                tensors=(
                    tuple(Tensor(1, dtype="float32") for _ in range(arity))
                    if tensors is None
                    else tensors
                ),
                backend="ascend",
                backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
            )
        ),
        output_dir=tmp_path,
        mode="aot",
    )


def _compile_dtype(application, tmp_path, *, arrangement, tensors, mode):
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )

    return DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode=mode)


def _assert_sizes(handle, reference):
    import torch

    for size in (0, 1, 255, 256, 257, 513):
        input = torch.linspace(-2.0, 2.0, size, device="npu", dtype=torch.float32)
        other = torch.linspace(1.0, 3.0, size, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)

        assert handle(input, other, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, reference(input, other))


@pytest.mark.parametrize(
    ("opcode", "application", "reference"),
    (
        ("arith.sub", _sub_application, lambda input, other: input - other),
        ("arith.mul", _mul_application, lambda input, other: input * other),
        ("arith.div", _div_application, lambda input, other: input / other),
    ),
)
def test_ascend_fp32_binary_opcode_tail_matrix(
    opcode, application, reference, tmp_path
):
    import torch_npu  # noqa: F401

    handle = _compile(application, 3, tmp_path)

    _assert_sizes(handle, reference)


@pytest.mark.parametrize(
    ("opcode", "application", "reference"),
    (
        ("arith.neg", _neg_application, lambda input: -input),
        ("arith.pos", _pos_application, lambda input: +input),
        ("arith.constant", _constant_application, lambda input: input + 1.25),
    ),
)
def test_ascend_fp32_unary_opcode_tail_matrix(opcode, application, reference, tmp_path):
    import torch
    import torch_npu  # noqa: F401

    handle = _compile(application, 2, tmp_path, _unary_arrangement)

    for size in (0, 1, 255, 256, 257, 513):
        input = torch.linspace(-2.0, 2.0, size, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)

        assert handle(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, reference(input))


@pytest.mark.parametrize(
    ("opcode", "application", "reference"),
    (
        ("cmp.eq/select.where", _select_eq_application, lambda x, y: x == y),
        ("cmp.ne/select.where", _select_ne_application, lambda x, y: x != y),
        ("cmp.lt/select.where", _select_lt_application, lambda x, y: x < y),
        ("cmp.le/select.where", _select_le_application, lambda x, y: x <= y),
        ("cmp.gt/select.where", _select_gt_application, lambda x, y: x > y),
        ("cmp.ge/select.where", _select_ge_application, lambda x, y: x >= y),
    ),
)
def test_ascend_fp32_comparison_select_tail_matrix(
    opcode, application, reference, tmp_path
):
    import torch
    import torch_npu  # noqa: F401

    handle = _compile(application, 3, tmp_path)

    for size in (0, 1, 255, 256, 257, 513):
        input = torch.linspace(-2.0, 2.0, size, device="npu", dtype=torch.float32)
        other = torch.linspace(2.0, -2.0, size, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)

        assert handle(input, other, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(
            output, torch.where(reference(input, other), input, other)
        )


def test_ascend_fp32_cast_noop_tail_matrix(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    handle = _compile(_cast_application, 2, tmp_path, _unary_arrangement)

    for size in (0, 1, 255, 256, 257, 513):
        input = torch.linspace(-2.0, 2.0, size, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)

        assert handle(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input)


def test_ascend_fp32_tensor_extract_to_scalar_output(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    handle = _compile(
        _scalar_extract_application,
        2,
        tmp_path,
        _scalar_extract_arrangement,
        tensors=(Tensor(1, dtype="float32"), Tensor(0, dtype="float32")),
    )
    input = torch.tensor([3.25], device="npu", dtype=torch.float32)
    output = torch.empty((), device="npu", dtype=torch.float32)

    assert handle(input, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input[0])

    reloaded = load_built_artifact(handle._built_artifact)
    output.zero_()
    assert reloaded(input, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input[0])


def test_ascend_fp32_scalar_input_tail_and_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    handle = _compile(
        _scalar_input_mul_application,
        3,
        tmp_path,
        _scalar_input_arrangement,
        tensors=(
            Tensor(1, dtype="float32"),
            Tensor(0, dtype="float32"),
            Tensor(1, dtype="float32"),
        ),
    )
    alpha = 1.75

    for size in (0, 1, 255, 256, 257, 513):
        input = torch.linspace(-2.0, 2.0, size, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)

        assert handle(input, alpha, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input * alpha)

    reloaded = load_built_artifact(handle._built_artifact)
    input = torch.arange(257, device="npu", dtype=torch.float32)
    output = torch.empty_like(input)
    assert reloaded(input, alpha, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input * alpha)


@pytest.mark.parametrize(
    ("shape", "arrangement"),
    (((17, 31), _matrix_arrangement), ((2, 17, 31), _volume_arrangement)),
)
def test_ascend_fp32_contiguous_multidimensional_jit_aot_reload(
    shape, arrangement, tmp_path
):
    import torch
    import torch_npu  # noqa: F401

    tensors = tuple(Tensor(len(shape), dtype="float32") for _ in range(3))
    jit = _compile_dtype(
        _application,
        tmp_path,
        arrangement=arrangement,
        tensors=tensors,
        mode="jit",
    )
    input = torch.arange(
        17 * 31 if len(shape) == 2 else 2 * 17 * 31, device="npu", dtype=torch.float32
    ).reshape(shape)
    other = torch.full_like(input, 2.0)
    output = torch.empty_like(input)

    assert jit(input, other, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input + other)

    aot = _compile_dtype(
        _application,
        tmp_path,
        arrangement=arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    output.zero_()
    assert reloaded(input, other, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input + other)


def test_ascend_fp32_matrix_row_broadcast_and_scalar_input_jit_aot_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    input = torch.arange(17 * 31, device="npu", dtype=torch.float32).reshape(17, 31)
    bias = torch.arange(31, device="npu", dtype=torch.float32).reshape(1, 31)
    output = torch.empty_like(input)
    jit = _compile_dtype(
        _broadcast_application,
        tmp_path,
        arrangement=_matrix_broadcast_arrangement,
        tensors=(
            Tensor(2, dtype="float32"),
            Tensor(2, shape=(1, 31), dtype="float32"),
            Tensor(2, dtype="float32"),
        ),
        mode="jit",
    )

    assert jit(input, bias, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input + bias)

    aot = _compile_dtype(
        _matrix_scalar_mul_application,
        tmp_path,
        arrangement=_matrix_scalar_arrangement,
        tensors=(
            Tensor(2, dtype="float32"),
            Tensor(0, dtype="float32"),
            Tensor(2, dtype="float32"),
        ),
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    alpha = 1.75
    output.zero_()
    assert reloaded(input, alpha, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, input * alpha)


@pytest.mark.parametrize(
    ("shape", "arrangement"),
    (
        ((17, 31), _multi_output_matrix_arrangement),
        ((2, 17, 31), _multi_output_volume_arrangement),
    ),
)
def test_ascend_fp32_multidimensional_multiple_outputs_jit_aot_reload(
    shape, arrangement, tmp_path
):
    import torch
    import torch_npu  # noqa: F401

    tensors = tuple(Tensor(len(shape), dtype="float32") for _ in range(4))
    elements = 17 * 31 if len(shape) == 2 else 2 * 17 * 31
    input = torch.arange(elements, device="npu", dtype=torch.float32).reshape(shape)
    other = torch.full_like(input, 2.0)
    out0 = torch.empty_like(input)
    out1 = torch.empty_like(input)
    jit = _compile_dtype(
        _multi_output_application,
        tmp_path,
        arrangement=arrangement,
        tensors=tensors,
        mode="jit",
    )

    assert jit(input, other, out0, out1) == (out0, out1)
    torch.npu.synchronize()
    torch.testing.assert_close(out0, input + other)
    torch.testing.assert_close(out1, input - other)

    aot = _compile_dtype(
        _multi_output_application,
        tmp_path,
        arrangement=arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    out0.zero_()
    out1.zero_()
    assert reloaded(input, other, out0, out1) == (out0, out1)
    torch.npu.synchronize()
    torch.testing.assert_close(out0, input + other)
    torch.testing.assert_close(out1, input - other)


def test_ascend_fp32_multiple_outputs_with_scalar_input_jit_aot_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    input = torch.arange(17 * 31, device="npu", dtype=torch.float32).reshape(17, 31)
    alpha = 1.75
    out0 = torch.empty_like(input)
    out1 = torch.empty_like(input)
    tensors = (
        Tensor(2, dtype="float32"),
        Tensor(0, dtype="float32"),
        Tensor(2, dtype="float32"),
        Tensor(2, dtype="float32"),
    )
    jit = _compile_dtype(
        _multi_output_scalar_application,
        tmp_path,
        arrangement=_multi_output_scalar_arrangement,
        tensors=tensors,
        mode="jit",
    )

    assert jit(input, alpha, out0, out1) == (out0, out1)
    torch.npu.synchronize()
    torch.testing.assert_close(out0, input * alpha)
    torch.testing.assert_close(out1, input + alpha)

    aot = _compile_dtype(
        _multi_output_scalar_application,
        tmp_path,
        arrangement=_multi_output_scalar_arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    out0.zero_()
    out1.zero_()
    assert reloaded(input, alpha, out0, out1) == (out0, out1)
    torch.npu.synchronize()
    torch.testing.assert_close(out0, input * alpha)
    torch.testing.assert_close(out1, input + alpha)


@pytest.mark.parametrize("dtype", ("float64", "int32"))
def test_ascend_unverified_dtype_fails_closed_before_materialization(dtype):
    with pytest.raises(ValueError, match="only FP16, BF16, and FP32 elementwise SSA"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=tuple(Tensor(1, dtype=dtype) for _ in range(3)),
                backend="ascend",
                backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
            )
        )


@pytest.mark.parametrize(
    ("ninetoothed_dtype", "torch_dtype", "rtol", "atol"),
    (
        (float16, "float16", 1e-3, 1e-3),
        (bfloat16, "bfloat16", 1e-2, 1e-2),
    ),
)
def test_ascend_low_precision_jit_aot_reload_and_scalar_output(
    ninetoothed_dtype, torch_dtype, rtol, atol, tmp_path
):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    tensor_dtype = getattr(torch, torch_dtype)
    tensors = tuple(Tensor(1, dtype=ninetoothed_dtype) for _ in range(3))
    jit = _compile_dtype(
        _application,
        tmp_path,
        arrangement=_arrangement,
        tensors=tensors,
        mode="jit",
    )

    for size in (0, 1, 255, 256, 257, 513):
        input = torch.linspace(-2.0, 2.0, size, device="npu", dtype=tensor_dtype)
        other = torch.linspace(1.0, 3.0, size, device="npu", dtype=tensor_dtype)
        output = torch.empty_like(input)

        assert jit(input, other, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input + other, rtol=rtol, atol=atol)

    aot = _compile_dtype(
        _application,
        tmp_path,
        arrangement=_arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)

    for size in (257, 513):
        input = torch.arange(size, device="npu", dtype=tensor_dtype)
        other = torch.full_like(input, 2)
        output = torch.empty_like(input)

        assert reloaded(input, other, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input + other, rtol=rtol, atol=atol)

    scalar = _compile_dtype(
        _scalar_extract_application,
        tmp_path,
        arrangement=_scalar_extract_arrangement,
        tensors=(
            Tensor(1, dtype=ninetoothed_dtype),
            Tensor(0, dtype=ninetoothed_dtype),
        ),
        mode="aot",
    )
    scalar_input = torch.tensor([3.25], device="npu", dtype=tensor_dtype)
    scalar_output = torch.empty((), device="npu", dtype=tensor_dtype)

    assert scalar(scalar_input, scalar_output) is scalar_output
    torch.npu.synchronize()
    torch.testing.assert_close(scalar_output, scalar_input[0], rtol=rtol, atol=atol)

    scalar_reloaded = load_built_artifact(scalar._built_artifact)
    scalar_output.zero_()
    assert scalar_reloaded(scalar_input, scalar_output) is scalar_output
    torch.npu.synchronize()
    torch.testing.assert_close(scalar_output, scalar_input[0], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "feature",
    (
        "runtime slice",
        "reshape or multidimensional view",
        "transpose",
        "negative stride",
        "as_strided",
        "in-place storage alias",
    ),
)
def test_ascend_unsupported_layouts_are_not_runtime_positive_cases(feature):
    pytest.skip(f"Ascend runtime positive coverage excludes unsupported {feature}.")


def test_ascend_jit_and_source_reload_execute_fp32_elementwise(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    x = torch.arange(513, device="npu", dtype=torch.float32)
    other = torch.full_like(x, 2.0)
    output = torch.empty_like(x)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=(
                Tensor(1, dtype="float32"),
                Tensor(1, dtype="float32"),
                Tensor(1, dtype="float32"),
            ),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )

    assert handle(x, other, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, x + other)

    reloaded = load_built_artifact(handle._built_artifact)
    output.zero_()
    assert reloaded(x, other, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, x + other)

    tail_input = torch.arange(257, device="npu", dtype=torch.float32)
    tail_other = torch.full_like(tail_input, 3.0)
    tail_output = torch.empty_like(tail_input)
    stream = torch.npu.Stream(device=tail_input.device)

    with torch.npu.stream(stream):
        assert reloaded(tail_input, tail_other, tail_output) is tail_output

    stream.synchronize()
    torch.testing.assert_close(tail_output, tail_input + tail_other)


def test_ascend_jit_and_source_reload_execute_fp32_singleton_broadcast(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    x = torch.arange(513, device="npu", dtype=torch.float32)
    bias = torch.tensor([2.0], device="npu", dtype=torch.float32)
    output = torch.empty_like(x)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_broadcast_arrangement,
            application=_broadcast_application,
            tensors=(
                Tensor(1, dtype="float32"),
                Tensor(1, shape=(1,), dtype="float32"),
                Tensor(1, dtype="float32"),
            ),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )

    assert handle(x, bias, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, x + bias)

    reloaded = load_built_artifact(handle._built_artifact)
    output.zero_()
    assert reloaded(x, bias, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, x + bias)


def test_ascend_tail_mask_preserves_nan_and_infinity_elementwise_semantics(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    x = torch.arange(257, dtype=torch.float32)
    other = torch.full_like(x, 2.0)
    x[0] = torch.nan
    x[1] = torch.inf
    x[2] = -torch.inf
    other[1] = -torch.inf
    other[2] = torch.inf
    other[3] = torch.nan
    x = x.to("npu")
    other = other.to("npu")
    output = torch.empty_like(x)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tuple(Tensor(1, dtype="float32") for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )

    assert handle(x, other, output) is output
    torch.npu.synchronize()
    expected = x + other
    torch.testing.assert_close(output, expected, equal_nan=True)

    reloaded = load_built_artifact(handle._built_artifact)
    output.zero_()
    assert reloaded(x, other, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output, expected, equal_nan=True)


def test_ascend_static_offset_logical_view_executes_and_rejects_alias(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    source = torch.arange(258, device="npu", dtype=torch.float32)
    output = torch.zeros_like(source)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_offset_arrangement,
            application=_offset_application,
            tensors=tuple(Tensor(1, dtype="float32") for _ in range(2)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    handle = DEFAULT_COMPILER.materialize(
        compilation,
        output_dir=tmp_path,
        mode="aot",
    )

    assert handle(source, output) is output
    torch.npu.synchronize()
    torch.testing.assert_close(output[1:], source[1:] + source[1:])
    assert output[0].item() == 0.0

    with pytest.raises(ValueError, match="storage overlap"):
        handle(source, source)
