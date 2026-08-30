"""Capability-gated integration tests for the first Ascend execution tier."""

import os

import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, bfloat16, float16, float32
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact
from ninetoothed.language import libdevice

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


def _pow_application(input, exponent, output):
    output = libdevice.pow(input, exponent)  # noqa: F841


def _fill_application(output):
    output = ntl.full(output.shape, 2.5, dtype=output.dtype)  # noqa: F841


def _copy_application(input, output):
    output = input  # noqa: F841


def _math_application(input, output):
    output = input.exp() + input.log()  # noqa: F841


def _mixed_scalar_application(input, alpha, output):
    output = input + alpha  # noqa: F841


def _nested_control_application(input, output):
    for i in range(input.shape[0]):
        if input[i] > 0.0:
            output[i] = input[i].exp()
        else:
            output[i] = (input[i] + 2.0).log()


def _transpose_arrangement(input, output):
    return input.tile((17, 31)), output.tile((31, 17))


def _transpose_application(input, output):
    output = input.T  # noqa: F841


def _rng_arrangement(output, seed, offset):
    return output.tile((257,)), seed, offset.tile((257,))


def _rng_application(output, seed, offset):
    output = ntl.rand(seed, offset)  # noqa: F841


def _atomic_arrangement(input, output):
    return input.tile((257,)), output.tile((1,))


def _atomic_application(input, output):
    ntl.atomic_add(output.source.data_ptr(), input)


def _fill_arrangement(output):
    return output.tile((513,))


def _copy_arrangement(input, output):
    return input.tile((513,)), output.tile((513,))


def _loop_application(input, output):
    for i in range(input.shape[0]):
        output[i] = input[i] + 1.0


def _reduction_matrix_arrangement(input, output):
    return input.tile((1, 127)), output.tile((1,))


def _reduction_volume_arrangement(input, output):
    return input.tile((2, 3, 127)), output.tile((2, 3))


def _reduction_application(input, output):
    output = ntl.sum(input, axis=-1)  # noqa: F841


def _large_reduction_arrangement(input, output):
    return input.tile((1, 513)), output.tile((1,))


def _large_reduction_application(input, output):
    output = ntl.sum(input, axis=-1)  # noqa: F841


def _reduction_scalar_arrangement(input, output):
    return input.tile((127,)), output


def _reduction_empty_scalar_arrangement(input, output):
    return input.tile((0,)), output


def _reduction_axis_zero_arrangement(input, output):
    return input.tile((127, 31)), output.tile((31,))


def _reduction_middle_axis_arrangement(input, output):
    return input.tile((2, 127, 31)), output.tile((2, 31))


def _reduction_scalar_sum(input, output):
    output = ntl.sum(input)  # noqa: F841


def _reduction_axis_zero_max(input, output):
    output = ntl.max(input, axis=0)  # noqa: F841


def _reduction_middle_axis_min(input, output):
    output = ntl.min(input, axis=1)  # noqa: F841


def _matmul_arrangement(lhs, rhs, output):
    return lhs.tile((17, 127)), rhs.tile((127, 31)), output.tile((17, 31))


def _matmul_small_arrangement(lhs, rhs, output):
    return lhs.tile((3, 127)), rhs.tile((127, 7)), output.tile((3, 7))


def _matmul_application(lhs, rhs, output):
    output = lhs @ rhs  # noqa: F841


def _matmul_large_k_arrangement(lhs, rhs, output):
    return lhs.tile((3, 513)), rhs.tile((513, 5)), output.tile((3, 5))


def _matmul_batched_arrangement(lhs, rhs, output):
    return lhs.tile((2, 3, 257)), rhs.tile((2, 257, 5)), output.tile((2, 3, 5))


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


def test_ascend_pow_jit_and_source_reload_tail(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_pow_application,
            tensors=tuple(Tensor(1, dtype="float32") for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        x = torch.linspace(0.25, 2.0, 257, device="npu")
        exponent = torch.full_like(x, 2.0)
        output = torch.empty_like(x)
        assert launch(x, exponent, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.pow(x, exponent))


def test_ascend_fill_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    tensors = (Tensor(1, dtype="float32"),)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_fill_arrangement,
            application=_fill_application,
            tensors=tensors,
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        output = torch.empty(257, device="npu", dtype=torch.float32)
        assert launch(output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 2.5))


def test_ascend_contiguous_copy_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    tensors = (Tensor(1, dtype="float32"), Tensor(1, dtype="float32"))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_copy_arrangement,
            application=_copy_application,
            tensors=tensors,
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        input = torch.arange(257, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input)


@pytest.mark.parametrize("feature", ("rank2 slice view", "rank3 slice view"))
def test_ascend_multidimensional_slice_views_are_capability_skipped(feature):
    pytest.skip(
        f"Ascend static-view contract rejects non-zero multidimensional offsets: {feature}"
    )


def test_ascend_control_flow_loop_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    tensors = (Tensor(1, dtype="float32"), Tensor(1, dtype="float32"))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_copy_arrangement,
            application=_loop_application,
            tensors=tensors,
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        input = torch.arange(257, device="npu", dtype=torch.float32)
        output = torch.empty_like(input)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input + 1.0)


def test_ascend_row_reduction_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    rank, arrangement, shape, output_shape = (
        2,
        _reduction_matrix_arrangement,
        (1, 127),
        (1,),
    )
    tensors = (Tensor(rank, dtype="float32"), Tensor(rank - 1, dtype="float32"))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_reduction_application,
            tensors=tensors,
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        input = torch.randn(shape, device="npu", dtype=torch.float32)
        output = torch.empty(output_shape, device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input.sum(dim=-1))


@pytest.mark.ascend_next_stage
def test_ascend_hierarchical_reduction_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_large_reduction_arrangement,
            application=_large_reduction_application,
            tensors=(Tensor(2, dtype="float32"), Tensor(1, dtype="float32")),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    for launch in (jit, reloaded):
        input = torch.randn((1, 513), device="npu", dtype=torch.float32)
        output = torch.empty((1,), device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input.sum(dim=-1), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("rank", "arrangement", "application", "shape", "output_shape", "expected"),
    (
        (
            1,
            _reduction_scalar_arrangement,
            _reduction_scalar_sum,
            (127,),
            (),
            lambda value: value.sum(),
        ),
        (
            2,
            _reduction_axis_zero_arrangement,
            _reduction_axis_zero_max,
            (127, 31),
            (31,),
            lambda value: value.max(dim=0).values,
        ),
        (
            3,
            _reduction_middle_axis_arrangement,
            _reduction_middle_axis_min,
            (2, 127, 31),
            (2, 31),
            lambda value: value.min(dim=1).values,
        ),
    ),
)
def test_ascend_ranked_reductions_jit_and_source_reload(
    tmp_path, rank, arrangement, application, shape, output_shape, expected
):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    tensors = (Tensor(rank, dtype="float32"), Tensor(rank - 1, dtype="float32"))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        input = torch.randn(shape, device="npu", dtype=torch.float32)
        output = torch.empty(output_shape, device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected(input))


@pytest.mark.skip(
    reason=(
        "Ascend empty scalar reduction requires an emittable row-vector domain; "
        "the shared reduction analysis currently selects scalar-fallback"
    )
)
def test_ascend_empty_scalar_reduction_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_reduction_empty_scalar_arrangement,
            application=_reduction_scalar_sum,
            tensors=(Tensor(1, dtype="float32"), Tensor(0, dtype="float32")),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        input = torch.empty((0,), device="npu", dtype=torch.float32)
        output = torch.empty((), device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.tensor(0.0, device="npu"))


@pytest.mark.parametrize(
    ("arrangement", "lhs_shape", "rhs_shape", "output_shape"),
    (
        (_matmul_arrangement, (17, 127), (127, 31), (17, 31)),
        (_matmul_small_arrangement, (3, 127), (127, 7), (3, 7)),
    ),
)
def test_ascend_matmul_jit_and_source_reload_tail(
    tmp_path, arrangement, lhs_shape, rhs_shape, output_shape
):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_matmul_arrangement,
            application=_matmul_application,
            tensors=tuple(Tensor(2, dtype="float32") for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        lhs = torch.randn(lhs_shape, device="npu", dtype=torch.float32)
        rhs = torch.randn(rhs_shape, device="npu", dtype=torch.float32)
        output = torch.empty(output_shape, device="npu", dtype=torch.float32)
        assert launch(lhs, rhs, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.matmul(lhs, rhs))


@pytest.mark.ascend_next_stage
@pytest.mark.parametrize(
    ("arrangement", "lhs_shape", "rhs_shape", "output_shape", "rank"),
    (
        (_matmul_large_k_arrangement, (3, 513), (513, 5), (3, 5), 2),
        (_matmul_batched_arrangement, (2, 3, 257), (2, 257, 5), (2, 3, 5), 3),
    ),
)
def test_ascend_tiled_batched_matmul_jit_and_source_reload(
    tmp_path, arrangement, lhs_shape, rhs_shape, output_shape, rank
):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_matmul_application,
            tensors=tuple(Tensor(rank, dtype="float32") for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    for launch in (jit, reloaded):
        lhs = torch.randn(lhs_shape, device="npu", dtype=torch.float32)
        rhs = torch.randn(rhs_shape, device="npu", dtype=torch.float32)
        output = torch.empty(output_shape, device="npu", dtype=torch.float32)
        assert launch(lhs, rhs, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.matmul(lhs, rhs), rtol=1e-2, atol=1e-2)


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


def test_ascend_math_jit_and_source_reload_tail(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    tensors = (Tensor(1, dtype="float32"), Tensor(1, dtype="float32"))
    jit = _compile_dtype(
        _math_application,
        tmp_path,
        arrangement=_unary_arrangement,
        tensors=tensors,
        mode="jit",
    )
    aot = _compile_dtype(
        _math_application,
        tmp_path,
        arrangement=_unary_arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    input = torch.linspace(0.25, 2.0, 513, device="npu", dtype=torch.float32)

    for launch in (jit, reloaded):
        output = torch.empty_like(input)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(
            output, input.exp() + input.log(), rtol=1e-4, atol=1e-4
        )


def test_ascend_mixed_scalar_vector_promotion_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    tensors = (
        Tensor(1, dtype="float16"),
        Tensor(0, dtype="float32"),
        Tensor(1, dtype="float32"),
    )
    jit = _compile_dtype(
        _mixed_scalar_application,
        tmp_path,
        arrangement=_scalar_input_arrangement,
        tensors=tensors,
        mode="jit",
    )
    aot = _compile_dtype(
        _mixed_scalar_application,
        tmp_path,
        arrangement=_scalar_input_arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    input = torch.linspace(-2.0, 2.0, 513, device="npu", dtype=torch.float16)

    for launch in (jit, reloaded):
        output = torch.empty(513, device="npu", dtype=torch.float32)
        assert launch(input, 1.25, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input.float() + 1.25, rtol=1e-3, atol=1e-3)


def test_ascend_transpose_jit_and_source_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    tensors = (Tensor(2, dtype="float32"), Tensor(2, dtype="float32"))
    jit = _compile_dtype(
        _transpose_application,
        tmp_path,
        arrangement=_transpose_arrangement,
        tensors=tensors,
        mode="jit",
    )
    aot = _compile_dtype(
        _transpose_application,
        tmp_path,
        arrangement=_transpose_arrangement,
        tensors=tensors,
        mode="aot",
    )
    reloaded = load_built_artifact(aot._built_artifact)
    input = torch.arange(17 * 31, device="npu", dtype=torch.float32).reshape(17, 31)

    for launch in (jit, reloaded):
        output = torch.empty((31, 17), device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input.T)


@pytest.mark.ascend_next_stage
def test_ascend_rng_jit_and_aot_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_rng_arrangement,
            application=_rng_application,
            tensors=(
                Tensor(1, dtype="float32"),
                Tensor(0, dtype="int32"),
                Tensor(1, dtype="int32"),
            ),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    offset = torch.arange(257, device="npu", dtype=torch.int32)

    for launch in (jit, reloaded):
        output = torch.empty(257, device="npu", dtype=torch.float32)
        assert launch(output, 17, offset) is output
        torch.npu.synchronize()
        assert torch.isfinite(output).all()
        assert bool(((output >= 0.0) & (output <= 1.0)).all())


@pytest.mark.ascend_next_stage
def test_ascend_fp32_atomic_add_jit_and_aot_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_atomic_arrangement,
            application=_atomic_application,
            tensors=(Tensor(1, dtype="float32"), Tensor(1, dtype="float32")),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    input = torch.linspace(-2.0, 2.0, 257, device="npu", dtype=torch.float32)

    for launch in (jit, reloaded):
        output = torch.zeros(1, device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input.sum().reshape(1), rtol=1e-4, atol=1e-4)


@pytest.mark.ascend_next_stage
def test_ascend_generic_dot_loop_jit_and_aot_reload(tmp_path):
    import functools

    import torch
    import torch_npu  # noqa: F401

    from tests import test_conv2d

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=functools.partial(
                test_conv2d.arrangement,
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=16,
                BLOCK_SIZE_K=16,
            ),
            application=test_conv2d.matmul.application,
            tensors=(
                Tensor(shape=(1, 2, 4, 4), dtype="float16"),
                Tensor(shape=(3, 2, 3, 3), dtype="float16"),
                Tensor(shape=(1, 3, 2, 2), dtype="float16"),
            ),
            backend="ascend",
            tensor_dtypes={
                "input": "float16",
                "filter": "float16",
                "output": "float16",
            },
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 65535},
        )
    )
    assert (
        compilation.artifact.metadata["ssa_metadata"]["schedule"]["ascend_dot_loop"][
            "mode"
        ]
        == "generic-dot-loop"
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    input = torch.randn((1, 2, 4, 4), device="npu", dtype=torch.float16)
    filter = torch.randn((3, 2, 3, 3), device="npu", dtype=torch.float16)

    for launch in (jit, reloaded):
        output = torch.empty((1, 3, 2, 2), device="npu", dtype=torch.float16)
        with pytest.raises(ValueError, match="generic dot-loop runtime is fail-closed"):
            launch(input, filter, output)


@pytest.mark.ascend_next_stage
def test_ascend_online_softmax_attention_jit_and_aot_reload(tmp_path):
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401

    from ninetoothed.backends.materializers.ascend import AscendMaterializer
    from tests import test_attention

    assert torch.npu.is_available()
    q, k, v, output = tuple(
        Tensor(shape=(1, 1, 16, 16), dtype="float32") for _ in range(4)
    )
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=test_attention.arrangement,
            application=test_attention.application,
            tensors=(q, k, v, Tensor(0, constexpr=True), output),
            backend="ascend",
            tensor_dtypes={
                "q": "float32",
                "k": "float32",
                "v": "float32",
                "o": "float32",
            },
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 65535},
        )
    )
    attention = compilation.artifact.metadata["ssa_metadata"]["schedule"][
        "ascend_attention_loop"
    ]
    assert attention["mode"] == "generic-online-softmax-loop"
    assert attention["status"] == "source-generated-cann-not-executed"
    materializer = AscendMaterializer()
    jit = materializer.jit_materialize(compilation, output_dir=tmp_path)
    aot = materializer.aot_build(compilation, output_dir=tmp_path)
    reloaded = load_built_artifact(aot._built_artifact)
    q_value, k_value, v_value = (
        torch.randn((1, 1, 16, 16), device="npu", dtype=torch.float32) for _ in range(3)
    )
    expected = functional.scaled_dot_product_attention(
        q_value, k_value, v_value, is_causal=False, scale=1
    )

    for launch in (jit, reloaded):
        output_value = torch.empty_like(expected)
        assert launch(q_value, k_value, v_value, False, output_value) is output_value
        torch.npu.synchronize()
        torch.testing.assert_close(output_value, expected, rtol=2.5e-2, atol=2.5e-2)
