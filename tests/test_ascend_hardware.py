"""Capability-gated integration tests for the first Ascend execution tier."""

import contextlib
import os
import signal
import time
from pathlib import Path

import pytest

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, bfloat16, float16, float32
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact
from ninetoothed.language import libdevice

pytestmark = pytest.mark.skipif(
    os.environ.get("NINETOOTHED_RUN_ASCEND_TESTS") != "1",
    reason="set NINETOOTHED_RUN_ASCEND_TESTS=1 on an Ascend runner",
)


@contextlib.contextmanager
def _ascend_conv2d_stage(name, *, timeout=None):
    """Print a bounded stage result so a stalled NPU phase is attributable."""
    limit = float(
        timeout
        if timeout is not None
        else os.environ.get("NINETOOTHED_ASCEND_STAGE_TIMEOUT", "180")
    )
    started = time.monotonic()
    print(f"ASCEND_CONV2D_STAGE START {name} timeout={limit:.1f}s", flush=True)
    previous = signal.getsignal(signal.SIGALRM)

    def alarm_handler(_signum, _frame):
        elapsed = time.monotonic() - started
        raise TimeoutError(
            f"Ascend conv2d stage {name!r} exceeded {limit:.1f}s "
            f"(elapsed={elapsed:.1f}s)."
        )

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    except Exception as exc:
        elapsed = time.monotonic() - started
        status = "TIMEOUT" if isinstance(exc, TimeoutError) else "FAIL"
        print(
            f"ASCEND_CONV2D_STAGE {status} {name} elapsed={elapsed:.1f}s "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        print(f"ASCEND_CONV2D_STAGE PASS {name} elapsed={elapsed:.1f}s", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _ascend_conv2d_stage_skip(name, reason):
    print(f"ASCEND_CONV2D_STAGE SKIP {name} reason={reason}", flush=True)


def _ascend_conv2d_target_compilation(kernel_name):
    from tests import test_conv2d

    sizes = {"n": 4, "c": 64, "h": 16, "w": 16, "k": 512, "r": 3, "s": 3}
    arrangement, application, tensors = test_conv2d.premake(
        **sizes, dtype="float16", block_size_m=64, block_size_n=64, block_size_k=64
    )
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="ascend",
            kernel_name=kernel_name,
            tensor_dtypes={
                "input": "float16",
                "filter": "float16",
                "output": "float16",
            },
            backend_options={"soc_version": "Ascend910B4", "max_core_dim": 65535},
        )
    )


def _ascend_conv2d_values(torch):
    shape = (4, 64, 16, 16)
    filter_shape = (512, 64, 3, 3)
    input_value = torch.randn(shape, device="npu", dtype=torch.float16)
    filter_value = torch.randn(filter_shape, device="npu", dtype=torch.float16)
    expected = torch.nn.functional.conv2d(input_value, filter_value)
    return input_value, filter_value, expected


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


def _softmax_arrangement(input, output):
    return input.tile((16, 127)), output.tile((16, 127))


def _softmax_application(input, output):
    maximum = ntl.max(input, axis=-1)
    numerator = (input - maximum[:, None]).exp()
    denominator = ntl.sum(numerator, axis=-1)
    output = numerator / denominator[:, None]  # noqa: F841


def _rmsnorm_arrangement(input, output):
    return input.tile((16, 127)), output.tile((16, 127))


def _rmsnorm_application(input, output):
    input_fp32 = input.to(float32)
    mean_square = ntl.sum(input_fp32 * input_fp32, axis=-1) / input.shape[-1]
    inverse_rms = (mean_square + 1e-5).rsqrt()
    output = input * inverse_rms[:, None]  # noqa: F841


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


def _reduction_keepdim_arrangement(input, output):
    return input.tile((127, 31)), output.tile((127, 1))


def _reduction_keepdim_sum(input, output):
    output = ntl.sum(input, axis=1, keepdim=True)  # noqa: F841


def _matmul_arrangement(lhs, rhs, output):
    return lhs.tile((17, 127)), rhs.tile((127, 31)), output.tile((17, 31))


def _matmul_small_arrangement(lhs, rhs, output):
    return lhs.tile((3, 127)), rhs.tile((127, 7)), output.tile((3, 7))


def _matmul_application(lhs, rhs, output):
    output = lhs @ rhs  # noqa: F841


def _matmul_epilogue_arrangement(lhs, rhs, bias, output):
    return (
        lhs.tile((3, 127)),
        rhs.tile((127, 5)),
        bias.tile((3, 5)),
        output.tile((3, 5)),
    )


def _matmul_bias_silu_residual_application(lhs, rhs, bias, output):
    matmul = lhs @ rhs
    activated = matmul + bias
    silu = activated * (1.0 / (1.0 + (-activated).exp()))
    output = silu + matmul  # noqa: F841


def _matmul_epilogue_vector_arrangement(lhs, rhs, scale, bias, residual, output):
    return tuple(
        tensor.tile((3, 5))
        if index >= 2
        else tensor.tile((3, 127) if index == 0 else (127, 5))
        for index, tensor in enumerate((lhs, rhs, scale, bias, residual, output))
    )


def _matmul_gelu_bias_residual_application(lhs, rhs, scale, bias, residual, output):
    matmul = lhs @ rhs
    cubic = matmul * matmul * matmul
    gelu = 0.5 * matmul * (1.0 + (0.79788456 * (matmul + 0.044715 * cubic)).tanh())
    output = gelu + bias + residual  # noqa: F841


def _matmul_leaky_relu_scale_bias_application(lhs, rhs, scale, bias, residual, output):
    matmul = lhs @ rhs
    activated = ntl.where(matmul > 0.0, matmul, 0.1 * matmul)
    output = activated * scale + bias  # noqa: F841


def _matmul_mul_add_application(lhs, rhs, scale, bias, residual, output):
    matmul = lhs @ rhs
    output = matmul * scale + bias  # noqa: F841


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


def _matrix_column_broadcast_arrangement(input, bias, output):
    return input.tile((17, 31)), bias.tile((17, 1)), output.tile((17, 31))


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


@pytest.mark.parametrize("dtype", ("float64", "int8"))
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


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_ascend_softmax_last_axis_jit_and_aot_reload(tmp_path, dtype):
    """Validate the one-block stable Softmax primitive against PyTorch."""
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_softmax_arrangement,
            application=_softmax_application,
            tensors=tuple(Tensor(2, dtype=dtype) for _ in range(2)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    torch_dtype = getattr(torch, dtype)
    rtol, atol = (5e-2, 5e-2) if dtype == "float16" else (1e-4, 1e-4)

    for launch in (jit, reloaded):
        input = torch.randn((16, 127), device="npu", dtype=torch_dtype)
        output = torch.empty_like(input)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(
            output, functional.softmax(input, dim=-1), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_ascend_rmsnorm_last_axis_jit_and_aot_reload(tmp_path, dtype):
    """Validate FP32-accumulated one-block RMSNorm core against PyTorch."""
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_rmsnorm_arrangement,
            application=_rmsnorm_application,
            tensors=tuple(Tensor(2, dtype=dtype) for _ in range(2)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    torch_dtype = getattr(torch, dtype)
    rtol, atol = (5e-2, 5e-2) if dtype == "float16" else (1e-4, 1e-4)

    for launch in (jit, reloaded):
        input = torch.randn((16, 127), device="npu", dtype=torch_dtype)
        output = torch.empty_like(input)
        expected = input * torch.rsqrt(
            input.float().square().mean(dim=-1, keepdim=True) + 1e-5
        ).to(input.dtype)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


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


def test_ascend_keepdim_reduction_jit_and_aot_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    shape = (127, 31)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_reduction_keepdim_arrangement,
            application=_reduction_keepdim_sum,
            tensors=(Tensor(2, dtype="float32"), Tensor(2, dtype="float32")),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    input = torch.randn(shape, device="npu", dtype=torch.float32)
    for launch in (jit, reloaded):
        output = torch.empty((127, 1), device="npu", dtype=torch.float32)
        assert launch(input, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input.sum(dim=1, keepdim=True))


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
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        ("float16", 5e-2, 5e-2),
        ("bfloat16", 1e-1, 1e-1),
        ("float32", 1e-2, 1e-2),
    ),
)
def test_ascend_matmul_jit_and_source_reload_tail(
    tmp_path, arrangement, lhs_shape, rhs_shape, output_shape, dtype, rtol, atol
):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=_matmul_application,
            tensors=tuple(Tensor(2, dtype=dtype) for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    torch_dtype = getattr(torch, dtype)

    for launch in (jit, reloaded):
        lhs = torch.randn(lhs_shape, device="npu", dtype=torch_dtype)
        rhs = torch.randn(rhs_shape, device="npu", dtype=torch_dtype)
        output = torch.empty(output_shape, device="npu", dtype=torch_dtype)
        assert launch(lhs, rhs, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.matmul(lhs, rhs), rtol=rtol, atol=atol)


def test_ascend_matmul_bias_silu_residual_epilogue_jit_and_aot_reload(tmp_path):
    import torch
    import torch_npu  # noqa: F401

    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_matmul_epilogue_arrangement,
            application=_matmul_bias_silu_residual_application,
            tensors=tuple(Tensor(2, dtype="float16") for _ in range(4)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    lhs = torch.randn((3, 127), device="npu", dtype=torch.float16)
    rhs = torch.randn((127, 5), device="npu", dtype=torch.float16)
    bias = torch.randn((3, 5), device="npu", dtype=torch.float16)
    matmul = torch.matmul(lhs, rhs)
    expected = (matmul + bias) * torch.sigmoid(matmul + bias) + matmul
    for launch in (jit, reloaded):
        output = torch.empty((3, 5), device="npu", dtype=torch.float16)
        assert launch(lhs, rhs, bias, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.ascend_next_stage
@pytest.mark.parametrize(
    "application",
    (
        _matmul_gelu_bias_residual_application,
        _matmul_leaky_relu_scale_bias_application,
        _matmul_mul_add_application,
    ),
)
def test_ascend_vector_epilogue_jit_and_aot_reload(tmp_path, application):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_matmul_epilogue_vector_arrangement,
            application=application,
            tensors=tuple(Tensor(2, dtype="float16") for _ in range(6)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    lhs = torch.randn((3, 127), device="npu", dtype=torch.float16)
    rhs = torch.randn((127, 5), device="npu", dtype=torch.float16)
    scale = torch.randn((3, 5), device="npu", dtype=torch.float16)
    bias = torch.randn((3, 5), device="npu", dtype=torch.float16)
    residual = torch.randn((3, 5), device="npu", dtype=torch.float16)
    matmul = torch.matmul(lhs, rhs)

    if application is _matmul_gelu_bias_residual_application:
        expected = (
            torch.nn.functional.gelu(matmul, approximate="tanh") + bias + residual
        )
    elif application is _matmul_leaky_relu_scale_bias_application:
        expected = (
            torch.nn.functional.leaky_relu(matmul, negative_slope=0.1) * scale + bias
        )
    else:
        expected = matmul * scale + bias

    for launch in (jit, reloaded):
        output = torch.empty((3, 5), device="npu", dtype=torch.float16)
        assert launch(lhs, rhs, scale, bias, residual, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.ascend_next_stage
@pytest.mark.parametrize(
    ("arrangement", "lhs_shape", "rhs_shape", "output_shape", "rank", "verified"),
    (
        (_matmul_large_k_arrangement, (3, 513), (513, 5), (3, 5), 2, True),
        (_matmul_batched_arrangement, (2, 3, 257), (2, 257, 5), (2, 3, 5), 3, True),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        ("float16", 5e-2, 5e-2),
        ("bfloat16", 1e-1, 1e-1),
        ("float32", 1e-2, 1e-2),
    ),
)
def test_ascend_tiled_batched_matmul_jit_and_source_reload(
    tmp_path,
    arrangement,
    lhs_shape,
    rhs_shape,
    output_shape,
    rank,
    verified,
    dtype,
    rtol,
    atol,
):
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    request = CompileRequest(
        arrangement=arrangement,
        application=_matmul_application,
        tensors=tuple(Tensor(rank, dtype=dtype) for _ in range(3)),
        backend="ascend",
        backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
    )
    if not verified:
        with pytest.raises(ValueError, match="batched matmul is fail-closed"):
            DEFAULT_COMPILER.compile(request)
        return

    compilation = DEFAULT_COMPILER.compile(request)
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    for launch in (jit, reloaded):
        torch_dtype = getattr(torch, dtype)
        lhs = torch.randn(lhs_shape, device="npu", dtype=torch_dtype)
        rhs = torch.randn(rhs_shape, device="npu", dtype=torch_dtype)
        output = torch.empty(output_shape, device="npu", dtype=torch_dtype)
        assert launch(lhs, rhs, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, torch.matmul(lhs, rhs), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("rank", "shapes"),
    (
        (2, ((17, 31, 127), (23, 7, 129))),
        (3, ((2, 17, 31, 127), (3, 5, 7, 129))),
    ),
)
def test_ascend_dynamic_matmul_same_artifact_jit_and_aot_reload(tmp_path, rank, shapes):
    """One dynamic-M/N/K artifact must execute multiple runtime shapes."""
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    symbols = tuple(
        Symbol(name) for name in ("dynamic_b", "dynamic_m", "dynamic_n", "dynamic_k")
    )
    if rank == 2:
        m, n, k = symbols[1:]
        tensor_shapes = ((m, k), (k, n), (m, n))
    else:
        b, m, n, k = symbols
        tensor_shapes = ((b, m, k), (b, k, n), (b, m, n))

    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=lambda lhs, rhs, output: (lhs, rhs, output),
            application=_matmul_application,
            tensors=tuple(
                Tensor(shape=shape, dtype="float32") for shape in tensor_shapes
            ),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    assert {"dynamic_m", "dynamic_n", "dynamic_k"}.issubset(
        set(compilation.launch_abi.shape_params)
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)

    for launch in (jit, reloaded):
        for shape in shapes:
            if rank == 2:
                m_value, n_value, k_value = shape
                lhs_shape, rhs_shape, output_shape = (
                    (m_value, k_value),
                    (k_value, n_value),
                    (m_value, n_value),
                )
            else:
                b_value, m_value, n_value, k_value = shape
                lhs_shape, rhs_shape, output_shape = (
                    (b_value, m_value, k_value),
                    (b_value, k_value, n_value),
                    (b_value, m_value, n_value),
                )
            lhs = torch.randn(lhs_shape, device="npu", dtype=torch.float32)
            rhs = torch.randn(rhs_shape, device="npu", dtype=torch.float32)
            output = torch.empty(output_shape, device="npu", dtype=torch.float32)
            assert launch(lhs, rhs, output) is output
            torch.npu.synchronize()
            torch.testing.assert_close(
                output, torch.matmul(lhs, rhs), rtol=1e-2, atol=1e-2
            )


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


def test_ascend_column_singleton_broadcast_jit_and_aot_reload(tmp_path):
    """Verify the `(M, 1)` broadcast coordinate contract on Ascend 910B3."""
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_matrix_column_broadcast_arrangement,
            application=_broadcast_application,
            tensors=tuple(Tensor(2, dtype="float32") for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    input = torch.randn((17, 31), device="npu", dtype=torch.float32)
    bias = torch.randn((17, 1), device="npu", dtype=torch.float32)

    for launch in (jit, reloaded):
        output = torch.empty_like(input)
        assert launch(input, bias, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, input + bias)


@pytest.mark.parametrize("dtype", ("float32", "int32"))
def test_ascend_positive_stride_and_integer_dtype_jit_aot(tmp_path, dtype):
    """Verify strided, non-overlapping input addressing and common int dtype."""
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tuple(Tensor(1, dtype=dtype) for _ in range(3)),
            backend="ascend",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="jit")
    aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path, mode="aot")
    reloaded = load_built_artifact(aot._built_artifact)
    torch_dtype = getattr(torch, dtype)
    for launch in (jit, reloaded):
        base = torch.arange(1026, device="npu", dtype=torch_dtype)
        lhs = base[::2]
        rhs = torch.full((513,), 3, device="npu", dtype=torch_dtype)
        output = torch.empty_like(rhs)
        assert launch(lhs, rhs, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, lhs + rhs)


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
    import torch.nn.functional as functional
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
    expected = functional.conv2d(input, filter)

    for launch in (jit, reloaded):
        output = torch.empty((1, 3, 2, 2), device="npu", dtype=torch.float16)
        assert launch(input, filter, output) is output
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.ascend_next_stage
def test_ascend_conv2d_target_scale_compile_only_diagnostics(tmp_path):
    """Report lowering, import, and CANN compile phases for target-size Conv2d."""
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    with _ascend_conv2d_stage("lowering"):
        compilation = _ascend_conv2d_target_compilation(
            "ascend_conv2d_target_scale_compile_only"
        )

    with _ascend_conv2d_stage("source generation"):
        source = compilation.artifact.primary_source
        assert source and "ascend_conv2d_target_scale_compile_only" in source
        assert compilation.artifact.metadata["generation_py_fallback"] is False

    from ninetoothed.backends.materializers.ascend import AscendMaterializer

    with _ascend_conv2d_stage("Triton import"):
        jit = AscendMaterializer().jit_materialize(
            compilation, output_dir=tmp_path / "jit"
        )
        raw_kernel = jit._kernel[1]

    input_value = torch.empty((4, 64, 16, 16), device="npu", dtype=torch.float16)
    filter_value = torch.empty((512, 64, 3, 3), device="npu", dtype=torch.float16)
    output_value = torch.empty((4, 512, 14, 14), device="npu", dtype=torch.float16)
    launch_grid = 13 * 8 * 16

    cann_blocked = False
    try:
        with _ascend_conv2d_stage("CANN compile"):
            compiled = raw_kernel.warmup(
                input_value,
                filter_value,
                output_value,
                *input_value.stride(),
                *filter_value.stride(),
                *output_value.stride(),
                grid=(launch_grid,),
                num_warps=4,
                num_stages=1,
            )
            assert {"ttir", "ttadapter", "npubin"}.issubset(compiled.asm)
    except Exception as exc:
        cann_blocked = True
        for name in ("kernel launch", "torch.npu.synchronize", "output comparison", "AOT reload"):
            _ascend_conv2d_stage_skip(name, "blocked by CANN compile")
        assert "ConvertLinalgRToBinary" in str(exc) or isinstance(exc, TimeoutError)
    if cann_blocked:
        return

    for name in ("kernel launch", "torch.npu.synchronize", "output comparison", "AOT reload"):
        _ascend_conv2d_stage_skip(name, "compile-only diagnostic")


@pytest.mark.parametrize("padding", ((0, 0), (1, 1), (0, 1), (2, 0)))
def test_ascend_conv2d_specialized_padding_compile_jit_aot_reload(tmp_path, padding):
    """Target-size regression for physical state and specialized source masks."""
    import functools

    import torch
    import torch_npu  # noqa: F401

    from ninetoothed.compiler.runtime import _bound_values
    from tests import test_conv2d

    out_h = 16 + 2 * padding[0] - 3 + 1
    out_w = 16 + 2 * padding[1] - 3 + 1
    with _ascend_conv2d_stage("lowering/source generation"):
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=functools.partial(
                    test_conv2d.arrangement, enable_padding=True,
                    BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                ),
                application=test_conv2d.matmul.application,
                tensors=(Tensor(shape=(4, 64, 16, 16), dtype="float16"),
                         Tensor(shape=(512, 64, 3, 3), dtype="float16"),
                         Tensor(shape=(4, 512, out_h, out_w), dtype="float16")),
                backend="ascend",
                specialization_values={
                    "ninetoothed_constexpr_prefix_padding_h": padding[0],
                    "ninetoothed_constexpr_prefix_padding_w": padding[1],
                },
            )
        )
    input_value = torch.rand((4, 64, 16, 16), device="npu", dtype=torch.float16)
    filter_value = torch.rand((512, 64, 3, 3), device="npu", dtype=torch.float16)
    output = torch.empty((4, 512, out_h, out_w), device="npu", dtype=torch.float16)
    with _ascend_conv2d_stage("Triton import"):
        jit = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path / "jit", mode="jit")
    public = {"lhs": input_value, "rhs": filter_value, "output": output}
    values, _keepalive = _bound_values(compilation.launch_abi, public, scalar_mode="value")
    grid = ((4 * out_h * out_w + 63) // 64) * 8 * 16
    with _ascend_conv2d_stage("CANN compile (warmup, no launch)"):
        compiled = jit._kernel[1].warmup(
            *values, grid=(grid,), multibuffer=False, num_stages=1,
        )
        assert "npubin" in compiled.asm
    with _ascend_conv2d_stage("AOT build"):
        aot = DEFAULT_COMPILER.materialize(compilation, output_dir=tmp_path / "aot", mode="aot")
    with _ascend_conv2d_stage("AOT reload"):
        reloaded = load_built_artifact(aot._built_artifact)
    expected = torch.nn.functional.conv2d(input_value, filter_value, padding=padding)
    for name, launch in (("JIT", jit), ("AOT reload", reloaded)):
        output.fill_(float("nan"))
        with _ascend_conv2d_stage(f"{name} kernel launch"):
            assert launch(input_value, filter_value, output) is output
        with _ascend_conv2d_stage(f"{name} torch.npu.synchronize"):
            torch.npu.synchronize()
        with _ascend_conv2d_stage(f"{name} output contract"):
            assert output.shape == expected.shape and output.dtype == torch.float16
        with _ascend_conv2d_stage(f"{name} numerical comparison"):
            torch.testing.assert_close(output, expected, rtol=0.001, atol=0.001)


@pytest.mark.ascend_next_stage
def test_ascend_conv2d_target_scale_jit_aot_stage_diagnostics(tmp_path):
    """Run target-size Conv2d and report every JIT/AOT runtime boundary."""
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    with _ascend_conv2d_stage("lowering"):
        compilation = _ascend_conv2d_target_compilation(
            "ascend_conv2d_target_scale_stage_diagnostics"
        )
    with _ascend_conv2d_stage("source generation"):
        source = compilation.artifact.primary_source
        assert source

    from ninetoothed.backends.materializers.ascend import AscendMaterializer
    from ninetoothed.compiler import load_built_artifact

    materializer = AscendMaterializer()
    with _ascend_conv2d_stage("Triton import"):
        jit = materializer.jit_materialize(compilation, output_dir=tmp_path / "jit")

    input_value, filter_value, expected = _ascend_conv2d_values(torch)

    def warmup(handle, label):
        raw_kernel = handle._kernel[1]
        schedule = compilation.artifact.metadata["ssa_metadata"]["schedule"]
        launch_grid = 13 * 8 * 16
        with _ascend_conv2d_stage(f"CANN compile ({label})"):
            compiled = raw_kernel.warmup(
                input_value,
                filter_value,
                torch.empty_like(expected),
                *input_value.stride(),
                *filter_value.stride(),
                *expected.stride(),
                grid=(launch_grid,),
                num_warps=4,
                num_stages=1,
            )
            assert {"ttir", "ttadapter", "npubin"}.issubset(compiled.asm)

    try:
        warmup(jit, "jit")
    except Exception as exc:
        for name in (
            "kernel launch (jit)",
            "torch.npu.synchronize (jit)",
            "output comparison (jit)",
            "AOT reload",
            "kernel launch (aot_reload)",
            "torch.npu.synchronize (aot_reload)",
            "output comparison (aot_reload)",
        ):
            _ascend_conv2d_stage_skip(name, "blocked by CANN compile (jit)")
        assert "ConvertLinalgRToBinary" in str(exc) or isinstance(exc, TimeoutError)
        return
    with _ascend_conv2d_stage("kernel launch (jit)"):
        jit_output = torch.empty_like(expected)
        assert jit(input_value, filter_value, jit_output) is jit_output
    with _ascend_conv2d_stage("torch.npu.synchronize (jit)"):
        torch.npu.synchronize()
    with _ascend_conv2d_stage("output comparison (jit)"):
        torch.testing.assert_close(jit_output, expected, rtol=1e-3, atol=1e-3)

    try:
        with _ascend_conv2d_stage("AOT reload"):
            aot = materializer.aot_build(compilation, output_dir=tmp_path / "aot")
            reloaded = load_built_artifact(aot._built_artifact)
    except Exception:
        for name in (
            "kernel launch (aot_reload)",
            "torch.npu.synchronize (aot_reload)",
            "output comparison (aot_reload)",
        ):
            _ascend_conv2d_stage_skip(name, "blocked by AOT reload")
        raise
    try:
        warmup(reloaded, "aot_reload")
    except Exception as exc:
        for name in (
            "kernel launch (aot_reload)",
            "torch.npu.synchronize (aot_reload)",
            "output comparison (aot_reload)",
        ):
            _ascend_conv2d_stage_skip(name, "blocked by CANN compile (aot_reload)")
        assert "ConvertLinalgRToBinary" in str(exc) or isinstance(exc, TimeoutError)
        return
    with _ascend_conv2d_stage("kernel launch (aot_reload)"):
        aot_output = torch.empty_like(expected)
        assert reloaded(input_value, filter_value, aot_output) is aot_output
    with _ascend_conv2d_stage("torch.npu.synchronize (aot_reload)"):
        torch.npu.synchronize()
    with _ascend_conv2d_stage("output comparison (aot_reload)"):
        torch.testing.assert_close(aot_output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.ascend_next_stage
def test_ascend_causal_attention_frontend_compile_only(tmp_path):
    """Compile the causal all-masked path through Triton and CANN without launch."""
    import importlib.util

    import torch
    import torch_npu  # noqa: F401

    from tests import test_attention

    assert torch.npu.is_available()
    shape = (2, 4, 1, 64)
    q, k, v, output = tuple(
        Tensor(shape=shape, dtype="float16") for _ in range(4)
    )
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=test_attention.arrangement,
            application=test_attention.application,
            tensors=(
                q,
                k,
                v,
                Tensor(ndim=0, constexpr=True, value=1),
                output,
            ),
            backend="ascend",
            kernel_name="ascend_attention_causal_compile_only",
            backend_options={"soc_version": "Ascend910B4", "max_core_dim": 65535},
        )
    )
    schedule = compilation.artifact.metadata["ssa_metadata"]["schedule"]
    plan = schedule["ascend_attention_plan"]
    resource = plan["resource_plan"]
    source = compilation.artifact.primary_source
    assert plan["mode"] == "causal"
    assert resource["selected_tile"] == {"m": 16, "n": 32, "k": 32}
    assert "axis=None) != 0" in source
    assert "128x64" not in source

    source_path = tmp_path / "ascend_attention_causal_compile_only.py"
    source_path.write_text(source)
    spec = importlib.util.spec_from_file_location(
        "ascend_attention_causal_compile_only", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.ascend_attention_causal_compile_only_kernel

    q_value = torch.empty(shape, device="npu", dtype=torch.float16)
    k_value = torch.empty_like(q_value)
    v_value = torch.empty_like(q_value)
    output_value = torch.empty_like(q_value)
    compiled = kernel.warmup(
        q_value,
        k_value,
        v_value,
        output_value,
        *q_value.stride(),
        *k_value.stride(),
        *v_value.stride(),
        True,
        *output_value.stride(),
        BLOCK=1,
        grid=(int(plan["grid"]),),
        num_warps=4,
        num_stages=1,
    )
    assert {"ttir", "ttadapter", "npubin"}.issubset(compiled.asm)


@pytest.mark.parametrize(
    ("dtype_name", "torch_dtype"),
    (
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
        ("float32", "float32"),
    ),
)
@pytest.mark.parametrize("sequence", (1, 1024))
@pytest.mark.parametrize("is_causal", (False, True))
def test_ascend_attention_fp16_bf16_fp32_jit_and_aot_matrix(
    tmp_path, dtype_name, torch_dtype, sequence, is_causal
):
    """Run the supported Attention matrix through JIT and AOT reload on NPU."""
    import json
    import traceback

    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401

    from ninetoothed.backends.ascend import (
        ASCEND_ATTENTION_DTYPE_REGISTRY,
        read_ascend_sidecar,
    )
    from ninetoothed.backends.materializers.ascend import AscendMaterializer
    from tests import test_attention

    assert torch.npu.is_available()
    shape = (2, 4, sequence, 64)
    q, k, v, output = tuple(
        Tensor(shape=shape, dtype=dtype_name) for _ in range(4)
    )
    kernel_name = (
        f"ascend_attention_{dtype_name}_s{sequence}_c{int(is_causal)}_jit_aot"
    )

    def fail_phase(phase, exc):
        chain = "\n".join(traceback.format_exception(exc))
        pytest.fail(f"{phase} failed for {kernel_name}:\n{chain}", pytrace=False)

    try:
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=test_attention.arrangement,
                application=test_attention.application,
                tensors=(q, k, v, Tensor(0, constexpr=True), output),
                backend="ascend",
                kernel_name=kernel_name,
                backend_options={
                    "soc_version": "Ascend910B4",
                    "max_core_dim": 65535,
                },
            )
        )
    except Exception as exc:
        fail_phase("runtime validation / Attention contract planning", exc)

    schedule = compilation.artifact.metadata["ssa_metadata"]["schedule"]
    plan = schedule["ascend_attention_plan"]
    resource_plan = plan["resource_plan"]
    selected_tile = resource_plan["selected_tile"]
    assert selected_tile == {"m": 16, "n": 32, "k": 32}
    assert int(resource_plan["ub_estimated_peak_bytes"]) <= int(
        resource_plan["ub_budget_bytes"]
    )

    materializer = AscendMaterializer()
    try:
        jit = materializer.jit_materialize(
            compilation, output_dir=tmp_path / "jit"
        )
        aot = materializer.aot_build(
            compilation, output_dir=tmp_path / "aot"
        )
        sidecar = read_ascend_sidecar(Path(aot._built_artifact.source_path))
        dtype_contract = sidecar["attention_dtype_registry"][dtype_name]
        expected_tolerance = ASCEND_ATTENTION_DTYPE_REGISTRY[dtype_name]["error"]
        assert json.dumps(sidecar["attention_plan"], sort_keys=True) == json.dumps(
            plan, sort_keys=True, default=dict
        )
        assert json.dumps(sidecar["attention_retile"], sort_keys=True) == json.dumps(
            schedule["ascend_attention_retile"], sort_keys=True, default=dict
        )
        assert dtype_contract["input"] == {
            "q": dtype_name,
            "k": dtype_name,
            "v": dtype_name,
            "o": dtype_name,
        }
        assert set(dtype_contract["internal"].values()) == {"float32"}
        assert dtype_contract["error"] == expected_tolerance
        assert dtype_contract["status"] == "verified-jit-aot-npu"
        assert dtype_contract["runtime"] == {
            "device": "Ascend910B4",
            "cann": "9.0.0",
            "torch_npu": "2.7.1",
            "triton": "3.2.0",
            "triton_ascend": "3.2.2",
        }
        reloaded = load_built_artifact(aot._built_artifact)
    except Exception as exc:
        fail_phase("AOT reload / dtype-resource sidecar contract", exc)

    torch.manual_seed(20261003 + sequence + int(is_causal))
    torch_dtype_value = getattr(torch, torch_dtype)
    q_value, k_value, v_value = (
        torch.randn(shape, device="npu", dtype=torch_dtype_value)
        for _ in range(3)
    )
    expected = functional.scaled_dot_product_attention(
        q_value, k_value, v_value, is_causal=is_causal, scale=1
    )
    assert expected.device.type == "npu"

    run_records = {}
    for label, materialized, launch in (
        ("jit", jit, jit),
        ("aot_reload", aot, reloaded),
    ):
        raw_kernel = materialized._kernel[1]
        try:
            compiled = raw_kernel.warmup(
                q_value,
                k_value,
                v_value,
                torch.empty_like(expected),
                *q_value.stride(),
                *k_value.stride(),
                *v_value.stride(),
                is_causal,
                *expected.stride(),
                BLOCK=1,
                grid=(int(plan["grid"]),),
                num_warps=4,
                num_stages=1,
            )
            missing = {"ttir", "ttadapter", "npubin"} - set(compiled.asm)
            assert not missing, f"compiled artifact is missing {sorted(missing)}"
        except Exception as exc:
            fail_phase(f"CANN compile ({label})", exc)

        output_value = torch.empty_like(expected)
        try:
            returned = launch(
                q_value, k_value, v_value, is_causal, output_value
            )
        except Exception as exc:
            message = str(exc)
            phase = (
                f"launch ({label})"
                if message.startswith("Ascend kernel launch failed")
                else f"runtime validation ({label})"
            )
            fail_phase(phase, exc)
        assert returned is output_value, f"{label} returned a different output buffer"
        try:
            torch.npu.synchronize()
        except Exception as exc:
            fail_phase(f"synchronize ({label})", exc)

        if output_value.dtype != torch_dtype_value or tuple(output_value.shape) != shape:
            pytest.fail(
                "output contract failed for "
                f"{kernel_name}/{label}: dtype={output_value.dtype}, "
                f"shape={tuple(output_value.shape)}, expected dtype={torch_dtype_value}, "
                f"shape={shape}",
                pytrace=False,
            )

        actual_fp32 = output_value.float()
        expected_fp32 = expected.float()
        absolute_error = (actual_fp32 - expected_fp32).abs()
        max_abs_error = float(absolute_error.max().item())
        max_relative_error = float(
            (absolute_error / expected_fp32.abs().clamp_min(1e-8)).max().item()
        )
        tolerance = expected_tolerance
        print(
            "ASCEND_ATTENTION_NUMERICAL_SAMPLE "
            + json.dumps(
                {
                    "case": kernel_name,
                    "path": label,
                    "q": q_value[0, 0, 0, :8].float().cpu().tolist(),
                    "k": k_value[0, 0, 0, :8].float().cpu().tolist(),
                    "v": v_value[0, 0, 0, :8].float().cpu().tolist(),
                    "actual": actual_fp32[0, 0, 0, :8].cpu().tolist(),
                    "expected": expected_fp32[0, 0, 0, :8].cpu().tolist(),
                },
                sort_keys=True,
            )
        )
        try:
            torch.testing.assert_close(
                output_value,
                expected,
                rtol=float(tolerance["rtol"]),
                atol=float(tolerance["atol"]),
            )
        except AssertionError as exc:
            pytest.fail(
                "numerical comparison failed for "
                f"{kernel_name}/{label}: max_abs_error={max_abs_error}, "
                f"max_relative_error={max_relative_error}, "
                f"rtol={tolerance['rtol']}, atol={tolerance['atol']}\n{exc}",
                pytrace=False,
            )
        run_records[label] = {
            "launch": "passed",
            "synchronize": "passed",
            "output_dtype": str(output_value.dtype),
            "output_shape": list(output_value.shape),
            "max_abs_error": max_abs_error,
            "max_relative_error": max_relative_error,
            "rtol": tolerance["rtol"],
            "atol": tolerance["atol"],
            "cann_artifacts": sorted(compiled.asm),
        }

    report = {
        "case": kernel_name,
        "device": torch.npu.get_device_name(q_value.device),
        "selected_mnk": [selected_tile["m"], selected_tile["n"], selected_tile["k"]],
        "ub_estimated_bytes": resource_plan["ub_estimated_peak_bytes"],
        "ub_budget_bytes": resource_plan["ub_budget_bytes"],
        "workspace_bytes": plan["workspace_bytes"],
        "jit": run_records["jit"],
        "aot_reload": run_records["aot_reload"],
    }
    print("ASCEND_ATTENTION_MATRIX_RESULT " + json.dumps(report, sort_keys=True))


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
    assert attention["status"] == "verified-static-public-online-softmax"
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


@pytest.mark.parametrize("is_causal", (False, True))
def test_ascend_online_attention_multi_batch_head_jit_and_aot_reload(
    tmp_path, is_causal
):
    """Validate the generic online-softmax loop across batch/head and causal modes."""
    import functools

    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401

    from tests import test_attention

    assert torch.npu.is_available()
    shape = (2, 2, 32, 16)
    q, k, v, output = tuple(Tensor(shape=shape, dtype="float32") for _ in range(4))
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=functools.partial(
                test_attention.arrangement, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16
            ),
            application=test_attention.application,
            tensors=(q, k, v, Tensor(0, constexpr=True), output),
            backend="ascend",
            kernel_name=(
                "ascend_attention_b2_h2_s32_causal"
                if is_causal
                else "ascend_attention_b2_h2_s32_noncausal"
            ),
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
    from ninetoothed.backends.materializers.ascend import AscendMaterializer

    materializer = AscendMaterializer()
    jit = materializer.jit_materialize(compilation, output_dir=tmp_path)
    aot = materializer.aot_build(compilation, output_dir=tmp_path)
    reloaded = load_built_artifact(aot._built_artifact)
    q_value, k_value, v_value = (
        torch.randn(shape, device="npu", dtype=torch.float32) for _ in range(3)
    )
    expected = functional.scaled_dot_product_attention(
        q_value, k_value, v_value, is_causal=is_causal, scale=1
    )

    for launch in (jit, reloaded):
        output_value = torch.empty_like(expected)
        assert (
            launch(q_value, k_value, v_value, is_causal, output_value) is output_value
        )
        torch.npu.synchronize()
        torch.testing.assert_close(output_value, expected, rtol=2.5e-2, atol=2.5e-2)


def test_ascend_aot_composite_matmul_rmsnorm_reload(tmp_path):
    """Run a reloaded Matmul artifact followed by a reloaded RMSNorm artifact."""
    import torch
    import torch_npu  # noqa: F401

    from ninetoothed.backends.ascend import ascend_capability_matrix
    from ninetoothed.backends.materializers.ascend import AscendMaterializer

    assert torch.npu.is_available()
    materializer = AscendMaterializer()
    matmul = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=lambda lhs, rhs, output: (
                lhs.tile((16, 31)),
                rhs.tile((31, 127)),
                output.tile((16, 127)),
            ),
            application=_matmul_application,
            tensors=tuple(Tensor(2, dtype="float32") for _ in range(3)),
            backend="ascend",
            kernel_name="ascend_e2e_matmul",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    rmsnorm = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_rmsnorm_arrangement,
            application=_rmsnorm_application,
            tensors=tuple(Tensor(2, dtype="float32") for _ in range(2)),
            backend="ascend",
            kernel_name="ascend_e2e_rmsnorm",
            backend_options={"soc_version": "Ascend910B3", "max_core_dim": 8},
        )
    )
    matmul_aot = materializer.aot_build(matmul, output_dir=tmp_path)
    rmsnorm_aot = materializer.aot_build(rmsnorm, output_dir=tmp_path)
    assert matmul_aot._built_artifact.binary_path is None
    assert rmsnorm_aot._built_artifact.binary_path is None
    for built in (matmul_aot._built_artifact, rmsnorm_aot._built_artifact):
        source = Path(built.source_path)
        assert source.is_file()
        assert source.with_suffix(".ascend-launch.json").is_file()
        assert Path(built.manifest_path).is_file()

    matmul_reload = load_built_artifact(matmul_aot._built_artifact)
    rmsnorm_reload = load_built_artifact(rmsnorm_aot._built_artifact)
    lhs = torch.randn((16, 31), device="npu", dtype=torch.float32)
    rhs = torch.randn((31, 127), device="npu", dtype=torch.float32)
    projected = torch.empty((16, 127), device="npu", dtype=torch.float32)
    output = torch.empty_like(projected)
    assert matmul_reload(lhs, rhs, projected) is projected
    assert rmsnorm_reload(projected, output) is output
    torch.npu.synchronize()
    expected_matmul = torch.matmul(lhs, rhs)
    expected = expected_matmul * torch.rsqrt(
        expected_matmul.square().mean(dim=-1, keepdim=True) + 1e-5
    )
    torch.testing.assert_close(projected, expected_matmul, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
    assert ascend_capability_matrix()["aot"]["sidecar_schema"] == 4
