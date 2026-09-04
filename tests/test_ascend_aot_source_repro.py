"""Static Ascend sidecar/source reproduction tests.

These tests intentionally stop after source emission and private sidecar reload.
They do not invoke the Triton-Ascend compiler or an NPU runtime.
"""

import ast
import functools

from ninetoothed import Tensor
from ninetoothed.backends.ascend import read_ascend_sidecar, write_ascend_sidecar
from ninetoothed.backends.core import Target
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from tests import test_attention, test_conv2d


def _write_and_reload_sidecar(tmp_path, compilation):
    source_path = tmp_path / compilation.artifact.primary_source_name
    source_path.write_text(compilation.artifact.primary_source, encoding="utf-8")
    write_ascend_sidecar(
        source_path,
        abi=compilation.launch_abi,
        specs=compilation.kernel.tensors,
        outputs=compilation.artifact.metadata["outputs"],
        metadata=compilation.artifact.metadata,
    )

    return read_ascend_sidecar(source_path)


def _dot_loop_request():
    return CompileRequest(
        arrangement=test_conv2d.arrangement,
        application=test_conv2d.matmul.application,
        tensors=(
            Tensor(shape=(1, 2, 4, 4), dtype="float32"),
            Tensor(shape=(3, 2, 3, 3), dtype="float32"),
            Tensor(shape=(1, 3, 2, 2), dtype="float32"),
        ),
        backend=Target.ASCEND,
        kernel_name="ascend_dot_loop_source_repro",
        tensor_dtypes={
            "input": "float32",
            "filter": "float32",
            "output": "float32",
        },
    )


def _padded_dot_loop_request():
    return CompileRequest(
        arrangement=functools.partial(
            test_conv2d.arrangement,
            enable_padding=True,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_K=16,
        ),
        application=test_conv2d.matmul.application,
        tensors=(
            Tensor(shape=(1, 2, 4, 4), dtype="float16"),
            Tensor(shape=(3, 2, 3, 3), dtype="float16"),
            Tensor(shape=(1, 3, 4, 4), dtype="float16"),
        ),
        backend=Target.ASCEND,
        kernel_name="ascend_padded_dot_loop_source_repro",
        tensor_dtypes={
            "input": "float16",
            "filter": "float16",
            "output": "float16",
        },
    )


def _attention_request():
    q, k, v, o = tuple(
        Tensor(
            shape=(1, 1, 16, 16),
            dtype="float32",
        )
        for _ in range(4)
    )
    return CompileRequest(
        arrangement=test_attention.arrangement,
        application=test_attention.application,
        tensors=(q, k, v, Tensor(0, constexpr=True), o),
        backend=Target.ASCEND,
        kernel_name="ascend_attention_source_repro",
        tensor_dtypes={"q": "float32", "k": "float32", "v": "float32", "o": "float32"},
    )


def test_ascend_dot_loop_sidecar_reload_reproduces_source(tmp_path):
    request = _dot_loop_request()
    first = DEFAULT_COMPILER.compile(request)
    first_schedule = first.artifact.metadata["ssa_metadata"]["schedule"]
    sidecar = _write_and_reload_sidecar(tmp_path, first)
    second = DEFAULT_COMPILER.compile(request)

    assert sidecar["dot_loop"] == first_schedule["ascend_dot_loop"]
    assert (
        second.artifact.metadata["ssa_metadata"]["schedule"]["ascend_dot_loop"]
        == sidecar["dot_loop"]
    )
    assert second.artifact.primary_source == first.artifact.primary_source


def test_ascend_padded_conv_access_template_clamps_invalid_addresses():
    source = DEFAULT_COMPILER.compile(
        _padded_dot_loop_request()
    ).artifact.primary_source

    # ``padding_*`` remains part of the public coordinate expression.  Ascend
    # additionally derives the physical pointer from that exact load predicate,
    # so invalid padded lanes cannot form negative addresses before masked load.
    assert "ninetoothed_constexpr_prefix_padding_h" in source
    assert "ninetoothed_constexpr_prefix_padding_w" in source
    assert "tl.load(lhs + tl.where((" in source
    assert "), 0), mask=(True &" in source
    ast.parse(source)


def test_ascend_attention_sidecar_reload_reproduces_source(tmp_path):
    request = _attention_request()
    first = DEFAULT_COMPILER.compile(request)
    first_schedule = first.artifact.metadata["ssa_metadata"]["schedule"]
    sidecar = _write_and_reload_sidecar(tmp_path, first)
    second = DEFAULT_COMPILER.compile(request)

    attention = first_schedule["ascend_attention_loop"]
    assert attention["mode"] == "generic-online-softmax-loop"
    assert attention["status"] == "verified-static-public-online-softmax"
    assert sidecar["attention_loop"] == attention
    assert (
        second.artifact.metadata["ssa_metadata"]["schedule"]["ascend_attention_loop"]
        == sidecar["attention_loop"]
    )
    assert second.artifact.primary_source == first.artifact.primary_source
