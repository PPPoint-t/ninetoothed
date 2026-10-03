"""Static Ascend sidecar/source reproduction tests.

These tests intentionally stop after source emission and private sidecar reload.
They do not invoke the Triton-Ascend compiler or an NPU runtime.
"""

import ast
import functools
import json
from collections.abc import Mapping

import pytest

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
            shape=(1, 1, 64, 64),
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

    assert sidecar["schema"] == 4
    attention = first_schedule["ascend_attention_loop"]
    attention_plan = first_schedule["ascend_attention_plan"]
    attention_retile = first_schedule["ascend_attention_retile"]
    assert sidecar["logical_domain"] is None
    assert attention["kind"] == "generic-online-softmax-loop"
    assert attention["mode"] in {"causal", "non-causal"}
    assert attention["status"] == "verified-static-public-online-softmax"
    assert sidecar["attention_loop"] == _json_compatible(attention)
    assert sidecar["attention_plan"] == _json_compatible(attention_plan)
    assert sidecar["attention_retile"] == _json_compatible(attention_retile)
    selected = attention_plan["resource_plan"]["selected_tile"]
    assert attention_plan["tile"] == selected
    assert {
        axis: attention_retile[f"block_{axis}"] for axis in ("m", "n", "k")
    } == selected
    assert first.artifact.metadata["ascend_tile_override"]["BLOCK_SIZE_M"] == selected["m"]
    assert first.artifact.metadata["ascend_tile_override"]["BLOCK_SIZE_N"] == selected["n"]
    assert first.artifact.metadata["ascend_tile_override"]["BLOCK_SIZE_K"] == selected["k"]

    source_tree = ast.parse(first.artifact.primary_source)
    source_contract = next(
        ast.literal_eval(statement.value)
        for statement in source_tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "__ninetoothed_ascend_attention_contract__"
            for target in statement.targets
        )
    )
    assert source_contract == sidecar["attention_source_contract"]
    assert source_contract["attention_plan"] == sidecar["attention_plan"]
    assert source_contract["attention_retile"] == sidecar["attention_retile"]
    assert (
        second.artifact.metadata["ssa_metadata"]["schedule"]["ascend_attention_loop"]
        == attention
    )
    assert second.artifact.primary_source == first.artifact.primary_source


def _json_compatible(value):
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def test_ascend_attention_sidecar_reload_rejects_stale_source_plan(tmp_path):
    compilation = DEFAULT_COMPILER.compile(_attention_request())
    source_path = tmp_path / compilation.artifact.primary_source_name
    source_path.write_text(compilation.artifact.primary_source, encoding="utf-8")
    write_ascend_sidecar(
        source_path,
        abi=compilation.launch_abi,
        specs=compilation.kernel.tensors,
        outputs=compilation.artifact.metadata["outputs"],
        metadata=compilation.artifact.metadata,
    )

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    contract_statement = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "__ninetoothed_ascend_attention_contract__"
            for target in statement.targets
        )
    )
    source_contract = ast.literal_eval(contract_statement.value)
    source_contract["attention_plan"]["tile"]["m"] += 16
    source = source_path.read_text(encoding="utf-8")
    original_contract = ast.get_source_segment(source, contract_statement)
    replacement = (
        "__ninetoothed_ascend_attention_contract__ = "
        f"{source_contract!r}"
    )
    source_path.write_text(source.replace(original_contract, replacement), encoding="utf-8")

    with pytest.raises(ValueError, match="Ascend Attention plan/source contract mismatch"):
        read_ascend_sidecar(source_path)


def test_ascend_attention_sidecar_reload_rejects_missing_plan(tmp_path):
    compilation = DEFAULT_COMPILER.compile(_attention_request())
    source_path = tmp_path / compilation.artifact.primary_source_name
    source_path.write_text(compilation.artifact.primary_source, encoding="utf-8")
    sidecar_path = write_ascend_sidecar(
        source_path,
        abi=compilation.launch_abi,
        specs=compilation.kernel.tensors,
        outputs=compilation.artifact.metadata["outputs"],
        metadata=compilation.artifact.metadata,
    )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload.pop("attention_plan")
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Ascend Attention plan/source contract mismatch"):
        read_ascend_sidecar(source_path)
