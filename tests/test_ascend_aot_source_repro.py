"""Static Ascend sidecar/source reproduction tests.

These tests intentionally stop after source emission and private sidecar reload.
They do not invoke the Triton-Ascend compiler or an NPU runtime.
"""

import ast
import functools
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ninetoothed import Tensor
from ninetoothed.backends.ascend import read_ascend_sidecar, write_ascend_sidecar
from ninetoothed.backends.core import Target
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.backends.materializers.ascend import AscendMaterializer
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


@pytest.mark.parametrize(
    "tile",
    (
        {"block_size_m": 64, "block_size_n": 64, "block_size_k": 64},
        {"block_size_m": 128, "block_size_n": 32, "block_size_k": 64},
    ),
)
def test_ascend_conv2d_static_aot_uses_verified_resource_tile(tile, tmp_path):
    """Static AOT Conv2d candidates share the dynamic JIT-safe tile contract."""
    sizes = {"n": 4, "c": 64, "h": 16, "w": 16, "k": 512, "r": 3, "s": 3}
    arrangement, application, tensors = test_conv2d.premake(
        **sizes, dtype="float16", **tile
    )
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend=Target.ASCEND,
            kernel_name="ascend_conv2d_static_resource_contract",
            backend_options={"soc_version": "Ascend910B4", "max_core_dim": 65535},
        )
    )
    schedule = compilation.artifact.metadata["ssa_metadata"]["schedule"]
    access = schedule["ascend_access_template_resources"]
    plan = schedule["ascend_conv2d_plan"]
    assert access["operator"] == "conv2d-im2col"
    assert access["tile"] == {"m": 16, "n": 16, "k": 16}
    assert plan["selected_tile"] == {"m": 16, "n": 16, "k": 16}
    assert plan["initial_tile"] == {
        "block_m": tile["block_size_m"],
        "block_n": tile["block_size_n"],
        "block_k": tile["block_size_k"],
    }
    assert plan["ub_estimated_peak_bytes"] <= plan["ub_budget_bytes"]

    sidecar = _write_and_reload_sidecar(tmp_path, compilation)
    assert sidecar["conv2d_plan"]["kind"] == plan["kind"]
    assert sidecar["conv2d_plan"]["selected_tile"] == {
        "m": 16,
        "n": 16,
        "k": 16,
    }
    assert sidecar["conv2d_plan"]["initial_tile"] == plan["initial_tile"]
    assert sidecar["conv2d_plan"]["ub_estimated_peak_bytes"] == plan[
        "ub_estimated_peak_bytes"
    ]
    assert sidecar["dot_loop"]["tile"] == {"m": 16, "n": 16, "k": 16}
    source = compilation.artifact.primary_source
    # The resource plan must reach physical source emission.  Conv2d uses a
    # 16x16 matrix tile and four K=16 reductions for the real K=64 operand;
    # no stale 64-wide vector lane may remain in the generated source.
    assert "tl.arange(0, 16)" in source
    assert "tl.arange(0, 64)" not in source
    assert "for nt_conv_k in range(0, 64, 16)" in source
    assert "nt_matrix_active" in source
    # Triton-Ascend changes cube fragment layout when an otherwise unused
    # generic BLOCK constexpr is present.  Native Conv2d has explicit
    # physical tile coordinates, so its kernel ABI must omit that parameter.
    assert "BLOCK: tl.constexpr" not in source
    assert "BLOCK=block" not in source


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


def test_ascend_attention_jit_aot_share_compile_contract(tmp_path):
    """JIT and AOT materialization consume one compiled Ascend plan."""
    request = _attention_request()
    compilation = DEFAULT_COMPILER.compile(request)
    schedule = compilation.artifact.metadata["ssa_metadata"]["schedule"]
    plan = schedule["ascend_attention_plan"]
    resource = plan["resource_plan"]
    materializer = AscendMaterializer()

    jit = materializer.jit_materialize(compilation, output_dir=tmp_path / "jit")
    aot = materializer.aot_build(compilation, output_dir=tmp_path / "aot")
    sidecar = read_ascend_sidecar(Path(aot._built_artifact.source_path))

    assert jit._compilation is compilation
    assert aot._compilation is compilation
    assert jit._compilation.artifact.metadata["ssa_metadata"]["schedule"] == schedule
    assert aot._compilation.artifact.metadata["ssa_metadata"]["schedule"] == schedule
    assert resource["selected_tile"] == plan["tile"]
    assert resource["ub_budget_bytes"] >= resource["ub_estimated_peak_bytes"]
    assert sidecar["attention_plan"] == _json_compatible(plan)
    assert sidecar["attention_retile"] == _json_compatible(
        schedule["ascend_attention_retile"]
    )

    source_tree = ast.parse(Path(aot._built_artifact.source_path).read_text())
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
    assert source_contract["attention_plan"] == _json_compatible(plan)


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
