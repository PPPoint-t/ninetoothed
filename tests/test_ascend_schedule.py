from dataclasses import replace

import pytest

from ninetoothed import Tensor
import ninetoothed.backends.ascend as ascend_backend
from ninetoothed.backends.ascend import (
    AscendUBPlan,
    _ascend_advanced_contract,
    _ascend_attention_loop_contract,
    _attention_ub_budget_bytes,
    _plan_ascend_attention_contract,
    _plan_ascend_attention_resources,
    ascend_capability_matrix,
    ascend_logical_domain,
    is_static_forward_view_offset,
    static_forward_view_offset,
    UnsupportedBackendOpError,
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


def test_ascend_attention_dtype_capability_matrix_is_explicit():
    attention = ascend_capability_matrix()["dtypes"]["attention"]
    registry = attention["registry"]
    assert set(attention["q"]) == {"float16", "bfloat16", "float32"}
    assert attention["score"] == "float32"
    assert attention["softmax"] == "float32"
    assert attention["m_i"] == "float32"
    assert attention["l_i"] == "float32"
    assert attention["accumulator"] == "float32"
    assert attention["error_tolerance"]["float32"] == {"rtol": 0.025, "atol": 0.025}
    for dtype in ("float16", "bfloat16", "float32"):
        assert registry[dtype]["status"] == "verified-jit-aot-npu"
        assert registry[dtype]["runtime"] == {
            "device": "Ascend910B4",
            "cann": "9.0.0",
            "torch_npu": "2.7.1",
            "triton": "3.2.0",
            "triton_ascend": "3.2.2",
        }
    assert registry["bfloat16"]["status"] == "verified-jit-aot-npu"
    assert registry["bfloat16"]["input"] == {
        "q": "bfloat16",
        "k": "bfloat16",
        "v": "bfloat16",
        "o": "bfloat16",
    }
    assert registry["bfloat16"]["internal"] == {
        "score": "float32",
        "softmax": "float32",
        "acc": "float32",
        "m_i": "float32",
        "l_i": "float32",
    }
    assert registry["bfloat16"]["error"] == {"rtol": 0.05, "atol": 0.1}


@pytest.mark.parametrize("dtype", ("float64", "int8"))
def test_ascend_attention_rejects_unverified_dtype_at_planning(dtype):
    from tests import test_attention

    q = k = v = output = Tensor(shape=(2, 4, 1, 64), dtype=dtype)
    with pytest.raises(UnsupportedBackendOpError, match="dtype capability is unsupported"):
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=test_attention.arrangement,
                application=test_attention.application,
                tensors=(q, k, v, Tensor(0, constexpr=True), output),
                backend=Target.ASCEND,
                kernel_name=f"attention_unsupported_{dtype}",
                backend_options={"soc_version": "Ascend910B4", "max_core_dim": 65535},
            )
        )


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


def _attention_resource_program():
    loop = ssa.Operation(
        opcode="scf.for",
        operands=("%lower", "%upper", "%step"),
        results=(
            ssa.Value(
                name="%loop_result",
                type=ssa.Type(
                    kind="tensor", shape=("32", "64"), dtype="float32"
                ),
            ),
        ),
    )
    return ssa.Program(
        kind="attention",
        blocks=(ssa.Block(operations=(loop,)),),
        metadata={"schedule": {"core_dim_limit": 65535}},
    )


def _attention_resource_contract():
    return {
        "kind": "generic-online-softmax-loop",
        "mode": "non-causal",
        "sequence": 64,
        "head_dim": 64,
        "tile": {"m": 32, "n": 32, "k": 32},
        "score_dot": {},
        "value_dot": {},
        "key_bounds_mask": {},
        "combined_key_valid": {},
        "all_masked_predicate": {"value": "%all_masked"},
        "v_value_mask": {},
        "loop_carried_state": {},
        "state_preserving_branch": {},
        "access_provenance": {
            "q": {"source_shape": ("2", "4", "64", "64")},
            "k": {"source_shape": ("2", "4", "64", "64")},
        },
    }


def _fake_ub_plan(tile, *, estimated_peak, safety_margin=1):
    return AscendUBPlan(
        safe_tile={
            "block_m": tile["m"],
            "block_n": tile["n"],
            "block_k": tile["k"],
        },
        estimated_peak_bytes=estimated_peak,
        workspace_bytes=tile["m"] * tile["n"] * 4,
        safety_margin_bytes=safety_margin,
    )


def test_ascend_attention_resource_plan_matches_ub_solver_selection(monkeypatch):
    candidate = {"m": 32, "n": 32, "k": 32}
    monkeypatch.setattr(
        ascend_backend, "calculate_ssa_ub_bytes", lambda program, tile: 40_000
    )
    monkeypatch.setattr(
        ascend_backend,
        "plan_ascend_ub",
        lambda program, tile: _fake_ub_plan(candidate, estimated_peak=40_000),
    )

    planned = _plan_ascend_attention_contract(
        _attention_resource_program(), _attention_resource_contract()
    )
    resource = planned["resource_plan"]
    record = next(
        candidate
        for candidate in resource["candidate_tiles"]
        if candidate["tile"] == resource["selected_tile"]
    )

    assert resource["selected_tile"] == record["solver_tile"]
    assert planned["tile"] == resource["selected_tile"]
    assert resource["ub_estimated_peak_bytes"] == record["solver_estimated_ub_bytes"]
    assert resource["ub_budget_bytes"] == _attention_ub_budget_bytes()
    assert resource["internal_dtype"] == "float32"
    assert planned["workspace_bytes"] == 32 * 64 * 4


def test_ascend_attention_resource_plan_rejects_over_budget_candidate(monkeypatch):
    candidate = {"m": 32, "n": 32, "k": 32}
    budget = _attention_ub_budget_bytes()
    monkeypatch.setattr(
        ascend_backend,
        "calculate_ssa_ub_bytes",
        lambda program, tile: budget + 1,
    )
    monkeypatch.setattr(
        ascend_backend,
        "plan_ascend_ub",
        lambda program, tile: _fake_ub_plan(candidate, estimated_peak=budget + 1),
    )

    resource = _plan_ascend_attention_resources(
        _attention_resource_program(), head_dim=64
    )

    assert resource["selected_tile"] is None
    assert all(not candidate["feasible"] for candidate in resource["candidate_tiles"])
    assert all(
        candidate["estimated_ub_bytes"] == budget + 1
        and candidate["ub_budget_bytes"] == budget
        for candidate in resource["candidate_tiles"]
    )
    assert resource["rejection_reason"]


def test_ascend_attention_no_feasible_tile_error_reports_tile_estimate_and_budget(
    monkeypatch,
):
    candidate = {"m": 16, "n": 32, "k": 32}
    estimate = 59_392
    budget = _attention_ub_budget_bytes()
    monkeypatch.setattr(
        ascend_backend,
        "calculate_ssa_ub_bytes",
        lambda program, tile: estimate,
    )
    monkeypatch.setattr(
        ascend_backend,
        "plan_ascend_ub",
        lambda program, tile: _fake_ub_plan(candidate, estimated_peak=33_792),
    )

    with pytest.raises(ValueError) as caught:
        _plan_ascend_attention_contract(
            _attention_resource_program(), _attention_resource_contract()
        )

    message = str(caught.value)
    assert "no UB-feasible verified tile" in message
    assert "'m': 32" in message and "'n': 32" in message and "'k': 32" in message
    assert str(estimate) in message
    assert str(budget) in message


@pytest.mark.parametrize("selected_m", (16, 32))
def test_ascend_attention_retile_consumes_selected_resource_tile(monkeypatch, selected_m):
    from tests.test_attention import application, arrangement

    budget = _attention_ub_budget_bytes()

    def estimate(program, tile):
        del program
        return 32_000 if int(tile["block_m"]) == selected_m else budget + 1

    def solve(program, tile):
        del program
        normalized = {
            axis: int(tile[f"block_{axis}"]) for axis in ("m", "n", "k")
        }
        value = estimate(None, tile)
        return _fake_ub_plan(
            normalized,
            estimated_peak=value,
            safety_margin=max(0, budget - value),
        )

    monkeypatch.setattr(ascend_backend, "calculate_ssa_ub_bytes", estimate)
    monkeypatch.setattr(ascend_backend, "plan_ascend_ub", solve)

    captured = {}
    structured_retile = ascend_backend._structured_retile_ascend_attention

    def capture_retile(kernel, provenance):
        retiled = structured_retile(kernel, provenance)
        captured["kernel"] = retiled
        return retiled

    monkeypatch.setattr(
        ascend_backend, "_structured_retile_ascend_attention", capture_retile
    )

    qkv_and_output = tuple(
        Tensor(shape=(2, 4, 65, 64), dtype="float32") for _ in range(4)
    )
    causal = Tensor(0, constexpr=True)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=(*qkv_and_output[:3], causal, qkv_and_output[3]),
            backend=Target.ASCEND,
        )
    )
    kernel = captured["kernel"]
    program = kernel.ssa
    schedule = dict(program.metadata["schedule"])
    attention_plan = schedule["ascend_attention_plan"]
    resource_plan = attention_plan["resource_plan"]
    tile = dict(resource_plan["selected_tile"])

    assert tile == {"m": selected_m, "n": 32, "k": 32}
    assert attention_plan["tile"] == tile
    assert schedule["ascend_attention_retile"]["block_m"] == selected_m
    key_valid_plan = schedule["ascend_attention_key_valid"]
    assert key_valid_plan["axis"] == "all_key_valid_elements"
    all_masked_reductions = tuple(
        operation
        for operation in ascend_backend._walk_operations(program.blocks)
        if operation.opcode == "reduce.all"
        and operation.attrs.get("ascend_attention_mask") == "all_masked"
    )
    assert len(all_masked_reductions) == 1
    assert all_masked_reductions[0].attrs["axis"] is None
    assert "axis=None) != 0" in compilation.artifact.primary_source

    types = {value.name: value.type for value in (*program.inputs, *program.outputs)}

    def collect_types(block):
        for argument in block.args:
            types[argument.name] = argument.type
        for operation in block.operations:
            for result in operation.results:
                types[result.name] = result.type
            for region in operation.regions:
                collect_types(region)

    for block in program.blocks:
        collect_types(block)

    dots = tuple(
        operation
        for operation in ascend_backend._walk_operations(program.blocks)
        if operation.opcode == "linalg.dot"
    )
    score_dot, value_dot = dots
    dot_shapes = tuple(
        tuple(str(dim) for dim in operation.results[0].type.shape)
        for operation in dots
    )
    assert dot_shapes == ((str(selected_m), "32"), (str(selected_m), "64"))
    assert tuple(str(dim) for dim in types[score_dot.operands[0]].shape) == (
        str(selected_m),
        "64",
    )
    assert tuple(str(dim) for dim in types[score_dot.operands[1]].shape) == (
        "64",
        "32",
    )
    assert tuple(str(dim) for dim in types[value_dot.operands[0]].shape) == (
        str(selected_m),
        "32",
    )
    assert tuple(str(dim) for dim in types[value_dot.operands[1]].shape) == (
        "32",
        "64",
    )

    qk_tile = dict(score_dot.attrs["ascend_attention_dot_tile"])
    pv_tile = dict(value_dot.attrs["ascend_attention_dot_tile"])
    assert qk_tile["resource_tile"] == tile
    assert {
        axis: score_dot.attrs[f"block_{axis}"] for axis in ("m", "n", "k")
    } == tile
    assert {
        axis: value_dot.attrs[f"block_{axis}"] for axis in ("m", "n", "k")
    } == tile
    assert qk_tile["tile_shapes"] == {
        "lhs": (str(selected_m), "32"),
        "rhs": ("32", "32"),
        "result": (str(selected_m), "32"),
    }
    assert qk_tile["logical_shapes"] == {
        "lhs": (str(selected_m), "64"),
        "rhs": ("64", "32"),
        "result": (str(selected_m), "32"),
    }
    assert qk_tile["reduction"] == {
        "extent": 64,
        "tile_extent": tile["k"],
        "tile_count": 2,
        "tail_mask": f"reduction_tile_index * {tile['k']} + reduction_lane < 64",
    }
    assert "query_lane < 65" in qk_tile["tail_masks"]["query"]
    assert "key_lane < 65" in qk_tile["tail_masks"]["key"]
    assert "reduction_lane < 64" in qk_tile["tail_masks"]["reduction"]
    assert pv_tile["resource_tile"] == tile
    assert pv_tile["tile_shapes"] == {
        "lhs": (str(selected_m), "32"),
        "rhs": ("32", "64"),
        "result": (str(selected_m), "64"),
    }
    assert pv_tile["reduction"] == {
        "extent": 32,
        "tile_extent": 32,
        "tile_count": 1,
        "tail_mask": "key_tile_index * 32 + key_lane < 65",
    }
    assert schedule["ascend_attention_retile"]["dot_tiles"] == attention_plan[
        "dot_tiles"
    ]

    changed_qk_tile = dict(qk_tile)
    changed_resource_tile = dict(changed_qk_tile["resource_tile"])
    changed_resource_tile["k"] = tile["k"] + 16
    changed_qk_tile["resource_tile"] = changed_resource_tile
    changed_dot = ssa.Operation(
        opcode=score_dot.opcode,
        operands=score_dot.operands,
        results=score_dot.results,
        attrs=dict(score_dot.attrs)
        | {"ascend_attention_dot_tile": changed_qk_tile},
        regions=score_dot.regions,
    )
    changed_program = ascend_backend._replace_operation(program, score_dot, changed_dot)
    with pytest.raises(ValueError, match="SSA dot tile"):
        ascend_backend._verify_ascend_attention_retile_plan(
            replace(kernel, ssa=changed_program), tile
        )

    loop = next(
        operation
        for operation in ascend_backend._walk_operations(program.blocks)
        if operation.opcode == "scf.for"
    )
    retile = dict(loop.attrs["ascend_attention_retile"])
    key_tiles = (65 + tile["n"] - 1) // tile["n"]
    query_tiles = (65 + tile["m"] - 1) // tile["m"]
    upper = next(
        operation
        for operation in ascend_backend._walk_operations(program.blocks)
        if operation.results and operation.results[0].name == loop.operands[1]
    )
    assert upper.opcode == "arith.constant"
    assert upper.attrs["value"] == key_tiles
    assert retile["key_loop_upper"] == f"(({65} + {tile['n'] - 1}) // {tile['n']})"
    assert retile["query_loop_upper"] == f"(({65} + {tile['m'] - 1}) // {tile['m']})"
    assert attention_plan["query_tiles"] == query_tiles
    assert attention_plan["key_tiles"] == key_tiles
    assert attention_plan["grid"] == 2 * 4 * query_tiles
    assert retile["grid"] == f"2 * 4 * triton.cdiv(65, {tile['m']})"
    assert f"grid = ({retile['grid']},)" in compilation.artifact.primary_source

    tensor_specs = {tensor.name: tensor for tensor in kernel.tensors}
    for name, expected in (("q", tile["m"]), ("o", tile["m"]), ("k", tile["n"]), ("v", tile["n"])):
        attrs = dict(tensor_specs[name].attrs)
        assert tuple(str(dim) for dim in attrs["dtype_shapes"][-1])[0] == str(expected)
        masks = tuple(template["mask"] for template in attrs["access_templates"])
        assert any(f"value_0 < {expected}" in str(mask) for mask in masks)

    causal_shapes = tuple(
        tuple(str(dim) for dim in operation.results[0].type.shape)
        for operation in ascend_backend._walk_operations(program.blocks)
        if operation.opcode == "cmp.ge" and operation.results
    )
    assert (str(selected_m), "32") in causal_shapes
    semantics = ascend_backend.verify_ascend_attention_mask_semantics(program)
    assert semantics["k_score_mask_before_max"]
    assert semantics["v_value_mask_zero_before_dot"]
    assert semantics["all_masked_state_preserving_branch"]


def test_ascend_attention_symbolic_key_loop_upper_uses_planned_n(monkeypatch):
    from tests.test_attention import application, arrangement

    captured = {}
    structured_retile = ascend_backend._structured_retile_ascend_attention

    def capture_retile(kernel, provenance):
        retiled = structured_retile(kernel, provenance)
        captured["kernel"] = retiled
        return retiled

    monkeypatch.setattr(
        ascend_backend, "_structured_retile_ascend_attention", capture_retile
    )
    qkv_and_output = tuple(
        Tensor(
            4,
            shape_options=(
                None,
                None,
                {"constexpr": True},
                {"constexpr": True, "upper_bound": 128},
            ),
        )
        for _ in range(4)
    )
    causal = Tensor(0, constexpr=True)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=(*qkv_and_output[:3], causal, qkv_and_output[3]),
            backend=Target.ASCEND,
        )
    )
    kernel = captured["kernel"]
    schedule = dict(kernel.ssa.metadata["schedule"])
    selected_tile = dict(schedule["ascend_attention_plan"]["resource_plan"]["selected_tile"])
    loop = next(
        operation
        for operation in ascend_backend._walk_operations(kernel.ssa.blocks)
        if operation.opcode == "scf.for"
    )
    producers = {
        result.name: operation
        for operation in ascend_backend._walk_operations(kernel.ssa.blocks)
        for result in operation.results
    }
    upper = producers[loop.operands[1]]
    numerator = producers[upper.operands[0]]
    sequence_dim = producers[numerator.operands[0]]
    rounding = producers[numerator.operands[1]]
    divisor = producers[upper.operands[1]]

    assert upper.opcode == "arith.floordiv"
    assert numerator.opcode == "arith.add"
    assert sequence_dim.opcode == "shape.dim"
    assert sequence_dim.attrs["source"] is True
    assert sequence_dim.attrs["dim"] == -2
    assert sequence_dim.operands == ("k",)
    assert rounding.attrs["value"] == selected_tile["n"] - 1
    assert divisor.attrs["value"] == selected_tile["n"]
    assert f"ceil_div(" in schedule["ascend_attention_retile"]["key_loop_upper"]
    assert f"triton.cdiv(" in compilation.artifact.primary_source
    assert str(selected_tile["m"]) in schedule["ascend_attention_retile"]["grid"]
    assert schedule["ascend_attention_mask_semantics"]["all_masked_state_preserving_branch"]

    provenance = schedule["ascend_tile_provenance"]
    roles = {entry["role"]: entry for entry in provenance["entries"]}
    assert set(roles) >= {"matmul-m", "matmul-n", "reduction-k"}
    assert all(entry["candidate_values"] for entry in roles.values())
    assert all(entry["resolved"] is not None for entry in roles.values())
    assert all(entry["ssa_operations"] for entry in roles.values())


def test_ascend_attention_retile_rejects_missing_resource_role():
    from tests.test_attention import application, arrangement

    qkv = tuple(Tensor(shape=(2, 4, 65, 64), dtype="float32") for _ in range(4))
    captured = {}
    original = ascend_backend._structured_retile_ascend_attention
    def capture(kernel, provenance):
        captured["kernel"] = kernel
        return original(kernel, provenance)
    # Stop at the private retile boundary so the test can mutate its input
    # provenance without depending on the later emitter artifact wrapper.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ascend_backend, "_structured_retile_ascend_attention", capture)
    DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=(*qkv[:3], Tensor(0, constexpr=True), qkv[3]),
            backend=Target.ASCEND,
        )
    )
    monkeypatch.undo()
    kernel = captured["kernel"]
    provenance = dict(kernel.ssa.metadata["schedule"]["ascend_tile_provenance"])
    provenance["entries"] = tuple(
        entry for entry in provenance["entries"] if entry["role"] != "matmul-n"
    )
    with pytest.raises(ValueError, match="requires M/N/K tile roles"):
        ascend_backend._structured_retile_ascend_attention(kernel, provenance)


def test_ascend_attention_retile_rejects_resource_plan_provenance_mismatch():
    from tests.test_attention import application, arrangement

    qkv = tuple(Tensor(shape=(2, 4, 65, 64), dtype="float32") for _ in range(4))
    captured = {}
    original = ascend_backend._structured_retile_ascend_attention
    def capture(kernel, provenance):
        captured["kernel"] = kernel
        return original(kernel, provenance)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ascend_backend, "_structured_retile_ascend_attention", capture)
    DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=(*qkv[:3], Tensor(0, constexpr=True), qkv[3]),
            backend=Target.ASCEND,
        )
    )
    monkeypatch.undo()
    kernel = captured["kernel"]
    provenance = dict(kernel.ssa.metadata["schedule"]["ascend_tile_provenance"])
    entries = []
    for entry in provenance["entries"]:
        entry = dict(entry)
        if entry["role"] == "matmul-m":
            entry["resolved"] = int(entry["resolved"]) + 16
        entries.append(entry)
    provenance["entries"] = tuple(entries)
    with pytest.raises(ValueError, match="does not match the resource plan"):
        ascend_backend._structured_retile_ascend_attention(kernel, provenance)


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
