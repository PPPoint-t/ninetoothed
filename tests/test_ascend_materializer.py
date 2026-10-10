import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ninetoothed.backends.ascend import (
    ASCEND_ATTENTION_DTYPE_REGISTRY,
    ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE,
    UnsupportedBackendOpError,
    ascend_attention_source_contract,
    read_ascend_sidecar,
    write_ascend_sidecar,
)
from ninetoothed.backends.core import Artifact, BuiltArtifact, Target
from ninetoothed.backends.materializers.ascend import (
    AscendMaterializer,
    _ascend_wrapper,
    _current_npu_stream,
    _canonical_contract_value,
    _load_source_module,
    _logical_offset,
    _validate_ascend_bindings,
    _validate_ascend_dtype_specs,
    _validate_dot_loop_runtime_shapes,
    _validate_attention_runtime_contract,
    validate_tile_ub_capacity,
)
from ninetoothed.ir import LaunchABI, LaunchBinding, TensorSpec


class _Storage:
    def __init__(self, elements):
        self._elements = elements

    def nbytes(self):
        return self._elements * 4


class _Tensor:
    def __init__(
        self,
        elements=256,
        *,
        shape=None,
        contiguous=True,
        device_type="npu",
        device_index=0,
        storage_elements=None,
        storage_offset=0,
        data_ptr=None,
        dtype="torch.float32",
        strides=None,
    ):
        self.shape = tuple(shape) if shape is not None else (elements,)
        self.device = SimpleNamespace(type=device_type, index=device_index)
        self._elements = elements
        self._contiguous = contiguous
        self._storage_elements = storage_elements or elements
        self._storage_offset = storage_offset
        self._data_ptr = data_ptr if data_ptr is not None else id(self) * 4096
        self.dtype = dtype
        default_strides = []
        stride = 1

        for size in reversed(self.shape):
            default_strides.append(stride)
            stride *= size

        self._strides = strides or tuple(reversed(default_strides))

    def element_size(self):
        return 4

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return self._elements

    def storage_offset(self):
        return self._storage_offset

    def untyped_storage(self):
        return _Storage(self._storage_elements)

    def data_ptr(self):
        return self._data_ptr

    def stride(self):
        return self._strides


@pytest.mark.parametrize("tile_m", (16, 32))
def test_attention_runtime_grid_and_workspace_follow_resource_tile(tile_m):
    specs, tensors, roles, plan, retile, source_contract = _attention_runtime_fixture(tile_m)
    query_tiles = (plan["sequence"] + tile_m - 1) // tile_m
    key_tiles = (plan["sequence"] + plan["tile"]["n"] - 1) // plan["tile"]["n"]
    assert plan["grid"] == plan["batch"] * plan["heads"] * query_tiles
    assert plan["workspace_bytes"] == tile_m * plan["head_dim"] * 4

    _validate_attention_runtime_contract(
        specs,
        tensors,
        {"version": 2, "access_provenance": roles},
        {
            "k_score_mask_before_max": True,
            "v_value_mask_zero_before_dot": True,
            "causal_query_key_compare": True,
            "loop_carried_state": ("acc", "m_i", "l_i"),
            "all_masked_state_preserving_branch": True,
        },
        {"predicate": "%key_valid"},
        {"all_masked": "%all_masked"},
        plan,
        max_core_dim=65535,
        attention_dtype_registry={"float32": {}},
        attention_retile=retile,
        attention_source_contract=source_contract,
    )
    assert plan["query_tiles"] == query_tiles
    assert plan["key_tiles"] == key_tiles


def test_attention_runtime_rejects_mixed_qkv_output_storage_dtype():
    specs, tensors, roles, plan, retile, source_contract = _attention_runtime_fixture(16)
    tensors = dict(tensors)
    tensors["v"].dtype = "torch.float16"

    with pytest.raises(UnsupportedBackendOpError, match="one storage dtype"):
        _validate_attention_runtime_contract(
            specs,
            tensors,
            {"version": 2, "access_provenance": roles},
            {
                "k_score_mask_before_max": True,
                "v_value_mask_zero_before_dot": True,
                "causal_query_key_compare": True,
                "loop_carried_state": ("acc", "m_i", "l_i"),
                "all_masked_state_preserving_branch": True,
            },
            {"predicate": "%key_valid"},
            {"all_masked": "%all_masked"},
            plan,
            max_core_dim=65535,
            attention_dtype_registry=ASCEND_ATTENTION_DTYPE_REGISTRY,
            attention_retile=retile,
            attention_source_contract=source_contract,
        )


def test_ascend_reload_abi_contract_normalizes_json_sequences():
    in_memory = {"public_args": ("q",), "kernel_args": ({"name": "q"},)}
    sidecar = {"public_args": ["q"], "kernel_args": [{"name": "q"}]}
    assert _canonical_contract_value(in_memory) == _canonical_contract_value(sidecar)


def test_attention_runtime_resolves_symbolic_workspace_from_bound_shape():
    specs, tensors, roles, plan, retile, _ = _attention_runtime_fixture(16)
    plan = dict(plan)
    resource_plan = dict(plan["resource_plan"])
    workspace_expression = "16 * (head_dim) * 4"
    candidate = dict(resource_plan["candidate_tiles"][0])
    candidate["workspace_bytes"] = workspace_expression
    resource_plan["candidate_tiles"] = [candidate]
    resource_plan["workspace_bytes"] = workspace_expression
    plan["resource_plan"] = resource_plan
    plan["workspace_bytes"] = workspace_expression
    plan["head_dim"] = None
    retile = dict(retile) | {"head_dim": "head_dim"}
    source_contract = ascend_attention_source_contract(plan, retile)

    _validate_attention_runtime_contract(
        specs,
        tensors,
        {"version": 2, "access_provenance": roles},
        {
            "k_score_mask_before_max": True,
            "v_value_mask_zero_before_dot": True,
            "causal_query_key_compare": True,
            "loop_carried_state": ("acc", "m_i", "l_i"),
            "all_masked_state_preserving_branch": True,
        },
        {"predicate": "%key_valid"},
        {"all_masked": "%all_masked"},
        plan,
        max_core_dim=65535,
        attention_dtype_registry={"float32": {}},
        attention_retile=retile,
        attention_source_contract=source_contract,
    )


def _attention_runtime_fixture(tile_m=16):
    source_shape = (2, 4, 65, 64)
    roles = {role: {"tensor": role} for role in ("q", "k", "v", "o")}
    specs = {
        role: TensorSpec(
            ndim=4,
            shape=tuple(str(dim) for dim in source_shape),
            dtype="float32",
            name=role,
            attrs={"source_shape": source_shape},
        )
        for role in roles
    }
    tensors = {
        role: _Tensor(
            elements=math.prod(source_shape),
            shape=source_shape,
            dtype="torch.float32",
        )
        for role in roles
    }
    selected_tile = {"m": tile_m, "n": 32, "k": 32}
    query_tiles = (source_shape[2] + tile_m - 1) // tile_m
    key_tiles = (source_shape[2] + selected_tile["n"] - 1) // selected_tile["n"]
    grid = source_shape[0] * source_shape[1] * query_tiles
    workspace = tile_m * source_shape[3] * 4
    ub_estimate = 1024
    ub_budget = 8192
    candidate = {
        "tile": selected_tile,
        "estimated_ub_bytes": ub_estimate,
        "solver_tile": selected_tile,
        "solver_estimated_ub_bytes": ub_estimate,
        "ub_budget_bytes": ub_budget,
        "workspace_bytes": workspace,
        "feasible": True,
        "rejection_reason": None,
    }
    resource_plan = {
        "version": 1,
        "candidate_tiles": [candidate],
        "selected_tile": selected_tile,
        "ub_estimated_peak_bytes": ub_estimate,
        "ub_budget_bytes": ub_budget,
        "internal_dtype": "float32",
        "workspace_bytes": workspace,
        "rejection_reason": None,
    }
    plan = {
        "status": "verified-static-single-block",
        "batch": source_shape[0],
        "heads": source_shape[1],
        "sequence": source_shape[2],
        "head_dim": source_shape[3],
        "tile": selected_tile,
        "resource_plan": resource_plan,
        "query_tiles": query_tiles,
        "key_tiles": key_tiles,
        "grid": grid,
        "core_grid_limit": 65535,
        "workspace_bytes": workspace,
        "dot_tiles": {},
    }
    key_upper = f"(({source_shape[2]} + {selected_tile['n'] - 1}) // {selected_tile['n']})"
    query_upper = f"(({source_shape[2]} + {selected_tile['m'] - 1}) // {selected_tile['m']})"
    retile = {
        "block_m": selected_tile["m"],
        "block_n": selected_tile["n"],
        "block_k": selected_tile["k"],
        "sequence": source_shape[2],
        "head_dim": source_shape[3],
        "query_tiles": query_tiles,
        "key_tiles": key_tiles,
        "query_loop_upper": query_upper,
        "key_loop_upper": key_upper,
        "loop_upper": key_upper,
        "grid": f"{source_shape[0]} * {source_shape[1]} * triton.cdiv({source_shape[2]}, {selected_tile['m']})",
        "dot_tiles": {},
    }
    return (
        specs,
        tensors,
        roles,
        plan,
        retile,
        ascend_attention_source_contract(plan, retile),
    )


def _attention_wrapper_for_test(specs_by_name, tensors, roles, plan, retile, source_contract, calls):
    module = SimpleNamespace(
        **{ASCEND_ATTENTION_SOURCE_CONTRACT_ATTRIBUTE: source_contract}
    )
    abi = LaunchABI(
        public_args=("q", "k", "v", "o"),
        kernel_args=tuple(
            LaunchBinding(
                name=role,
                kind="tensor",
                source=role,
                access="write" if role == "o" else "read",
            )
            for role in ("q", "k", "v", "o")
        ),
        outputs=("o",),
    )
    return _ascend_wrapper(
        lambda *args: calls.append(args),
        abi,
        tuple(specs_by_name.values()),
        source_path=Path("attention.ascend.py"),
        kernel_name="attention",
        max_core_dim=65535,
        module=module,
        attention_loop={"version": 2, "kind": "generic-online-softmax-loop", "access_provenance": roles},
        attention_mask_semantics={
            "k_score_mask_before_max": True,
            "v_value_mask_zero_before_dot": True,
            "causal_query_key_compare": True,
            "loop_carried_state": ("acc", "m_i", "l_i"),
            "all_masked_state_preserving_branch": True,
        },
        attention_key_valid={"predicate": "%key_valid"},
        attention_loop_state={"all_masked": "%all_masked"},
        attention_plan=plan,
        attention_retile=retile,
        attention_dtype_registry={"float32": {}},
    )


def test_attention_runtime_rejects_stale_grid_in_plan_before_launch():
    specs, tensors, roles, plan, retile, source_contract = _attention_runtime_fixture(16)
    stale_plan = dict(plan) | {"grid": plan["grid"] + 1}
    with pytest.raises(UnsupportedBackendOpError, match="plan/source contract mismatch"):
        _validate_attention_runtime_contract(
            specs,
            tensors,
            {"version": 2, "access_provenance": roles},
            {
                "k_score_mask_before_max": True,
                "v_value_mask_zero_before_dot": True,
                "causal_query_key_compare": True,
                "loop_carried_state": ("acc", "m_i", "l_i"),
                "all_masked_state_preserving_branch": True,
            },
            {"predicate": "%key_valid"},
            {"all_masked": "%all_masked"},
            stale_plan,
            max_core_dim=65535,
            attention_dtype_registry={"float32": {}},
            attention_retile=retile,
            attention_source_contract=source_contract,
        )


def test_attention_runtime_rejects_stale_workspace_before_launch():
    specs, tensors, roles, plan, retile, source_contract = _attention_runtime_fixture(16)
    stale_plan = dict(plan) | {"workspace_bytes": plan["workspace_bytes"] + 4}
    with pytest.raises(UnsupportedBackendOpError, match="plan/source contract mismatch"):
        _validate_attention_runtime_contract(
            specs,
            tensors,
            {"version": 2, "access_provenance": roles},
            {
                "k_score_mask_before_max": True,
                "v_value_mask_zero_before_dot": True,
                "causal_query_key_compare": True,
                "loop_carried_state": ("acc", "m_i", "l_i"),
                "all_masked_state_preserving_branch": True,
            },
            {"predicate": "%key_valid"},
            {"all_masked": "%all_masked"},
            stale_plan,
            max_core_dim=65535,
            attention_dtype_registry={"float32": {}},
            attention_retile=retile,
            attention_source_contract=source_contract,
        )


def test_attention_wrapper_rejects_source_plan_mismatch_before_launch():
    specs_by_name, tensors, roles, plan, retile, source_contract = _attention_runtime_fixture(16)
    calls = []
    bad_source_contract = dict(source_contract)
    bad_source_plan = dict(source_contract["attention_plan"])
    bad_source_plan["tile"] = {"m": 32, "n": 32, "k": 32}
    bad_source_contract["attention_plan"] = bad_source_plan
    launch = _attention_wrapper_for_test(
        specs_by_name,
        tensors,
        roles,
        plan,
        retile,
        bad_source_contract,
        calls,
    )

    with pytest.raises(UnsupportedBackendOpError, match="plan/source contract mismatch"):
        launch(*(tensors[role] for role in ("q", "k", "v", "o")))

    assert calls == []


def test_attention_wrapper_rejects_runtime_shape_mismatch_before_launch():
    specs_by_name, tensors, roles, plan, retile, source_contract = _attention_runtime_fixture(16)
    tensors = dict(tensors)
    tensors["q"] = _Tensor(
        elements=math.prod((2, 4, 64, 64)),
        shape=(2, 4, 64, 64),
        dtype="torch.float32",
    )
    calls = []
    launch = _attention_wrapper_for_test(
        specs_by_name,
        tensors,
        roles,
        plan,
        retile,
        source_contract,
        calls,
    )

    with pytest.raises(TypeError, match="expected dimension 2 to be 65"):
        launch(*(tensors[role] for role in ("q", "k", "v", "o")))

    assert calls == []


def test_attention_wrapper_rejects_runtime_binding_that_differs_from_plan_before_launch():
    specs_by_name, tensors, roles, plan, retile, _ = _attention_runtime_fixture(16)
    plan = dict(plan)
    plan["batch"] = 3
    plan["grid"] = plan["batch"] * plan["heads"] * plan["query_tiles"]
    retile = dict(retile)
    retile["grid"] = (
        f"{plan['batch']} * {plan['heads']} * "
        f"triton.cdiv({plan['sequence']}, {plan['tile']['m']})"
    )
    source_contract = ascend_attention_source_contract(plan, retile)
    calls = []
    launch = _attention_wrapper_for_test(
        specs_by_name,
        tensors,
        roles,
        plan,
        retile,
        source_contract,
        calls,
    )

    with pytest.raises(UnsupportedBackendOpError, match="runtime shape does not match the compiled plan"):
        launch(*(tensors[role] for role in ("q", "k", "v", "o")))

    assert calls == []


def _abi(*, with_access=False):
    return LaunchABI(
        public_args=("x", "out"),
        kernel_args=(
            LaunchBinding(
                name="x",
                kind="tensor",
                source="x",
                access="read" if with_access else None,
            ),
            LaunchBinding(
                name="out",
                kind="tensor",
                source="out",
                access="write" if with_access else None,
            ),
        ),
        outputs=("out",),
    )


def _specs():
    return (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
    )


def test_ascend_sidecar_round_trips_dot_loop_schedule(tmp_path):
    source = tmp_path / "dot_loop.ascend.py"
    source.write_text("def launch_dot_loop():\n    return None\n", encoding="utf-8")
    contract = {
        "version": 1,
        "mode": "generic-dot-loop",
        "loop_carried": True,
        "layout": "public-access-template",
        "tile": {"m": 16, "n": 16, "k": 64},
    }
    write_ascend_sidecar(
        source,
        abi=_abi(),
        specs=_specs(),
        outputs=("out",),
        metadata={"ssa_metadata": {"schedule": {"ascend_dot_loop": contract}}},
    )
    assert read_ascend_sidecar(source)["dot_loop"] == contract


def test_ascend_binding_validator_accepts_offset_and_positive_strides():
    abi = _abi()
    specs = _specs()

    _validate_ascend_bindings(
        abi,
        {"x": _Tensor(), "out": _Tensor()},
        specs,
        max_core_dim=1,
    )

    _validate_ascend_bindings(
        abi,
        {"x": _Tensor(storage_offset=1, storage_elements=257), "out": _Tensor()},
        specs,
        max_core_dim=1,
    )

    _validate_ascend_bindings(
        abi,
        {
            "x": _Tensor(contiguous=False, strides=(2,), storage_elements=511),
            "out": _Tensor(),
        },
        specs,
        max_core_dim=1,
    )

    with pytest.raises(ValueError, match="requires 256 elements"):
        _validate_ascend_bindings(
            abi,
            {"x": _Tensor(128), "out": _Tensor()},
            specs,
            max_core_dim=1,
        )


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_ascend_binding_validator_accepts_verified_dtypes(dtype):
    specs = tuple(
        TensorSpec(ndim=1, shape=("n",), dtype=dtype, name=name)
        for name in ("x", "out")
    )

    _validate_ascend_bindings(
        _abi(),
        {
            "x": _Tensor(dtype=f"torch.{dtype}"),
            "out": _Tensor(dtype=f"torch.{dtype}"),
        },
        specs,
        max_core_dim=1,
    )


def test_ascend_binding_validator_rejects_runtime_dtype_mismatch():
    with pytest.raises(TypeError, match="dtype float32; expected float16"):
        _validate_ascend_bindings(
            _abi(with_access=True),
            {"x": _Tensor(), "out": _Tensor()},
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float16", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float16", name="out"),
            ),
            max_core_dim=1,
        )


@pytest.mark.parametrize("dtype", ("float64", None))
def test_ascend_materializer_rejects_unverified_dtype_specs(dtype):
    specs = (TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="x"),)

    with pytest.raises(ValueError, match="verified FP16, BF16, FP32, and INT32"):
        _validate_ascend_dtype_specs(specs)


def test_ascend_binding_validator_accepts_one_dimensional_singleton_broadcast():
    abi = LaunchABI(
        public_args=("x", "bias", "out"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x"),
            LaunchBinding(name="bias", kind="tensor", source="bias"),
            LaunchBinding(name="out", kind="tensor", source="out"),
        ),
        outputs=("out",),
    )
    specs = (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("1",), dtype="float32", name="bias"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
    )

    _validate_ascend_bindings(
        abi,
        {"x": _Tensor(256), "bias": _Tensor(1), "out": _Tensor(256)},
        specs,
        max_core_dim=1,
    )


def test_ascend_binding_validator_accepts_contiguous_matrix_and_row_broadcast():
    abi = LaunchABI(
        public_args=("x", "bias", "out"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x"),
            LaunchBinding(name="bias", kind="tensor", source="bias"),
            LaunchBinding(name="out", kind="tensor", source="out"),
        ),
        outputs=("out",),
    )
    specs = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("1", "n"), dtype="float32", name="bias"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )

    _validate_ascend_bindings(
        abi,
        {
            "x": _Tensor(527, shape=(17, 31), strides=(31, 1)),
            "bias": _Tensor(31, shape=(1, 31), strides=(31, 1)),
            "out": _Tensor(527, shape=(17, 31), strides=(31, 1)),
        },
        specs,
        max_core_dim=3,
        logical_domain=527,
    )


def test_ascend_binding_validator_accepts_fused_row_reduction_value_domain():
    specs = tuple(
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name=name)
        for name in ("x", "out")
    )
    reduction = {
        "mode": "row-vector",
        "axis": 1,
        "value_shape": ("m", "n"),
        "result_shape": ("m",),
    }

    _validate_ascend_bindings(
        _abi(),
        {
            "x": _Tensor(527, shape=(17, 31), strides=(31, 1)),
            "out": _Tensor(527, shape=(17, 31), strides=(31, 1)),
        },
        specs,
        max_core_dim=3,
        logical_domain=527,
        reduction_schedule=reduction,
    )


def test_ascend_binding_validator_accepts_multiple_outputs_and_rejects_output_alias():
    abi = LaunchABI(
        public_args=("x", "out0", "out1"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x", access="read"),
            LaunchBinding(name="out0", kind="tensor", source="out0", access="write"),
            LaunchBinding(name="out1", kind="tensor", source="out1", access="write"),
        ),
        outputs=("out0", "out1"),
    )
    specs = tuple(
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name=name)
        for name in ("x", "out0", "out1")
    )
    public = {
        "x": _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=1024),
        "out0": _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=4096),
        "out1": _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=8192),
    }

    _validate_ascend_bindings(abi, public, specs, max_core_dim=3, logical_domain=527)

    public["out1"] = _Tensor(
        0,
        shape=(17, 31),
        strides=(31, 1),
        storage_elements=527,
        data_ptr=8192,
    )

    with pytest.raises(ValueError, match="requires every output to contain"):
        _validate_ascend_bindings(
            abi, public, specs, max_core_dim=3, logical_domain=527
        )

    public["out1"] = _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=4096)

    with pytest.raises(ValueError, match="storage overlap between writers"):
        _validate_ascend_bindings(
            abi, public, specs, max_core_dim=3, logical_domain=527
        )


def test_ascend_binding_validator_accepts_multidimensional_noncontiguous_and_rejects_alias():
    specs = tuple(
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name=name)
        for name in ("x", "out")
    )

    _validate_ascend_bindings(
        _abi(),
        {
            "x": _Tensor(527, shape=(17, 31), contiguous=False, strides=(1, 17)),
            "out": _Tensor(527, shape=(17, 31), strides=(31, 1)),
        },
        specs,
        max_core_dim=3,
        logical_domain=527,
    )

    with pytest.raises(ValueError, match="storage overlap"):
        _validate_ascend_bindings(
            _abi(with_access=True),
            {
                "x": _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=4096),
                "out": _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=4096),
            },
            specs,
            max_core_dim=3,
            logical_domain=527,
        )


@pytest.mark.parametrize("shape", ((17, 1), (1, 31), (1, 1), (31,), (1,)))
def test_ascend_binding_validator_accepts_trailing_singleton_broadcast(shape):
    specs = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )

    _validate_ascend_bindings(
        _abi(),
        {
            "x": _Tensor(527, shape=shape),
            "out": _Tensor(527, shape=(17, 31)),
        },
        specs,
        max_core_dim=3,
        logical_domain=527,
    )


def test_ascend_binding_validator_accepts_readonly_zero_stride_expand():
    specs = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )

    _validate_ascend_bindings(
        _abi(with_access=True),
        {
            "x": _Tensor(31, shape=(17, 31), strides=(0, 1)),
            "out": _Tensor(527, shape=(17, 31)),
        },
        specs,
        max_core_dim=3,
        logical_domain=527,
    )


@pytest.mark.parametrize("shape", ((17,), (2, 31), (2, 17, 31)))
def test_ascend_binding_validator_rejects_incompatible_broadcast(shape):
    specs = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )

    with pytest.raises(ValueError, match="match the output rank|broadcast|requires"):
        _validate_ascend_bindings(
            _abi(),
            {
                "x": _Tensor(
                    527,
                    shape=shape,
                    storage_elements=1054 if len(shape) == 3 else None,
                ),
                "out": _Tensor(527, shape=(17, 31)),
            },
            specs,
            max_core_dim=3,
            logical_domain=527,
        )


def test_ascend_binding_validator_accepts_scalar_output_pointer():
    abi = LaunchABI(
        public_args=("x", "out"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x", access="read"),
            LaunchBinding(name="out", kind="tensor", source="out", access="write"),
        ),
        outputs=("out",),
    )
    specs = (
        TensorSpec(ndim=1, shape=("1",), dtype="float32", name="x"),
        TensorSpec(ndim=0, shape=(), dtype="float32", name="out"),
    )

    _validate_ascend_bindings(
        abi,
        {"x": _Tensor(1), "out": _Tensor(1, shape=())},
        specs,
        max_core_dim=1,
        logical_domain=1,
    )


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "axis"),
    (
        ((127,), (), 0),
        ((127, 31), (31,), 0),
        ((2, 127, 31), (2, 31), 1),
    ),
)
def test_ascend_binding_validator_accepts_ranked_row_reduction_domains(
    input_shape, output_shape, axis
):
    input_elements = 1

    for extent in input_shape:
        input_elements *= extent

    output_elements = 1

    for extent in output_shape:
        output_elements *= extent

    specs = (
        TensorSpec(ndim=len(input_shape), shape=input_shape, dtype="float32", name="x"),
        TensorSpec(
            ndim=len(output_shape), shape=output_shape, dtype="float32", name="out"
        ),
    )
    _validate_ascend_bindings(
        _abi(with_access=True),
        {
            "x": _Tensor(input_elements, shape=input_shape, data_ptr=1024),
            "out": _Tensor(output_elements, shape=output_shape, data_ptr=1048576),
        },
        specs,
        max_core_dim=max(1, (output_elements + 255) // 256),
        logical_domain=output_elements,
        reduction_schedule={"mode": "row-vector", "axis": axis},
    )


def test_ascend_binding_validator_uses_per_binding_access_templates():
    abi = LaunchABI(
        public_args=("a", "b", "out"),
        kernel_args=(
            LaunchBinding(name="a", kind="tensor", source="a", access="read"),
            LaunchBinding(name="b", kind="tensor", source="b", access="read"),
            LaunchBinding(name="out", kind="tensor", source="out", access="write"),
        ),
        outputs=("out",),
    )
    template = {
        "offsets": ("value_0", "value_1"),
        "linear_offset": "value_0 * 127 + value_1",
        "mask": "True",
    }
    specs = (
        TensorSpec(
            ndim=2,
            shape=("m", "k"),
            dtype="float32",
            name="a",
            attrs={"source_ndim": 2, "access_templates": (template,)},
        ),
        TensorSpec(
            ndim=2,
            shape=("k", "n"),
            dtype="float32",
            name="b",
            attrs={"source_ndim": 2, "access_templates": (template,)},
        ),
        TensorSpec(
            ndim=2,
            shape=("m", "n"),
            dtype="float32",
            name="out",
            attrs={"source_ndim": 2, "access_templates": (template,)},
        ),
    )
    public = {
        "a": _Tensor(2159, shape=(17, 127), data_ptr=1024),
        "b": _Tensor(3937, shape=(127, 31), data_ptr=1048576),
        "out": _Tensor(527, shape=(17, 31), data_ptr=2097152),
    }

    _validate_ascend_bindings(
        abi,
        public,
        specs,
        max_core_dim=3,
        logical_domain=527,
    )

    public["a"] = _Tensor(
        2159,
        shape=(17, 127),
        storage_elements=2158,
        data_ptr=1024,
    )
    with pytest.raises(TypeError, match="exceeds its underlying storage span"):
        _validate_ascend_bindings(
            abi,
            public,
            specs,
            max_core_dim=3,
            logical_domain=527,
        )


def test_ascend_binding_validator_uses_output_for_core_limit():
    abi = LaunchABI(
        public_args=("x", "bias", "out"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x"),
            LaunchBinding(name="bias", kind="tensor", source="bias"),
            LaunchBinding(name="out", kind="tensor", source="out"),
        ),
        outputs=("out",),
    )
    specs = (
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
        TensorSpec(ndim=1, shape=("1",), dtype="float32", name="bias"),
        TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
    )

    with pytest.raises(ValueError, match="required 2, limit 1"):
        _validate_ascend_bindings(
            abi,
            {"x": _Tensor(257), "bias": _Tensor(1), "out": _Tensor(257)},
            specs,
            max_core_dim=1,
        )


def test_ascend_binding_validator_rejects_grid_over_core_limit():
    with pytest.raises(ValueError, match="required 2, limit 1"):
        _validate_ascend_bindings(
            _abi(),
            {"x": _Tensor(257), "out": _Tensor(257)},
            _specs(),
            max_core_dim=1,
        )


def test_ascend_source_loader_does_not_register_a_global_module(tmp_path):
    source = tmp_path / "artifact.ascend.py"
    source.write_text("def launch_test():\n    return 'ok'\n", encoding="utf-8")

    module = _load_source_module(source, "test")

    assert module.launch_test() == "ok"
    assert module.__name__ == "_ninetoothed_ascend_test"


def test_ascend_source_loader_reports_missing_toolchain_dependency(tmp_path):
    source = tmp_path / "missing_dependency.ascend.py"
    source.write_text("import triton_ascend_missing_for_test\n", encoding="utf-8")

    with pytest.raises(ImportError, match="Triton Ascend and CANN runtime"):
        _load_source_module(source, "missing_dependency")


def test_ascend_built_source_artifact_reloads_without_a_binary(tmp_path):
    source_path = tmp_path / "reload.ascend.py"
    source_path.write_text("def launch_reload():\n    return None\n", encoding="utf-8")
    artifact = Artifact(
        backend=Target.ASCEND,
        kernel_name="reload",
        language="python/triton",
        sources={"reload.ascend.py": source_path.read_text(encoding="utf-8")},
        entrypoint="launch_reload",
        metadata={"ssa_schedule": {"core_dim_limit": 1}},
    )
    built = BuiltArtifact(
        source=artifact,
        cache_key="reload-key",
        source_path=str(source_path),
        binary_path=None,
        manifest_path=str(tmp_path / "reload.manifest.json"),
        abi={},
    )
    source_path.with_suffix(".ascend-launch.json").write_text(
        json.dumps(
            {
                "schema": 4,
                "launch_abi": {
                    "public_args": [],
                    "kernel_args": [],
                    "outputs": [],
                    "shape_params": [],
                },
                "logical_domain": "1",
                "outputs": [],
                "max_core_dim": 1,
                "reduction_schedule": None,
                "linalg_contract": None,
                "layout_contract": None,
                "block_meta": {},
            }
        ),
        encoding="utf-8",
    )

    assert AscendMaterializer().load_built_artifact(built)() is None


def test_ascend_wrapper_rejects_non_npu_before_launch():
    calls = []
    launch = _ascend_wrapper(
        lambda *values: calls.append(values),
        _abi(),
        _specs(),
        source_path=SimpleNamespace(),
        kernel_name="test",
        max_core_dim=1,
        module=object(),
    )

    with pytest.raises(TypeError, match="NPU device"):
        launch(_Tensor(device_type="cuda"), _Tensor(device_type="cuda"))

    assert calls == []


def test_ascend_wrapper_rejects_mixed_npu_devices_before_launch():
    calls = []
    launch = _ascend_wrapper(
        lambda *values: calls.append(values),
        _abi(),
        _specs(),
        source_path=SimpleNamespace(),
        kernel_name="test",
        max_core_dim=1,
        module=object(),
    )

    with pytest.raises(TypeError, match="same NPU device"):
        launch(_Tensor(device_index=0), _Tensor(device_index=1))

    assert calls == []


def test_ascend_empty_tensor_returns_without_launch_or_stream_lookup():
    calls = []
    output = _Tensor(elements=0)
    launch = _ascend_wrapper(
        lambda *values: calls.append(values),
        _abi(),
        _specs(),
        source_path=SimpleNamespace(),
        kernel_name="test",
        max_core_dim=1,
        module=object(),
    )

    assert launch(_Tensor(elements=0), output) is output
    assert calls == []


def test_ascend_binding_validator_rejects_invalid_storage_offset():
    with pytest.raises(TypeError, match="negative storage offsets"):
        _validate_ascend_bindings(
            _abi(),
            {"x": _Tensor(storage_offset=-1), "out": _Tensor()},
            _specs(),
            max_core_dim=1,
        )


def test_ascend_binding_validator_uses_logical_domain_for_offset_view():
    specs = (
        TensorSpec(
            ndim=1,
            shape=("257",),
            dtype="float32",
            name="x",
            attrs={"view_offsets": ("index + 1",)},
        ),
        TensorSpec(
            ndim=1,
            shape=("257",),
            dtype="float32",
            name="out",
            attrs={"view_offsets": ("index + 1",)},
        ),
    )

    _validate_ascend_bindings(
        _abi(),
        {
            "x": _Tensor(257, storage_offset=1, storage_elements=258),
            "out": _Tensor(257, storage_offset=1, storage_elements=258),
        },
        specs,
        max_core_dim=2,
        logical_domain=257,
    )

    with pytest.raises(ValueError, match="requires 257 elements"):
        _validate_ascend_bindings(
            _abi(),
            {
                "x": _Tensor(256, storage_offset=1, storage_elements=257),
                "out": _Tensor(257),
            },
            specs,
            max_core_dim=2,
            logical_domain=257,
        )


@pytest.mark.parametrize(
    ("expression", "offset"),
    (("0", 0), ("index", 0), ("index + 2", 2)),
)
def test_ascend_materializer_uses_static_view_offset_contract(expression, offset):
    spec = TensorSpec(
        ndim=1,
        shape=("n",),
        dtype="float32",
        name="x",
        attrs={"view_offsets": (expression,)},
    )

    assert _logical_offset(spec, _abi(), {}) == offset


@pytest.mark.parametrize("expression", ("-1", "index - 1", "index + n"))
def test_ascend_materializer_rejects_invalid_static_view_offsets(expression):
    spec = TensorSpec(
        ndim=1,
        shape=("n",),
        dtype="float32",
        name="x",
        attrs={"view_offsets": (expression,)},
    )

    with pytest.raises(ValueError, match="static forward offset"):
        _logical_offset(spec, _abi(), {})


def test_ascend_binding_validator_rejects_writer_reader_storage_alias():
    abi = LaunchABI(
        public_args=("x", "out"),
        kernel_args=(
            LaunchBinding(name="x", kind="tensor", source="x", access="read"),
            LaunchBinding(name="out", kind="tensor", source="out", access="write"),
        ),
        outputs=("out",),
    )

    with pytest.raises(ValueError, match="storage overlap.*writer 'out'.*reader 'x'"):
        _validate_ascend_bindings(
            abi,
            {
                "x": _Tensor(data_ptr=4096),
                "out": _Tensor(data_ptr=4096),
            },
            _specs(),
            max_core_dim=1,
            logical_domain=256,
        )


def test_ascend_stream_error_identifies_missing_torch_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    with pytest.raises(RuntimeError, match="torch_npu"):
        _current_npu_stream({"x": _Tensor()})


def test_dynamic_source_shape_is_checked_only_at_runtime():
    spec = TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x")
    value = SimpleNamespace(shape=(37,), storage_offset=lambda: 0)
    _validate_dot_loop_runtime_shapes(
        {"x": spec},
        {"x": value},
        {
            "mode": "generic-dot-loop",
            "layout": "public-access-template",
            "loop_carried": True,
        },
    )


def test_tile_ub_capacity_rejects_oversized_tile():
    with pytest.raises(ValueError, match="UB bytes"):
        validate_tile_ub_capacity({"m": 256, "n": 256, "k": 256})
