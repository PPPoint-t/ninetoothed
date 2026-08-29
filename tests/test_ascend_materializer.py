import sys
from types import SimpleNamespace

import pytest

from ninetoothed.backends.core import Artifact, BuiltArtifact, Target
from ninetoothed.backends.materializers.ascend import (
    AscendMaterializer,
    _ascend_wrapper,
    _current_npu_stream,
    _load_source_module,
    _logical_offset,
    _matmul_inputs,
    _validate_ascend_bindings,
    _validate_ascend_dtype_specs,
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
        self._data_ptr = data_ptr if data_ptr is not None else id(self) * 8
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


def test_ascend_binding_validator_requires_contiguous_broadcastable_tensors():
    abi = _abi()
    specs = _specs()

    _validate_ascend_bindings(
        abi,
        {"x": _Tensor(), "out": _Tensor()},
        specs,
        max_core_dim=1,
    )

    with pytest.raises(TypeError, match="must be contiguous"):
        _validate_ascend_bindings(
            abi,
            {"x": _Tensor(contiguous=False), "out": _Tensor()},
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


@pytest.mark.parametrize("dtype", ("float64", "int32", None))
def test_ascend_materializer_rejects_unverified_dtype_specs(dtype):
    specs = (TensorSpec(ndim=1, shape=("n",), dtype=dtype, name="x"),)

    with pytest.raises(ValueError, match="verified FP16, BF16, and FP32"):
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

    public["out1"] = _Tensor(0, shape=(17, 31), strides=(31, 1), data_ptr=8192)

    with pytest.raises(ValueError, match="requires every output to contain"):
        _validate_ascend_bindings(
            abi, public, specs, max_core_dim=3, logical_domain=527
        )

    public["out1"] = _Tensor(527, shape=(17, 31), strides=(31, 1), data_ptr=4096)

    with pytest.raises(ValueError, match="storage overlap between writers"):
        _validate_ascend_bindings(
            abi, public, specs, max_core_dim=3, logical_domain=527
        )


def test_ascend_binding_validator_rejects_multidimensional_noncontiguous_and_alias():
    specs = tuple(
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name=name)
        for name in ("x", "out")
    )

    with pytest.raises(TypeError, match="must be contiguous"):
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


@pytest.mark.parametrize("shape", ((17, 1), (17,), (2, 17, 31)))
def test_ascend_binding_validator_rejects_unsupported_multidimensional_broadcast(shape):
    specs = (
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="x"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )

    with pytest.raises(
        ValueError, match="match the output rank|broadcast|requires 527 elements"
    ):
        _validate_ascend_bindings(
            _abi(),
            {
                "x": _Tensor(17, shape=shape),
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
        TensorSpec(ndim=len(output_shape), shape=output_shape, dtype="float32", name="out"),
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


def test_ascend_binding_validator_rejects_unimplemented_partial_reduction():
    with pytest.raises(ValueError, match="exceeds BLOCK=256"):
        _validate_ascend_bindings(
            _abi(with_access=True),
            {
                "x": _Tensor(257, shape=(257,)),
                "out": _Tensor(1, shape=()),
            },
            (
                TensorSpec(ndim=1, shape=("257",), dtype="float32", name="x"),
                TensorSpec(ndim=0, shape=(), dtype="float32", name="out"),
            ),
            max_core_dim=1,
            logical_domain=1,
            reduction_schedule={"mode": "row-vector", "axis": 0},
        )


def test_ascend_binding_validator_accepts_contiguous_matmul_and_rejects_bad_k():
    abi = LaunchABI(
        public_args=("a", "b", "out"),
        kernel_args=(
            LaunchBinding(name="a", kind="tensor", source="a", access="read"),
            LaunchBinding(name="b", kind="tensor", source="b", access="read"),
            LaunchBinding(name="out", kind="tensor", source="out", access="write"),
        ),
        outputs=("out",),
    )
    specs = (
        TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
        TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
        TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
    )
    contract = {"mode": "matrix-scalar-loop", "lhs": "a", "rhs": "b"}
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
        linalg_contract=contract,
    )

    public["b"] = _Tensor(4064, shape=(128, 31), data_ptr=1048576)

    with pytest.raises(ValueError, match=r"lhs\[M,K\] @ rhs\[K,N\]"):
        _validate_ascend_bindings(
            abi,
            public,
            specs,
            max_core_dim=3,
            logical_domain=527,
            linalg_contract=contract,
        )


def test_ascend_matmul_contract_rejects_rank_and_dtype_mismatches():
    contract = {"mode": "matrix-scalar-loop", "lhs": "a", "rhs": "b"}
    tensors = {
        "a": _Tensor(2159, shape=(17, 127), dtype="torch.float32"),
        "b": _Tensor(3937, shape=(127, 31), dtype="torch.float16"),
        "out": _Tensor(527, shape=(17, 31), dtype="torch.float32"),
    }

    with pytest.raises(TypeError, match="matching input/output dtypes"):
        _matmul_inputs(tensors, ("out",), (17, 31), contract)

    tensors["b"] = _Tensor(127, shape=(127,), dtype="torch.float32")

    with pytest.raises(ValueError, match="rank-2 contiguous inputs"):
        _matmul_inputs(tensors, ("out",), (17, 31), contract)


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
    with pytest.raises(ValueError, match="storage span"):
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
        {"x": _Tensor(258), "out": _Tensor(258)},
        specs,
        max_core_dim=2,
        logical_domain=257,
    )

    with pytest.raises(ValueError, match="requires 258 elements"):
        _validate_ascend_bindings(
            _abi(),
            {"x": _Tensor(257), "out": _Tensor(258)},
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
