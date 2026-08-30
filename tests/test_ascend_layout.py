from types import SimpleNamespace

import pytest

from ninetoothed.backends.ascend_layout import (
    AscendLayoutCapabilityError,
    admit_tensor_layout,
    reject_write_overlap,
)


class _Storage:
    def __init__(self, elements):
        self.elements = elements

    def nbytes(self):
        return self.elements * 4


class _Tensor:
    def __init__(self, shape, strides, *, offset=0, storage_elements=256, pointer=1024):
        self.shape = shape
        self._strides = strides
        self._offset = offset
        self._storage = _Storage(storage_elements)
        self._pointer = pointer
        self.device = SimpleNamespace(type="npu", index=0)

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self._offset

    def untyped_storage(self):
        return self._storage

    def element_size(self):
        return 4

    def data_ptr(self):
        return self._pointer


def test_ascend_layout_admits_positive_non_overlapping_transpose_metadata():
    layout = admit_tensor_layout("x", _Tensor((17, 31), (1, 17), storage_elements=527))

    assert not layout.contiguous


@pytest.mark.parametrize("strides", ((0,), (1, 1), (-1,)))
def test_ascend_layout_rejects_overlapping_or_negative_strides(strides):
    shape = (2, 2) if len(strides) == 2 else (2,)

    with pytest.raises(AscendLayoutCapabilityError, match="overlapping|negative"):
        admit_tensor_layout("x", _Tensor(shape, strides))


def test_ascend_layout_rejects_writer_reader_overlap():
    tensor = _Tensor((16,), (1,), storage_elements=16)

    with pytest.raises(AscendLayoutCapabilityError, match="writer 'out'.*reader 'x'"):
        reject_write_overlap(
            {"x": tensor, "out": tensor}, {"x": "read", "out": "write"}
        )
