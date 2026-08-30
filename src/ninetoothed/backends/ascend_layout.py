"""Runtime layout admission for the Ascend Triton backend.

The common frontend preserves tensor layout metadata.  This module owns the
NPU-specific decision about whether a concrete strided view can be launched.
"""

from dataclasses import dataclass
from typing import Any

from ninetoothed.compiler.layout_runtime import (
    _tensor_has_non_overlapping_strides,
    memory_spans_overlap,
    tensor_memory_span,
)


class AscendLayoutCapabilityError(ValueError):
    """Raised when a concrete layout is outside the verified NPU contract."""


@dataclass(frozen=True)
class AscendLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    storage_offset: int

    @property
    def contiguous(self) -> bool:
        expected = 1

        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size > 1 and stride != expected:
                return False
            expected *= size

        return True


def admit_tensor_layout(
    name: str, value: Any, *, allow_rank4_access_template: bool = False
) -> AscendLayout:
    """Validate rank, positive strides, span, and non-overlapping elements."""
    try:
        shape = tuple(int(size) for size in value.shape)
        strides = tuple(int(stride) for stride in value.stride())
        storage_offset = int(value.storage_offset())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` must expose shape, stride, and storage offset."
        ) from exc

    allowed_ranks = {0, 1, 2, 3, 4} if allow_rank4_access_template else {0, 1, 2, 3}
    if len(shape) not in allowed_ranks:
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` supports only rank 0 through 3; received rank {len(shape)}."
        )

    if len(shape) != len(strides) or any(size < 0 for size in shape):
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` has invalid shape/stride metadata."
        )

    if storage_offset < 0 or any(stride < 0 for stride in strides):
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` does not support negative storage offsets or strides."
        )

    if not _tensor_has_non_overlapping_strides(value):
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` has overlapping strides unsupported by Triton-Ascend."
        )

    _validate_storage_span(name, value, shape, strides)

    return AscendLayout(shape=shape, strides=strides, storage_offset=storage_offset)


def reject_write_overlap(tensors, access_by_name) -> None:
    """Reject storage overlap involving an Ascend writer."""
    items = tuple(tensors.items())

    for index, (first_name, first) in enumerate(items):
        first_access = access_by_name.get(first_name, "read") or "read"

        for second_name, second in items[index + 1 :]:
            second_access = access_by_name.get(second_name, "read") or "read"

            if (
                first_access == "read"
                and second_access == "read"
                or not memory_spans_overlap(
                    tensor_memory_span(first), tensor_memory_span(second)
                )
            ):
                continue

            if first_access == "write" and second_access == "read":
                writer, reader = first_name, second_name
            elif second_access == "write" and first_access == "read":
                writer, reader = second_name, first_name
            else:
                writer = reader = None

            if reader is None:
                raise AscendLayoutCapabilityError(
                    "Ascend launch rejects storage overlap between writers "
                    f"`{first_name}` and `{second_name}`."
                )

            raise AscendLayoutCapabilityError(
                "Ascend launch rejects storage overlap involving writer "
                f"'{writer}' and reader '{reader}'."
            )


def _validate_storage_span(name: str, value: Any, shape, strides) -> None:
    if any(size == 0 for size in shape):
        return

    try:
        capacity = value.untyped_storage().nbytes() // value.element_size()
        end = (
            int(value.storage_offset())
            + 1
            + sum((size - 1) * stride for size, stride in zip(shape, strides))
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` cannot validate its storage span."
        ) from exc

    if end > capacity:
        raise AscendLayoutCapabilityError(
            f"Ascend layout for `{name}` exceeds its underlying storage span."
        )
