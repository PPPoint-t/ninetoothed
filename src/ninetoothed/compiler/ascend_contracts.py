"""Structured Ascend contracts shared by lowering and launch planning."""

from typing import Any

from ninetoothed.ir import IndexExpr

ASCEND_ELEMENTWISE_DTYPES = frozenset({"float16", "bfloat16", "float32"})


def normalize_ascend_dtype(dtype: str | None) -> str | None:
    """Return a canonical dtype spelling without defaulting an absent dtype."""
    if dtype is None:
        return None

    value = str(dtype).strip().lower()

    if "." in value:
        value = value.rsplit(".", 1)[-1]

    return {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }.get(value, value)


def unsupported_ascend_elementwise_dtypes(
    dtypes: tuple[str | None, ...],
) -> tuple[str, ...]:
    """Return canonical dtype names outside the verified Ascend tier."""
    return tuple(
        sorted(
            {
                "unspecified" if dtype is None else normalize_ascend_dtype(dtype)
                for dtype in dtypes
                if normalize_ascend_dtype(dtype) not in ASCEND_ELEMENTWISE_DTYPES
            }
        )
    )


def static_forward_view_offset(value: Any) -> int:
    """Resolve the supported one-dimensional logical-view offset to an integer."""
    expression = IndexExpr.parse(value)

    if expression.op == "constant" and _is_nonnegative_int(expression.value):
        return expression.value

    if expression.op == "symbol" and expression.value == "index":
        return 0

    if (
        expression.op == "add"
        and _is_index(expression.operands[0])
        and _is_nonnegative_int(expression.operands[1].value)
    ):
        return expression.operands[1].value

    raise ValueError(
        "Ascend logical views require a static forward offset: a non-negative "
        "constant, `index`, or `index + k`."
    )


def is_static_forward_view_offset(value: Any) -> bool:
    """Return whether ``value`` belongs to the Ascend static-view contract."""
    try:
        static_forward_view_offset(value)
    except (SyntaxError, ValueError):
        return False

    return True


def _is_index(expression: IndexExpr) -> bool:
    return expression.op == "symbol" and expression.value == "index"


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0
