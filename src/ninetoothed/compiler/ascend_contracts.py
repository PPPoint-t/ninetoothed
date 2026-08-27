"""Structured Ascend contracts shared by lowering and launch planning."""

from typing import Any

from ninetoothed.ir import IndexExpr


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
