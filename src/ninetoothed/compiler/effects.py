"""Shared SSA effect analysis used to construct launch ABI access modes."""

from ninetoothed.ir import ssa


def tensor_access_modes(
    program: ssa.Program, *, pointer_names: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Return conservative read/write modes for tensor program inputs."""
    tensor_names = {
        value.name for value in program.inputs if value.type.kind == "tensor"
    } | set(pointer_names)
    operations = tuple(
        operation
        for block in program.blocks
        for operation in _walk_operations(block.operations)
    )
    producers = {
        result.name: operation
        for operation in operations
        for result in operation.results
    }
    block_sources = {}

    for operation in operations:
        if operation.opcode != "scf.for" or not operation.regions:
            continue

        block_args = operation.regions[0].args

        if not block_args:
            continue

        block_sources[block_args[0].name] = operation.operands[:3]
        yield_operation = next(
            (
                nested
                for nested in reversed(operation.regions[0].operations)
                if nested.opcode == "scf.yield"
            ),
            None,
        )
        yielded = yield_operation.operands if yield_operation is not None else ()

        for block_arg, initial, result in zip(
            block_args[1:], operation.operands[3:], yielded
        ):
            block_sources[block_arg.name] = (initial, result)

    dependencies: dict[str, frozenset[str]] = {}

    def data_dependencies(name, visiting=frozenset()):
        if name in tensor_names:
            return frozenset((name,))

        if name in dependencies:
            return dependencies[name]

        if name in visiting:
            return frozenset()

        sources = block_sources.get(name)

        if sources is not None:
            result = frozenset().union(
                *(data_dependencies(source, visiting | {name}) for source in sources)
            )
            dependencies[name] = result

            return result

        producer = producers.get(name)

        if producer is None or producer.opcode.startswith(("index.", "shape.")):
            return frozenset()

        nested_yields = (
            operand
            for region in producer.regions
            for operation in _walk_operations(region.operations)
            if operation.opcode == "scf.yield"
            for operand in operation.operands
        )
        result = frozenset().union(
            *(
                data_dependencies(operand, visiting | {name})
                for operand in (*producer.operands, *nested_yields)
            )
        )
        dependencies[name] = result

        return result

    reads = set()
    writes = set()

    def visit_effects(effect_operations, control_operands=()):
        for operation in effect_operations:
            if operation.opcode == "mem.store" and len(operation.operands) >= 2:
                reads.update(data_dependencies(operation.operands[0]))

                for index in operation.attrs.get("indices", ()):
                    reads.update(data_dependencies(str(index)))

                writes.update(data_dependencies(operation.operands[1]))
            elif operation.opcode == "mem.atomic_add" and operation.operands:
                target = data_dependencies(operation.operands[0])
                reads.update(target)
                writes.update(target)

                for operand in operation.operands[1:]:
                    reads.update(data_dependencies(operand))

            if operation.opcode in {"mem.store", "mem.atomic_add"}:
                for operand in control_operands:
                    reads.update(data_dependencies(operand))

            control_arity = {"scf.if": 1, "scf.for": 3}.get(operation.opcode, 0)
            nested_controls = (*control_operands, *operation.operands[:control_arity])

            for region in operation.regions:
                visit_effects(region.operations, nested_controls)

    visit_effects(program.blocks[0].operations)
    writes.update(value.name for value in program.outputs if value.name in tensor_names)

    return {
        name: (
            "read_write"
            if name in reads and name in writes
            else "write"
            if name in writes
            else "read"
        )
        for name in tensor_names
        if name in reads or name in writes
    }


def _walk_operations(operations):
    for operation in operations:
        yield operation

        for region in operation.regions:
            yield from _walk_operations(region.operations)
