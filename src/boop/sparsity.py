"""Structural equality analysis and sparse LDL scheduling."""

from dataclasses import dataclass

import sympy

BooleanPattern = tuple[tuple[bool, ...], ...]
GramContribution = tuple[int, int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class LDLColumn:
    """Describe the operations emitted for one LDL factor column.

    Attributes
    ----------
    diagonal_terms
        Earlier factor columns contributing to this pivot.
    rows
        Lower rows and their earlier shared factor columns.
    """

    diagonal_terms: tuple[int, ...]
    rows: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class SparseLDLProgram:
    """Describe the fill pattern and numerical LDL execution schedule.

    Attributes
    ----------
    permutation
        Factor-order to original-row permutation.
    pattern
        Filled lower-factor structural pattern.
    columns
        Numerical update schedule for each factor column.
    """

    permutation: tuple[int, ...]
    pattern: BooleanPattern
    columns: tuple[LDLColumn, ...]

    @property
    def dimension(self) -> int:
        """Return the factorization dimension."""
        return len(self.permutation)


@dataclass(frozen=True, slots=True)
class EqualitySparsityIR:
    """Collect structural data used by equality C emitters.

    Attributes
    ----------
    jacobian_pattern
        Structural nonzeros of the equality Jacobian.
    gram_contributions
        Gram entries and Jacobian columns contributing to them.
    ldl
        Fill-reduced factorization schedule for the Gram matrix.
    """

    jacobian_pattern: BooleanPattern
    gram_contributions: tuple[GramContribution, ...]
    ldl: SparseLDLProgram


def analyze_equality_sparsity(jacobian: sympy.ImmutableDenseMatrix) -> EqualitySparsityIR:
    """Derive Gram assembly and LDL schedules from a Jacobian structure."""
    jacobian_pattern = _jacobian_pattern(jacobian)
    gram_pattern = _gram_pattern(jacobian_pattern)
    permutation = _minimum_degree_order(gram_pattern)
    return EqualitySparsityIR(
        jacobian_pattern=jacobian_pattern,
        gram_contributions=_gram_contributions(jacobian_pattern),
        ldl=_ldl_program(gram_pattern, permutation),
    )


def _jacobian_pattern(jacobian: sympy.ImmutableDenseMatrix) -> BooleanPattern:
    rows, columns = jacobian.shape
    return tuple(tuple(not (jacobian[row, column] == 0 or jacobian[row, column].is_zero is True) for column in range(columns)) for row in range(rows))


def _gram_contributions(jacobian: BooleanPattern) -> tuple[GramContribution, ...]:
    contributions = []
    for row, left in enumerate(jacobian):
        for column, right in enumerate(jacobian[: row + 1]):
            shared = tuple(index for index, pair in enumerate(zip(left, right, strict=True)) if all(pair))
            if row == column or shared:
                contributions.append((row, column, shared))
    return tuple(contributions)


def _gram_pattern(jacobian: BooleanPattern) -> BooleanPattern:
    return tuple(
        tuple(row == column or any(left and right for left, right in zip(jacobian[row], jacobian[column], strict=True)) for column in range(len(jacobian)))
        for row in range(len(jacobian))
    )


def _minimum_degree_order(pattern: BooleanPattern) -> tuple[int, ...]:
    remaining = set(range(len(pattern)))
    graph = {row: {column for column in remaining if row != column and pattern[row][column]} for row in remaining}
    order = []

    while remaining:
        node = min(remaining, key=lambda item: (len(graph[item] & remaining), item))
        neighbors = list(graph[node] & remaining)
        for offset, left in enumerate(neighbors):
            for right in neighbors[offset + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
        remaining.remove(node)
        order.append(node)

    return tuple(order)


def _ldl_program(pattern: BooleanPattern, permutation: tuple[int, ...]) -> SparseLDLProgram:
    dimension = len(permutation)
    filled = [[pattern[permutation[row]][permutation[column]] for column in range(dimension)] for row in range(dimension)]

    for column in range(dimension):
        neighbors = [row for row in range(column + 1, dimension) if filled[row][column]]
        for offset, left in enumerate(neighbors):
            for right in neighbors[offset + 1 :]:
                filled[left][right] = filled[right][left] = True

    columns = []
    for column in range(dimension):
        diagonal_terms = tuple(previous for previous in range(column) if filled[column][previous])
        rows = tuple(
            (
                row,
                tuple(previous for previous in range(column) if filled[row][previous] and filled[column][previous]),
            )
            for row in range(column + 1, dimension)
            if filled[row][column]
        )
        columns.append(LDLColumn(diagonal_terms, rows))

    return SparseLDLProgram(permutation, tuple(tuple(row) for row in filled), tuple(columns))
