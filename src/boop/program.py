"""Generated evaluator and sparse LDL program representation."""

from dataclasses import dataclass

import sympy

from .model import NonlinearProgram


@dataclass(frozen=True, slots=True)
class EvaluatorIR:
    """Store the symbolic expressions defining an NLP evaluator.

    Attributes
    ----------
    objective
        Scalar objective expression.
    equalities
        Column matrix of equality residual expressions.
    gradient
        Objective-gradient column matrix.
    hessian
        Objective-Hessian matrix.
    equality_jacobian
        Jacobian matrix of the equality residuals.
    lower_bounds
        Column matrix of variable lower-bound expressions.
    upper_bounds
        Column matrix of variable upper-bound expressions.
    """

    objective: sympy.Expr
    equalities: sympy.ImmutableDenseMatrix
    gradient: sympy.ImmutableDenseMatrix
    hessian: sympy.ImmutableDenseMatrix
    equality_jacobian: sympy.ImmutableDenseMatrix
    lower_bounds: sympy.ImmutableDenseMatrix
    upper_bounds: sympy.ImmutableDenseMatrix


@dataclass(frozen=True, slots=True)
class LDLColumn:
    """Describe the generated operations for one LDL factorization column.

    Attributes
    ----------
    diagonal_terms
        Earlier columns contributing to the diagonal update.
    rows
        Pairs containing a lower-triangular row and the earlier columns shared
        by that row and this column.
    """

    diagonal_terms: tuple[int, ...]
    rows: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class SparseLDLProgram:
    """Describe a symbolic fill pattern and numerical LDL execution schedule.

    Attributes
    ----------
    permutation
        Fill-reducing permutation from factor order to original matrix order.
    pattern
        Symmetric structural nonzero pattern after symbolic fill-in.
    columns
        Per-column numerical factorization schedules.
    """

    permutation: tuple[int, ...]
    pattern: tuple[tuple[bool, ...], ...]
    columns: tuple[LDLColumn, ...]

    @property
    def dimension(self) -> int:
        """Return the factorization dimension."""
        return len(self.permutation)


class GeneratedProgram:
    """Problem-specific symbolic evaluator and sparse linear-algebra schedule."""

    def __init__(self, nlp: NonlinearProgram) -> None:
        self.nlp = nlp

        # Derivatives remain symbolic until this evaluator is instantiated.
        variables = sympy.ImmutableDenseMatrix(nlp.dimension, 1, nlp.x)
        equalities = sympy.ImmutableDenseMatrix(nlp.equality_dimension, 1, nlp.g)
        objective = nlp.f
        gradient = sympy.ImmutableDenseMatrix([sympy.diff(objective, item) for item in nlp.x])
        hessian = sympy.ImmutableDenseMatrix(sympy.hessian(objective, nlp.x))
        jacobian = sympy.ImmutableDenseMatrix(equalities.jacobian(variables)) if nlp.equality_dimension else sympy.ImmutableDenseMatrix.zeros(0, nlp.dimension)
        self.ir = EvaluatorIR(
            objective=objective,
            equalities=equalities,
            gradient=gradient,
            hessian=hessian,
            equality_jacobian=jacobian,
            lower_bounds=sympy.ImmutableDenseMatrix(nlp.dimension, 1, nlp.lb),
            upper_bounds=sympy.ImmutableDenseMatrix(nlp.dimension, 1, nlp.ub),
        )
        # Only free columns shared by two equality rows contribute to E_F E_F.T.
        self.jacobian_pattern = _jacobian_pattern(jacobian)
        variables_set = set(nlp.x)
        self.has_nonlinear_equalities = any(entry.free_symbols & variables_set for entry in jacobian)
        self.gram_contributions = _gram_contributions(self.jacobian_pattern)
        gram_pattern = _gram_pattern(self.jacobian_pattern)
        permutation = _minimum_degree_order(gram_pattern)
        self.ldl = _ldl_program(gram_pattern, permutation)


def _structurally_zero(expression: sympy.Expr) -> bool:
    return expression == 0 or expression.is_zero is True


def _jacobian_pattern(jacobian: sympy.ImmutableDenseMatrix) -> tuple[tuple[bool, ...], ...]:
    rows, columns = jacobian.shape
    return tuple(tuple(not _structurally_zero(jacobian[i, j]) for j in range(columns)) for i in range(rows))


def _gram_contributions(jacobian: tuple[tuple[bool, ...], ...]) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    contributions: list[tuple[int, int, tuple[int, ...]]] = []

    for i, row_i in enumerate(jacobian):
        for j, row_j in enumerate(jacobian[: i + 1]):
            shared_columns = tuple(k for k, (left, right) in enumerate(zip(row_i, row_j, strict=True)) if left and right)
            if i == j or shared_columns:
                contributions.append((i, j, shared_columns))

    return tuple(contributions)


def _gram_pattern(jacobian: tuple[tuple[bool, ...], ...]) -> tuple[tuple[bool, ...], ...]:
    m = len(jacobian)

    if not m:
        return ()

    return tuple(tuple(i == j or any(a and b for a, b in zip(jacobian[i], jacobian[j], strict=True)) for j in range(m)) for i in range(m))


def _minimum_degree_order(pattern: tuple[tuple[bool, ...], ...]) -> tuple[int, ...]:
    remaining = set(range(len(pattern)))
    graph = {i: {j for j in remaining if i != j and pattern[i][j]} for i in remaining}
    order: list[int] = []

    while remaining:
        # Eliminate the least connected row; connect its future neighbors.
        node = min(remaining, key=lambda item: (len(graph[item] & remaining), item))
        neighbors = list(graph[node] & remaining)

        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)

        remaining.remove(node)
        order.append(node)

    return tuple(order)


def _ldl_program(pattern: tuple[tuple[bool, ...], ...], permutation: tuple[int, ...]) -> SparseLDLProgram:
    n = len(permutation)
    filled = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            filled[i][j] = pattern[permutation[i]][permutation[j]]

    for k in range(n):
        # Symbolic elimination adds the clique induced by column k.
        neighbors = [i for i in range(k + 1, n) if filled[i][k]]

        for offset, left in enumerate(neighbors):
            for right in neighbors[offset + 1 :]:
                filled[left][right] = filled[right][left] = True

    columns: list[LDLColumn] = []

    for k in range(n):
        diagonal_terms = tuple(j for j in range(k) if filled[k][j])
        rows = []

        for i in range(k + 1, n):
            if filled[i][k]:
                common = tuple(j for j in range(k) if filled[i][j] and filled[k][j])
                rows.append((i, common))

        columns.append(LDLColumn(diagonal_terms=diagonal_terms, rows=tuple(rows)))

    return SparseLDLProgram(permutation, tuple(tuple(row) for row in filled), tuple(columns))
