"""Generated evaluator and sparse LDL program representation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy
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

    def factor(self, matrix: numpy.ndarray, tolerance: float) -> LDLFactor:
        """Numerically factor a matrix using the generated sparse schedule."""
        n = self.dimension

        if n == 0:
            return LDLFactor(self, numpy.empty((0, 0)), numpy.empty(0))

        permuted = matrix[numpy.ix_(self.permutation, self.permutation)]
        lower = numpy.eye(n)
        diagonal = numpy.empty(n)

        for k, column in enumerate(self.columns):
            # Numeric LDL follows the fill schedule found during generation.
            diagonal_update = sum(lower[k, j] ** 2 * diagonal[j] for j in column.diagonal_terms)
            diagonal[k] = permuted[k, k] - diagonal_update

            if not numpy.isfinite(diagonal[k]) or diagonal[k] <= tolerance:
                message = f"non-positive LDL pivot {k}: {diagonal[k]}"
                raise numpy.linalg.LinAlgError(message)

            for i, common in column.rows:
                row_update = sum(lower[i, j] * lower[k, j] * diagonal[j] for j in common)
                lower[i, k] = (permuted[i, k] - row_update) / diagonal[k]

        return LDLFactor(self, lower, diagonal)


@dataclass(frozen=True, slots=True)
class LDLFactor:
    """Store a numerical factorization produced by :class:`SparseLDLProgram`.

    Attributes
    ----------
    program
        Symbolic program that determined the permutation and factor pattern.
    lower
        Unit lower-triangular factor in permuted order.
    diagonal
        Diagonal factor in permuted order.
    """

    program: SparseLDLProgram
    lower: numpy.ndarray
    diagonal: numpy.ndarray

    def solve(self, rhs: numpy.ndarray) -> numpy.ndarray:
        """Solve a linear system using this numerical factorization."""
        if self.program.dimension == 0:
            return numpy.empty_like(rhs)

        permutation = numpy.asarray(self.program.permutation)
        inverse = numpy.argsort(permutation)
        work = numpy.asarray(rhs, dtype=float)[permutation].copy()
        n = len(work)

        # Solve L z = P b, D y = z, and L.T w = y.
        for i in range(n):
            work[i] -= self.lower[i, :i] @ work[:i]

        work /= self.diagonal

        for i in range(n - 1, -1, -1):
            work[i] -= self.lower[i + 1 :, i] @ work[i + 1 :]

        return cast(numpy.ndarray, work[inverse])


@dataclass(slots=True)
class ModelValues:
    """Store one numerical evaluation of the nonlinear program.

    Attributes
    ----------
    objective
        Objective value.
    equalities
        Equality residual vector.
    gradient
        Objective-gradient vector.
    hessian
        Objective-Hessian matrix.
    equality_jacobian
        Equality-Jacobian matrix.
    lower_bounds
        Evaluated variable lower bounds.
    upper_bounds
        Evaluated variable upper bounds.
    """

    objective: float
    equalities: numpy.ndarray
    gradient: numpy.ndarray
    hessian: numpy.ndarray
    equality_jacobian: numpy.ndarray
    lower_bounds: numpy.ndarray
    upper_bounds: numpy.ndarray


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
        expressions = (
            objective,
            equalities,
            gradient,
            hessian,
            jacobian,
            self.ir.lower_bounds,
            self.ir.upper_bounds,
        )
        self._evaluator = sympy.lambdify((*nlp.x, *nlp.parameters), expressions, modules="numpy", cse=True)
        self._equality_hessian_evaluator: Callable[..., object] | None = None

        # Only free columns shared by two equality rows contribute to E_F E_F.T.
        self.jacobian_pattern = _jacobian_pattern(jacobian)
        self.gram_contributions = _gram_contributions(self.jacobian_pattern)
        gram_pattern = _gram_pattern(self.jacobian_pattern)
        permutation = _minimum_degree_order(gram_pattern)
        self.ldl = _ldl_program(gram_pattern, permutation)

    def assemble_equality_gram(
        self,
        jacobian: numpy.ndarray,
        free: numpy.ndarray,
        regularization: float,
    ) -> numpy.ndarray:
        """Assemble ``E_F E_F.T + regularization*I`` using generated sparsity."""
        m = self.nlp.equality_dimension
        gram = numpy.zeros((m, m))

        for i, j, columns in self.gram_contributions:
            # The active box set appears only through this free-column mask.
            value = sum(jacobian[i, k] * jacobian[j, k] for k in columns if free[k])
            gram[i, j] = gram[j, i] = value

        if m:
            gram.flat[:: m + 1] += regularization

        return gram

    def evaluate(self, x: numpy.ndarray, parameters: numpy.ndarray) -> ModelValues:
        """Evaluate the generated model and its derivatives numerically."""
        raw = self._evaluator(*numpy.asarray(x, dtype=float), *numpy.asarray(parameters, dtype=float))
        n = self.nlp.dimension
        m = self.nlp.equality_dimension
        values = ModelValues(
            objective=float(numpy.asarray(raw[0]).reshape(())),
            equalities=numpy.asarray(raw[1], dtype=float).reshape(m),
            gradient=numpy.asarray(raw[2], dtype=float).reshape(n),
            hessian=numpy.asarray(raw[3], dtype=float).reshape(n, n),
            equality_jacobian=numpy.asarray(raw[4], dtype=float).reshape(m, n),
            lower_bounds=numpy.asarray(raw[5], dtype=float).reshape(n),
            upper_bounds=numpy.asarray(raw[6], dtype=float).reshape(n),
        )

        if numpy.any(values.lower_bounds > values.upper_bounds):
            message = "evaluated lower bound exceeds its upper bound"
            raise ValueError(message)

        arrays = (
            numpy.asarray(values.objective),
            values.equalities,
            values.gradient,
            values.hessian,
            values.equality_jacobian,
        )

        if not all(numpy.all(numpy.isfinite(item)) for item in arrays):
            message = "NLP evaluation produced a non-finite value or derivative"
            raise FloatingPointError(message)

        if numpy.any(numpy.isnan(values.lower_bounds)) or numpy.any(numpy.isnan(values.upper_bounds)):
            message = "NLP bounds evaluated to NaN"
            raise FloatingPointError(message)

        return values

    def evaluate_equality_hessians(self, x: numpy.ndarray, parameters: numpy.ndarray) -> numpy.ndarray:
        """Evaluate equality Hessians only when second-order validation needs them."""
        if self._equality_hessian_evaluator is None:
            expressions = tuple(sympy.ImmutableDenseMatrix(sympy.hessian(item, self.nlp.x)) for item in self.nlp.g)
            self._equality_hessian_evaluator = sympy.lambdify(
                (*self.nlp.x, *self.nlp.parameters),
                expressions,
                modules="numpy",
                cse=True,
            )
        raw = self._equality_hessian_evaluator(*numpy.asarray(x, dtype=float), *numpy.asarray(parameters, dtype=float))
        return numpy.asarray(raw, dtype=float).reshape(self.nlp.equality_dimension, self.nlp.dimension, self.nlp.dimension)


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
