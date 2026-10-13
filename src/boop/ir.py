"""Symbolic evaluator intermediate representation."""

from dataclasses import dataclass

import sympy

from .model import NonlinearProgram


@dataclass(frozen=True, slots=True)
class EvaluatorIR:
    """Store the symbolic expressions evaluated by the generated solver.

    Attributes
    ----------
    objective
        Scalar objective expression.
    equalities
        Equality residual column vector.
    gradient, hessian
        First and second objective derivatives.
    equality_jacobian
        First equality derivatives.
    lower_bounds, upper_bounds
        Symbolic box-bound column vectors.
    """

    objective: sympy.Expr
    equalities: sympy.ImmutableDenseMatrix
    gradient: sympy.ImmutableDenseMatrix
    hessian: sympy.ImmutableDenseMatrix
    equality_jacobian: sympy.ImmutableDenseMatrix
    lower_bounds: sympy.ImmutableDenseMatrix
    upper_bounds: sympy.ImmutableDenseMatrix


def build_evaluator_ir(nlp: NonlinearProgram) -> EvaluatorIR:
    """Differentiate an NLP into the expressions needed by the C runtime."""
    equalities = sympy.ImmutableDenseMatrix(nlp.g)
    # Cache symbol dependencies per expression, including repeated stage terms.
    dependencies: dict[sympy.Expr, set[sympy.Symbol]] = {}

    def derivative(expression: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr:
        if expression not in dependencies:
            dependencies[expression] = expression.free_symbols
        if variable not in dependencies[expression]:
            return sympy.S.Zero
        return sympy.diff(expression, variable, simplify=False)

    gradient = [derivative(nlp.f, variable) for variable in nlp.x]
    # Reuse first derivatives and symmetry; never differentiate the full
    # objective a second time or request mixed-derivative simplification.
    hessian = [[sympy.S.Zero] * nlp.dimension for _ in nlp.x]
    for row, first in enumerate(gradient):
        for column in range(row, nlp.dimension):
            entry = derivative(first, nlp.x[column])
            hessian[row][column] = hessian[column][row] = entry
    jacobian = sympy.ImmutableDenseMatrix(
        len(nlp.g), nlp.dimension,
        [derivative(expression, variable) for expression in nlp.g for variable in nlp.x],
    )

    return EvaluatorIR(
        objective=nlp.f,
        equalities=equalities,
        gradient=sympy.ImmutableDenseMatrix(gradient),
        hessian=sympy.ImmutableDenseMatrix(nlp.dimension, nlp.dimension, [entry for row in hessian for entry in row]),
        equality_jacobian=sympy.ImmutableDenseMatrix(jacobian),
        lower_bounds=sympy.ImmutableDenseMatrix(nlp.lb),
        upper_bounds=sympy.ImmutableDenseMatrix(nlp.ub),
    )
