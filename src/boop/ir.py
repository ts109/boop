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
    variables = sympy.ImmutableDenseMatrix(nlp.x)
    equalities = sympy.ImmutableDenseMatrix(nlp.g)
    jacobian = equalities.jacobian(variables) if nlp.g else sympy.zeros(0, nlp.dimension)

    return EvaluatorIR(
        objective=nlp.f,
        equalities=equalities,
        gradient=sympy.ImmutableDenseMatrix([sympy.diff(nlp.f, variable) for variable in nlp.x]),
        hessian=sympy.ImmutableDenseMatrix(sympy.hessian(nlp.f, nlp.x)),
        equality_jacobian=sympy.ImmutableDenseMatrix(jacobian),
        lower_bounds=sympy.ImmutableDenseMatrix(nlp.lb),
        upper_bounds=sympy.ImmutableDenseMatrix(nlp.ub),
    )
