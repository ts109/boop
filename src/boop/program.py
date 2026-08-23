"""Top-level symbolic solver program construction."""

from dataclasses import dataclass

from .ir import EvaluatorIR, build_evaluator_ir
from .model import NonlinearProgram
from .sparsity import EqualitySparsityIR, analyze_equality_sparsity


@dataclass(frozen=True, slots=True)
class GeneratedProgram:
    """Bundle all target-independent information consumed by code generation.

    Attributes
    ----------
    nlp
        Canonical symbolic user input.
    evaluator
        Objective, constraints, bounds, and required derivatives.
    equality_sparsity
        Structural Gram assembly and sparse LDL schedules.
    has_nonlinear_equalities
        Whether a second-order correction can be useful.
    """

    nlp: NonlinearProgram
    evaluator: EvaluatorIR
    equality_sparsity: EqualitySparsityIR
    has_nonlinear_equalities: bool


def generate_program(nlp: NonlinearProgram) -> GeneratedProgram:
    """Build the target-independent IR and numerical execution schedules."""
    evaluator = build_evaluator_ir(nlp)
    variables = set(nlp.x)
    nonlinear_equalities = any(entry.free_symbols & variables for entry in evaluator.equality_jacobian)
    return GeneratedProgram(
        nlp=nlp,
        evaluator=evaluator,
        equality_sparsity=analyze_equality_sparsity(evaluator.equality_jacobian),
        has_nonlinear_equalities=nonlinear_equalities,
    )
