"""Tests for the detailed solver report."""

from io import StringIO

import numpy
import sympy

from boop import NonlinearProgram, SolverDiagnostics, SolverOptions, create_solver, format_solver_diagnostics, print_solver_diagnostics


def test_detailed_report_exposes_numerical_history() -> None:
    """Report model, linear algebra, working-set, CG, filter, and trial data."""
    x0, x1 = sympy.symbols("x0 x1")
    problem = NonlinearProgram(
        x=(x0, x1),
        f=(x0 - 1) ** 2 + (x1 - 1) ** 2,
        g=(x0**2 + x1 - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    solver = create_solver(problem, SolverOptions(sqp_iterations=4))
    result = solver(numpy.array([-3.0, 2.0]), diagnostics=True)

    report = format_solver_diagnostics(result.diagnostics)

    assert "Boop solver diagnostics" in report
    assert "initial projection norm" in report
    assert "diagnostic flags" in report
    assert "equality LDL pivots" in report
    assert "Steihaug CG" in report
    assert "filter trials" in report
    assert "active lower / upper" in report
    assert len(result.diagnostics.iterations) == 4
    assert result.diagnostics.initial_guess.tolist() == [-3.0, 2.0]
    assert result.diagnostics.projected_initial_guess.tolist() == [0.0, 2.0]


def test_print_report_accepts_a_stream_and_empty_history() -> None:
    """The printer supports capture and a deliberately empty diagnostic set."""
    diagnostics = SolverDiagnostics()
    stream = StringIO()

    print_solver_diagnostics(diagnostics, stream=stream)

    assert stream.getvalue() == "Boop diagnostics: no SQP iterations recorded\n"
