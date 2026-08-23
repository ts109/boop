"""Tests for native, lazily exposed solver diagnostics."""

import sympy

from boop import NonlinearProgram, SolverOptions, create_compiled_solver


def test_native_diagnostics_expose_numerical_history() -> None:
    """Expose each fixed-size native history through an on-demand getter."""
    x0, x1 = sympy.symbols("x0 x1")
    problem = NonlinearProgram(
        x=(x0, x1),
        f=(x0 - 1) ** 2 + (x1 - 1) ** 2,
        g=(x0**2 + x1 - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    result = create_compiled_solver(problem, SolverOptions(sqp_iterations=4))(
        [-3.0, 2.0],
        diagnostics=True,
    )

    diagnostics = result.diagnostics
    assert len(diagnostics.procedure) == 4
    assert len(diagnostics.x) == 4
    assert len(diagnostics.active_bounds) == 4
    assert len(diagnostics.cg_iterations) == 4
    assert all(0.0 <= value <= 2.0 for value in diagnostics.x[0])
    assert all(radius > 0.0 for radius in diagnostics.trust_radius)


def test_compiled_solver_returns_plain_python_lists() -> None:
    """Keep the compiled target independent of third-party array types."""
    x = sympy.symbols("x")
    solver = create_compiled_solver(NonlinearProgram(x=(x,), f=(x - 1) ** 2))

    solution = solver([3.0])

    assert isinstance(solution, list)
    assert all(isinstance(value, float) for value in solution)
