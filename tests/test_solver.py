import numpy
import sympy

from boop import NonlinearProgram, Procedure, SolverOptions, create_solver


def test_equality_constrained_quadratic():
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(
        x=(x, y),
        f=(x - sympy.Rational(1, 2)) ** 2 + (y - sympy.Rational(1, 2)) ** 2,
        g=(x + y - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    solver = create_solver(model, SolverOptions(sqp_iterations=6, cg_iterations=4))
    result = solver([0.8, 0.8], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-8)
    assert len(result.diagnostics.iterations) == 6
    assert result.diagnostics.iterations[0].accepted


def test_active_upper_bound_is_identified_and_enforced():
    x = sympy.symbols("x")
    model = NonlinearProgram(x=(x,), f=(x - 2) ** 2, lb=(0,), ub=(1,))
    solver = create_solver(
        model,
        SolverOptions(sqp_iterations=8, cg_iterations=3, activation_delay=1),
    )
    result = solver([0.0], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [1.0], atol=1e-8)
    assert any(item.active_bounds[0] == 1 for item in result.diagnostics.iterations)


def test_steihaug_handles_negative_curvature_at_boundary():
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(x=(x, y), f=-(x**2) + y**2)
    options = SolverOptions(sqp_iterations=1, cg_iterations=5, initial_trust_radius=0.25)
    result = create_solver(model, options)([0.1, 0.1], diagnostics=True)
    step = result.x - numpy.array([0.1, 0.1])
    assert numpy.linalg.norm(step) <= 0.25 + 1e-12
    assert result.diagnostics.iterations[0].procedure in {
        Procedure.BYRD_OMOJOKUN,
        Procedure.BYRD_OMOJOKUN_SOC,
    }


def test_nonlinear_equality_with_second_order_correction():
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(
        x=(x, y),
        f=x**2 + y**2,
        g=(x**2 + y - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    result = create_solver(
        model,
        SolverOptions(sqp_iterations=15, cg_iterations=8, initial_trust_radius=0.5),
    )([0.8, 0.4], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [numpy.sqrt(0.5), 0.5], atol=2e-5)
    assert abs(result.x[0] ** 2 + result.x[1] - 1.0) < 1e-10
