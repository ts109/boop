"""Tests for numerical SQP behavior."""

import numpy
import pytest
import sympy

from boop import NonlinearProgram, SolverOptions, create_compiled_solver


def test_equality_constrained_quadratic() -> None:
    """Solve a well-conditioned equality-constrained quadratic."""
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(
        x=(x, y),
        f=(x - sympy.Rational(1, 2)) ** 2 + (y - sympy.Rational(1, 2)) ** 2,
        g=(x + y - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    solver = create_compiled_solver(model, SolverOptions(sqp_iterations=6, cg_iterations=4))
    result = solver([0.8, 0.8], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-8)
    assert len(result.diagnostics.procedure) == 6
    assert result.diagnostics.accepted[0]
    assert set(result.diagnostics.procedure) <= {"byrd-omojukun", "rejected"}


def test_active_upper_bound_is_identified_and_enforced() -> None:
    """Identify an optimal upper bound and enforce it exactly."""
    x = sympy.symbols("x")
    model = NonlinearProgram(x=(x,), f=(x - 2) ** 2, lb=(0,), ub=(1,))
    solver = create_compiled_solver(
        model,
        SolverOptions(sqp_iterations=8, cg_iterations=3),
    )
    result = solver([0.0], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [1.0], atol=1e-8)
    assert any(item[0] == 1 for item in result.diagnostics.active_bounds)


def test_blocking_bound_consumes_an_iteration_without_moving() -> None:
    """Keep numerical state fixed during an active-set transition."""
    x = sympy.symbols("x")
    model = NonlinearProgram(x=(x,), f=(x - 2) ** 2, ub=(1,))
    solver = create_compiled_solver(model, SolverOptions(sqp_iterations=2, initial_trust_radius=1.0))
    result = solver([0.0], diagnostics=True)

    assert result.diagnostics.procedure[0] == "active-set-update"
    assert not result.diagnostics.accepted[0]
    numpy.testing.assert_array_equal(result.diagnostics.x[0], [0.0])
    assert result.diagnostics.trust_radius[0] == 1.0
    assert result.diagnostics.active_bounds[0][0] == 1
    assert result.diagnostics.accepted[1]
    numpy.testing.assert_allclose(result.x, [1.0])


def test_bound_is_not_released_before_its_face_is_stationary() -> None:
    """Finish the reduced solve before using multipliers to change faces."""
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(
        x=(x, y),
        f=100 * (y - x**2) ** 2 + (1 - x) ** 2,
        lb=(0, 1.5),
    )
    result = create_compiled_solver(model)([1.224958, 1.5], diagnostics=True)

    numpy.testing.assert_allclose(result.x, [1.224370748736353, 1.5], atol=1e-10)
    transitions = [i for i, procedure in enumerate(result.diagnostics.procedure) if procedure == "active-set-update"]
    assert len(transitions) == 1
    assert result.diagnostics.active_bounds[transitions[0]][1] == -1


def test_initial_guess_is_projected_onto_box() -> None:
    """Project an infeasible initial guess before the first SQP iteration."""
    x = sympy.symbols("x")
    model = NonlinearProgram(x=(x,), f=(x - 0.5) ** 2, lb=(0,), ub=(1,))
    result = create_compiled_solver(model, SolverOptions(sqp_iterations=1))([5.0], diagnostics=True)
    assert 0.0 <= result.x[0] <= 1.0
    assert 0.0 <= result.diagnostics.x[0][0] <= 1.0


def test_steihaug_handles_negative_curvature_at_boundary() -> None:
    """Stop Steihaug CG at the trust boundary under negative curvature."""
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(x=(x, y), f=-(x**2) + y**2)
    options = SolverOptions(sqp_iterations=1, cg_iterations=5, initial_trust_radius=0.25)
    result = create_compiled_solver(model, options)([0.1, 0.1], diagnostics=True)
    step = numpy.asarray(result.x) - numpy.array([0.1, 0.1])
    assert numpy.linalg.norm(step) <= 0.25 + 1e-12
    assert result.diagnostics.procedure[0] in {"byrd-omojukun", "byrd-omojukun-soc"}


def test_nonlinear_equality_with_second_order_correction() -> None:
    """Solve a nonlinear equality problem using second-order corrections."""
    x, y = sympy.symbols("x y")
    model = NonlinearProgram(
        x=(x, y),
        f=x**2 + y**2,
        g=(x**2 + y - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    result = create_compiled_solver(
        model,
        SolverOptions(sqp_iterations=15, cg_iterations=8, initial_trust_radius=0.5),
    )([0.8, 0.4], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [numpy.sqrt(0.5), 0.5], atol=2e-5)
    assert abs(result.x[0] ** 2 + result.x[1] - 1.0) < 1e-10
    assert "byrd-omojukun-soc" in result.diagnostics.procedure


def test_tied_bounds_do_not_overconstrain_equality_working_set():
    x, y = sympy.symbols('x y')
    model = NonlinearProgram(x=(x, y), f=(x-2)**2+(y-2)**2,
                             g=(y-x,), lb=(0, 0), ub=(1, 1))
    result = create_compiled_solver(model, SolverOptions(sqp_iterations=12))([0., 0.], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [1., 1.], atol=1e-9)
    assert all(numpy.count_nonzero(active) <= 1 for active in result.diagnostics.active_bounds)
    assert all(numpy.all((numpy.asarray(point) >= 0) & (numpy.asarray(point) <= 1))
               for point in result.diagnostics.x)


def test_full_working_set_keeps_trials_box_feasible_on_infeasible_equalities():
    x, y = sympy.symbols('x y')
    model = NonlinearProgram(x=(x, y), f=(x-2)**2+(y-2)**2,
                             g=(x+y-3,), lb=(0, 0), ub=(1, 1))
    result = create_compiled_solver(model, SolverOptions(sqp_iterations=12))([.5, .5], diagnostics=True)
    assert result.diagnostics.violation[-1] >= 1.
    assert all(numpy.count_nonzero(active) <= 1 for active in result.diagnostics.active_bounds)
    assert all(numpy.all((numpy.asarray(point) >= 0) & (numpy.asarray(point) <= 1))
               for point in result.diagnostics.x)


def test_invalid_fixed_working_set_exposes_failure_state():
    x = sympy.symbols('x')
    model = NonlinearProgram(x=(x,), f=x**2, g=(x-1,), lb=(0,), ub=(0,))
    with pytest.raises(RuntimeError, match='invalid active set') as failed:
        create_compiled_solver(model)([0.])
    assert failed.value.solver_state == dict(status=3, equalities=1, x=[0.], active_bounds=[-1])


def test_dependent_tied_bounds_preserve_equality_rank_below_capacity():
    x, y, z = sympy.symbols('x y z')
    model = NonlinearProgram(x=(x, y, z), f=(x-2)**2+(y+2)**2+z**2,
                             g=(x+y,), lb=(-1, -1, -sympy.oo), ub=(1, 1, sympy.oo))
    result = create_compiled_solver(model, SolverOptions(sqp_iterations=8))([0., 0., 0.], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [1., -1., 0.], atol=1e-8)
    # Two bounds fit the count budget, but fixing both x and y loses the equality's rank.
    assert all(numpy.count_nonzero(active[:2]) <= 1 for active in result.diagnostics.active_bounds)


def test_second_order_correction_stops_at_box_instead_of_being_discarded():
    x, y = sympy.symbols('x y')
    model = NonlinearProgram(x=(x, y), f=-x-y, g=(y-x**2,),
                             lb=(0, -sympy.oo), ub=(sympy.oo, sympy.Rational(351, 1000)))
    result = create_compiled_solver(model, SolverOptions(sqp_iterations=1, initial_trust_radius=numpy.sqrt(.02)))([.5, .25], diagnostics=True)
    numpy.testing.assert_allclose(result.x, [.599, .351], atol=1e-8)
    assert result.diagnostics.procedure == ['byrd-omojukun-soc']
    assert result.diagnostics.violation[0] < .01
