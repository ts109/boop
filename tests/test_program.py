"""Tests for generated evaluators and sparse LDL programs."""

import numpy
import sympy

from boop import GeneratedProgram, NonlinearProgram


def test_generated_sparse_ldl_matches_numpy_solve() -> None:
    """Match NumPy when solving with a generated sparse LDL schedule."""
    x = sympy.symbols("x:5")
    model = NonlinearProgram(
        x=x,
        f=sum(item**2 for item in x),
        g=(x[0] + x[1], x[1] + x[2], x[3] + x[4]),
    )
    program = GeneratedProgram(model)
    values = program.evaluate(numpy.arange(1.0, 6.0), numpy.empty(0))
    free = numpy.array([True, False, True, True, True])
    gram = program.assemble_equality_gram(values.equality_jacobian, free, 0.25)
    factor = program.ldl.factor(gram, 1e-14)
    rhs = numpy.array([1.0, -2.0, 3.0])
    numpy.testing.assert_allclose(factor.solve(rhs), numpy.linalg.solve(gram, rhs), rtol=1e-12, atol=1e-12)


def test_generated_derivatives_and_parameters() -> None:
    """Evaluate generated derivatives and parameterized bounds correctly."""
    x, y, target = sympy.symbols("x y target")
    model = NonlinearProgram(
        x=(x, y),
        f=(x - target) ** 2 + y**2,
        g=(x + y - target,),
        lb=(0, -sympy.oo),
        ub=(target, sympy.oo),
        parameters=(target,),
    )
    values = GeneratedProgram(model).evaluate(numpy.array([1.0, 2.0]), numpy.array([3.0]))
    assert values.objective == 8.0
    numpy.testing.assert_allclose(values.gradient, [-4.0, 4.0])
    numpy.testing.assert_allclose(values.hessian, 2.0 * numpy.eye(2))
    numpy.testing.assert_allclose(values.equality_jacobian, [[1.0, 1.0]])
    numpy.testing.assert_allclose(values.upper_bounds, [3.0, numpy.inf])
