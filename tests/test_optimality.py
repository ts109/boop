"""Tests for strict local-minimum certification."""

import numpy
import pytest
import sympy

from boop import LocalMinimumError, NonlinearProgram, certify_local_minimum, create_solver


def test_certificate_uses_lagrangian_curvature() -> None:
    """Accept the minimum and reject a stationary maximum on one manifold."""
    x, y = sympy.symbols("x y")
    nlp = NonlinearProgram(
        x=(x, y),
        f=x**2 + y**2,
        g=(x**2 + y - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    solver = create_solver(nlp)

    minimum = numpy.array([numpy.sqrt(0.5), 0.5])
    certificate = certify_local_minimum(solver, minimum)
    assert certificate.minimum_reduced_curvature > 0.0

    with pytest.raises(LocalMinimumError, match="minimum reduced curvature"):
        certify_local_minimum(solver, numpy.array([0.0, 1.0]))
