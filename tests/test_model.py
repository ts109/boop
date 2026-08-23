"""Tests for symbolic nonlinear-program validation."""

import pytest
import sympy

from boop import NonlinearProgram


def test_model_rejects_general_or_undeclared_symbols() -> None:
    """Reject expressions containing symbols absent from the NLP signature."""
    x, y = sympy.symbols("x y")
    with pytest.raises(ValueError, match="undeclared"):
        NonlinearProgram(x=(x,), f=x + y)


def test_model_validates_bound_dimensions() -> None:
    """Require one lower and upper bound per decision variable."""
    x, y = sympy.symbols("x y")
    with pytest.raises(ValueError, match="one entry"):
        NonlinearProgram(x=(x, y), f=x, lb=(0,), ub=(1, 1))


def test_model_accepts_parameterized_bounds() -> None:
    """Accept bounds that depend on declared NLP parameters."""
    x, limit = sympy.symbols("x limit")
    model = NonlinearProgram(x=(x,), f=x**2, lb=(-limit,), ub=(limit,), parameters=(limit,))
    assert model.dimension == 1
    assert model.equality_dimension == 0
