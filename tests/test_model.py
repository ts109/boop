import pytest
import sympy

from boop import NonlinearProgram


def test_model_rejects_general_or_undeclared_symbols():
    x, y = sympy.symbols("x y")
    with pytest.raises(ValueError, match="undeclared"):
        NonlinearProgram(x=(x,), f=x + y)


def test_model_validates_bound_dimensions():
    x, y = sympy.symbols("x y")
    with pytest.raises(ValueError, match="one entry"):
        NonlinearProgram(x=(x, y), f=x, lb=(0,), ub=(1, 1))


def test_model_accepts_parameterized_bounds():
    x, limit = sympy.symbols("x limit")
    model = NonlinearProgram(x=(x,), f=x**2, lb=(-limit,), ub=(limit,), parameters=(limit,))
    assert model.dimension == 1
    assert model.equality_dimension == 0
