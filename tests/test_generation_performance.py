"""Guard generation paths against expensive symbolic graph transformations."""

from pathlib import Path
from typing import Never

import numpy
import pytest
import sympy

from boop import NonlinearProgram, create_compiled_solver, generate_c_solver, generate_program
from boop.c_evaluator import _shared_subexpressions


def test_generation_avoids_symbolic_simplification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject accidental reintroduction of simplification or algebraic CSE."""
    def forbidden(*_args: object, **_kwargs: object) -> Never:
        message = "symbolic simplification is forbidden"
        raise AssertionError(message)

    monkeypatch.setattr(sympy, "simplify", forbidden)
    monkeypatch.setattr(sympy, "cse", forbidden)
    x, y, p = sympy.symbols("x y p")
    nlp = NonlinearProgram(x=(x, y), f=(x - sympy.sin(y))**2,
                           g=(x + y,), lb=(-p, -1), ub=(p, 1), parameters=(p,))
    source = generate_c_solver(nlp).sources["boop_problem.c"]
    assert "temporary_" in source
    assert "model->hessian" in source


def test_sparse_derivatives_match_sympy() -> None:
    """Compare sparse Hessian construction against independent derivatives."""
    x = sympy.symbols("x:8")
    f = sum((x[i] - sympy.sin(x[i+1]))**2 for i in range(7))
    g = (x[0]**2 + x[1], sympy.exp(x[5]) - x[6])
    ir = generate_program(NonlinearProgram(x=x, f=f, g=g)).evaluator
    values = dict(zip(x, (0.1, 0.2, -0.3, 0.5, 1, -1, 0.7, 0.8), strict=True))
    numpy.testing.assert_allclose(numpy.array(ir.hessian.subs(values), dtype=float),
                                  numpy.array(sympy.hessian(f, x).subs(values), dtype=float))
    assert ir.equality_jacobian == sympy.ImmutableDenseMatrix(g).jacobian(x)


def test_structural_sharing_is_topological_and_keeps_branches_local() -> None:
    """Keep dependency ordering and conditional domain restrictions intact."""
    x = sympy.Symbol("x")
    inner = sympy.sin(x)
    outer = sympy.exp(inner)
    assert _shared_subexpressions([outer + inner, outer]) == [inner, outer]
    branch = sympy.Piecewise((sympy.log(x) + sympy.log(x)**2, x > 0), (0, True))
    assert _shared_subexpressions([branch]) == []


@pytest.mark.parametrize("share", [False, True])
def test_sharing_modes_compile_and_solve(tmp_path: Path, share: bool) -> None:
    """Validate both emission modes through the native solver."""
    # Input names deliberately overlap generated aliases.
    x, y = sympy.symbols("boop_x_1 boop_x_0")
    nlp = NonlinearProgram(x=(x, y), f=(x-1)**2 + (y+2)**2)
    solver = create_compiled_solver(nlp, cache_directory=tmp_path, share_subexpressions=share)
    numpy.testing.assert_allclose(solver([4, -5]), [1, -2], atol=1e-9)
