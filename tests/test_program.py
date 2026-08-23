"""Tests for symbolic evaluator and sparse-LDL planning."""

import sympy

from boop import GeneratedProgram, NonlinearProgram


def test_sparse_ldl_schedule_reflects_equality_structure() -> None:
    """Build a valid fill schedule without performing numerical work in Python."""
    x = sympy.symbols("x:5")
    program = GeneratedProgram(
        NonlinearProgram(
            x=x,
            f=sum(item**2 for item in x),
            g=(x[0] + x[1], x[1] + x[2], x[3] + x[4]),
        )
    )

    assert program.jacobian_pattern == (
        (True, True, False, False, False),
        (False, True, True, False, False),
        (False, False, False, True, True),
    )
    assert sorted(program.ldl.permutation) == [0, 1, 2]
    assert program.ldl.dimension == 3


def test_generated_ir_contains_derivatives_and_parameterized_bounds() -> None:
    """Keep evaluator inputs and symbolic derivatives in the generated IR."""
    x, y, target = sympy.symbols("x y target")
    program = GeneratedProgram(
        NonlinearProgram(
            x=(x, y),
            f=(x - target) ** 2 + y**2,
            g=(x + y - target,),
            lb=(0, -sympy.oo),
            ub=(target, sympy.oo),
            parameters=(target,),
        )
    )

    assert program.ir.gradient == sympy.ImmutableDenseMatrix((2 * x - 2 * target, 2 * y))
    assert program.ir.hessian == 2 * sympy.eye(2)
    assert program.ir.equality_jacobian == sympy.ImmutableDenseMatrix(((1, 1),))
    assert program.ir.upper_bounds == sympy.ImmutableDenseMatrix((target, sympy.oo))
