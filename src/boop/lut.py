"""Fixed-shape symbolic lookup tables."""

from dataclasses import dataclass, field
from numbers import Number

import sympy
from sympy.core.function import ArgumentIndexError


class LUTCall(sympy.Function):
    """Internal table identity, derivative orders, and query coordinates."""

    def fdiff(self, argindex: int = 4) -> sympy.Expr:
        """Return a symbolic partial derivative with respect to a query."""
        if not 4 <= argindex <= len(self.args):
            raise ArgumentIndexError(self, argindex)
        identity, dx, dy, *queries = self.args
        orders = [dx, dy]
        orders[argindex - 4] += 1
        return LUTCall(identity, *orders, *queries)


@dataclass(frozen=True, eq=False, slots=True)
class LUT:
    """Declare a scalar natural cubic spline on a 1D or 2D rectilinear grid.

    Runtime inputs follow declaration order in ``NonlinearProgram.luts``.
    Values outside the grid extend the boundary polynomial.
    """

    shape: tuple[int, ...]
    identity: sympy.Dummy = field(default_factory=lambda: sympy.Dummy("lut"), init=False, repr=False)

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if len(shape) not in (1, 2) or any(type(n) is not int or n < 2 for n in shape):
            message = "LUT shape must contain one or two integer sizes >= 2"
            raise ValueError(message)
        object.__setattr__(self, "shape", shape)

    def __call__(self, *queries: sympy.Expr | Number) -> sympy.Expr:
        """Evaluate the table symbolically at the supplied coordinates."""
        if len(queries) != len(self.shape):
            message = "LUT query count must match its shape"
            raise ValueError(message)
        return LUTCall(self.identity, 0, 0, *map(sympy.sympify, queries))
