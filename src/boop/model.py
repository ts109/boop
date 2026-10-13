"""Symbolic nonlinear-program definition."""

from dataclasses import dataclass, field
from numbers import Number

import sympy

from .lut import LUT, LUTCall


@dataclass(frozen=True, slots=True)
class NonlinearProgram:
    """Represent a symbolic equality-constrained nonlinear program.

    Parameters
    ----------
    x
        Decision-variable symbols.
    f
        Scalar objective expression.
    g
        Equality expressions interpreted as ``g == 0``.
    lb
        Lower bounds for ``x``. If omitted, all variables are unbounded below.
    ub
        Upper bounds for ``x``. If omitted, all variables are unbounded above.
    parameters
        Symbols whose numerical values are supplied when the solver is called.
    luts
        Fixed-shape lookup tables, in runtime input order.
    """

    x: tuple[sympy.Symbol, ...]
    f: sympy.Expr
    g: tuple[sympy.Expr, ...] = field(default_factory=tuple)
    lb: tuple[sympy.Expr | Number, ...] | None = None
    ub: tuple[sympy.Expr | Number, ...] | None = None
    parameters: tuple[sympy.Symbol, ...] = field(default_factory=tuple)

    luts: tuple[LUT, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Normalize absent and numeric bounds before validating the signature.
        lower_bounds = (sympy.S.NegativeInfinity,) * len(self.x) if self.lb is None else tuple(sympy.sympify(item) for item in self.lb)
        upper_bounds = (sympy.S.Infinity,) * len(self.x) if self.ub is None else tuple(sympy.sympify(item) for item in self.ub)
        object.__setattr__(self, "lb", lower_bounds)
        object.__setattr__(self, "ub", upper_bounds)

        if len(lower_bounds) != len(self.x) or len(upper_bounds) != len(self.x):
            message = "lb and ub must have one entry per variable"
            raise ValueError(message)

        for lower, upper in zip(lower_bounds, upper_bounds, strict=True):
            difference = upper - lower

            if difference.is_negative:
                message = "a lower bound is greater than its upper bound"
                raise ValueError(message)

        if len({*self.x, *self.parameters}) != len(self.x) + len(self.parameters):
            message = "variables and parameters must be unique"
            raise ValueError(message)

        if len(self.g) > len(self.x):
            message = "the number of equalities cannot exceed the variables"
            raise ValueError(message)

        # Generated evaluators have no implicit inputs.
        if len(set(self.luts)) != len(self.luts):
            message = "LUT declarations must be unique"
            raise ValueError(message)
        identities = {table.identity for table in self.luts}
        allowed = {*self.x, *self.parameters, *identities}
        expressions = (self.f, *self.g, *lower_bounds, *upper_bounds)
        unknown = set().union(*(expr.free_symbols for expr in expressions)) - allowed

        for expr in expressions:
            for call in expr.atoms(LUTCall):
                if call.args[0] not in identities:
                    message = "expressions contain an undeclared LUT"
                    raise ValueError(message)

        if unknown:
            message = f"expressions contain undeclared symbols: {sorted(map(str, unknown))}"
            raise ValueError(message)

    @property
    def dimension(self) -> int:
        """Return the number of decision variables."""
        return len(self.x)

    @property
    def equality_dimension(self) -> int:
        """Return the number of nonlinear equality constraints."""
        return len(self.g)
