from dataclasses import dataclass, field
from numbers import Number

import sympy


@dataclass(frozen=True, slots=True)
class NonlinearProgram:
    """A symbolic equality-constrained NLP with bounds ``lb <= x <= ub``."""

    x: tuple[sympy.Symbol, ...]
    f: sympy.Expr
    g: tuple[sympy.Expr, ...] = field(default_factory=tuple)
    lb: tuple[sympy.Expr | Number, ...] | None = None
    ub: tuple[sympy.Expr | Number, ...] | None = None
    parameters: tuple[sympy.Symbol, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.lb is None:
            object.__setattr__(self, "lb", (sympy.S.NegativeInfinity,) * len(self.x))
        else:
            object.__setattr__(self, "lb", tuple(sympy.sympify(item) for item in self.lb))

        if self.ub is None:
            object.__setattr__(self, "ub", (sympy.S.Infinity,) * len(self.x))
        else:
            object.__setattr__(self, "ub", tuple(sympy.sympify(item) for item in self.ub))

        if len(self.lb) != len(self.x) or len(self.ub) != len(self.x):
            raise ValueError("lb and ub must have one entry per variable")

        for lower, upper in zip(self.lb, self.ub, strict=True):
            difference = sympy.simplify(upper - lower)

            if difference.is_negative:
                raise ValueError("a lower bound is greater than its upper bound")

        if len({*self.x, *self.parameters}) != len(self.x) + len(self.parameters):
            raise ValueError("variables and parameters must be unique")

        if len(self.g) > len(self.x):
            raise ValueError("the number of equalities cannot exceed the variables")

        allowed = {*self.x, *self.parameters}
        expressions = (self.f, *self.g, *self.lb, *self.ub)
        unknown = set().union(*(expr.free_symbols for expr in expressions)) - allowed

        if unknown:
            raise ValueError(f"expressions contain undeclared symbols: {sorted(map(str, unknown))}")

    @property
    def dimension(self) -> int:
        return len(self.x)

    @property
    def equality_dimension(self) -> int:
        return len(self.g)
