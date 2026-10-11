"""Emit the problem-specific C evaluator from symbolic expressions."""

import sympy
from sympy.printing.c import C99CodePrinter

from .lut import LUTCall
from .program import GeneratedProgram


def emit_evaluator(program: GeneratedProgram) -> str:
    """Emit objective, constraint, bound, and derivative assignments."""
    nlp = program.nlp
    variables = sympy.symbols(f"boop_x_0:{nlp.dimension}")
    parameters = sympy.symbols(f"boop_parameter_0:{len(nlp.parameters)}")
    substitutions = dict(zip((*nlp.x, *nlp.parameters), (*variables, *parameters), strict=True))
    substitutions.update((table.identity, sympy.Integer(index)) for index, table in enumerate(nlp.luts))
    expressions = [expression.xreplace(substitutions) for expression in _flatten_evaluator(program)]
    replacements, reduced = sympy.cse(expressions, symbols=sympy.numbered_symbols("temporary_"))

    lines = ["BoopStatus boop_evaluate(const double *x, const double *parameters, const BoopWorkspace *workspace, BoopModel *model) {"]
    lines.extend(("  (void)x;", "  (void)parameters;", "  (void)workspace;"))
    lines.extend(f"  const double {symbol} = x[{index}];" for index, symbol in enumerate(variables))
    lines.extend(f"  const double {symbol} = parameters[{index}];" for index, symbol in enumerate(parameters))
    lines.extend(f"  const double {symbol} = {_ccode(expression)};" for symbol, expression in replacements)
    lines.extend(f"  {target} = {_ccode(expression)};" for target, expression in zip(_evaluator_targets(program), reduced, strict=True))
    lines.extend(_finite_checks())
    lines.extend(("  return BOOP_OK;", "}"))
    return "\n".join(lines)


def _flatten_evaluator(program: GeneratedProgram) -> list[sympy.Expr]:
    ir = program.evaluator
    return [
        ir.objective,
        *ir.equalities,
        *ir.gradient,
        *ir.hessian,
        *ir.equality_jacobian,
        *ir.lower_bounds,
        *ir.upper_bounds,
    ]


def _evaluator_targets(program: GeneratedProgram) -> list[str]:
    n = program.nlp.dimension
    m = program.nlp.equality_dimension
    targets = ["model->objective"]
    targets.extend(f"model->equalities[{index}]" for index in range(m))
    targets.extend(f"model->gradient[{index}]" for index in range(n))
    targets.extend(f"model->hessian[{index}]" for index in range(n * n))
    targets.extend(f"model->jacobian[{index}]" for index in range(m * n))
    targets.extend(f"model->lower[{index}]" for index in range(n))
    targets.extend(f"model->upper[{index}]" for index in range(n))
    return targets


def _finite_checks() -> tuple[str, ...]:
    return (
        "  if (!isfinite(model->objective)) return BOOP_NONFINITE_EVALUATION;",
        "  for (int i = 0; i < BOOP_M; ++i) if (!isfinite(model->equalities[i])) return BOOP_NONFINITE_EVALUATION;",
        "  for (int i = 0; i < BOOP_N; ++i) if (!isfinite(model->gradient[i]) || isnan(model->lower[i]) || isnan(model->upper[i])) return BOOP_NONFINITE_EVALUATION;",
        "  for (int i = 0; i < BOOP_N * BOOP_N; ++i) if (!isfinite(model->hessian[i])) return BOOP_NONFINITE_EVALUATION;",
        "  for (int i = 0; i < BOOP_M * BOOP_N; ++i) if (!isfinite(model->jacobian[i])) return BOOP_NONFINITE_EVALUATION;",
    )


class _Printer(C99CodePrinter):
    def _print_LUTCall(self, expression: LUTCall) -> str:  # noqa: N802
        identity, dx, dy, *queries = expression.args
        orders = (dx, dy)
        if any(order > 2 for order in orders) or sum(orders) > 2:
            message = "LUT derivatives above total order two are unsupported"
            raise ValueError(message)
        x, y = (*queries, sympy.Integer(0))[:2]
        return f"boop_lut_eval(workspace, {identity}, {dx}, {dy}, {self._print(x)}, {self._print(y)})"


def _ccode(expression: sympy.Expr) -> str:
    return _Printer().doprint(expression)
