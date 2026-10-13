"""Emit the problem-specific C evaluator from symbolic expressions."""

import sympy
from sympy.printing.c import C99CodePrinter

from .lut import LUTCall
from .program import GeneratedProgram


def emit_evaluator(program: GeneratedProgram, *, share_subexpressions: bool = True) -> str:
    """Emit objective, constraint, bound, and derivative assignments."""
    nlp = program.nlp
    variables = sympy.symbols(f"boop_x_0:{nlp.dimension}")
    parameters = sympy.symbols(f"boop_parameter_0:{len(nlp.parameters)}")
    assignments = [
        (target, expression)
        for target, expression in zip(_evaluator_targets(program), _flatten_evaluator(program), strict=True)
        if expression != 0 or not target.startswith(("model->hessian[", "model->jacobian["))
    ]
    expressions = [expression for _, expression in assignments]
    printer = _Printer()
    printer.references = dict(zip((*nlp.x, *nlp.parameters), map(str, (*variables, *parameters)), strict=True))
    printer.lut_indices = {table.identity: index for index, table in enumerate(nlp.luts)}

    lines = ["BoopStatus boop_evaluate(const double *x, const double *parameters, const BoopWorkspace *workspace, BoopModel *model) {"]
    lines.extend(("  (void)x;", "  (void)parameters;", "  (void)workspace;"))
    lines.extend((
        "  for (int i = 0; i < BOOP_N * BOOP_N; ++i) model->hessian[i] = 0.0;",
        "  for (int i = 0; i < BOOP_M * BOOP_N; ++i) model->jacobian[i] = 0.0;",
    ))
    lines.extend(f"  const double {symbol} = x[{index}];" for index, symbol in enumerate(variables))
    lines.extend(f"  const double {symbol} = parameters[{index}];" for index, symbol in enumerate(parameters))
    if share_subexpressions:
        for index, expression in enumerate(_shared_subexpressions(expressions)):
            name = f"temporary_{index}"
            lines.append(f"  const double {name} = {printer.doprint(expression)};")
            printer.references[expression] = name
    lines.extend(f"  {target} = {printer.doprint(expression)};" for target, expression in assignments)
    lines.extend(_finite_checks())
    lines.extend(("  return BOOP_OK;", "}"))
    return "\n".join(lines)


def _shared_subexpressions(expressions: list[sympy.Expr]) -> list[sympy.Expr]:
    """Find exact repeated DAG nodes in dependency order without rewriting.

    Visit each distinct node once, counting incoming edges and output uses.
    Unlike algebraic CSE this does not factor, expand, or rebuild expressions.
    """
    counts: dict[sympy.Basic, int] = {}
    visited: set[sympy.Basic] = set()
    ordered: list[sympy.Basic] = []
    for root in expressions:
        counts[root] = counts.get(root, 0) + 1
        stack = [(root, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                ordered.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                # Branch-local work must retain conditional evaluation.
                children = () if isinstance(node, sympy.Piecewise) else node.args
                for child in reversed(children):
                    counts[child] = counts.get(child, 0) + 1
                    stack.append((child, False))
    return [node for node in ordered if isinstance(node, sympy.Expr) and node.args and counts[node] > 1]


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
    def __init__(self) -> None:
        super().__init__({"order": "none"})
        self.references: dict[sympy.Basic, str] = {}
        self.lut_indices: dict[sympy.Basic, int] = {}

    def _print(self, expression: sympy.Basic, **kwargs: object) -> str:
        reference = self.references.get(expression)
        if reference is not None:
            return reference
        return super()._print(expression, **kwargs)

    def _print_LUTCall(self, expression: LUTCall) -> str:  # noqa: N802
        identity, dx, dy, *queries = expression.args
        orders = (dx, dy)
        if any(order > 2 for order in orders) or sum(orders) > 2:
            message = "LUT derivatives above total order two are unsupported"
            raise ValueError(message)
        x, y = (*queries, sympy.Integer(0))[:2]
        return f"boop_lut_eval(workspace, {self.lut_indices[identity]}, {dx}, {dy}, {self._print(x)}, {self._print(y)})"
