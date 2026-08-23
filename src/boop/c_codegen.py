"""C source generation for problem-specific Boop solvers."""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Self

import sympy

from .model import NonlinearProgram
from .options import SolverOptions
from .program import GeneratedProgram


@dataclass(frozen=True, slots=True)
class GeneratedCCode:
    """Contain a complete, standalone C solver source bundle.

    Attributes
    ----------
    headers
        Header filenames mapped to their contents.
    sources
        C source filenames mapped to their contents.
    """

    headers: dict[str, str]
    sources: dict[str, str]

    def write(self, directory: str | Path) -> Self:
        """Write the source bundle into an existing or new directory."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in self.headers.items():
            (destination / name).write_text(content)
        for name, content in self.sources.items():
            (destination / name).write_text(content)
        return self


def generate_c_solver(nlp: NonlinearProgram, options: SolverOptions | None = None) -> GeneratedCCode:
    """Generate portable C11 sources for an NLP-specific solver."""
    selected_options = options or SolverOptions()
    program = GeneratedProgram(nlp)
    runtime = resources.files("boop").joinpath("c_runtime")
    return GeneratedCCode(
        headers={
            "boop_config.h": _config_header(program, selected_options),
            "boop_runtime.h": runtime.joinpath("boop_runtime.h").read_text(),
        },
        sources={
            "boop_problem.c": _problem_source(program),
            "boop_runtime.c": runtime.joinpath("boop_runtime.c").read_text(),
        },
    )


def _config_header(program: GeneratedProgram, options: SolverOptions) -> str:
    nonlinear = int(program.has_nonlinear_equalities)
    return f"""#ifndef BOOP_CONFIG_H
#define BOOP_CONFIG_H
#define BOOP_N {program.nlp.dimension}
#define BOOP_M {program.nlp.equality_dimension}
#define BOOP_P {len(program.nlp.parameters)}
#define BOOP_SQP_ITERATIONS {options.sqp_iterations}
#define BOOP_CG_ITERATIONS {options.cg_iterations}
#define BOOP_NONLINEAR_EQUALITIES {nonlinear}
#define BOOP_INITIAL_TRUST_RADIUS {options.initial_trust_radius:.17g}
#define BOOP_TRUST_EXPAND {options.trust_expand:.17g}
#define BOOP_TRUST_SHRINK {options.trust_shrink:.17g}
#define BOOP_EQUALITY_REGULARIZATION {options.equality_regularization:.17g}
#define BOOP_TANGENTIAL_DAMPING {options.tangential_damping:.17g}
#define BOOP_CURVATURE_FLOOR {options.curvature_floor:.17g}
#define BOOP_FACTORIZATION_TOLERANCE {options.factorization_tolerance:.17g}
#define BOOP_FILTER_BETA {options.filter_beta:.17g}
#define BOOP_FILTER_GAMMA {options.filter_gamma:.17g}
#define BOOP_BOUND_TOLERANCE {options.bound_tolerance:.17g}
#define BOOP_ACTIVE_SET_STATIONARITY_TOLERANCE {options.active_set_stationarity_tolerance:.17g}
#endif
"""


def _problem_source(program: GeneratedProgram) -> str:
    nlp = program.nlp
    generated_variables = sympy.symbols(f"boop_x_0:{nlp.dimension}")
    generated_parameters = sympy.symbols(f"boop_parameter_0:{len(nlp.parameters)}")
    substitutions = dict(zip((*nlp.x, *nlp.parameters), (*generated_variables, *generated_parameters), strict=True))
    arguments = (*generated_variables, *generated_parameters)
    expressions = [
        program.ir.objective,
        *program.ir.equalities,
        *program.ir.gradient,
        *program.ir.hessian,
        *program.ir.equality_jacobian,
        *program.ir.lower_bounds,
        *program.ir.upper_bounds,
    ]
    replacements, reduced = sympy.cse([item.xreplace(substitutions) for item in expressions], symbols=sympy.numbered_symbols("temporary_"))
    lines = ['#include "boop_runtime.h"', "", "#include <math.h>", "#include <string.h>", ""]
    lines.append("BoopStatus boop_evaluate(const double *x, const double *parameters, BoopModel *model) {")
    lines.extend(("  (void)x;", "  (void)parameters;"))
    for index, symbol in enumerate(generated_variables):
        lines.append(f"  const double {symbol} = x[{index}];")
    for index, symbol in enumerate(generated_parameters):
        lines.append(f"  const double {symbol} = parameters[{index}];")
    for symbol, expression in replacements:
        lines.append(f"  const double {symbol} = {_ccode(expression, arguments)};")
    targets = ["model->objective"]
    targets.extend(f"model->equalities[{i}]" for i in range(nlp.equality_dimension))
    targets.extend(f"model->gradient[{i}]" for i in range(nlp.dimension))
    targets.extend(f"model->hessian[{i}]" for i in range(nlp.dimension * nlp.dimension))
    targets.extend(f"model->jacobian[{i}]" for i in range(nlp.equality_dimension * nlp.dimension))
    targets.extend(f"model->lower[{i}]" for i in range(nlp.dimension))
    targets.extend(f"model->upper[{i}]" for i in range(nlp.dimension))
    for target, expression in zip(targets, reduced, strict=True):
        lines.append(f"  {target} = {_ccode(expression, arguments)};")
    lines.append("  if (!isfinite(model->objective)) return BOOP_NONFINITE_EVALUATION;")
    lines.append("  for (int i = 0; i < BOOP_M; ++i) if (!isfinite(model->equalities[i])) return BOOP_NONFINITE_EVALUATION;")
    lines.append("  for (int i = 0; i < BOOP_N; ++i) if (!isfinite(model->gradient[i]) || isnan(model->lower[i]) || isnan(model->upper[i])) return BOOP_NONFINITE_EVALUATION;")
    lines.append("  for (int i = 0; i < BOOP_N * BOOP_N; ++i) if (!isfinite(model->hessian[i])) return BOOP_NONFINITE_EVALUATION;")
    lines.append("  for (int i = 0; i < BOOP_M * BOOP_N; ++i) if (!isfinite(model->jacobian[i])) return BOOP_NONFINITE_EVALUATION;")
    lines.append("  return BOOP_OK;")
    lines.append("}")
    lines.extend(("", _gram_source(program), "", _ldl_source(program)))
    return "\n".join(lines) + "\n"


def _gram_source(program: GeneratedProgram) -> str:
    lines = [
        "void boop_assemble_gram(const BoopModel *model, const unsigned char *free_variables, double *gram) {",
        "  (void)model;",
        "  (void)free_variables;",
        "  memset(gram, 0, BOOP_STORAGE(BOOP_M * BOOP_M) * sizeof(double));",
    ]
    n = program.nlp.dimension
    m = program.nlp.equality_dimension
    for i, j, columns in program.gram_contributions:
        terms = " + ".join(f"(free_variables[{k}] ? model->jacobian[{i * n + k}] * model->jacobian[{j * n + k}] : 0.0)" for k in columns) or "0.0"
        lines.append(f"  gram[{i * m + j}] = {terms};")
        if i != j:
            lines.append(f"  gram[{j * m + i}] = gram[{i * m + j}];")
    for i in range(m):
        lines.append(f"  gram[{i * m + i}] += BOOP_EQUALITY_REGULARIZATION;")
    lines.append("}")
    return "\n".join(lines)


def _ldl_source(program: GeneratedProgram) -> str:
    m = program.nlp.equality_dimension
    permutation = program.ldl.permutation
    lines = ["BoopStatus boop_ldl_factor(const double *gram, double *lower, double *diagonal) {", "  memset(lower, 0, BOOP_STORAGE(BOOP_M * BOOP_M) * sizeof(double));"]
    lines.extend(("  (void)gram;", "  (void)diagonal;"))
    for i in range(m):
        lines.append(f"  lower[{i * m + i}] = 1.0;")
    for k, column in enumerate(program.ldl.columns):
        diagonal_terms = " + ".join(f"lower[{k * m + j}] * lower[{k * m + j}] * diagonal[{j}]" for j in column.diagonal_terms) or "0.0"
        lines.append(f"  diagonal[{k}] = gram[{permutation[k] * m + permutation[k]}] - ({diagonal_terms});")
        lines.append(f"  if (!isfinite(diagonal[{k}]) || diagonal[{k}] <= BOOP_FACTORIZATION_TOLERANCE) return BOOP_FACTOR_FAILURE;")
        for i, common in column.rows:
            updates = " + ".join(f"lower[{i * m + j}] * lower[{k * m + j}] * diagonal[{j}]" for j in common) or "0.0"
            lines.append(f"  lower[{i * m + k}] = (gram[{permutation[i] * m + permutation[k]}] - ({updates})) / diagonal[{k}];")
    lines.extend(("  return BOOP_OK;", "}"))
    lines.append("void boop_ldl_solve(const double *lower, const double *diagonal, const double *rhs, double *solution) {")
    lines.append("  double work[BOOP_STORAGE(BOOP_M)] = {0.0};")
    lines.extend(("  (void)lower;", "  (void)diagonal;", "  (void)rhs;", "  (void)solution;", "  (void)work;"))
    for i, original in enumerate(permutation):
        lines.append(f"  work[{i}] = rhs[{original}];")
        for j in range(i):
            if program.ldl.pattern[i][j]:
                lines.append(f"  work[{i}] -= lower[{i * m + j}] * work[{j}];")
    for i in range(m):
        lines.append(f"  work[{i}] /= diagonal[{i}];")
    for i in range(m - 1, -1, -1):
        for j in range(i + 1, m):
            if program.ldl.pattern[j][i]:
                lines.append(f"  work[{i}] -= lower[{j * m + i}] * work[{j}];")
    for i, original in enumerate(permutation):
        lines.append(f"  solution[{original}] = work[{i}];")
    lines.append("}")
    return "\n".join(lines)


def _ccode(expression: sympy.Expr, _arguments: tuple[sympy.Symbol, ...]) -> str:
    return str(sympy.ccode(expression, standard="c99"))
