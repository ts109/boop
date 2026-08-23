"""Emit problem-specific sparse equality linear algebra."""

from .program import GeneratedProgram


def emit_equality_linear_algebra(program: GeneratedProgram) -> str:
    """Emit sparse Gram assembly, LDL factorization, and triangular solves."""
    return f"{_emit_gram_assembly(program)}\n\n{_emit_ldl(program)}"


def _emit_gram_assembly(program: GeneratedProgram) -> str:
    n = program.nlp.dimension
    m = program.nlp.equality_dimension
    lines = [
        "void boop_assemble_gram(const BoopModel *model, const unsigned char *free_variables, double *gram) {",
        "  (void)model;",
        "  (void)free_variables;",
        "  for (int i = 0; i < BOOP_STORAGE(BOOP_M * BOOP_M); ++i) gram[i] = 0.0;",
    ]
    for row, column, variables in program.equality_sparsity.gram_contributions:
        terms = (
            " + ".join(f"(free_variables[{variable}] ? model->jacobian[{row * n + variable}] * model->jacobian[{column * n + variable}] : 0.0)" for variable in variables) or "0.0"
        )
        lines.append(f"  gram[{row * m + column}] = {terms};")
        if row != column:
            lines.append(f"  gram[{column * m + row}] = gram[{row * m + column}];")
    lines.extend(f"  gram[{index * m + index}] += BOOP_EQUALITY_REGULARIZATION;" for index in range(m))
    lines.append("}")
    return "\n".join(lines)


def _emit_ldl(program: GeneratedProgram) -> str:
    dimension = program.nlp.equality_dimension
    factor = program.equality_sparsity.ldl
    lines = [
        "BoopStatus boop_ldl_factor(const double *gram, double *lower, double *diagonal) {",
        "  for (int i = 0; i < BOOP_STORAGE(BOOP_M * BOOP_M); ++i) lower[i] = 0.0;",
        "  (void)gram;",
        "  (void)diagonal;",
    ]
    lines.extend(f"  lower[{index * dimension + index}] = 1.0;" for index in range(dimension))
    for column, schedule in enumerate(factor.columns):
        updates = (
            " + ".join(f"lower[{column * dimension + previous}] * lower[{column * dimension + previous}] * diagonal[{previous}]" for previous in schedule.diagonal_terms) or "0.0"
        )
        original = factor.permutation[column]
        lines.append(f"  diagonal[{column}] = gram[{original * dimension + original}] - ({updates});")
        lines.append(f"  if (!isfinite(diagonal[{column}]) || diagonal[{column}] <= BOOP_FACTORIZATION_TOLERANCE) return BOOP_FACTOR_FAILURE;")
        for row, shared in schedule.rows:
            updates = " + ".join(f"lower[{row * dimension + previous}] * lower[{column * dimension + previous}] * diagonal[{previous}]" for previous in shared) or "0.0"
            lines.append(f"  lower[{row * dimension + column}] = (gram[{factor.permutation[row] * dimension + original}] - ({updates})) / diagonal[{column}];")
    lines.extend(("  return BOOP_OK;", "}", _emit_ldl_solve(program)))
    return "\n".join(lines)


def _emit_ldl_solve(program: GeneratedProgram) -> str:
    dimension = program.nlp.equality_dimension
    factor = program.equality_sparsity.ldl
    lines = [
        "void boop_ldl_solve(const double *lower, const double *diagonal, const double *rhs, double *solution) {",
        "  double work[BOOP_STORAGE(BOOP_M)] = {0.0};",
        "  (void)lower;",
        "  (void)diagonal;",
        "  (void)rhs;",
        "  (void)solution;",
        "  (void)work;",
    ]
    for row, original in enumerate(factor.permutation):
        lines.append(f"  work[{row}] = rhs[{original}];")
        lines.extend(f"  work[{row}] -= lower[{row * dimension + column}] * work[{column}];" for column in range(row) if factor.pattern[row][column])
    lines.extend(f"  work[{index}] /= diagonal[{index}];" for index in range(dimension))
    for row in range(dimension - 1, -1, -1):
        lines.extend(f"  work[{row}] -= lower[{column * dimension + row}] * work[{column}];" for column in range(row + 1, dimension) if factor.pattern[column][row])
    lines.extend(f"  solution[{original}] = work[{row}];" for row, original in enumerate(factor.permutation))
    lines.append("}")
    return "\n".join(lines)
