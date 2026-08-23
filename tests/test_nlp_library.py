"""Readable library of small NLPs and an executable Boop benchmark."""

from statistics import median
from time import perf_counter

import numpy
import pytest
import sympy

from boop import NonlinearProgram, Solver, SolverOptions, create_solver

ProblemAndSolution = tuple[NonlinearProgram, numpy.ndarray]
BenchmarkResult = tuple[str, int, int, int, float, float, int, float]

# Every problem uses a prefix of this vector. Each table value consists only of
# the mathematical problem and its known solution.
x = sympy.symbols("x:20")

NLP_LIBRARY = {
    "unconstrained_quadratic": (
        NonlinearProgram(
            x=x[:4],
            f=(x[0] - 1) ** 2 + (x[1] + 2) ** 2 + (x[2] - 0.5) ** 2 + (x[3] - 3) ** 2,
        ),
        numpy.array([1.0, -2.0, 0.5, 3.0]),
    ),
    "linear_equalities": (
        NonlinearProgram(
            x=x[:6],
            f=(x[0] + 1) ** 2 + (x[1] + 0.5) ** 2 + x[2] ** 2 + (x[3] - 0.5) ** 2 + (x[4] - 1) ** 2 + (x[5] - 1.5) ** 2,
            g=(
                x[0] + x[1] + 1.5,
                x[2] - 2 * x[3] + x[4],
                x[1] + x[4] - x[5] + 1,
            ),
        ),
        numpy.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
    ),
    "some_one_and_two_sided_bounds": (
        NonlinearProgram(
            x=x[:5],
            f=(x[0] + 2) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 3) ** 2 + (x[3] + 0.5) ** 2 + (x[4] - 1) ** 2,
            lb=(0, -sympy.oo, -sympy.oo, -1, -sympy.oo),
            ub=(sympy.oo, sympy.oo, 2, 1, sympy.oo),
        ),
        numpy.array([0.0, 0.5, 2.0, -0.5, 1.0]),
    ),
    "all_variables_two_sided": (
        NonlinearProgram(
            x=x[:8],
            f=(x[0] + 3) ** 2 + (x[1] - 2.5) ** 2 + (x[2] + 2) ** 2 + (x[3] - 4) ** 2 + (x[4] - 0.2) ** 2 + (x[5] + 0.3) ** 2 + (x[6] - 3) ** 2 + (x[7] + 4) ** 2,
            lb=(-1,) * 8,
            ub=(1,) * 8,
        ),
        numpy.array([-1.0, 1.0, -1.0, 1.0, 0.2, -0.3, 1.0, -1.0]),
    ),
    "nonlinear_equality": (
        NonlinearProgram(
            x=x[:2],
            f=x[0] ** 2 + x[1] ** 2,
            g=(x[0] ** 2 + x[1] - 1,),
            lb=(0, 0),
            ub=(2, 2),
        ),
        numpy.array([numpy.sqrt(0.5), 0.5]),
    ),
    "nearly_parallel_equalities": (
        NonlinearProgram(
            x=x[:5],
            f=(x[0] - 0.2) ** 2 + (x[1] + 0.4) ** 2 + (x[2] - 0.7) ** 2 + (x[3] + 0.1) ** 2 + (x[4] - 0.5) ** 2,
            g=(
                (x[0] - 0.2) + (x[1] + 0.4),
                (x[0] - 0.2) + 1.0002 * (x[1] + 0.4) + 0.0002 * (x[2] - 0.7),
                (x[3] + 0.1) + (x[4] - 0.5),
            ),
        ),
        numpy.array([0.2, -0.4, 0.7, -0.1, 0.5]),
    ),
    "nonlinear_equality_infeasible_start": (
        NonlinearProgram(
            x=x[:2],
            f=(x[0] - 1) ** 2 + x[1] ** 2,
            g=(x[0] ** 2 + x[1] ** 2 - 1,),
            lb=(0, -sympy.oo),
            ub=(1, sympy.oo),
        ),
        numpy.array([1.0, 0.0]),
    ),
    "forty_box_inequalities": (
        NonlinearProgram(
            x=x[:20],
            f=sum((x[i] - (-2 + 4 * i / 19)) ** 2 for i in range(20)),
            lb=(-0.75,) * 20,
            ub=(0.75,) * 20,
        ),
        numpy.clip(numpy.linspace(-2, 2, 20), -0.75, 0.75),
    ),
    "hs001": (
        NonlinearProgram(
            x=x[:2],
            f=100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lb=(-sympy.oo, -1.5),
        ),
        numpy.array([1.0, 1.0]),
    ),
    "hs002r": (
        NonlinearProgram(
            x=x[:2],
            f=100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lb=(0, 1.5),
        ),
        numpy.array([1.224370748736353, 1.5]),
    ),
}


# All problems deliberately receive the same style of poor initial guess and
# the default solver configuration. This keeps behavior comparisons meaningful.
OPTIONS = SolverOptions()


def initial_guess(solution: numpy.ndarray) -> numpy.ndarray:
    """Move every component far from the known solution with alternating sign."""
    signs = numpy.where(numpy.arange(solution.size) % 2, -1.0, 1.0)
    return solution + 4.0 * signs


def violation(solver: Solver, point: numpy.ndarray) -> float:
    """Return the infinity norm of all equality and bound violations."""
    values = solver.program.evaluate(point, numpy.empty(0))
    equality = float(numpy.max(numpy.abs(values.equalities))) if values.equalities.size else 0.0
    lower = float(numpy.max(numpy.maximum(values.lower_bounds - point, 0.0)))
    upper = float(numpy.max(numpy.maximum(point - values.upper_bounds, 0.0)))
    return max(equality, lower, upper)


def solve_and_validate(
    name: str,
    problem: NonlinearProgram,
    solution: numpy.ndarray,
    repeats: int = 1,
) -> BenchmarkResult:
    """Compile, solve, validate, and benchmark one table entry."""
    compile_start = perf_counter()
    solver = create_solver(problem, OPTIONS)
    compile_seconds = perf_counter() - compile_start

    solve_seconds = []
    results = []
    for _ in range(repeats):
        start = perf_counter()
        results.append(solver(initial_guess(solution), diagnostics=True))
        solve_seconds.append(perf_counter() - start)

    result = results[-1]
    for repeated in results[:-1]:
        numpy.testing.assert_array_equal(repeated.x, result.x)
    numpy.testing.assert_allclose(result.x, solution, atol=5e-4, rtol=0)
    final_violation = violation(solver, result.x)
    assert final_violation <= 1e-7
    assert len(result.diagnostics.iterations) == OPTIONS.sqp_iterations
    assert numpy.all(numpy.isfinite(result.x))
    assert all(numpy.isfinite(item.objective) for item in result.diagnostics.iterations)
    assert all(numpy.isfinite(item.violation) for item in result.diagnostics.iterations)
    assert all(item.trust_radius > 0 for item in result.diagnostics.iterations)

    bounds = solver.program.evaluate(result.x, numpy.empty(0))
    for item in result.diagnostics.iterations:
        assert numpy.all(item.x >= bounds.lower_bounds - OPTIONS.bound_tolerance)
        assert numpy.all(item.x <= bounds.upper_bounds + OPTIONS.bound_tolerance)

    values = bounds
    number_bounds = numpy.count_nonzero(numpy.isfinite(values.lower_bounds)) + numpy.count_nonzero(numpy.isfinite(values.upper_bounds))
    return (
        name,
        problem.dimension,
        problem.equality_dimension,
        int(number_bounds),
        compile_seconds,
        median(solve_seconds),
        sum(item.accepted for item in result.diagnostics.iterations),
        final_violation,
    )


@pytest.mark.parametrize(
    ("name", "problem_and_solution"),
    NLP_LIBRARY.items(),
    ids=NLP_LIBRARY.keys(),
)
def test_nlp_library(name: str, problem_and_solution: ProblemAndSolution) -> None:
    """Solve and validate one declarative benchmark-library entry."""
    problem, solution = problem_and_solution
    solve_and_validate(name, problem, solution)


def main() -> None:
    """Run the library and print its benchmark table."""
    results = [solve_and_validate(name, problem, solution, repeats=3) for name, (problem, solution) in NLP_LIBRARY.items()]
    print("case                                    n  eq  bounds  compile_ms  solve_ms  accepted  violation")
    for name, n, equalities, bounds, compile_time, solve_time, accepted, final_violation in results:
        print(f"{name:37} {n:3d} {equalities:3d} {bounds:7d} {compile_time * 1e3:11.3f} {solve_time * 1e3:9.3f} {accepted:9d} {final_violation:10.2e}")


if __name__ == "__main__":
    main()
