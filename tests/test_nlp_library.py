"""Readable library of small NLPs and an executable Boop benchmark."""

import fnmatch
import os
import sys
from statistics import median
from time import perf_counter

import numpy
import pytest
import sympy
from optimality_check import certify_local_minimum

from boop import NonlinearProgram, SolverOptions, create_compiled_solver

ProblemAndInitializer = tuple[NonlinearProgram, numpy.ndarray]
BenchmarkResult = tuple[str, int, int, int, float, float, int, float]

# Every problem uses a prefix of this vector. Each table entry keeps the
# mathematical problem beside the initializer from which it is normally run.
x = sympy.symbols("x:100")

NLP_LIBRARY = {
    "unconstrained_quadratic": (
        NonlinearProgram(
            x=x[:4],
            f=(x[0] - 1) ** 2 + (x[1] + 2) ** 2 + (x[2] - 0.5) ** 2 + (x[3] - 3) ** 2,
        ),
        numpy.array([4.0, -5.0, 2.0, -1.0]),
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
        numpy.array([2.0, -3.0, 2.0, -2.0, 3.0, -2.0]),
    ),
    "some_one_and_two_sided_bounds": (
        NonlinearProgram(
            x=x[:5],
            f=(x[0] + 2) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 3) ** 2 + (x[3] + 0.5) ** 2 + (x[4] - 1) ** 2,
            lb=(0, -sympy.oo, -sympy.oo, -1, -sympy.oo),
            ub=(sympy.oo, sympy.oo, 2, 1, sympy.oo),
        ),
        numpy.array([-3.0, 2.0, 4.0, 2.0, -2.0]),
    ),
    "all_variables_two_sided": (
        NonlinearProgram(
            x=x[:8],
            f=(x[0] + 3) ** 2 + (x[1] - 2.5) ** 2 + (x[2] + 2) ** 2 + (x[3] - 4) ** 2 + (x[4] - 0.2) ** 2 + (x[5] + 0.3) ** 2 + (x[6] - 3) ** 2 + (x[7] + 4) ** 2,
            lb=(-1,) * 8,
            ub=(1,) * 8,
        ),
        numpy.array([2.0, -2.0, 2.0, -2.0, 1.5, -1.5, -2.0, 2.0]),
    ),
    "nonlinear_equality": (
        NonlinearProgram(
            x=x[:2],
            f=x[0] ** 2 + x[1] ** 2,
            g=(x[0] ** 2 + x[1] - 1,),
            lb=(0, 0),
            ub=(2, 2),
        ),
        numpy.array([0.8, 0.4]),
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
        numpy.array([2.0, -2.0, 1.0, 2.0, -2.0]),
    ),
    "anisotropic_objective": (
        NonlinearProgram(
            x=x[:6],
            f=(1e-6 * (x[0] - 1) ** 2 + 1e-3 * (x[1] + 1) ** 2 + (x[2] - 0.5) ** 2 + 1e2 * (x[3] + 0.5) ** 2 + 1e4 * (x[4] - 2) ** 2 + 1e6 * (x[5] + 2) ** 2),
        ),
        numpy.array([1.2, -0.8, 0.8, -0.2, 2.2, -1.8]),
    ),
    "mixed_transcendental_objective": (
        NonlinearProgram(
            x=x[:4],
            f=(
                sympy.exp(3 * (x[0] - sympy.Rational(1, 5)))
                - 3 * (x[0] - sympy.Rational(1, 5))
                - 1
                + 10 * (-sympy.log(x[1]) + x[1] - 1)
                + 1 / (x[2] + 1)
                + (x[2] + 1) / 4
                - 1
                + (sympy.cosh(4 * (x[3] + sympy.Rational(3, 10))) - 1) / 100
            ),
            lb=(-1, sympy.Rational(1, 10), sympy.Rational(-1, 2), -1),
            ub=(1, 3, 3, 1),
        ),
        numpy.array([0.35, 1.2, 0.8, -0.1]),
    ),
    "mixed_scale_nonlinear_equalities": (
        NonlinearProgram(
            x=x[:5],
            f=(x[0] - 0.5) ** 2 + (x[1] + 0.5) ** 2 + (x[2] - 1) ** 2 + (x[3] - 0.25) ** 2 + (x[4] + 0.25) ** 2,
            g=(
                1e4 * (sympy.sin(x[0]) + x[1] - sympy.sin(sympy.Rational(1, 2)) + sympy.Rational(1, 2)),
                1e-4 * (x[2] ** 2 + x[3] + x[4] - 1),
            ),
            lb=(-1, -2, 0, -1, -1),
            ub=(2, 1, 2, 1, 1),
        ),
        numpy.array([0.65, -0.6, 0.85, 0.4, -0.15]),
    ),
    "transcendental_manifold": (
        NonlinearProgram(
            x=x[:4],
            f=(x[0] - 0.4) ** 2 + 3 * (x[1] + 0.2) ** 2 + (x[2] - 0.7) ** 4 + (x[2] - 0.7) ** 2 + 2 * (x[3] + 0.1) ** 2,
            g=(
                sympy.sin(2 * x[0]) + x[1] ** 3 - sympy.sin(sympy.Rational(4, 5)) + sympy.Rational(1, 125),
                sympy.exp(x[1] / 2) + x[2] * x[3] - sympy.exp(sympy.Rational(-1, 10)) + sympy.Rational(7, 100),
            ),
            lb=(-1, -1, 0, -1),
            ub=(1, 1, 2, 1),
        ),
        numpy.array([0.5, -0.3, 0.8, -0.2]),
    ),
    "coupled_rosenbrock_manifold": (
        NonlinearProgram(
            x=x[:5],
            f=sum(50 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(4)),
            g=(
                x[0] * x[2] + x[4] - 2,
                x[1] + sympy.sin(x[3]) - 1 - sympy.sin(1),
            ),
            lb=(0, 0, 0, 0, 0),
            ub=(2, 2, 2, 2, 2),
        ),
        numpy.array([1.08, 1.12, 1.1, 1.05, 0.9]),
    ),
    "nonlinear_equality_infeasible_start": (
        NonlinearProgram(
            x=x[:2],
            f=(x[0] - 1) ** 2 + x[1] ** 2,
            g=(x[0] ** 2 + x[1] ** 2 - 1,),
            lb=(0, -sympy.oo),
            ub=(1, sympy.oo),
        ),
        numpy.array([0.4, 1.5]),
    ),
    "forty_box_inequalities": (
        NonlinearProgram(
            x=x[:20],
            f=sum((x[i] - (-2 + 4 * i / 19)) ** 2 for i in range(20)),
            lb=(-0.75,) * 20,
            ub=(0.75,) * 20,
        ),
        numpy.linspace(2.0, -2.0, 20),
    ),
    "thirty_variables_three_dof_linear": (
        NonlinearProgram(
            x=x[:30],
            f=sum((1 + i / 30) * x[i] ** 2 for i in range(30)),
            g=tuple(x[i] - sympy.Rational(1, 5) * x[i + 3] for i in range(27)),
            lb=(-1,) * 30,
            ub=(1,) * 30,
        ),
        numpy.linspace(0.3, -0.3, 30),
    ),
    "sixty_variables_four_dof_nonlinear": (
        NonlinearProgram(
            x=x[:60],
            f=sum((1 + i / 60) * x[i] ** 2 for i in range(60)),
            g=tuple(x[i] - sympy.Rational(1, 10) * sympy.sin(x[i + 4]) for i in range(56)),
            lb=(-1,) * 60,
            ub=(1,) * 60,
        ),
        numpy.linspace(-0.25, 0.25, 60),
    ),
    "hundred_variables_three_dof_mixed": (
        NonlinearProgram(
            x=x[:100],
            f=sum((1 + i / 100) * x[i] ** 2 for i in range(100)),
            g=tuple(x[i] - sympy.Rational(1, 10) * x[i + 3] for i in range(48)) + tuple(x[i] - sympy.Rational(1, 20) * x[i + 3] ** 2 for i in range(48, 97)),
            lb=(-1,) * 100,
            ub=(1,) * 100,
        ),
        numpy.linspace(0.2, -0.2, 100),
    ),
    "hs001": (
        NonlinearProgram(
            x=x[:2],
            f=100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lb=(-sympy.oo, -1.5),
        ),
        numpy.array([1.2, 1.2]),
    ),
    "hs002r": (
        NonlinearProgram(
            x=x[:2],
            f=100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
            lb=(0, 1.5),
        ),
        numpy.array([2.0, 2.0]),
    ),
}


OPTIONS = SolverOptions()
PERTURBATION_SCALE = 0.01


def solve_and_validate(
    name: str,
    problem: NonlinearProgram,
    initializer: numpy.ndarray,
    repeats: int = 10,
) -> BenchmarkResult:
    """Compile, solve, validate, and benchmark one table entry."""
    compile_start = perf_counter()
    solver = create_compiled_solver(problem, OPTIONS)
    compile_seconds = perf_counter() - compile_start

    rng = numpy.random.default_rng(sum(name.encode()))

    solve_seconds = []
    results = []
    initial_guesses = [initializer]
    initial_guesses.extend(
        rng.normal(
            loc=initializer,
            scale=PERTURBATION_SCALE * numpy.fmax(1.0, numpy.abs(initializer)),
        )
        for _ in range(repeats - 1)
    )

    for initial_guess in initial_guesses:
        start = perf_counter()
        results.append(solver(initial_guess.tolist(), diagnostics=True))
        solve_seconds.append(perf_counter() - start)

    result = results[-1]
    if os.getenv("BOOP_TEST_DIAGNOSTICS"):
        print(f"\nDiagnostics for benchmark problem {name!r}")
        print(result.diagnostics.procedure)

    certificates = [certify_local_minimum(problem, repeated.x) for repeated in results]

    assert len(result.diagnostics.procedure) == OPTIONS.sqp_iterations
    assert numpy.all(numpy.isfinite(result.x))
    assert all(numpy.isfinite(item) for item in result.diagnostics.objective)
    assert all(numpy.isfinite(item) for item in result.diagnostics.violation)
    assert all(item > 0 for item in result.diagnostics.trust_radius)

    bounds = (*(problem.lb or ()), *(problem.ub or ()))
    number_bounds = sum(sympy.sympify(bound).is_finite is not False for bound in bounds)
    return (
        name,
        problem.dimension,
        problem.equality_dimension,
        int(number_bounds),
        compile_seconds,
        median(solve_seconds),
        sum(result.diagnostics.accepted),
        certificates[-1].primal_residual,
    )


@pytest.mark.parametrize(
    ("name", "problem_and_initializer"),
    NLP_LIBRARY.items(),
    ids=NLP_LIBRARY.keys(),
)
def test_nlp_library(name: str, problem_and_initializer: ProblemAndInitializer) -> None:
    """Solve and validate one declarative benchmark-library entry."""
    problem, initializer = problem_and_initializer
    repeats = 3 if problem.dimension >= 30 else 10
    solve_and_validate(name, problem, initializer, repeats)


def main() -> None:
    """Run the library and print its benchmark table."""
    match len(sys.argv):
        case 1:
            pattern = "*"
        case 2:
            pattern = sys.argv[1]
        case _:
            msg = "Unsupported number of arguments"
            raise ValueError(msg)

    results = []

    for name, (problem, initializer) in NLP_LIBRARY.items():
        if fnmatch.fnmatch(name, pattern):
            results.append(solve_and_validate(name, problem, initializer))

    print("case                                    n  eq  bounds  compile_ms  solve_ms  accepted  violation")
    for name, n, equalities, bounds, compile_time, solve_time, accepted, final_violation in results:
        print(f"{name:37} {n:3d} {equalities:3d} {bounds:7d} {compile_time * 1e3:11.3f} {solve_time * 1e3:9.3f} {accepted:9d} {final_violation:10.2e}")


if __name__ == "__main__":
    main()
