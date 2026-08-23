"""Tests for standalone and JIT-compiled C solvers."""

import subprocess
from pathlib import Path

import numpy
import sympy

from boop import NonlinearProgram, SolverOptions, create_compiled_solver, generate_c_solver


def test_generated_c_builds_as_a_standalone_solver(tmp_path: Path) -> None:
    """Compile and execute the public allocation-free C API."""
    x, y = sympy.symbols("x y")
    nlp = NonlinearProgram(x=(x, y), f=(x - 1) ** 2 + (y + 2) ** 2)
    generated = generate_c_solver(nlp)
    assert "string.h" not in "".join(generated.sources.values())
    generated.write(tmp_path)
    harness = tmp_path / "main.c"
    harness.write_text(
        """#include <math.h>
#include "boop_runtime.h"
int main(void) {
  BoopWorkspace workspace;
  double initial[BOOP_N] = {4.0, -5.0};
  double parameters[BOOP_STORAGE(BOOP_P)] = {0.0};
  double solution[BOOP_N];
  if (boop_solve(&workspace, initial, parameters, solution, NULL) != BOOP_OK) return 1;
  return fabs(solution[0] - 1.0) < 1e-9 && fabs(solution[1] + 2.0) < 1e-9 ? 0 : 2;
}
"""
    )
    executable = tmp_path / "solver"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(tmp_path),
            str(tmp_path / "boop_runtime.c"),
            str(tmp_path / "boop_problem.c"),
            str(harness),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_compiled_solver_matches_reference_and_wraps_native_diagnostics(tmp_path: Path) -> None:
    """Use sequence inputs while materializing diagnostic fields only on access."""
    x, y = sympy.symbols("x y")
    nlp = NonlinearProgram(
        x=(x, y),
        f=x**2 + y**2,
        g=(x**2 + y - 1,),
        lb=(0, 0),
        ub=(2, 2),
    )
    compiled = create_compiled_solver(nlp, cache_directory=tmp_path)

    # NumPy arrays work through the ordinary runtime sequence protocol; the
    # compiled target itself neither imports nor links against NumPy.
    result = compiled(numpy.array([0.8, 0.4]), diagnostics=True)  # type: ignore[call-overload]

    assert isinstance(result.x, list)
    numpy.testing.assert_allclose(result.x, [numpy.sqrt(0.5), 0.5], atol=2e-5, rtol=0)
    assert type(result.diagnostics).__module__.startswith("boop_jit_")
    iterations = SolverOptions().sqp_iterations
    assert len(result.diagnostics.procedure) == iterations
    assert len(result.diagnostics.objective) == iterations
    assert len(result.diagnostics.x) == iterations
    assert len(result.diagnostics.active_bounds) == iterations
    assert len(result.diagnostics.cg_final_rayleigh) == iterations
    extension_source = next(tmp_path.glob("*/boop_extension.c")).read_text()
    assert "numpy" not in extension_source.lower()
