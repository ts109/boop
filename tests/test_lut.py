"""Check spline evaluation, analytic derivatives, and runtime input contracts."""

import ctypes
import subprocess
from pathlib import Path
from typing import Any

import numpy
import pytest
import sympy

from boop import LUT, NonlinearProgram, create_compiled_solver, generate_c_solver


def reference(nodes: numpy.ndarray, values: numpy.ndarray, query: float, order: int = 0) -> float:
    """Evaluate a natural spline using an independent dense linear solve."""
    n = len(nodes)
    h = numpy.diff(nodes)
    matrix = numpy.eye(n)
    rhs = numpy.zeros(n)
    for i in range(1, n - 1):
        matrix[i, i - 1:i + 2] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1])
    second = numpy.linalg.solve(matrix, rhs)
    i = min(max(numpy.searchsorted(nodes, query, side="right") - 1, 0), n - 2)
    z = query - nodes[i]
    a = values[i]
    b = (values[i + 1] - a) / h[i] - h[i] * (2 * second[i] + second[i + 1]) / 6
    c = second[i] / 2
    d = (second[i + 1] - second[i]) / (6 * h[i])
    return (a + z * (b + z * (c + z * d)), b + z * (2 * c + z * 3 * d), 2 * c + z * 6 * d)[order]


@pytest.mark.parametrize("shape", [(2,), (5,), (2, 2), (5, 4)])
def test_native_spline_derivatives(tmp_path: Path, shape: tuple[int, ...]) -> None:
    """Match independent natural splines at knots, within cells, and outside."""
    variables = sympy.symbols(f"x:0{len(shape)}")
    table = LUT(shape)
    nlp = NonlinearProgram(x=variables, f=table(*variables), luts=(table,))
    generate_c_solver(nlp).write(tmp_path)
    (tmp_path / "probe.c").write_text('''#include "boop_runtime.h"
BoopStatus boop_prepare_luts(BoopWorkspace *, const BoopLutInput *);
int probe(const double *x, const double *nx, const double *ny,
          const double *values, double *output) {
  BoopWorkspace work;
  BoopModel model;
  BoopLutInput input = {{nx, ny}, values};
  int status = boop_prepare_luts(&work, &input);
  if (status) return status;
  status = boop_evaluate(x, NULL, &work, &model);
  output[0] = model.objective;
  for (int i = 0; i < BOOP_N; ++i) output[1+i] = model.gradient[i];
  for (int i = 0; i < BOOP_N*BOOP_N; ++i) output[1+BOOP_N+i] = model.hessian[i];
  return status;
}
''')
    library = tmp_path / "probe.so"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", str(tmp_path / "probe.c"), str(tmp_path / "boop_problem.c"), "-lm", "-o", str(library)], check=True)
    probe = ctypes.CDLL(str(library)).probe
    pointer = ctypes.POINTER(ctypes.c_double)
    probe.argtypes = [pointer] * 5
    nodes = [numpy.cumsum(numpy.linspace(0.3, 1.2, n)) for n in shape]
    values = numpy.random.default_rng(42).normal(size=shape)
    def ptr(array: numpy.ndarray) -> Any:
        return array.ctypes.data_as(pointer)
    for query_x in [nodes[0][0] - 0.3, *nodes[0], nodes[0][0] + 0.2, nodes[0][-1] + 0.3]:
        query = numpy.array([query_x] if len(shape) == 1 else [query_x, nodes[1][1] + 0.1])
        output = numpy.zeros(1 + len(shape) + len(shape)**2)
        assert probe(ptr(query), ptr(nodes[0]), ptr(nodes[1]) if len(shape) == 2 else None, ptr(values), ptr(output)) == 0
        if len(shape) == 1:
            expected = [reference(nodes[0], values, query_x, d) for d in range(3)]
        else:
            def partial(dx: int, dy: int, query: numpy.ndarray = query, query_x: float = query_x) -> float:
                along_y = numpy.array([reference(nodes[1], row, query[1], dy) for row in values])
                return reference(nodes[0], along_y, query_x, dx)
            expected = [partial(0, 0), partial(1, 0), partial(0, 1), partial(2, 0), partial(1, 1), partial(1, 1), partial(0, 2)]
        numpy.testing.assert_allclose(output, expected, rtol=2e-12, atol=2e-12)


def test_lut_solver_inputs() -> None:
    """Prepare inside solve, refresh data, and reject malformed inputs."""
    x = sympy.Symbol("x")
    table = LUT((3,))
    solver = create_compiled_solver(NonlinearProgram(x=(x,), f=(table(x) - 1)**2, luts=(table,)))
    assert solver([0.2], luts=[(([0, 1, 2],), [0, 1, 2])])[0] == pytest.approx(1)
    assert solver([0.2], luts=[(([0, 1, 2],), [0, 2, 4])])[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        solver([0.2])
    with pytest.raises(ValueError):
        solver([0.2], luts=[(([0, 1],), [0, 1, 2])])
    with pytest.raises(RuntimeError, match="status 4"):
        solver([0.2], luts=[(([0, 1, 1],), [0, 1, 2])])


def test_lut_declarations() -> None:
    """Reject invalid shapes, query arity, and missing declarations."""
    x = sympy.Symbol("x")
    for shape in [(), (1,), (2, 2, 2), (2.5,)]:
        with pytest.raises(ValueError):
            LUT(shape)
    table = LUT((3,))
    with pytest.raises(ValueError):
        table(x, x)
    with pytest.raises(ValueError, match="undeclared LUT"):
        NonlinearProgram(x=(x,), f=table(x))
    with pytest.raises(ValueError, match="unique"):
        NonlinearProgram(x=(x,), f=table(x), luts=(table, table))


def test_multiple_luts_in_composed_objective_and_constraint() -> None:
    """Keep slot identities distinct and apply chain rules in objectives and equalities."""
    x, y = sympy.symbols("x y")
    curve, surface = LUT((3,)), LUT((2, 2))
    nlp = NonlinearProgram(
        x=(x, y), f=(curve(x**2) - 4)**2,
        g=(surface(x, y) - 2,), lb=(0.1, 0.1), ub=(3, 3),
        luts=(curve, surface),
    )
    solver = create_compiled_solver(nlp)
    result = solver([1.8, 1.1], luts=[
        (([0, 1, 4],), [0, 1, 4]),
        (([0, 3], [0, 2]), [[0, 0], [0, 6]]),
    ])
    numpy.testing.assert_allclose(result, [2, 1], atol=1e-6)
    with pytest.raises(ValueError, match="LUT row"):
        solver([1.8, 1.1], luts=[
            (([0, 1, 4],), [0, 1, 4]),
            (([0, 3], [0, 2]), [[0], [0]]),
        ])


def test_standalone_lut_solver_has_no_allocator_dependencies(tmp_path: Path) -> None:
    """Exercise the C API and verify preparation and solve do not reference allocators."""
    x, y = sympy.symbols("x y")
    table = LUT((2, 2))
    nlp = NonlinearProgram(x=(x, y), f=(table(x, y) - 2)**2 + (y - 1)**2, luts=(table,))
    generate_c_solver(nlp).write(tmp_path)
    objects = []
    for name in ("boop_problem", "boop_runtime"):
        source, obj = tmp_path / f"{name}.c", tmp_path / f"{name}.o"
        subprocess.run(["cc", "-std=c11", "-pedantic-errors", "-Wall", "-Wextra", "-Werror", "-c", str(source), "-o", str(obj)], check=True)
        symbols = subprocess.run(["nm", "-u", str(obj)], check=True, capture_output=True, text=True).stdout
        assert not {"malloc", "calloc", "realloc", "free", "aligned_alloc"} & {line.split()[-1] for line in symbols.splitlines()}
        objects.append(str(obj))
    main = tmp_path / "main.c"
    main.write_text('''#include <math.h>
#include "boop_runtime.h"
int main(void) {
  static BoopWorkspace workspace;
  const double nodes_x[2] = {0, 3}, nodes_y[2] = {0, 2};
  const double values[4] = {0, 0, 0, 6};
  const double initial[2] = {1.8, 1.1};
  BoopLutInput inputs[1] = {{{nodes_x, nodes_y}, values}};
  double solution[2];
  if (boop_solve(&workspace, initial, NULL, inputs, solution, NULL) != BOOP_OK) return 1;
  if (fabs(solution[0] - 2) > 1e-5 || fabs(solution[1] - 1) > 1e-5) return 2;
  return 0;
}
''')
    executable = tmp_path / "solver"
    subprocess.run(["cc", "-std=c11", "-pedantic-errors", str(main), *objects, "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
