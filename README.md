# Boop

Boop (Byrd--Omojokun Optimizer) derives problem-specific SQP solvers from
SymPy nonlinear programs. The current prototype supports nonlinear equality
constraints and parameterized box constraints.

Boop targets small, smooth nonlinear programs for which predictable execution
time and a small deployment footprint matter. It is particularly well suited to
problems with:

- roughly 100 decision variables or fewer (this is a practical target rather
  than a hard limit, and larger problems may also work well);
- twice-continuously differentiable (`C²`) objective and constraint functions;
  and
- a low-dimensional equality-constraint tangent space, meaning that the number
  of independent equality constraints is close to the number of decision
  variables.

This structure occurs, for example, in model predictive control (MPC) and
trajectory optimization when the system dynamics are represented explicitly as
equality constraints. Boop also supports lower and upper bounds on the decision
variables; the bounds may depend on run-time parameters.

Boop is designed for hard real-time use. The generated solver performs a fixed
number of outer SQP iterations, with a fixed upper bound on the work performed
by each inner conjugate-gradient solve. Consequently, the solver has a bounded
algorithmic workload and does not depend on convergence-based stopping tests.
Actual worst-case execution time still needs to be established for the chosen
compiler, target hardware, and integration environment.

For suitable problems, the specialized generated solver can also achieve
sub-millisecond solve times on typical desktop CPUs. Performance depends on the
problem dimensions and structure, solver settings, compiler, and processor, so
applications should benchmark their own generated solver.

Boop uses ahead-of-time code generation: it translates the symbolic nonlinear
program into standard C11 source code. All solver storage is statically sized,
and the generated runtime uses only a small subset of the C standard library.
Problem-specific linear-algebra operations—including sparse equality-system
assembly and factorization—are emitted directly as C code, so the generated
solver does not require BLAS, LAPACK, or another numerical library. This makes
it portable across desktop and embedded targets with a conforming C11 toolchain
and well suited to resource-constrained real-time applications.

```python
import sympy

from boop import NonlinearProgram, SolverOptions, create_compiled_solver

x, y = sympy.symbols("x y")
nlp = NonlinearProgram(
    x=(x, y),
    f=x**2 + y**2,
    g=(x**2 + y - 1,),
    lb=(0, 0),
    ub=(2, 2),
)
solver = create_compiled_solver(nlp, SolverOptions(sqp_iterations=20))
solution = solver([0.8, 0.4])
```

For a full numerical trace, request native diagnostics:

```python
result = solver([0.8, 0.4], diagnostics=True)
print(result.diagnostics.procedure)
print(result.diagnostics.violation)
```

Diagnostic histories remain in the native C structure until their corresponding
Python properties are accessed.

The generation pipeline is deliberately explicit:

1. `build_evaluator_ir` differentiates the normalized `NonlinearProgram`.
2. `analyze_equality_sparsity` plans Gram assembly, fill reduction, and LDL.
3. `generate_program` bundles those target-independent representations.
4. The evaluator and linear-algebra emitters translate their respective IRs.
5. `generate_c_solver` combines generated code with the handwritten C runtime.
6. `create_compiled_solver` optionally compiles and caches a CPython extension.

The generated program contains SymPy expressions for the evaluator and a
problem-specific sparse LDLᵀ schedule. At runtime Boop:

1. evaluates the objective, equalities, objective gradient and Hessian, and
   equality Jacobian;
2. eliminates active bound variables;
3. assembles and factors `E_F E_F.T + rho I`, regularizing only equalities;
4. computes a Byrd--Omojokun normal step;
5. computes the tangential step with projected Steihaug CG;
6. converts a blocking box constraint into a transition-only SQP iteration;
7. tries a box-feasible second-order correction and the uncorrected step; and
8. globalizes objective value against equality violation with the full-history
   filter.

The initial guess is projected onto the box. Bound crossings never reach the
filter: blocking bounds update the working set while the iterate, trust radius,
and filter remain unchanged. Bounds with invalid multipliers are removed in the
same transition-only fashion.

The outer SQP and inner CG iteration counts are fixed. Numerical zero checks
inside linear algebra prevent undefined divisions but are not optimization
termination criteria.

Boop has no interpreted numerical backend. `generate_c_solver(nlp).write(directory)`
emits a standalone C11 solver. Its
public API uses caller-owned workspace and diagnostic structures and performs
no dynamic allocation. `create_compiled_solver(nlp)` builds and caches a CPython
extension around the same runtime. The extension accepts ordinary sequences
(including NumPy arrays through Python's sequence protocol) without using the
NumPy C API, and solutions are returned as plain `list[float]` values. When
diagnostics are requested, it returns an extension-backed object whose
individual histories are converted to Python lists only when their properties
are accessed.

Lookup tables are declared alongside scalar parameters. Their shapes are fixed
when generating the solver; coordinates and samples are supplied at runtime:

```python
from boop import LUT

curve = LUT(shape=(4,))
surface = LUT(shape=(3, 2))
nlp = NonlinearProgram(
    x=(x, y),
    f=(curve(x) - 1)**2 + (surface(x, y) - 2)**2,
    luts=(curve, surface),
)
solver = create_compiled_solver(nlp)
solution = solver(
    [0.8, 0.4],
    luts=[
        (([0, 0.5, 1, 2],), [0, 0.2, 1, 3]),
        (([0, 1, 2], [0, 2]), [[0, 2], [1, 3], [4, 6]]),
    ],
)
```

Each runtime slot is `(nodes, values)` in `nlp.luts` declaration order. Nodes
must be finite and strictly increasing, with at least two nodes per axis.
Samples must be finite and match the declared shape. In 2D, `values[i][j]`
belongs to the first axis's node `i` and the second axis's node `j`.

Boop uses natural cubic splines in 1D and their C² tensor product in 2D.
Outside the grid, it extends the boundary polynomial. Spline construction is
linear in the number of samples and happens inside every solve, before the SQP
iterations. Axis factorizations are reused within construction. The solver
stores values and first derivatives at nodes (including mixed derivatives in
2D), then evaluates local Hermite polynomials with analytic gradients and
Hessians. No user derivative implementation is required.

The generated C API accepts `const BoopLutInput *luts` between the scalar
parameters and solution arguments of `boop_solve`. Each declaration-order input
contains `nodes[2]` and `values` pointers; 1D inputs leave `nodes[1]` null. A 2D
sample array is contiguous in row-major order, `values[i * ny + j]`. The
`BOOP_LUT_COUNT` and `BOOP_LUT_SHAPES` configuration macros describe the required
inputs. Pass `NULL` when no LUTs are declared. All spline fields and scratch
storage are statically sized in the caller-owned workspace. Preparation has a
bounded workload fixed by the declared shapes, and each query uses binary
search followed by fixed-size polynomial arithmetic.
