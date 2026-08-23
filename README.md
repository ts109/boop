# Boop

Boop (Byrd--Omojokun Optimizer) derives problem-specific SQP solvers from
SymPy nonlinear programs. The current prototype supports nonlinear equality
constraints and parameterized box constraints.

```python
import sympy as sp

from boop import NonlinearProgram, SolverOptions, create_compiled_solver

x, y = sp.symbols("x y")
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
