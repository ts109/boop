# Boop

Boop (Byrd--Omojokun Optimizer) derives problem-specific SQP solvers from
SymPy nonlinear programs. The current prototype supports nonlinear equality
constraints and parameterized box constraints.

```python
import sympy as sp

from boop import NonlinearProgram, SolverOptions, create_solver

x, y = sp.symbols("x y")
nlp = NonlinearProgram(
    x=(x, y),
    f=x**2 + y**2,
    g=(x**2 + y - 1,),
    lb=(0, 0),
    ub=(2, 2),
)
solver = create_solver(nlp, SolverOptions(sqp_iterations=20))
solution = solver([0.8, 0.4])
```

For a full numerical trace, request diagnostics and print the report:

```python
from boop import print_solver_diagnostics

result = solver([0.8, 0.4], diagnostics=True)
print_solver_diagnostics(result.diagnostics)
```

The report includes every attempted filter candidate, step component, working-
set transition, multiplier estimate, Steihaug stopping reason, trust-radius
change, equality factorization condition estimate, Hessian spectrum, and primal
and stationarity residual. To print this report for every declarative benchmark,
run `BOOP_TEST_DIAGNOSTICS=1 pytest -s tests/test_nlp_library.py`.

The generated program contains SymPy expressions for the evaluator and a
problem-specific sparse LDLᵀ schedule. At runtime Boop:

1. evaluates the objective, equalities, objective gradient and Hessian, and
   equality Jacobian;
2. eliminates active bound variables;
3. assembles and factors `E_F E_F.T + rho I`, regularizing only equalities;
4. computes a Byrd--Omojokun normal step;
5. computes the tangential step with projected Steihaug CG;
6. converts a blocking box constraint into a transition-only SQP iteration;
7. tries a box-feasible second-order correction and fallback steps; and
8. globalizes objective value against equality violation with the full-history
   filter.

The initial guess is projected onto the box. Bound crossings never reach the
filter: blocking bounds update the working set while the iterate, trust radius,
and filter remain unchanged. Bounds with invalid multipliers are removed in the
same transition-only fashion.

The outer SQP and inner CG iteration counts are fixed. Numerical zero checks
inside linear algebra prevent undefined divisions but are not optimization
termination criteria.

The initial execution backend uses NumPy. Native source printing and compiled
extension caching are intentionally separate from the numerical prototype.
