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

The generated program contains SymPy expressions for the evaluator and a
problem-specific sparse LDLᵀ schedule. At runtime Boop:

1. evaluates the objective, equalities, objective gradient and Hessian, and
   equality Jacobian;
2. eliminates active bound variables;
3. assembles and factors `E_F E_F.T + rho I`, regularizing only equalities;
4. computes a Byrd--Omojokun normal step;
5. computes the tangential step with projected Steihaug CG;
6. tries a second-order correction and the configured fallback steps;
7. globalizes with the full-history objective/violation filter; and
8. updates the primal-dual bound active set after accepted candidates.

The outer SQP and inner CG iteration counts are fixed. Numerical zero checks
inside linear algebra prevent undefined divisions but are not optimization
termination criteria.

The initial execution backend uses NumPy. Native source printing and compiled
extension caching are intentionally separate from the numerical prototype.
