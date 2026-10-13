# Code-generation performance audit

The audit covers model validation, derivative IR, sparsity planning, evaluator
printing, linear-algebra emission, JIT compilation/cache lookup, and the C
runtime. No symbolic simplification is used by the updated generation pipeline.
SymPy still applies its normal constructor canonicalization and derivative
rules; this is not a replacement of SymPy with unevaluated arithmetic.

## Changes implemented

- Removed `sympy.simplify(upper - lower)` from bound validation. Contradictions
  recognizable from the ordinary difference still raise; contradictions that
  require simplification are not established at construction time.
- Compute objective first derivatives once and reuse them for the upper
  Hessian triangle, mirroring the result. Cache expression symbol dependencies
  and skip derivatives for absent variables. Every differentiation explicitly
  uses `simplify=False`.
- Determine Jacobian sparsity from literal zeros rather than recursive
  `is_zero` assumptions. Unproven zeros remain structural nonzeros, preserving
  correctness at a possible cost in factorization size.
- Compute Gram supports with intersections of sparse row index sets and reuse
  the resulting contributions to construct the pattern. Previously pattern and
  contribution construction each scanned every variable for every row pair.
- Replace SymPy CSE with exact subtree sharing: count DAG edges and output
  references, visit each distinct node once, then emit shared nodes in
  dependency order. There is no factoring, expansion, or expression rewriting.
  Piecewise branches are kept local to preserve conditional evaluation.
- Print original symbols through a mapping instead of `xreplace` on every
  evaluator output. Reuse one C printer and disable expression-order sorting.
- Initialize dense Hessian/Jacobian storage with C loops and emit only literal
  nonzero assignments, reducing Python printing and compiler input.
- Add `share_subexpressions=False` to standalone and JIT generation. Both modes
  compile and solve in regression tests. Default sharing avoids redundant
  transcendental and LUT evaluations, but does not find algebraically related
  expressions unless their subtrees are already equal.

## Measurements

A sparse MPC-like family uses `n` variables, nearest-neighbor nonlinear stage
costs, and `n - 2` local dynamics equalities. This represents multiple shooting;
it does not represent a deeply nested single-shooting graph. Measurements use
Python 3.14 and SymPy 1.14, clearing the SymPy cache before each path. These are
single cold runs on the local machine, not guaranteed speedups for other models.
Baseline: commit `a456001549b95db1b45acdc5c0447c3255ffa907`.

| Variables | Baseline IR + planning | Updated IR + planning | Baseline emission | Updated emission | Baseline evaluator bytes | Updated evaluator bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 0.602 s | 0.371 s | 0.366 s | 0.032 s | 71,200 | 25,950 |
| 60 | 2.308 s | 0.838 s | 0.923 s | 0.074 s | 247,270 | 52,814 |
| 100 | 7.915 s | 1.598 s | 2.182 s | 0.125 s | 641,990 | 88,582 |

For 100 variables the measured combined IR/planning/emission time decreased
from 10.10 s to 1.72 s (about 5.9 times faster). Model construction, C compilation,
and execution are excluded from this table. Reproduce with:

```sh
PYTHONPATH=src python benchmarks/codegen.py --baseline-ref a456001 --repeats 3
```

A separate evaluator-only native check used GCC 15.2.0 with `-O2 -std=c11`,
linking the evaluator and a small C harness without the solver runtime. It
called the evaluator 20,000 times at `x[i] = 0.01 * i`, accumulating the objective
in a volatile checksum. This was one compile/run per mode, not a repeated
statistical measurement or an end-to-end solve benchmark.

| 100-variable evaluator | Compile + link | Evaluation per call |
| --- | ---: | ---: |
| Baseline SymPy CSE | 30.370 s | 18.342 µs |
| Updated exact sharing | 1.668 s | 13.737 µs |
| Updated sharing disabled | 1.522 s | 14.962 µs |

All three objective checksums agreed for this input. Disabling sharing slightly
shortened compilation but increased evaluation time in this example; the tradeoff
must be measured on the actual MPC model. The baseline's many zero assignments
account for a substantial part of its compiler input.

## Remaining opportunities, in priority order

1. **DAG automatic differentiation or stage-wise derivative assembly.** The
   largest remaining measured stage is differentiation. Each objective
   derivative still walks the full stage-cost sum. Deep shared expression
   graphs may also suffer repeated recursive `free_symbols`, derivative, and
   validation (`atoms`) traversals. A separate operation DAG with memoized local
   derivative rules would avoid SymPy derivative graph growth altogether.
   Alternatively, accept explicit stage blocks and differentiate each small
   block once, then assemble sparse derivatives. Both require explicit support
   for LUT derivatives, powers, functions, and conditional expressions; a
   restricted expression vocabulary or a fallback is necessary. Moving AD to
   generated C can cut generation further, with additional runtime arithmetic
   and derivative workspace. Exact Hessians should be retained unless a
   separately documented approximation is explicitly selected.
2. **Cache before symbolic generation.** `create_compiled_solver` generates the
   full source bundle before checking its extension cache. A repeated call
   therefore repeats differentiation, planning, and printing even when the
   binary already exists. Reusing the returned solver is the immediate remedy.
   A bounded process-local cache of `GeneratedProgram`, or accepting a prebuilt
   program/code bundle, would also help multiple options/compilers. Persistent
   graph cache keys need structural serialization, package/SymPy versioning,
   LUT identities/order and all generation settings. Avoid `str(expr)` as a
   cheap cache key: it can expand a shared graph into a huge string.
3. **Table-driven Gram/LDL kernels.** `c_linalg.py` emits every numerical update
   directly as C. Dense equality coupling or fill can make this enormous even
   after evaluator improvements. An optional static-index schedule interpreted
   by compact C loops trades indexing and loop overhead for much smaller source
   and lower compiler optimization cost. A hybrid that unrolls only short rows
   is also possible. Minimum-degree ordering and the filled schedule still use
   dense Python patterns; sparse adjacency/factor structures would reduce
   planning memory and work for larger horizons.
4. **Separate trial residual evaluation from derivatives.** The C runtime calls
   the full evaluator for both the ordinary and second-order-correction trial,
   although filter decisions use objective/constraint values. Dedicated
   objective/residual evaluation and lazy derivatives for accepted iterates
   could reduce solve time substantially. This requires reviewing nonfinite
   derivative handling and accepted-state bookkeeping. Independent evaluator
   functions reduce the scope of sharing but can improve compiler behavior.
   The runtime also repeats evaluations around initial projection and iteration
   boundaries; reuse needs to track whether `x` or model inputs changed.
5. **Sparse derivative runtime storage/actions.** Generated IR and the C runtime
   still store dense Hessian/Jacobian arrays and perform dense products.
   Zero-initialization loops reduce source size but not the O(n²) memory traffic
   or CG matrix-vector work. CSR/block storage or generated sparse actions would
   improve larger sparse MPC problems. This is an ABI/runtime change rather
   than just a printer optimization. Hessian-vector AD avoids materializing a
   Hessian but repeats differentiation arithmetic in each CG action.
6. **Stage templates and parameter-only work.** Repeated horizon stages can
   share one generated C function called with stage offsets rather than emit
   separate expressions for every stage. This needs explicit stage metadata;
   guessing templates from arbitrary SymPy graphs is expensive. Parameter-only
   expressions and fixed bounds can be evaluated once per solve. LUT spline
   preparation already occurs once per solve; repeated solves with identical
   LUT data could reuse preparation through an explicit prepared-table API.
7. **Compiler settings and translation-unit splitting.** Existing JIT supports
   `CFLAGS` and includes build settings in the cache key. Compare `-O0`, `-O1`,
   `-O2` and `-O3` for the actual target, measuring compilation and solve time
   separately. Lower optimization can shorten builds and slow execution.
   Split very large evaluator/kernel functions or translation units only when
   profiling shows compiler optimization dominates; doing so can inhibit
   inlining and sharing. Avoid changing floating-point semantics merely to
   accelerate compilation: fast-math can invalidate nonfinite checks.

## Practical limits

Sharing reduces duplicate work but still uses SymPy structural hashing/equality;
very deep graphs can reach recursive printer or differentiation limits. The
fallback without sharing can emit much larger source and is not guaranteed to
be faster. Scope/lifetime partitioning of temporaries may help compiler register
allocation for large evaluators, at the cost of recalculating shared work.

A broad performance claim needs the customer's slow MPC graph, including its
construction time and whether it uses single or multiple shooting. The supplied
benchmark isolates generation, and regression tests check derivatives, LUTs,
solver behavior, both sharing modes, symbol-name collisions, and branch-local
sharing. Runtime numerical results need tolerance-based comparisons because
printer ordering and sharing can change floating-point rounding.
