"""Measure sparse MPC generation; optionally compare with a local Git revision.

Run: PYTHONPATH=src python benchmarks/codegen.py --baseline-ref <commit>
Cold SymPy caches are used for each run. C compilation and execution are not timed.
"""

import argparse
import statistics
import subprocess
import time
from collections.abc import Callable

import sympy

from boop import GeneratedProgram, NonlinearProgram, generate_program
from boop.c_evaluator import emit_evaluator


def historical_functions(ref: str) -> tuple[Callable[[NonlinearProgram], GeneratedProgram], Callable[[GeneratedProgram], str]]:
    """Load generation functions from an explicitly selected local revision."""
    namespaces = {}
    for name in ("ir", "program", "c_evaluator"):
        namespace = {"__name__": f"boop.benchmark_{name}", "__package__": "boop"}
        source = subprocess.check_output(["git", "show", f"{ref}:src/boop/{name}.py"], text=True)
        exec(source, namespace)
        namespaces[name] = namespace
    namespaces["program"]["build_evaluator_ir"] = namespaces["ir"]["build_evaluator_ir"]
    return namespaces["program"]["generate_program"], namespaces["c_evaluator"]["emit_evaluator"]


def main() -> None:
    """Print median planning/emission times and generated evaluator size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[30, 60, 100])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    paths = [("current", generate_program, emit_evaluator)]
    if args.baseline_ref:
        paths.insert(0, ("baseline", *historical_functions(args.baseline_ref)))
    for n in args.dimensions:
        x = sympy.symbols(f"x:{n}")
        nlp = NonlinearProgram(
            x=x,
            f=sum((x[i] - sympy.sin(x[i+1]))**2 + sympy.exp(x[i]/10) for i in range(n-1)),
            g=tuple(x[i+2]-x[i]-sympy.sin(x[i+1])/10 for i in range(n-2)),
        )
        for label, build, emit in paths:
            timings = []
            for _ in range(args.repeats):
                sympy.core.cache.clear_cache()
                start = time.perf_counter()
                program = build(nlp)
                middle = time.perf_counter()
                source = emit(program)
                end = time.perf_counter()
                timings.append((middle-start, end-middle))
            planning, emission = map(statistics.median, zip(*timings, strict=True))
            print(f"n={n:3} {label:8} planning={planning:.3f}s emission={emission:.3f}s bytes={len(source)}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
