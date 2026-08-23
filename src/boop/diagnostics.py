"""Human-readable solver diagnostics."""

import sys
from collections import Counter
from io import StringIO
from typing import TextIO

import numpy

from .solver import IterationDiagnostics, SolverDiagnostics


def format_solver_diagnostics(diagnostics: SolverDiagnostics, *, precision: int = 6) -> str:
    """Format a detailed convergence and numerical-health report.

    Parameters
    ----------
    diagnostics
        Diagnostics returned by a solver invocation.
    precision
        Significant digits used for scalar and vector values.

    Returns
    -------
    str
        Multiline diagnostic report.
    """
    stream = StringIO()
    print_solver_diagnostics(diagnostics, stream=stream, precision=precision)
    return stream.getvalue()


def print_solver_diagnostics(
    diagnostics: SolverDiagnostics,
    *,
    stream: TextIO | None = None,
    precision: int = 6,
) -> None:
    """Print a detailed convergence and numerical-health report.

    Parameters
    ----------
    diagnostics
        Diagnostics returned by a solver invocation.
    stream
        Destination text stream. Defaults to standard output.
    precision
        Significant digits used for scalar and vector values.
    """
    output = sys.stdout if stream is None else stream
    iterations = diagnostics.iterations

    if not iterations:
        print("Boop diagnostics: no SQP iterations recorded", file=output)
        return

    procedures = Counter(item.procedure.value for item in iterations)
    accepted = sum(item.accepted for item in iterations)
    transitions = procedures.get("active-set-update", 0)
    rejected = procedures.get("rejected", 0)
    initial = iterations[0]
    final = iterations[-1]
    best_violation = min(item.violation for item in iterations)
    best_objective = min(item.objective for item in iterations)
    projection = diagnostics.projected_initial_guess - diagnostics.initial_guess

    print("Boop solver diagnostics", file=output)
    print("=" * 80, file=output)
    print(f"iterations             : {len(iterations)}", file=output)
    print(f"accepted / rejected    : {accepted} / {rejected}", file=output)
    print(f"active-set transitions : {transitions}", file=output)
    print(f"procedures             : {dict(procedures)}", file=output)
    print(f"supplied initial x     : {_array(diagnostics.initial_guess, precision)}", file=output)
    print(f"box-projected initial x: {_array(diagnostics.projected_initial_guess, precision)}", file=output)
    print(f"initial projection norm: {_scalar(float(numpy.linalg.norm(projection)), precision)}", file=output)
    print(f"parameters             : {_array(diagnostics.parameters, precision)}", file=output)
    print(f"objective initial/final: {_scalar(initial.objective_before, precision)} -> {_scalar(final.objective, precision)}", file=output)
    print(f"objective best         : {_scalar(best_objective, precision)}", file=output)
    print(f"violation initial/final: {_scalar(initial.violation_before, precision)} -> {_scalar(final.violation, precision)}", file=output)
    print(f"violation best         : {_scalar(best_violation, precision)}", file=output)
    print(f"last model stationarity: {_scalar(final.stationarity_norm, precision)}", file=output)
    print(f"final trust radius     : {_scalar(final.trust_radius, precision)}", file=output)
    print(f"final x                : {_array(final.x, precision)}", file=output)
    print(f"diagnostic flags       : {'; '.join(_diagnostic_flags(iterations))}", file=output)

    for index, item in enumerate(iterations):
        _print_iteration(output, index, item, precision)


def _print_iteration(stream: TextIO, index: int, item: IterationDiagnostics, precision: int) -> None:
    status = "accepted" if item.accepted else "no iterate change"
    objective_change = item.objective - item.objective_before
    violation_change = item.violation - item.violation_before
    active_lower_before = numpy.flatnonzero(item.active_bounds_before < 0)
    active_upper_before = numpy.flatnonzero(item.active_bounds_before > 0)
    active_lower = numpy.flatnonzero(item.active_bounds < 0)
    active_upper = numpy.flatnonzero(item.active_bounds > 0)

    print("\n" + "-" * 80, file=stream)
    print(f"iteration {index:03d}: {item.procedure.value} ({status})", file=stream)
    print(
        "model/result           : "
        f"f {_scalar(item.objective_before, precision)} -> {_scalar(item.objective, precision)} "
        f"(delta {_scalar(objective_change, precision)}), "
        f"theta {_scalar(item.violation_before, precision)} -> {_scalar(item.violation, precision)} "
        f"(delta {_scalar(violation_change, precision)})",
        file=stream,
    )
    print(
        f"trust radius           : {_scalar(item.trust_radius_before, precision)} -> {_scalar(item.trust_radius, precision)}",
        file=stream,
    )
    print(f"x before               : {_array(item.x_before, precision)}", file=stream)
    print(f"x after                : {_array(item.x, precision)}", file=stream)
    print(f"accepted step          : {_array(item.accepted_step, precision)}", file=stream)
    print(f"equality residuals     : {_array(item.equality_residuals, precision)}", file=stream)
    print(f"lower-bound slack      : {_array(item.lower_bound_slack, precision)}", file=stream)
    print(f"upper-bound slack      : {_array(item.upper_bound_slack, precision)}", file=stream)
    print(
        f"active lower / upper   : {active_lower_before.tolist()} / {active_upper_before.tolist()} -> {active_lower.tolist()} / {active_upper.tolist()}",
        file=stream,
    )
    print(f"equality multipliers   : {_array(item.equality_multipliers, precision)}", file=stream)
    print(f"bound multipliers      : {_array(item.bound_multipliers, precision)}", file=stream)
    print(
        "gradient norms         : "
        f"raw={_scalar(item.gradient_norm, precision)}, "
        f"projected={_scalar(item.projected_gradient_norm, precision)}, "
        f"stationarity={_scalar(item.stationarity_norm, precision)}",
        file=stream,
    )
    print(
        f"Hessian diagonal       : min={_scalar(item.hessian_min_diagonal, precision)}, max={_scalar(item.hessian_max_diagonal, precision)}",
        file=stream,
    )
    print(
        f"equality LDL pivots    : [{_scalar(item.ldl_min_pivot, precision)}, {_scalar(item.ldl_max_pivot, precision)}]",
        file=stream,
    )
    print(
        "step norms             : "
        f"normal={_scalar(item.normal_step_norm, precision)}, "
        f"tangential={_scalar(item.tangential_step_norm, precision)}, "
        f"SOC={_scalar(item.correction_step_norm, precision)}",
        file=stream,
    )
    print(f"normal step            : {_array(item.normal_step, precision)}", file=stream)
    print(f"tangential step        : {_array(item.tangential_step, precision)}", file=stream)
    print(f"second-order correction: {_array(item.correction_step, precision)}", file=stream)
    print(
        "Steihaug CG            : "
        f"{item.cg.stop_reason.value}, iterations={item.cg.iterations}, "
        f"radius={_scalar(item.cg.radius, precision)}, "
        f"residual={_scalar(item.cg.initial_residual_norm, precision)} -> "
        f"{_scalar(item.cg.final_residual_norm, precision)}, "
        f"last curvature={_optional_scalar(item.cg.final_curvature, precision)}, "
        f"Rayleigh quotient={_optional_scalar(item.cg.final_rayleigh_quotient, precision)}",
        file=stream,
    )
    print(f"filter entries         : {item.filter_entries_before} -> {item.filter_entries_after}", file=stream)

    if not item.trials:
        print("filter trials          : none", file=stream)
    else:
        print("filter trials          :", file=stream)
        for trial_index, trial in enumerate(item.trials):
            verdict = "accepted" if trial.filter_accepted else "rejected"
            print(
                f"  [{trial_index}] {trial.procedure.value}: {verdict}, "
                f"f={_scalar(trial.objective, precision)}, "
                f"theta={_scalar(trial.violation, precision)}, "
                f"norm={_scalar(float(numpy.linalg.norm(trial.step)), precision)}, "
                f"step={_array(trial.step, precision)}",
                file=stream,
            )


def _array(value: numpy.ndarray, precision: int) -> str:
    return str(numpy.array2string(value, precision=precision, suppress_small=False, max_line_width=160))


def _scalar(value: float, precision: int) -> str:
    return f"{value:.{precision}g}"


def _optional_scalar(value: float | None, precision: int) -> str:
    return "n/a" if value is None else _scalar(value, precision)


def _diagnostic_flags(iterations: list[IterationDiagnostics]) -> list[str]:
    """Summarize conspicuous behavior without imposing a stop criterion."""
    flags = []
    final = iterations[-1]
    rejected = sum(item.procedure.value == "rejected" for item in iterations)
    transitions = sum(item.procedure.value == "active-set-update" for item in iterations)
    negative_curvature = sum(item.cg.stop_reason.value == "negative-curvature" for item in iterations)
    cg_limits = sum(item.cg.stop_reason.value == "iteration-limit" for item in iterations)
    finite_pivot_ratios = [item.ldl_min_pivot / item.ldl_max_pivot for item in iterations if item.ldl_max_pivot > 0.0]

    if final.violation > 1e-6:
        flags.append(f"material final equality violation ({final.violation:.3g})")
    if final.stationarity_norm > 1e-5:
        flags.append(f"material last-model stationarity residual ({final.stationarity_norm:.3g})")
    if rejected:
        flags.append(f"{rejected} filter-rejected iterations")
    if transitions:
        flags.append(f"{transitions} active-set transitions")
    if negative_curvature:
        flags.append(f"negative curvature in {negative_curvature} CG solves")
    if cg_limits:
        flags.append(f"CG iteration limit reached {cg_limits} times")
    if finite_pivot_ratios and min(finite_pivot_ratios) < 1e-12:
        flags.append(f"widely scaled equality LDL pivots (smallest ratio {min(finite_pivot_ratios):.3g})")
    if final.trust_radius < 1e-10:
        flags.append(f"collapsed trust radius ({final.trust_radius:.3g})")
    return flags or ["none detected by report heuristics"]
