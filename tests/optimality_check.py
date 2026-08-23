"""Strict local-minimum certification for solved nonlinear programs."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy
import sympy

from boop import NonlinearProgram


@dataclass(frozen=True, slots=True)
class LocalMinimumCertificate:
    """Summarize a strict local-minimum check.

    Attributes
    ----------
    primal_residual
        Largest equality or box violation.
    stationarity_residual
        Infinity norm of the final-point Lagrangian gradient.
    equality_rank
        Numerical rank of the equality Jacobian.
    equality_multipliers
        Least-squares equality multipliers.
    lower_bound_multipliers, upper_bound_multipliers
        Multipliers for numerically active box sides.
    fixed_bound_multipliers
        Free-sign multipliers for variables fixed by coincident box sides.
    minimum_reduced_curvature
        Smallest Lagrangian-Hessian eigenvalue on the equality tangent space.
        Box directions are deliberately retained, giving a conservative
        sufficient second-order test.
    """

    primal_residual: float
    stationarity_residual: float
    equality_rank: int
    equality_multipliers: numpy.ndarray
    lower_bound_multipliers: numpy.ndarray
    upper_bound_multipliers: numpy.ndarray
    fixed_bound_multipliers: numpy.ndarray
    minimum_reduced_curvature: float


class LocalMinimumError(AssertionError):
    """Indicate that a point could not be certified as a strict local minimum."""


def certify_local_minimum(  # noqa: PLR0915
    nlp: NonlinearProgram,
    point: Sequence[float] | numpy.ndarray,
    parameters: Sequence[float] | numpy.ndarray = (),
    *,
    feasibility_tolerance: float = 1e-7,
    stationarity_tolerance: float = 1e-4,
    activity_tolerance: float = 1e-7,
    rank_tolerance: float = 1e-10,
    curvature_tolerance: float = 1e-8,
) -> LocalMinimumCertificate:
    """Certify feasibility, KKT consistency, and strict local curvature.

    The curvature test uses the equality tangent space without deleting active
    box directions. Positive curvature on this larger space is conservative
    but sufficient for every feasible box direction, including weakly active
    and degenerate bounds.

    Parameters
    ----------
    nlp
        Symbolic nonlinear program to certify.
    point
        Candidate decision vector.
    parameters
        Numerical NLP parameters.
    feasibility_tolerance
        Maximum equality or box violation.
    stationarity_tolerance
        Relative infinity-norm tolerance for Lagrangian stationarity.
    activity_tolerance
        Distance at which a box side participates in the KKT system.
    rank_tolerance
        Relative singular-value threshold for equality independence.
    curvature_tolerance
        Strict lower bound for reduced Lagrangian curvature.

    Returns
    -------
    LocalMinimumCertificate
        Numerical measures supporting the strict local-minimum conclusion.

    Raises
    ------
    LocalMinimumError
        If any part of the sufficient local-minimum certificate fails.
    """
    x = numpy.asarray(point, dtype=float).reshape(-1)
    params = numpy.asarray(parameters, dtype=float).reshape(-1)
    n = nlp.dimension
    m = nlp.equality_dimension
    variables = sympy.ImmutableDenseMatrix(nlp.x)
    equalities = sympy.ImmutableDenseMatrix(nlp.g)
    expressions = (
        nlp.f,
        equalities,
        sympy.ImmutableDenseMatrix([sympy.diff(nlp.f, item) for item in nlp.x]),
        sympy.hessian(nlp.f, nlp.x),
        equalities.jacobian(variables) if m else sympy.zeros(0, n),
        sympy.ImmutableDenseMatrix(nlp.lb),
        sympy.ImmutableDenseMatrix(nlp.ub),
    )
    raw = sympy.lambdify((*nlp.x, *nlp.parameters), expressions, modules="numpy")(*x, *params)
    values = SimpleNamespace(
        objective=float(raw[0]),
        equalities=numpy.asarray(raw[1], dtype=float).reshape(m),
        gradient=numpy.asarray(raw[2], dtype=float).reshape(n),
        hessian=numpy.asarray(raw[3], dtype=float).reshape(n, n),
        equality_jacobian=numpy.asarray(raw[4], dtype=float).reshape(m, n),
        lower_bounds=numpy.asarray(raw[5], dtype=float).reshape(n),
        upper_bounds=numpy.asarray(raw[6], dtype=float).reshape(n),
    )

    equality_violation = float(numpy.max(numpy.abs(values.equalities))) if m else 0.0
    lower_violation = float(numpy.max(numpy.maximum(values.lower_bounds - x, 0.0)))
    upper_violation = float(numpy.max(numpy.maximum(x - values.upper_bounds, 0.0)))
    primal_residual = max(equality_violation, lower_violation, upper_violation)

    fixed = numpy.isfinite(values.lower_bounds) & numpy.isfinite(values.upper_bounds) & (numpy.abs(values.upper_bounds - values.lower_bounds) <= activity_tolerance)
    lower_active = numpy.isfinite(values.lower_bounds) & ~fixed & (x - values.lower_bounds <= activity_tolerance)
    upper_active = numpy.isfinite(values.upper_bounds) & ~fixed & (values.upper_bounds - x <= activity_tolerance)
    fixed_indices = numpy.flatnonzero(fixed)
    lower_indices = numpy.flatnonzero(lower_active)
    upper_indices = numpy.flatnonzero(upper_active)

    # Equalities and fixed variables have free multipliers; one-sided rows use
    # c(x) <= 0 so their multipliers must be nonnegative.
    rows = [*values.equality_jacobian]
    rows.extend(numpy.eye(n)[fixed_indices])
    rows.extend(-numpy.eye(n)[lower_indices])
    rows.extend(numpy.eye(n)[upper_indices])
    active_jacobian = numpy.asarray(rows, dtype=float).reshape(-1, n)
    multipliers = numpy.linalg.lstsq(active_jacobian.T, -values.gradient, rcond=rank_tolerance)[0] if rows else numpy.empty(0)
    stationarity = values.gradient + active_jacobian.T @ multipliers

    equality_multipliers = multipliers[:m]
    offset = m + fixed_indices.size
    fixed_values = multipliers[m:offset]
    lower_values = multipliers[offset : offset + lower_indices.size]
    upper_values = multipliers[offset + lower_indices.size :]
    lower_multipliers = numpy.zeros(n)
    upper_multipliers = numpy.zeros(n)
    fixed_multipliers = numpy.zeros(n)
    fixed_multipliers[fixed_indices] = fixed_values
    lower_multipliers[lower_indices] = lower_values
    upper_multipliers[upper_indices] = upper_values

    equality_rank = numpy.linalg.matrix_rank(values.equality_jacobian, tol=rank_tolerance)
    lagrangian_hessian = values.hessian.copy()
    equality_hessians = tuple(numpy.asarray(sympy.lambdify((*nlp.x, *nlp.parameters), sympy.hessian(item, nlp.x), modules="numpy")(*x, *params), dtype=float) for item in nlp.g)
    for multiplier, hessian in zip(equality_multipliers, equality_hessians, strict=True):
        lagrangian_hessian += multiplier * hessian

    _, singular_values, right_vectors = numpy.linalg.svd(values.equality_jacobian, full_matrices=True)
    numerical_rank = int(numpy.count_nonzero(singular_values > rank_tolerance * max(1.0, singular_values[0]))) if singular_values.size else 0
    tangent = right_vectors[numerical_rank:].T
    reduced_hessian = tangent.T @ lagrangian_hessian @ tangent
    minimum_curvature = float(numpy.min(numpy.linalg.eigvalsh(reduced_hessian))) if reduced_hessian.size else numpy.inf

    certificate = LocalMinimumCertificate(
        primal_residual=primal_residual,
        stationarity_residual=float(numpy.linalg.norm(stationarity, ord=numpy.inf)),
        equality_rank=int(equality_rank),
        equality_multipliers=equality_multipliers,
        lower_bound_multipliers=lower_multipliers,
        upper_bound_multipliers=upper_multipliers,
        fixed_bound_multipliers=fixed_multipliers,
        minimum_reduced_curvature=minimum_curvature,
    )

    scale = max(1.0, float(numpy.linalg.norm(values.gradient, ord=numpy.inf)))
    failures = []

    if certificate.primal_residual > feasibility_tolerance:
        failures.append(f"primal residual {certificate.primal_residual:.3g} exceeds {feasibility_tolerance:.3g}")

    if certificate.equality_rank != m:
        failures.append(f"equality Jacobian rank is {certificate.equality_rank}, expected {m}")

    if certificate.stationarity_residual > stationarity_tolerance * scale:
        failures.append(f"stationarity residual {certificate.stationarity_residual:.3g} exceeds {stationarity_tolerance * scale:.3g}")

    if lower_values.size and float(numpy.min(lower_values)) < -stationarity_tolerance:
        failures.append(f"lower-bound multiplier is negative ({float(numpy.min(lower_values)):.3g})")

    if upper_values.size and float(numpy.min(upper_values)) < -stationarity_tolerance:
        failures.append(f"upper-bound multiplier is negative ({float(numpy.min(upper_values)):.3g})")

    if certificate.minimum_reduced_curvature <= curvature_tolerance:
        failures.append(f"minimum reduced curvature {certificate.minimum_reduced_curvature:.3g} is not greater than {curvature_tolerance:.3g}")

    if failures:
        raise LocalMinimumError("; ".join(failures))

    return certificate
