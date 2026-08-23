"""Numerical Byrd--Omojokun SQP solver."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast, overload

import numpy

from .model import NonlinearProgram
from .options import SolverOptions
from .program import EvaluatorIR, GeneratedProgram, LDLFactor, ModelValues


class Procedure(StrEnum):
    """Identify the procedure performed during an SQP iteration."""

    ACTIVE_SET_UPDATE = "active-set-update"
    BYRD_OMOJOKUN_SOC = "byrd-omojukun-soc"
    BYRD_OMOJOKUN = "byrd-omojukun"
    REJECTED = "rejected"


class CGStopReason(StrEnum):
    """Identify why projected Steihaug CG stopped."""

    ZERO_RADIUS = "zero-radius"
    ZERO_RESIDUAL = "zero-residual"
    NEGATIVE_CURVATURE = "negative-curvature"
    TRUST_BOUNDARY = "trust-boundary"
    CONVERGED = "converged"
    ITERATION_LIMIT = "iteration-limit"


class ActiveSetError(RuntimeError):
    """Indicate that the bound working set is structurally invalid."""


@dataclass(slots=True)
class CGDiagnostics:
    """Describe one projected Steihaug solve.

    Attributes
    ----------
    iterations
        Number of Hessian-vector products performed.
    stop_reason
        Numerical reason for terminating CG.
    initial_residual_norm
        Norm of the projected residual before the first iteration.
    final_residual_norm
        Norm of the last projected residual.
    final_curvature
        Directional curvature observed in the last CG iteration.
    final_rayleigh_quotient
        Last directional curvature divided by the squared direction norm.
    radius
        Tangential trust-region radius.
    """

    iterations: int
    stop_reason: CGStopReason
    initial_residual_norm: float
    final_residual_norm: float
    final_curvature: float | None
    final_rayleigh_quotient: float | None
    radius: float


@dataclass(slots=True)
class TrialDiagnostics:
    """Describe one box-feasible candidate offered to the filter.

    Attributes
    ----------
    procedure
        Step construction used for the candidate.
    step
        Full displacement from the current iterate.
    objective
        Candidate objective value.
    violation
        Candidate equality violation.
    filter_accepted
        Whether this candidate was accepted by the filter.
    """

    procedure: Procedure
    step: numpy.ndarray
    objective: float
    violation: float
    filter_accepted: bool


@dataclass(slots=True)
class IterationDiagnostics:
    """Record the outcome of one SQP iteration.

    Attributes
    ----------
    procedure
        Step procedure selected by the filter.
    accepted
        Whether any trial point was accepted.
    x
        Accepted iterate, or the unchanged current iterate after rejection.
    objective
        Objective value at ``x``.
    violation
        Infinity norm of the equality residuals at ``x``. Box feasibility is
        maintained directly and is not part of the filter.
    trust_radius
        Trust-region radius prepared for the following iteration.
    normal_step_norm
        Euclidean norm of the normal step.
    tangential_step_norm
        Euclidean norm of the tangential step.
    correction_step_norm
        Euclidean norm of the second-order correction.
    active_bounds
        Bound working set, using ``-1`` for lower, ``0`` for inactive, and
        ``1`` for upper bounds.
    active_bounds_before
        Bound working set used to form the model. This differs from
        ``active_bounds`` during a working-set transition.
    equality_multipliers
        Estimated equality multipliers.
    bound_multipliers
        Estimated active-bound multipliers.
    x_before
        Iterate at which the quadratic model was formed.
    trust_radius_before
        Trust radius used to construct candidate steps.
    objective_before
        Objective value at the model point.
    violation_before
        Equality violation at the model point.
    equality_residuals
        Equality residual vector at the resulting iterate.
    lower_bound_slack, upper_bound_slack
        Signed distances from the resulting iterate to each box side.
    gradient_norm
        Objective-gradient norm at the model point.
    projected_gradient_norm
        Equality-nullspace gradient norm at the model point.
    stationarity_norm
        Primal-dual stationarity residual norm at the model point.
    hessian_min_diagonal, hessian_max_diagonal
        Extreme objective-Hessian diagonal entries. These are inexpensive
        curvature hints, not eigenvalue bounds.
    ldl_min_pivot, ldl_max_pivot
        Extreme diagonal pivots in its LDL factorization.
    normal_step, tangential_step, correction_step
        Components produced by the Byrd--Omojokun construction.
    accepted_step
        Step committed to the iterate; zero after rejection or transition.
    cg
        Projected Steihaug diagnostic record.
    trials
        Box-feasible candidates evaluated by the filter in attempted order.
    filter_entries_before, filter_entries_after
        Filter sizes bracketing this iteration.
    """

    procedure: Procedure
    accepted: bool
    x: numpy.ndarray
    objective: float
    violation: float
    trust_radius: float
    normal_step_norm: float
    tangential_step_norm: float
    correction_step_norm: float
    active_bounds: numpy.ndarray
    active_bounds_before: numpy.ndarray
    equality_multipliers: numpy.ndarray
    bound_multipliers: numpy.ndarray
    x_before: numpy.ndarray
    trust_radius_before: float
    objective_before: float
    violation_before: float
    equality_residuals: numpy.ndarray
    lower_bound_slack: numpy.ndarray
    upper_bound_slack: numpy.ndarray
    gradient_norm: float
    projected_gradient_norm: float
    stationarity_norm: float
    hessian_min_diagonal: float
    hessian_max_diagonal: float
    ldl_min_pivot: float
    ldl_max_pivot: float
    normal_step: numpy.ndarray
    tangential_step: numpy.ndarray
    correction_step: numpy.ndarray
    accepted_step: numpy.ndarray
    cg: CGDiagnostics
    trials: list[TrialDiagnostics]
    filter_entries_before: int
    filter_entries_after: int


@dataclass(slots=True)
class SolverDiagnostics:
    """Collect diagnostics produced by a complete solver invocation.

    Attributes
    ----------
    iterations
        Diagnostics in SQP iteration order.
    initial_guess
        Unmodified point supplied by the caller.
    projected_initial_guess
        Initial point after projection into the box.
    parameters
        Numerical parameter vector supplied to the generated evaluator.
    """

    iterations: list[IterationDiagnostics] = field(default_factory=list)
    initial_guess: numpy.ndarray = field(default_factory=lambda: numpy.empty(0))
    projected_initial_guess: numpy.ndarray = field(default_factory=lambda: numpy.empty(0))
    parameters: numpy.ndarray = field(default_factory=lambda: numpy.empty(0))


@dataclass(slots=True)
class SolverResult:
    """Store a solution together with its iteration diagnostics.

    Attributes
    ----------
    x
        Final decision-variable vector.
    diagnostics
        Diagnostics collected during the fixed SQP iteration sequence.
    """

    x: numpy.ndarray
    diagnostics: SolverDiagnostics


@dataclass(slots=True)
class _Linearization:
    """Bundle a model linearization and its reusable equality factorization.

    Attributes
    ----------
    values
        Objective, derivative, constraint, and bound values at the iterate.
    active
        Bound working set at the iterate.
    free
        Boolean mask selecting variables without active bounds.
    free_jacobian
        Equality Jacobian restricted to free variables.
    factor
        LDL factorization of the regularized free-variable equality Gramian.
    """

    values: ModelValues
    active: numpy.ndarray
    free: numpy.ndarray
    free_jacobian: numpy.ndarray
    gram: numpy.ndarray
    factor: LDLFactor


@dataclass(slots=True)
class _Step:
    """Bundle the normal and tangential components of the primary step.

    Attributes
    ----------
    compound
        Sum of the normal and tangential steps.
    """

    compound: numpy.ndarray
    normal: numpy.ndarray
    tangential: numpy.ndarray
    cg: CGDiagnostics


@dataclass(slots=True)
class _Candidate:
    """Bundle an evaluated step offered to the filter.

    Attributes
    ----------
    procedure
        Construction used for the candidate.
    step
        Displacement from the current iterate.
    values
        NLP evaluation at the candidate point.
    """

    procedure: Procedure
    step: numpy.ndarray
    values: ModelValues


class _Filter:
    def __init__(self, beta: float, gamma: float, feasibility_floor: float) -> None:
        self.beta = beta
        self.gamma = gamma
        self.feasibility_floor = feasibility_floor
        self.entries: list[tuple[float, float]] = []

    def add_initial(self, objective: float, violation: float) -> None:
        self.entries.append((objective, violation))

    def accepts(self, objective: float, violation: float) -> bool:
        # A candidate must improve objective or feasibility over every entry.
        for old_objective, old_violation in self.entries:
            improves_objective = objective + self.gamma * violation <= old_objective
            improves_feasibility = old_violation > self.feasibility_floor and violation <= (1.0 - self.beta) * old_violation

            if not (improves_objective or improves_feasibility):
                return False

        self.entries.append((objective, violation))
        return True


class Solver:
    """Callable generated Byrd--Omojokun optimizer."""

    def __init__(self, program: GeneratedProgram, options: SolverOptions) -> None:
        self.program = program
        self.options = options

    @property
    def nlp(self) -> NonlinearProgram:
        """Return the nonlinear program represented by this solver."""
        return self.program.nlp

    @property
    def ast(self) -> EvaluatorIR:
        """The SymPy evaluator IR used by this generated solver."""
        return self.program.ir

    @overload
    def __call__(
        self,
        initial_guess: Sequence[float] | numpy.ndarray,
        parameters: Sequence[float] | numpy.ndarray = (),
        *,
        diagnostics: Literal[False] = False,
    ) -> numpy.ndarray: ...

    @overload
    def __call__(
        self,
        initial_guess: Sequence[float] | numpy.ndarray,
        parameters: Sequence[float] | numpy.ndarray = (),
        *,
        diagnostics: Literal[True],
    ) -> SolverResult: ...

    def __call__(
        self,
        initial_guess: Sequence[float] | numpy.ndarray,
        parameters: Sequence[float] | numpy.ndarray = (),
        *,
        diagnostics: bool = False,
    ) -> numpy.ndarray | SolverResult:
        """Solve the NLP and optionally return iteration diagnostics."""
        result = self.solve(initial_guess, parameters)
        return result if diagnostics else result.x

    def solve(  # noqa: PLR0915
        self,
        initial_guess: Sequence[float] | numpy.ndarray,
        parameters: Sequence[float] | numpy.ndarray = (),
    ) -> SolverResult:
        """Execute the fixed number of generated SQP iterations."""
        x = numpy.asarray(initial_guess, dtype=float).reshape(-1).copy()
        supplied_initial_guess = x.copy()
        params = numpy.asarray(parameters, dtype=float).reshape(-1)

        if x.size != self.nlp.dimension:
            message = f"expected {self.nlp.dimension} initial values, got {x.size}"
            raise ValueError(message)

        if params.size != len(self.nlp.parameters):
            message = f"expected {len(self.nlp.parameters)} parameters, got {params.size}"
            raise ValueError(message)

        if not numpy.all(numpy.isfinite(x)) or not numpy.all(numpy.isfinite(params)):
            message = "initial values and parameters must be finite"
            raise ValueError(message)

        # Bounds are enforced geometrically and never delegated to the filter.
        initial = self.program.evaluate(x, params)
        x = numpy.minimum(numpy.maximum(x, initial.lower_bounds), initial.upper_bounds)
        initial = self.program.evaluate(x, params)

        active = numpy.zeros(self.nlp.dimension, dtype=numpy.int8)
        fixed = (
            numpy.isfinite(initial.lower_bounds) & numpy.isfinite(initial.upper_bounds) & (numpy.abs(initial.upper_bounds - initial.lower_bounds) <= self.options.bound_tolerance)
        )
        active[fixed] = -1

        trust_radius = self.options.initial_trust_radius
        candidate_filter = _Filter(
            self.options.filter_beta,
            self.options.filter_gamma,
            self.options.bound_tolerance,
        )
        candidate_filter.add_initial(initial.objective, _equality_violation(initial))
        diagnostics = SolverDiagnostics(
            initial_guess=supplied_initial_guess,
            projected_initial_guess=x.copy(),
            parameters=params.copy(),
        )

        for _ in range(self.options.sqp_iterations):
            # This factorization is shared by normal, tangential, and dual solves.
            x_before = x.copy()
            trust_radius_before = trust_radius
            filter_entries_before = len(candidate_filter.entries)
            current = self.program.evaluate(x, params)
            linearization = self._linearize(current, active)
            equality_multipliers, bound_multipliers = self._multipliers(linearization)
            projected_gradient = self._project(current.gradient, linearization)
            projected_gradient_norm = float(numpy.linalg.norm(projected_gradient))
            removable = self._removable_bound(
                current,
                x,
                linearization,
                fixed=fixed,
                multipliers=bound_multipliers,
                projected_gradient_norm=projected_gradient_norm,
            )

            if removable is not None:
                # Working-set changes consume a predictable outer iteration.
                active[removable] = 0
                diagnostics.iterations.append(
                    self._iteration_diagnostics(
                        procedure=Procedure.ACTIVE_SET_UPDATE,
                        accepted=False,
                        x_before=x_before,
                        x_after=x,
                        current=current,
                        result=current,
                        trust_radius_before=trust_radius_before,
                        trust_radius_after=trust_radius,
                        linearization=linearization,
                        step=None,
                        correction_step=numpy.zeros_like(x),
                        accepted_step=numpy.zeros_like(x),
                        equality_multipliers=equality_multipliers.copy(),
                        bound_multipliers=bound_multipliers.copy(),
                        active=active,
                        projected_gradient_norm=projected_gradient_norm,
                        trials=[],
                        filter_entries_before=filter_entries_before,
                        filter_entries_after=len(candidate_filter.entries),
                    )
                )
                continue

            step = self._step(x, linearization, trust_radius)
            blocking = self._blocking_bounds(x, step.compound, current, active)

            if blocking.size:
                # Recompute next iteration with all tied blockers fixed.
                for index in blocking:
                    active[index] = -1 if step.compound[index] < 0.0 else 1
                if self.nlp.equality_dimension + numpy.count_nonzero(active) > self.nlp.dimension:
                    message = "blocking bounds overdetermined the working set"
                    raise ActiveSetError(message)
                diagnostics.iterations.append(
                    self._iteration_diagnostics(
                        procedure=Procedure.ACTIVE_SET_UPDATE,
                        accepted=False,
                        x_before=x_before,
                        x_after=x,
                        current=current,
                        result=current,
                        trust_radius_before=trust_radius_before,
                        trust_radius_after=trust_radius,
                        linearization=linearization,
                        step=step,
                        correction_step=numpy.zeros_like(x),
                        accepted_step=numpy.zeros_like(x),
                        equality_multipliers=equality_multipliers.copy(),
                        bound_multipliers=bound_multipliers.copy(),
                        active=active,
                        projected_gradient_norm=projected_gradient_norm,
                        trials=[],
                        filter_entries_before=filter_entries_before,
                        filter_entries_after=len(candidate_filter.entries),
                    )
                )
                continue

            # Evaluate BO once; nonlinear equalities add one corrected candidate.
            bo_values = self.program.evaluate(x + step.compound, params)
            correction = numpy.zeros_like(x)
            if self.program.has_nonlinear_equalities:
                correction = self._second_order_correction(x, step.compound, bo_values, linearization, trust_radius)
            candidates = self._candidates(x, step.compound, correction, bo_values, params)
            accepted = False
            chosen_procedure = Procedure.REJECTED
            chosen_step = numpy.zeros_like(x)
            chosen_values = current
            chosen_x = x
            trial_diagnostics = []

            for candidate in candidates:
                trial_violation = _equality_violation(candidate.values)
                filter_accepted = candidate_filter.accepts(candidate.values.objective, trial_violation)
                trial_diagnostics.append(
                    TrialDiagnostics(
                        procedure=candidate.procedure,
                        step=candidate.step.copy(),
                        objective=candidate.values.objective,
                        violation=trial_violation,
                        filter_accepted=filter_accepted,
                    )
                )

                if filter_accepted:
                    accepted = True
                    chosen_procedure = candidate.procedure
                    chosen_step = candidate.step
                    chosen_values = candidate.values
                    chosen_x = x + candidate.step
                    break

            if accepted:
                x = chosen_x.copy()
                actual_norm = float(numpy.linalg.norm(chosen_step))
                trust_radius = self.options.trust_expand * actual_norm
            else:
                # Scheduled but unreached bounds may not block the smaller step.
                trust_radius *= self.options.trust_shrink
                scheduled = (active != 0) & ~fixed & ~self._geometrically_active(current, x, active)
                active[scheduled] = 0

            trust_radius = max(trust_radius, numpy.finfo(float).eps)
            diagnostics.iterations.append(
                self._iteration_diagnostics(
                    procedure=chosen_procedure,
                    accepted=accepted,
                    x_before=x_before,
                    x_after=x,
                    current=current,
                    result=chosen_values if accepted else current,
                    trust_radius_before=trust_radius_before,
                    trust_radius_after=trust_radius,
                    linearization=linearization,
                    step=step,
                    correction_step=correction,
                    accepted_step=chosen_step if accepted else numpy.zeros_like(x),
                    equality_multipliers=equality_multipliers.copy(),
                    bound_multipliers=bound_multipliers.copy(),
                    active=active,
                    projected_gradient_norm=projected_gradient_norm,
                    trials=trial_diagnostics,
                    filter_entries_before=filter_entries_before,
                    filter_entries_after=len(candidate_filter.entries),
                )
            )

        return SolverResult(x, diagnostics)

    def _iteration_diagnostics(
        self,
        *,
        procedure: Procedure,
        accepted: bool,
        x_before: numpy.ndarray,
        x_after: numpy.ndarray,
        current: ModelValues,
        result: ModelValues,
        trust_radius_before: float,
        trust_radius_after: float,
        linearization: _Linearization,
        step: _Step | None,
        correction_step: numpy.ndarray,
        accepted_step: numpy.ndarray,
        equality_multipliers: numpy.ndarray,
        bound_multipliers: numpy.ndarray,
        active: numpy.ndarray,
        projected_gradient_norm: float,
        trials: list[TrialDiagnostics],
        filter_entries_before: int,
        filter_entries_after: int,
    ) -> IterationDiagnostics:
        """Collect diagnostic measures without matrix decompositions."""
        zero = numpy.zeros(self.nlp.dimension)
        normal = zero if step is None else step.normal
        tangential = zero if step is None else step.tangential
        cg = _empty_cg() if step is None else step.cg

        stationarity = current.gradient + current.equality_jacobian.T @ equality_multipliers
        stationarity = stationarity.copy()
        # Multipliers belong to the model working set, before any transition.
        model_active = linearization.active
        stationarity[model_active < 0] -= bound_multipliers[model_active < 0]
        stationarity[model_active > 0] += bound_multipliers[model_active > 0]

        hessian_diagonal = numpy.diag(current.hessian)
        pivots = linearization.factor.diagonal

        return IterationDiagnostics(
            procedure=procedure,
            accepted=accepted,
            x=x_after.copy(),
            objective=result.objective,
            violation=_equality_violation(result),
            trust_radius=trust_radius_after,
            normal_step_norm=float(numpy.linalg.norm(normal)),
            tangential_step_norm=float(numpy.linalg.norm(tangential)),
            correction_step_norm=float(numpy.linalg.norm(correction_step)),
            active_bounds=active.copy(),
            active_bounds_before=model_active.copy(),
            equality_multipliers=equality_multipliers.copy(),
            bound_multipliers=bound_multipliers.copy(),
            x_before=x_before.copy(),
            trust_radius_before=trust_radius_before,
            objective_before=current.objective,
            violation_before=_equality_violation(current),
            equality_residuals=result.equalities.copy(),
            lower_bound_slack=x_after - result.lower_bounds,
            upper_bound_slack=result.upper_bounds - x_after,
            gradient_norm=float(numpy.linalg.norm(current.gradient)),
            projected_gradient_norm=projected_gradient_norm,
            stationarity_norm=float(numpy.linalg.norm(stationarity)),
            hessian_min_diagonal=float(numpy.min(hessian_diagonal)),
            hessian_max_diagonal=float(numpy.max(hessian_diagonal)),
            ldl_min_pivot=float(numpy.min(pivots)) if pivots.size else numpy.nan,
            ldl_max_pivot=float(numpy.max(pivots)) if pivots.size else numpy.nan,
            normal_step=normal.copy(),
            tangential_step=tangential.copy(),
            correction_step=correction_step.copy(),
            accepted_step=accepted_step.copy(),
            cg=cg,
            trials=trials,
            filter_entries_before=filter_entries_before,
            filter_entries_after=filter_entries_after,
        )

    def _linearize(self, values: ModelValues, active: numpy.ndarray) -> _Linearization:
        number_active = int(numpy.count_nonzero(active))
        if self.nlp.equality_dimension + number_active > self.nlp.dimension:
            message = "equalities plus active bounds exceed the number of variables"
            raise ActiveSetError(message)
        free = active == 0
        free_jacobian = values.equality_jacobian[:, free]

        # Active box rows reduce to deleting columns from the equality Jacobian.
        gram = self.program.assemble_equality_gram(
            values.equality_jacobian,
            free,
            self.options.equality_regularization,
        )
        factor = self.program.ldl.factor(gram, self.options.factorization_tolerance)
        return _Linearization(
            values=values,
            active=active.copy(),
            free=free,
            free_jacobian=free_jacobian,
            gram=gram,
            factor=factor,
        )

    def _project(self, vector: numpy.ndarray, linearization: _Linearization) -> numpy.ndarray:
        """Project a vector into the free-variable equality nullspace."""
        projected = numpy.zeros_like(vector)
        free_values = vector[linearization.free]

        if self.nlp.equality_dimension:
            correction = linearization.factor.solve(linearization.free_jacobian @ free_values)
            free_values = free_values - linearization.free_jacobian.T @ correction

        projected[linearization.free] = free_values
        return projected

    def _step(
        self,
        x: numpy.ndarray,
        linearization: _Linearization,
        trust_radius: float,
    ) -> _Step:
        """Construct the primary Byrd--Omojokun step."""
        values = linearization.values
        normal = self._normal_step(x, linearization)
        normal_norm = float(numpy.linalg.norm(normal))

        if normal_norm > trust_radius:
            normal *= trust_radius / normal_norm
            normal_norm = trust_radius

        # Orthogonality leaves this radius for motion in the nullspace.
        tangential_radius = float(numpy.sqrt(max(0.0, trust_radius * trust_radius - normal_norm * normal_norm)))
        gradient_at_normal = values.gradient + values.hessian @ normal
        tangent, cg = self._steihaug(gradient_at_normal, values.hessian, linearization, tangential_radius)
        compound = normal + tangent

        return _Step(
            compound=compound,
            normal=normal,
            tangential=tangent,
            cg=cg,
        )

    def _normal_step(self, x: numpy.ndarray, linearization: _Linearization) -> numpy.ndarray:
        """Find the minimum-norm correction of active bounds and equalities."""
        values = linearization.values
        normal = numpy.zeros_like(x)
        lower_active = linearization.active < 0
        upper_active = linearization.active > 0

        # Bound components are known exactly; solve equality correction on F.
        normal[lower_active] = values.lower_bounds[lower_active] - x[lower_active]
        normal[upper_active] = values.upper_bounds[upper_active] - x[upper_active]

        if self.nlp.equality_dimension:
            fixed_effect = values.equality_jacobian[:, ~linearization.free] @ normal[~linearization.free]
            dual_normal = linearization.factor.solve(-values.equalities - fixed_effect)
            normal[linearization.free] = linearization.free_jacobian.T @ dual_normal

        return normal

    def _second_order_correction(
        self,
        x: numpy.ndarray,
        compound: numpy.ndarray,
        trial: ModelValues,
        linearization: _Linearization,
        trust_radius: float,
    ) -> numpy.ndarray:
        """Correct nonlinear equality error left by the linearized step."""
        values = linearization.values
        trial_x = x + compound
        correction = numpy.zeros_like(x)
        lower_active = linearization.active < 0
        upper_active = linearization.active > 0

        correction[lower_active] = values.lower_bounds[lower_active] - trial_x[lower_active]
        correction[upper_active] = values.upper_bounds[upper_active] - trial_x[upper_active]

        if self.nlp.equality_dimension:
            fixed_effect = values.equality_jacobian[:, ~linearization.free] @ correction[~linearization.free]
            dual_correction = linearization.factor.solve(-trial.equalities - fixed_effect)
            correction[linearization.free] = linearization.free_jacobian.T @ dual_correction

        correction_norm = float(numpy.linalg.norm(correction))
        if correction_norm > trust_radius:
            correction *= trust_radius / correction_norm

        return correction

    def _candidates(
        self,
        x: numpy.ndarray,
        compound: numpy.ndarray,
        correction: numpy.ndarray,
        bo_values: ModelValues,
        parameters: numpy.ndarray,
    ) -> list[_Candidate]:
        """Evaluate the fixed SOC-first trial sequence for this NLP."""
        candidates = []

        if self.program.has_nonlinear_equalities:
            corrected = compound + correction
            soc_values = self.program.evaluate(x + corrected, parameters)
            if self._meaningful_step(x, corrected) and self._box_feasible(x + corrected, bo_values):
                candidates.append(_Candidate(Procedure.BYRD_OMOJOKUN_SOC, corrected, soc_values))

        if self._meaningful_step(x, compound):
            candidates.append(_Candidate(Procedure.BYRD_OMOJOKUN, compound, bo_values))

        return candidates

    def _steihaug(
        self,
        gradient: numpy.ndarray,
        hessian: numpy.ndarray,
        linearization: _Linearization,
        radius: float,
    ) -> tuple[numpy.ndarray, CGDiagnostics]:
        """Approximately minimize the projected quadratic in a trust ball."""
        tangent = numpy.zeros(self.nlp.dimension)

        if radius <= numpy.finfo(float).eps:
            return tangent, _empty_cg(radius)

        residual = -self._project(gradient, linearization)
        direction = residual.copy()
        residual_norm_sq = float(residual @ residual)
        initial_residual_norm = float(numpy.sqrt(residual_norm_sq))

        if residual_norm_sq <= numpy.finfo(float).eps:
            return tangent, CGDiagnostics(
                iterations=0,
                stop_reason=CGStopReason.ZERO_RESIDUAL,
                initial_residual_norm=initial_residual_norm,
                final_residual_norm=initial_residual_norm,
                final_curvature=None,
                final_rayleigh_quotient=None,
                radius=radius,
            )

        hessian_norm = float(numpy.linalg.norm(hessian))
        curvature = None

        for iteration in range(1, self.options.cg_iterations + 1):
            # Projection implements P H P without materializing a nullspace basis.
            action = hessian @ direction + self.options.tangential_damping * hessian_norm * direction
            action = self._project(action, linearization)
            curvature = float(direction @ action)
            direction_norm_sq = float(direction @ direction)

            if curvature <= self.options.curvature_floor * direction_norm_sq:
                # Negative curvature is useful only up to the trust boundary.
                tangent += _boundary_distance(tangent, direction, radius) * direction
                diagnostics = CGDiagnostics(
                    iterations=iteration,
                    stop_reason=CGStopReason.NEGATIVE_CURVATURE,
                    initial_residual_norm=initial_residual_norm,
                    final_residual_norm=float(numpy.sqrt(residual_norm_sq)),
                    final_curvature=curvature,
                    final_rayleigh_quotient=curvature / direction_norm_sq,
                    radius=radius,
                )
                return tangent, diagnostics

            alpha = residual_norm_sq / curvature
            trial = tangent + alpha * direction

            if float(trial @ trial) >= radius * radius:
                # Interpolate the current CG ray to the trust boundary.
                tangent += _boundary_distance(tangent, direction, radius) * direction
                diagnostics = CGDiagnostics(
                    iterations=iteration,
                    stop_reason=CGStopReason.TRUST_BOUNDARY,
                    initial_residual_norm=initial_residual_norm,
                    final_residual_norm=float(numpy.sqrt(residual_norm_sq)),
                    final_curvature=curvature,
                    final_rayleigh_quotient=curvature / direction_norm_sq,
                    radius=radius,
                )
                return tangent, diagnostics

            tangent = trial
            new_residual = self._project(residual - alpha * action, linearization)
            new_norm_sq = float(new_residual @ new_residual)

            if new_norm_sq <= numpy.finfo(float).eps:
                diagnostics = CGDiagnostics(
                    iterations=iteration,
                    stop_reason=CGStopReason.CONVERGED,
                    initial_residual_norm=initial_residual_norm,
                    final_residual_norm=float(numpy.sqrt(new_norm_sq)),
                    final_curvature=curvature,
                    final_rayleigh_quotient=curvature / direction_norm_sq,
                    radius=radius,
                )
                return tangent, diagnostics

            direction = new_residual + (new_norm_sq / residual_norm_sq) * direction
            residual = new_residual
            residual_norm_sq = new_norm_sq

        assert curvature is not None
        diagnostics = CGDiagnostics(
            iterations=self.options.cg_iterations,
            stop_reason=CGStopReason.ITERATION_LIMIT,
            initial_residual_norm=initial_residual_norm,
            final_residual_norm=float(numpy.sqrt(residual_norm_sq)),
            final_curvature=curvature,
            final_rayleigh_quotient=curvature / direction_norm_sq,
            radius=radius,
        )
        return tangent, diagnostics

    def _multipliers(self, linearization: _Linearization) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Estimate equality and signed active-bound multipliers."""
        values = linearization.values

        if self.nlp.equality_dimension:
            equality = linearization.factor.solve(-linearization.free_jacobian @ values.gradient[linearization.free])
        else:
            equality = numpy.empty(0)

        stationarity = values.gradient + values.equality_jacobian.T @ equality
        bounds = numpy.zeros(self.nlp.dimension)

        # Positive values satisfy the KKT sign on either side of the box.
        bounds[linearization.active < 0] = stationarity[linearization.active < 0]
        bounds[linearization.active > 0] = -stationarity[linearization.active > 0]

        return equality, bounds

    def _geometrically_active(self, values: ModelValues, x: numpy.ndarray, active: numpy.ndarray) -> numpy.ndarray:
        """Distinguish reached bounds from bounds scheduled by a blocker."""
        tolerance = self.options.bound_tolerance
        lower = (active < 0) & (numpy.abs(x - values.lower_bounds) <= tolerance)
        upper = (active > 0) & (numpy.abs(x - values.upper_bounds) <= tolerance)
        return cast(numpy.ndarray, lower | upper)

    def _removable_bound(
        self,
        values: ModelValues,
        x: numpy.ndarray,
        linearization: _Linearization,
        *,
        fixed: numpy.ndarray,
        multipliers: numpy.ndarray,
        projected_gradient_norm: float,
    ) -> int | None:
        """Release the worst invalid bound after optimizing its current face."""
        # Multiplier signs identify the correct neighboring face only after the
        # current face is stationary. This also prevents remove/re-add cycles.
        gradient_scale = max(1.0, float(numpy.linalg.norm(values.gradient)))
        if projected_gradient_norm > self.options.active_set_stationarity_tolerance * gradient_scale:
            return None

        active = linearization.active
        removable = self._geometrically_active(values, x, active) & ~fixed & (multipliers < -self.options.bound_tolerance)
        indices = numpy.flatnonzero(removable)
        if not indices.size:
            return None
        return int(indices[numpy.argmin(multipliers[indices])])

    def _blocking_bounds(self, x: numpy.ndarray, step: numpy.ndarray, values: ModelValues, active: numpy.ndarray) -> numpy.ndarray:
        """Return all free bounds first reached by the proposed step."""
        tolerance = self.options.bound_tolerance
        fractions = numpy.full(self.nlp.dimension, numpy.inf)
        free = active == 0
        toward_upper = free & (step > tolerance) & numpy.isfinite(values.upper_bounds)
        toward_lower = free & (step < -tolerance) & numpy.isfinite(values.lower_bounds)
        fractions[toward_upper] = (values.upper_bounds[toward_upper] - x[toward_upper]) / step[toward_upper]
        fractions[toward_lower] = (values.lower_bounds[toward_lower] - x[toward_lower]) / step[toward_lower]

        first = float(numpy.min(fractions))
        if not numpy.isfinite(first) or first > 1.0 + tolerance:
            return numpy.empty(0, dtype=int)
        tie_tolerance = tolerance * max(1.0, abs(first))
        return numpy.flatnonzero(numpy.abs(fractions - first) <= tie_tolerance)

    def _box_feasible(self, x: numpy.ndarray, values: ModelValues) -> bool:
        """Check the invariant required before evaluating a filter candidate."""
        tolerance = self.options.bound_tolerance
        return bool(numpy.all(x >= values.lower_bounds - tolerance) and numpy.all(x <= values.upper_bounds + tolerance))

    @staticmethod
    def _meaningful_step(x: numpy.ndarray, step: numpy.ndarray) -> bool:
        """Keep an unchanged iterate out of the filter and radius update."""
        scale = max(1.0, float(numpy.linalg.norm(x)))
        return bool(float(numpy.linalg.norm(step)) > numpy.finfo(float).eps * scale)


def _boundary_distance(point: numpy.ndarray, direction: numpy.ndarray, radius: float) -> float:
    """Return the nonnegative ray parameter intersecting a Euclidean ball."""
    a = float(direction @ direction)

    if a == 0.0:
        return 0.0

    b = float(point @ direction)
    discriminant = max(0.0, b * b + a * (radius * radius - float(point @ point)))

    return float(max(0.0, (-b + numpy.sqrt(discriminant)) / a))


def _empty_cg(radius: float = 0.0) -> CGDiagnostics:
    """Return the CG record used by iterations that construct no step."""
    return CGDiagnostics(
        iterations=0,
        stop_reason=CGStopReason.ZERO_RADIUS,
        initial_residual_norm=0.0,
        final_residual_norm=0.0,
        final_curvature=None,
        final_rayleigh_quotient=None,
        radius=radius,
    )


def _equality_violation(values: ModelValues) -> float:
    """Return the filter's infinity-norm feasibility measure."""
    return float(numpy.max(numpy.abs(values.equalities))) if values.equalities.size else 0.0


def create_solver(nlp: NonlinearProgram, options: SolverOptions | None = None) -> Solver:
    """Symbolically derive and create a callable, problem-specific SQP solver."""
    return Solver(GeneratedProgram(nlp), options or SolverOptions())
