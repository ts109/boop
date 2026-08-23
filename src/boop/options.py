"""Solver configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Configure generation and execution of a Boop solver.

    Parameters
    ----------
    sqp_iterations
        Fixed number of outer SQP iterations.
    cg_iterations
        Maximum number of Steihaug CG iterations per tangential step.
    initial_trust_radius
        Trust-region radius used for the first SQP iteration.
    trust_expand
        Factor relating an accepted step norm to the next trust radius.
    trust_shrink
        Factor applied to the trust radius after rejection or active-set change.
    equality_regularization
        Diagonal regularization added to the equality Gram matrix.
    tangential_damping
        Objective-Hessian damping used by the tangential subproblem.
    curvature_floor
        Relative curvature threshold used to detect negative or negligible
        curvature in Steihaug CG.
    factorization_tolerance
        Smallest acceptable diagonal pivot in the LDL factorization.
    filter_beta
        Fractional constraint-violation improvement required by the filter.
    filter_gamma
        Constraint-violation margin used by the filter objective test.
    bound_tolerance
        Numerical tolerance for bound activity, blocking, and fixed-bound
        detection.
    """

    sqp_iterations: int = 20
    cg_iterations: int = 10
    initial_trust_radius: float = 1.0
    trust_expand: float = 2.0
    trust_shrink: float = 0.5
    equality_regularization: float = 1e-10
    tangential_damping: float = 1e-10
    curvature_floor: float = 1e-12
    factorization_tolerance: float = 1e-14
    filter_beta: float = 1e-2
    filter_gamma: float = 1e-5
    bound_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.sqp_iterations <= 0 or self.cg_iterations <= 0:
            message = "iteration counts must be positive"
            raise ValueError(message)
        if self.initial_trust_radius <= 0.0:
            message = "initial_trust_radius must be positive"
            raise ValueError(message)
        if self.trust_expand <= 1.0:
            message = "trust_expand must be greater than one"
            raise ValueError(message)
        if not 0.0 < self.trust_shrink < 1.0:
            message = "trust_shrink must lie between zero and one"
            raise ValueError(message)
        if (
            min(
                self.equality_regularization,
                self.tangential_damping,
                self.curvature_floor,
                self.factorization_tolerance,
                self.filter_beta,
                self.filter_gamma,
                self.bound_tolerance,
            )
            < 0.0
        ):
            message = "regularization and tolerance options must be nonnegative"
            raise ValueError(message)
