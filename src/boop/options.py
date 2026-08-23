from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverOptions:
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
    active_set_freeze: int = 2
    activation_delay: int = 1
    bound_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.sqp_iterations <= 0 or self.cg_iterations <= 0:
            raise ValueError("iteration counts must be positive")
        if self.initial_trust_radius <= 0.0:
            raise ValueError("initial_trust_radius must be positive")
        if self.trust_expand <= 1.0:
            raise ValueError("trust_expand must be greater than one")
        if not 0.0 < self.trust_shrink < 1.0:
            raise ValueError("trust_shrink must lie between zero and one")
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
            raise ValueError("regularization and tolerance options must be nonnegative")
        if self.active_set_freeze < 0 or self.activation_delay < 0:
            raise ValueError("active-set delays must be nonnegative")
