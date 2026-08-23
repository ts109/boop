"""Byrd--Omojokun Optimizer."""

from .diagnostics import format_solver_diagnostics, print_solver_diagnostics
from .model import NonlinearProgram
from .optimality import LocalMinimumCertificate, LocalMinimumError, certify_local_minimum
from .options import SolverOptions
from .program import EvaluatorIR, GeneratedProgram, SparseLDLProgram
from .solver import (
    ActiveSetError,
    CGDiagnostics,
    CGStopReason,
    IterationDiagnostics,
    Procedure,
    Solver,
    SolverDiagnostics,
    SolverResult,
    TrialDiagnostics,
    create_solver,
)

__all__ = [
    "ActiveSetError",
    "CGDiagnostics",
    "CGStopReason",
    "EvaluatorIR",
    "GeneratedProgram",
    "IterationDiagnostics",
    "LocalMinimumCertificate",
    "LocalMinimumError",
    "NonlinearProgram",
    "Procedure",
    "Solver",
    "SolverDiagnostics",
    "SolverOptions",
    "SolverResult",
    "SparseLDLProgram",
    "TrialDiagnostics",
    "certify_local_minimum",
    "create_solver",
    "format_solver_diagnostics",
    "print_solver_diagnostics",
]
