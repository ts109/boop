"""JIT compilation and Python adaptation for generated C solvers."""

import hashlib
import importlib.util
import os
import shlex
import subprocess
import sysconfig
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast, overload

from .c_codegen import GeneratedCCode, generate_c_solver
from .model import NonlinearProgram
from .options import SolverOptions


class NativeSolverDiagnostics(Protocol):
    """Expose lazily converted fields of the native diagnostic structure."""

    @property
    def procedure(self) -> list[str]:
        """Return iteration procedure names."""
        ...

    @property
    def accepted(self) -> list[int]:
        """Return iteration acceptance flags."""
        ...

    @property
    def x(self) -> list[list[float]]:
        """Return the iterate history."""
        ...

    @property
    def objective(self) -> list[float]:
        """Return the objective history."""
        ...

    @property
    def violation(self) -> list[float]:
        """Return the equality-violation history."""
        ...

    @property
    def trust_radius(self) -> list[float]:
        """Return the trust-radius history."""
        ...

    @property
    def normal_step_norm(self) -> list[float]:
        """Return normal-step norms."""
        ...

    @property
    def tangential_step_norm(self) -> list[float]:
        """Return tangential-step norms."""
        ...

    @property
    def correction_step_norm(self) -> list[float]:
        """Return second-order-correction norms."""
        ...

    @property
    def active_bounds(self) -> list[list[int]]:
        """Return active-bound signs by iteration."""
        ...

    @property
    def cg_iterations(self) -> list[int]:
        """Return tangential CG iteration counts."""
        ...

    @property
    def cg_stop_reason(self) -> list[int]:
        """Return native tangential CG stop codes."""
        ...

    @property
    def cg_initial_residual(self) -> list[float]:
        """Return initial tangential CG residual norms."""
        ...

    @property
    def cg_final_residual(self) -> list[float]:
        """Return final tangential CG residual norms."""
        ...

    @property
    def cg_final_curvature(self) -> list[float]:
        """Return the final curvature sampled by CG."""
        ...

    @property
    def cg_final_rayleigh(self) -> list[float]:
        """Return the final Rayleigh quotient sampled by CG."""
        ...


class _NativeSolver(Protocol):
    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float], diagnostics: bool = False) -> Any: ...


@dataclass(frozen=True, slots=True)
class CompiledSolverResult:
    """Store a compiled solution and its native lazy diagnostics.

    Attributes
    ----------
    x
        Final decision vector.
    diagnostics
        Extension-backed diagnostic structure.
    """

    x: list[float]
    diagnostics: NativeSolverDiagnostics


class CompiledSolver:
    """Present a typed facade over the NumPy-independent extension."""

    def __init__(self, native: _NativeSolver) -> None:
        self._native = native

    @overload
    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float] = (), *, diagnostics: Literal[False] = False) -> list[float]: ...

    @overload
    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float] = (), *, diagnostics: Literal[True]) -> CompiledSolverResult: ...

    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float] = (), *, diagnostics: bool = False) -> list[float] | CompiledSolverResult:
        """Run the native solver using ordinary Python sequence inputs."""
        raw = self._native(initial_guess, parameters, diagnostics)
        if not diagnostics:
            return cast(list[float], raw)
        solution, native_diagnostics = raw
        return CompiledSolverResult(cast(list[float], solution), cast(NativeSolverDiagnostics, native_diagnostics))


def create_compiled_solver(nlp: NonlinearProgram, options: SolverOptions | None = None, *, cache_directory: str | Path | None = None) -> CompiledSolver:
    """Generate, compile, load, and instantiate a native solver extension."""
    code = generate_c_solver(nlp, options)
    module = _compile_extension(code, cache_directory)
    native_type = cast(Any, module).Solver
    return CompiledSolver(cast(_NativeSolver, native_type()))


def _compile_extension(code: GeneratedCCode, cache_directory: str | Path | None) -> ModuleType:
    extension_template = resources.files("boop").joinpath("c_runtime", "boop_extension.c").read_text()
    digest = hashlib.sha256()
    for name, content in sorted((*code.headers.items(), *code.sources.items())):
        digest.update(name.encode())
        digest.update(content.encode())
    digest.update(extension_template.encode())
    digest.update(str(sysconfig.get_config_var("SOABI")).encode())
    digest.update(str(sysconfig.get_config_var("LDSHARED")).encode())
    key = digest.hexdigest()[:20]
    module_name = f"boop_jit_{key}"
    root = Path(cache_directory) if cache_directory is not None else Path(tempfile.gettempdir()) / "boop-jit"
    build = root / key
    extension_suffix = cast(str, sysconfig.get_config_var("EXT_SUFFIX"))
    library = build / f"{module_name}{extension_suffix}"
    if not library.exists():
        build.mkdir(parents=True, exist_ok=True)
        code.write(build)
        extension = f'#define BOOP_MODULE_TOKEN {module_name}\n#define BOOP_MODULE_STRING "{module_name}"\n{extension_template}'
        (build / "boop_extension.c").write_text(extension)
        compiler = shlex.split(cast(str, sysconfig.get_config_var("LDSHARED")))
        include_python = cast(str, sysconfig.get_config_var("INCLUDEPY"))
        temporary_library = library.with_name(f"{library.name}.{os.getpid()}.tmp")
        command = [
            *compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-I",
            str(build),
            "-I",
            include_python,
            str(build / "boop_runtime.c"),
            str(build / "boop_problem.c"),
            str(build / "boop_extension.c"),
            "-lm",
            "-o",
            str(temporary_library),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            message = f"C extension compilation failed:\n{error.stderr}"
            raise RuntimeError(message) from error
        temporary_library.replace(library)
    specification = importlib.util.spec_from_file_location(module_name, library)
    if specification is None or specification.loader is None:
        message = f"could not load compiled extension {library}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module
