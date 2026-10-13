"""JIT compilation and Python adaptation for generated C solvers."""

import hashlib
import importlib.util
import json
import os
import sysconfig
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast, overload

import setuptools
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

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
    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float], diagnostics: bool = False, luts: Sequence[Any] = ()) -> Any: ...


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
    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float] = (), *, luts: Sequence[Any] = (), diagnostics: Literal[False] = False) -> list[float]: ...

    @overload
    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float] = (), *, luts: Sequence[Any] = (), diagnostics: Literal[True]) -> CompiledSolverResult: ...

    def __call__(self, initial_guess: Sequence[float], parameters: Sequence[float] = (), *, luts: Sequence[Any] = (), diagnostics: bool = False) -> list[float] | CompiledSolverResult:
        """Run the native solver using ordinary Python sequence inputs.

        Each entry in ``luts`` is ``(nodes, values)``. ``nodes`` contains one
        sequence per axis; 2D ``values`` contains rows indexed by the first
        axis. Shapes and slot ordering come from the generated C interface.
        """
        raw = self._native(initial_guess, parameters, diagnostics, luts)
        if not diagnostics:
            return cast(list[float], raw)
        solution, native_diagnostics = raw
        return CompiledSolverResult(cast(list[float], solution), cast(NativeSolverDiagnostics, native_diagnostics))


def create_compiled_solver(nlp: NonlinearProgram, options: SolverOptions | None = None, *, cache_directory: str | Path | None = None, share_subexpressions: bool = True) -> CompiledSolver:
    """Generate, compile, load, and instantiate a native solver extension."""
    code = generate_c_solver(nlp, options, share_subexpressions=share_subexpressions)
    module = _compile_extension(code, cache_directory)
    native_type = cast(Any, module).Solver
    return CompiledSolver(cast(_NativeSolver, native_type()))


def _compile_extension(code: GeneratedCCode, cache_directory: str | Path | None) -> ModuleType:
    extension_template = resources.files("boop").joinpath("c_runtime", "boop_extension.c").read_text()
    key = _extension_key(code, extension_template)
    module_name = f"boop_jit_{key}"
    root = Path(cache_directory) if cache_directory is not None else Path(tempfile.gettempdir()) / "boop-jit"
    build_directory = root / key
    suffix = cast(str, sysconfig.get_config_var("EXT_SUFFIX"))
    library = build_directory / f"{module_name}{suffix}"

    if not library.exists():
        _build_extension(code, extension_template, module_name, build_directory, library)
    return _load_extension(module_name, library)


def _extension_key(code: GeneratedCCode, extension_template: str) -> str:
    """Identify sources, Python ABI, and the selected build configuration."""
    digest = hashlib.sha256()
    for name, content in sorted((*code.headers.items(), *code.sources.items())):
        digest.update(name.encode())
        digest.update(content.encode())
    digest.update(extension_template.encode())
    digest.update(str(sysconfig.get_config_var("SOABI")).encode())
    configuration = {
        "python": sysconfig.get_config_vars(),
        "setuptools": setuptools.__version__,
        "environment": {name: os.environ.get(name) for name in (
            "CC", "CPP", "CXX", "LDSHARED", "CFLAGS", "CPPFLAGS", "LDFLAGS",
            "AR", "ARFLAGS", "PATH", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET",
            "ARCHFLAGS", "CL", "_CL_", "INCLUDE", "LIB", "LIBPATH",
        )},
    }
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    return digest.hexdigest()[:20]


def _build_extension(code: GeneratedCCode, template: str, module_name: str, build: Path, library: Path) -> None:
    """Materialize and compile one content-addressed extension module."""
    build.mkdir(parents=True, exist_ok=True)
    code.write(build)
    extension = f'#define BOOP_MODULE_TOKEN {module_name}\n#define BOOP_MODULE_STRING "{module_name}"\n{template}'
    (build / "boop_extension.c").write_text(extension)
    # Each build gets its own objects and output, including concurrent builds
    # of the same cache entry. Publish only a successfully linked extension.
    with tempfile.TemporaryDirectory(prefix="build-", dir=build) as temporary:
        temporary_build = Path(temporary)
        code.write(temporary_build)
        (temporary_build / "boop_extension.c").write_text(extension)
        extension_module = Extension(
            module_name,
            sources=[str(temporary_build / name) for name in (
                "boop_runtime.c", "boop_problem.c", "boop_extension.c",
            )],
            include_dirs=[str(temporary_build)],
            libraries=[] if os.name == "nt" else ["m"],
        )
        distribution = Distribution({"ext_modules": [extension_module]})
        command = build_ext(distribution)
        command.ensure_finalized()
        command.build_lib = str(temporary_build / "lib")
        command.build_temp = str(temporary_build / "objects")
        command.force = True
        command.run()
        Path(command.get_ext_fullpath(module_name)).replace(library)


def _load_extension(module_name: str, library: Path) -> ModuleType:
    """Load a compiled extension from its content-addressed cache path."""
    specification = importlib.util.spec_from_file_location(module_name, library)
    if specification is None or specification.loader is None:
        message = f"could not load compiled extension {library}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module
