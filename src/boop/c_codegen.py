"""C source generation for problem-specific Boop solvers."""

import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Self

from .c_evaluator import emit_evaluator
from .c_linalg import emit_equality_linear_algebra
from .model import NonlinearProgram
from .options import SolverOptions
from .program import GeneratedProgram, generate_program


@dataclass(frozen=True, slots=True)
class GeneratedCCode:
    """Contain a complete, standalone C solver source bundle.

    Attributes
    ----------
    headers
        Header filenames mapped to their contents.
    sources
        C source filenames mapped to their contents.
    """

    headers: dict[str, str]
    sources: dict[str, str]

    def write(self, directory: str | Path) -> Self:
        """Write the source bundle into an existing or new directory."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in self.headers.items():
            (destination / name).write_text(content)
        for name, content in self.sources.items():
            (destination / name).write_text(content)
        return self


def generate_c_solver(nlp: NonlinearProgram, options: SolverOptions | None = None) -> GeneratedCCode:
    """Generate portable C11 sources for an NLP-specific solver."""
    selected_options = options or SolverOptions()
    program = generate_program(nlp)
    runtime = resources.files("boop").joinpath("c_runtime")
    return GeneratedCCode(
        headers={
            "boop_config.h": _config_header(program, selected_options),
            "boop_runtime.h": runtime.joinpath("boop_runtime.h").read_text(),
        },
        sources={
            "boop_problem.c": _problem_source(program),
            "boop_runtime.c": runtime.joinpath("boop_runtime.c").read_text(),
        },
    )


def _config_header(program: GeneratedProgram, options: SolverOptions) -> str:
    nonlinear = int(program.has_nonlinear_equalities)
    return f"""#ifndef BOOP_CONFIG_H
#define BOOP_CONFIG_H
#define BOOP_N {program.nlp.dimension}
#define BOOP_M {program.nlp.equality_dimension}
#define BOOP_P {len(program.nlp.parameters)}
#define BOOP_LUT_COUNT {len(program.nlp.luts)}
#define BOOP_LUT_SHAPES {{{', '.join('{' + ', '.join(map(str, (*lut.shape, 1)[:2])) + '}' for lut in program.nlp.luts) or '{0, 0}'}}}
#define BOOP_LUT_INPUT_SIZE {sum(sum(lut.shape) + math.prod(lut.shape) for lut in program.nlp.luts)}
#define BOOP_LUT_FIELDS {sum((2 if len(lut.shape) == 1 else 4) * math.prod(lut.shape) for lut in program.nlp.luts)}
#define BOOP_LUT_MAX_AXIS {max((n for lut in program.nlp.luts for n in lut.shape), default=1)}
#define BOOP_SQP_ITERATIONS {options.sqp_iterations}
#define BOOP_CG_ITERATIONS {options.cg_iterations}
#define BOOP_NONLINEAR_EQUALITIES {nonlinear}
#define BOOP_INITIAL_TRUST_RADIUS {options.initial_trust_radius:.17g}
#define BOOP_TRUST_EXPAND {options.trust_expand:.17g}
#define BOOP_TRUST_SHRINK {options.trust_shrink:.17g}
#define BOOP_EQUALITY_REGULARIZATION {options.equality_regularization:.17g}
#define BOOP_TANGENTIAL_DAMPING {options.tangential_damping:.17g}
#define BOOP_CURVATURE_FLOOR {options.curvature_floor:.17g}
#define BOOP_FACTORIZATION_TOLERANCE {options.factorization_tolerance:.17g}
#define BOOP_FILTER_BETA {options.filter_beta:.17g}
#define BOOP_FILTER_GAMMA {options.filter_gamma:.17g}
#define BOOP_BOUND_TOLERANCE {options.bound_tolerance:.17g}
#define BOOP_ACTIVE_SET_STATIONARITY_TOLERANCE {options.active_set_stationarity_tolerance:.17g}
#endif
"""


def _problem_source(program: GeneratedProgram) -> str:
    sections = (
        '#include "boop_runtime.h"',
        "#include <math.h>",
        _lut_source(program),
        emit_evaluator(program),
        emit_equality_linear_algebra(program),
    )
    return "\n\n".join(sections) + "\n"


def _lut_source(program: GeneratedProgram) -> str:
    if not program.nlp.luts:
        return ""
    offsets = []
    offset = 0
    for lut in program.nlp.luts:
        offsets.append(str(offset))
        offset += (2 if len(lut.shape) == 1 else 4) * math.prod(lut.shape)
    runtime = resources.files("boop").joinpath("c_runtime", "boop_lut.c").read_text()
    return ("static const int boop_lut_shapes[BOOP_LUT_COUNT][2] = BOOP_LUT_SHAPES;\n"
            f"static const int boop_lut_offsets[BOOP_LUT_COUNT] = {{{', '.join(offsets)}}};\n" + runtime)
