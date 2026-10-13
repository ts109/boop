"""Check the JIT's standard compiler configuration and cache behavior."""

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest
import sympy

from boop import NonlinearProgram, create_compiled_solver


@pytest.mark.skipif(os.name == "nt", reason="Exercises Unix compiler environment conventions")
def test_jit_honors_build_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Build through a compiler wrapper, then rebuild with different flags."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("Requires a C compiler")
    log = tmp_path / "commands.jsonl"
    wrapper = tmp_path / "compiler wrapper.py"
    wrapper.write_text(
        "import json, subprocess, sys\n"
        f"with open({str(log)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"sys.exit(subprocess.call([{compiler!r}, *sys.argv[1:]]))\n"
    )
    includes = tmp_path / "include directory"
    includes.mkdir()
    (includes / "probe.h").write_text(
        "#ifndef BOOP_FLAG_PROBE\n#error CFLAGS was ignored\n#endif\n"
        "#if defined(__OPTIMIZE__)\n#error Optimization override was ignored\n#endif\n"
    )
    monkeypatch.setenv("CC", f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))}")
    monkeypatch.delenv("LDSHARED", raising=False)
    monkeypatch.setenv("CFLAGS", "-O0 -g -DBOOP_FLAG_PROBE")
    monkeypatch.setenv("CPPFLAGS", f"-I{shlex.quote(str(includes))} -include probe.h")
    monkeypatch.setenv("LDFLAGS", f"-L{shlex.quote(str(includes))}")
    x = sympy.Symbol("x")
    problem = NonlinearProgram(x=(x,), f=(x - 1) ** 2)
    cache = tmp_path / "cache"
    assert create_compiled_solver(problem, cache_directory=cache)([0.0]) == pytest.approx([1.0])
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert any("-c" in command and "-O0" in command for command in commands)
    assert any("-c" not in command and f"-L{includes}" in command for command in commands)
    # An identical configuration reuses the binary without invoking the wrapper.
    create_compiled_solver(problem, cache_directory=cache)
    assert len(log.read_text().splitlines()) == len(commands)
    monkeypatch.setenv("CFLAGS", "-O0 -g -DBOOP_FLAG_PROBE -DBOOP_SECOND_BUILD")
    create_compiled_solver(problem, cache_directory=cache)
    assert len(list(cache.iterdir())) == 2
    assert len(log.read_text().splitlines()) > len(commands)
