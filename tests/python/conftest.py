"""Shared fixtures for the hce Python test suite.

The suite tests whichever ``hce`` package Python imports: an installed wheel,
or a build tree put on the path with PYTHONPATH=<build>/python (what
``ctest --preset python`` does). Tests marked ``cli`` also need the hce
command-line tool, found through the HCE_CLI environment variable, then the
interpreter's scripts directory (where a wheel installs it), PATH, and
build/release/hce; they are skipped when it is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import Callable, List, Optional

import pytest
from hypothesis import HealthCheck, settings

import hce

REPO = Path(__file__).resolve().parents[2]

# Property tests run a fixed, derandomized set of examples so a failure
# reproduces on every machine; HYPOTHESIS_PROFILE=thorough explores further.
settings.register_profile(
    "ci",
    derandomize=True,
    deadline=None,
    max_examples=60,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("thorough", parent=settings.get_profile("ci"), derandomize=False, max_examples=1000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def pytest_report_header(config: pytest.Config) -> List[str]:
    facts = hce.info()
    cli = find_cli()
    return [
        f"hce {hce.__version__} from {Path(hce.__file__).parent} "
        f"(simd={facts['simd_path']}, aio={facts['aio_backend']})",
        f"hce CLI: {cli or 'not found; cli tests will be skipped'}",
    ]


def find_cli() -> Optional[Path]:
    env = os.environ.get("HCE_CLI")
    if env:
        return Path(env)
    candidates = [Path(sysconfig.get_path("scripts")) / "hce"]
    found = shutil.which("hce")
    if found:
        candidates.append(Path(found))
    candidates.append(REPO / "build" / "release" / "hce")
    return next((c for c in candidates if c.is_file() and os.access(c, os.X_OK)), None)


@pytest.fixture(scope="session")
def cli() -> Path:
    path = find_cli()
    if path is None:
        pytest.skip("the hce command-line tool was not found (set HCE_CLI)")
    if not (path.is_file() and os.access(path, os.X_OK)):
        pytest.fail(f"HCE_CLI={path} is not an executable file")
    return path


@pytest.fixture(scope="session")
def run_cli(cli: Path) -> Callable[..., subprocess.CompletedProcess]:
    """Runs the CLI with bytes on stdin; returns the CompletedProcess (never raises on exit status)."""

    def run(*args: str, stdin: bytes = b"", timeout: float = 300) -> subprocess.CompletedProcess:
        return subprocess.run([str(cli), *args], input=stdin, capture_output=True, timeout=timeout, check=False)

    return run


@pytest.fixture(scope="session")
def cpus() -> int:
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
