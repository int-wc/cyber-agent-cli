from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import subprocess

from . import __version__


@lru_cache(maxsize=1)
def get_build_revision() -> str | None:
    """Return the short source revision when available."""
    env_revision = os.getenv("CYBER_AGENT_BUILD_SHA", "").strip()
    if env_revision:
        return env_revision[:12]

    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision or None


def get_version_display() -> str:
    """Return the user-facing version string."""
    revision = get_build_revision()
    if revision:
        return f"{__version__} ({revision})"
    return __version__
