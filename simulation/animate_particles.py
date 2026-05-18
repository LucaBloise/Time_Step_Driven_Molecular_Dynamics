#!/usr/bin/env python3
"""Launcher for particle animation from the simulation module.

This wrapper delegates to visualization/animate_particles.py so you can run:
    python simulation/animate_particles.py ...
"""

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "visualization" / "animate_particles.py"

    if not target.exists():
        raise FileNotFoundError(f"Missing script: {target}")

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
