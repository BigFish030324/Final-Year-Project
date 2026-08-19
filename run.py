# --------------------
# Cross Platform Launcher
# Prepare the existing .venv, verify its libraries and start main.py on Windows, macOS or Linux.
# --------------------
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MINIMUM_PYTHON = (3, 10)
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIRECTORY = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
APPLICATION_FILE = PROJECT_ROOT / "main.py"
REQUIRED_IMPORTS = (
    "flask",
    "pandas",
    "numpy",
    "sklearn",
    "openpyxl",
    "reportlab",
    "PIL",
)


class LauncherError(RuntimeError):
    """An environment problem that can be explained to the local operator."""


def environment_python() -> Path:
    """Return the virtual-environment interpreter for the current platform."""
    if os.name == "nt":
        return VENV_DIRECTORY / "Scripts" / "python.exe"
    return VENV_DIRECTORY / "bin" / "python"


def require_supported_python() -> None:
    if sys.version_info < MINIMUM_PYTHON:
        current = ".".join(str(part) for part in sys.version_info[:3])
        required = ".".join(str(part) for part in MINIMUM_PYTHON)
        raise LauncherError(
            f"Python {required} or newer is required, but Python {current} is running. "
            "Install Python 3.11 or 3.12, then run this launcher with that Python."
        )


def create_environment(python_path: Path) -> None:
    if python_path.exists():
        return
    print("Creating the local .venv environment...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIRECTORY)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise LauncherError(
            "Python could not create .venv. Ensure the standard venv module "
            "is installed and that this project folder is writable."
        ) from exc
    if not python_path.exists():
        raise LauncherError("The virtual environment was created without a usable Python interpreter.")


def dependencies_available(python_path: Path) -> bool:
    import_statement = ", ".join(REQUIRED_IMPORTS)
    result = subprocess.run(
        [str(python_path), "-c", f"import {import_statement}"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def install_dependencies(python_path: Path) -> None:
    if dependencies_available(python_path):
        return
    if not REQUIREMENTS_FILE.exists():
        raise LauncherError("requirements.txt is missing, so dependencies cannot be prepared.")
    print("Installing or repairing the local analysis dependencies...")
    try:
        subprocess.run(
            [str(python_path), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS_FILE)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise LauncherError(
            "Dependency installation failed. Check the internet/proxy connection and "
            "confirm that the selected Python version is supported by the packages. "
            "If necessary, remove only .venv and run this launcher again."
        ) from exc
    if not dependencies_available(python_path):
        raise LauncherError(
            "Dependencies were installed but cannot be imported. Remove only .venv, "
            "confirm Python 3.10+ is available, and run this launcher again."
        )


def prepare_environment() -> Path:
    require_supported_python()
    python_path = environment_python()
    create_environment(python_path)
    install_dependencies(python_path)
    return python_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and start the local DataComparison website.")
    parser.add_argument("--check", action="store_true", help="Verify the environment without starting Flask.")
    arguments = parser.parse_args()
    try:
        python_path = prepare_environment()
        if arguments.check:
            print(f"DataComparison environment is ready: {python_path}")
            return 0
        print("Starting DataComparison at http://127.0.0.1:8000")
        return subprocess.call([str(python_path), str(APPLICATION_FILE)], cwd=PROJECT_ROOT)
    except LauncherError as exc:
        print(f"DataComparison could not start: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
