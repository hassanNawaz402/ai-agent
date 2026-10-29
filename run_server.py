#!/usr/bin/env python3
"""Bootstrap the conda env and launch the Advantage Removals LLM API server.

First run:  creates `dev_advantage_removals` (or `prod_advantage_removals`),
            installs the platform-appropriate requirements file, then re-execs
            inside that env to start the HTTP server.

Subsequent runs: detects that it is already inside the target env and
                 launches the server directly.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "llm_configs.yml"
MAC_REQS = PROJECT_ROOT / "requirements-mac.txt"
LINUX_REQS = PROJECT_ROOT / "requirements-linux.txt"
PYTHON_VERSION = "3.11"


def read_active_env() -> str:
    """Stdlib-only scan for env.active so we can run before PyYAML is installed."""
    text = CONFIG_PATH.read_text()
    match = re.search(
        r'^\s*active:\s*["\']?(dev|prod)["\']?\s*$', text, re.MULTILINE
    )
    if not match:
        raise RuntimeError(
            "Could not find env.active (dev|prod) in llm_configs.yml"
        )
    return match.group(1)


def conda_env_name(active_env: str) -> str:
    return f"{active_env}_advantage_removals"


def find_conda() -> str:
    exe = shutil.which("conda")
    if exe:
        return exe
    for candidate in [
        Path.home() / "miniforge3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path("/opt/homebrew/Caskroom/miniforge/base/bin/conda"),
        Path("/opt/miniconda3/bin/conda"),
    ]:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "conda not found. Install Miniforge on Apple Silicon: "
        "https://github.com/conda-forge/miniforge"
    )


def conda_env_exists(conda: str, name: str) -> bool:
    out = subprocess.run(
        [conda, "env", "list"], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts and parts[0] == name:
            return True
    return False


def create_conda_env(conda: str, name: str) -> None:
    print(f"[bootstrap] creating conda env '{name}' (python={PYTHON_VERSION})")
    subprocess.run(
        [conda, "create", "-n", name, f"python={PYTHON_VERSION}", "-y"],
        check=True,
    )


def requirements_file() -> Path:
    system = platform.system()
    if system == "Darwin":
        return MAC_REQS
    if system == "Linux":
        return LINUX_REQS
    raise RuntimeError(
        f"unsupported platform: {system}. Only macOS and Linux are supported."
    )


def pip_install_requirements(conda: str, name: str) -> None:
    reqs = requirements_file()
    print(f"[bootstrap] installing deps from {reqs.name} into '{name}'")
    subprocess.run(
        [conda, "run", "-n", name, "pip", "install", "-r", str(reqs)],
        check=True,
    )


def env_python_path(conda: str, name: str) -> str:
    out = subprocess.run(
        [
            conda, "run", "-n", name, "python", "-c",
            "import sys; print(sys.executable)",
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out


def currently_in_env(name: str) -> bool:
    return os.environ.get("CONDA_DEFAULT_ENV") == name


def launch_server(active_env: str) -> None:
    # Reached only when running inside the target conda env.
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass

    from app.config import load_config  # noqa: E402

    config = load_config()
    missing = config.check_required_env_vars()
    if missing:
        raise RuntimeError(
            "missing required env vars for active profile: "
            + ", ".join(missing)
            + ". Set them (e.g. in .env) and retry."
        )

    bin_dir = Path(sys.executable).parent
    host, port = config.server.host, config.server.port

    if active_env == "prod":
        cmd = [
            str(bin_dir / "gunicorn"), "app.main:app",
            "-k", "uvicorn.workers.UvicornWorker",
            "-w", "2",
            "--bind", f"{host}:{port}",
            "--access-logfile", "-",
        ]
    else:
        cmd = [
            str(bin_dir / "uvicorn"), "app.main:app",
            "--host", host,
            "--port", str(port),
            "--reload",
        ]

    print(f"[server] launching: {' '.join(cmd)}")
    os.execv(cmd[0], cmd)


def bootstrap_and_relaunch(
    active_env: str, args: argparse.Namespace
) -> None:
    name = conda_env_name(active_env)
    conda = find_conda()

    if not conda_env_exists(conda, name):
        create_conda_env(conda, name)
    if not args.skip_install:
        pip_install_requirements(conda, name)

    python = env_python_path(conda, name)
    print(f"[bootstrap] re-executing inside '{name}' via {python}")
    os.execv(
        python,
        [python, str(Path(__file__).resolve()), "--skip-bootstrap", *sys.argv[1:]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap and run the Advantage Removals LLM API."
    )
    parser.add_argument(
        "--skip-bootstrap", action="store_true",
        help="Skip conda env setup; launch the server directly.",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip pip install but still check/create the conda env.",
    )
    args = parser.parse_args()

    active_env = read_active_env()
    name = conda_env_name(active_env)

    if args.skip_bootstrap or currently_in_env(name):
        launch_server(active_env)
    else:
        bootstrap_and_relaunch(active_env, args)


if __name__ == "__main__":
    main()
