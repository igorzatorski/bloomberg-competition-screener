"""One-command Windows setup: create the project venv and install dependencies."""

import subprocess
import sys
import venv
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    venv_dir = root / ".venv"
    python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print("Creating .venv ...")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    print("Installing project dependencies ...")
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "-e", ".[dev]"],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            return completed.returncode
    print("Setup complete. You do not need to activate the virtual environment.")
    print("\nSTEP 0 COMPLETE — now run: python run\\01_download_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
