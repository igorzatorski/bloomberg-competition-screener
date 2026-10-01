"""Click Run to run the three-year weekly backtest."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        print("Project .venv is missing. Follow README.md setup instructions.")
        return 1
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + (";" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.call([str(python), "-m", "competition_screener.backtest", *sys.argv[1:]], cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
