#!/usr/bin/env python3
"""Entry point. Run ``python play.py --help`` for the available commands;
with no arguments the graphical interface opens."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bases.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
