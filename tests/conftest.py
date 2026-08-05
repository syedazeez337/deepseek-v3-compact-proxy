import sys
from pathlib import Path

# The compact_v3 package is installed from src/, but the CLI entry points
# (v3_cli.py, complete.py) live at the repo root and are imported directly by
# tests, so the root still needs to be importable.
sys.path.insert(0, str(Path(__file__).parents[1]))
