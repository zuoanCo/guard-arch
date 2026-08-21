"""Guard Arch: a Claude Code-style agent runtime built on PydanticAI."""

from pathlib import Path

__version__ = "0.1.0"

# src/guard_arch/__init__.py -> parents[2] is the project root (source checkout).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
