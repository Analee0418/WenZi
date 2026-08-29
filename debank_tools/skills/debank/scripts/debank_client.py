#!/usr/bin/env python3
"""Skill-local wrapper for the DeBank CLI."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "cli" / "debank_client.py"

if __name__ == "__main__":
    runpy.run_path(str(CLI), run_name="__main__")
