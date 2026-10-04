"""JSON and neighbor-file helpers for activation matching."""

from __future__ import annotations

import json

import os

from pathlib import Path

from typing import Any

GEMINI_MODEL = "gemini_m16384_k64"

NEMOTRON_MODEL = "nemotron_m16384_k64"

def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload

def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
