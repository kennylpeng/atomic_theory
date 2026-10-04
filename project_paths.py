"""Central access to clone-local paths from config/paths.yaml."""
from __future__ import annotations
import argparse
import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "paths.yaml"


@lru_cache(maxsize=8)
def _path_mapping(filename: str):
    return json.loads(Path(filename).read_text())


def resource_path(*parts: Any) -> Path:
    """Resolve resource paths recorded in saved manifests.

    Used by the isolated paper runner. Ordinary use is identical to Path().
    Longest-prefix matching permits overriding one file independently of its
    containing resource root; relative paths retain their ordinary semantics.
    """
    path = Path(*parts)
    remap = os.environ.get("ATOMIC_PATH_REMAP")
    if not remap or not path.is_absolute():
        return path
    mapping = _path_mapping(remap)
    value = str(path)
    # Consider destination roots as identity mappings. The most specific root
    # wins, so an unrelated broad destination cannot hide a model-file override.
    effective = {destination: destination for destination in mapping.values()}
    effective.update(mapping)
    for source in sorted(effective, key=len, reverse=True):
        if value == source or value.startswith(source.rstrip("/") + "/"):
            return Path(effective[source]) / value[len(source):].lstrip("/")
    return path


def resource_location(value: Any) -> Any:
    """Resolve a filename for file APIs while preserving handles/descriptors."""
    return resource_path(value) if isinstance(value, (str, os.PathLike)) else value


def load_paths(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or os.environ.get("ATOMIC_THEORY_PATHS", DEFAULT_CONFIG))
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Path config must be a mapping: {path}")
    return {str(k): v for k, v in data.items()}

def get_path(key: str, config_path: str | Path | None = None) -> str:
    data = load_paths(config_path)
    if key not in data:
        raise KeyError(f"Missing path key {key!r} in path config")
    return str(data[key])

def expand(value: Any, paths: dict[str, Any] | None = None) -> Any:
    paths = paths or load_paths()
    if isinstance(value, str):
        for key, replacement in paths.items():
            value = value.replace("${" + key + "}", str(replacement))
        return value
    if isinstance(value, list):
        return [expand(v, paths) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, paths) for k, v in value.items()}
    return value

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("--config")
    args = parser.parse_args()
    print(get_path(args.key, args.config))

if __name__ == "__main__":
    main()
