"""Filesystem configuration for hierarchy activation probes."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location

from pathlib import Path


HIERARCHY_DATA_DIR = _paper_path("/resources/hierarchy_data_dir")
GBIF_METADATA_DIR = _paper_path("/resources/gbif_metadata_dir")
PROBE_RESULTS_DIR = HIERARCHY_DATA_DIR / "probe_results"