#!/usr/bin/env python3
"""Run every hierarchy row-table preparation step in order."""
from project_paths import resource_path as _paper_path, resource_location as _paper_location
import subprocess, sys
from pathlib import Path

def main():
    root = _paper_path(__file__).resolve().parent
    for name in ("prepare_geonames.py", "prepare_wordfreq.py", "prepare_gbif.py", "prepare_gbif_probe_metadata.py", "prepare_geonames_questions.py"):
        subprocess.run([sys.executable, str(root / name)], check=True)

if __name__ == "__main__": main()
