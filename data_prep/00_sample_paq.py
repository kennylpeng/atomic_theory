from project_paths import resource_path as _paper_path, resource_location as _paper_location
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import argparse
import os

from datasets import DatasetDict, load_from_disk

DEFAULT_INPUT_DIR = str(_paper_path(get_path("datasets_dir")) / "paq")
DEFAULT_OUTPUT_DIR = str(_paper_path(get_path("datasets_dir")) / "paq_10m")
DEFAULT_SAMPLE_SIZE = 10_000_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a fixed number of rows from the saved PAQ dataset."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Path to PAQ dataset saved with datasets.save_to_disk.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Destination path for sampled PAQ dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to sample from the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1")

    if os.path.exists(args.output_dir) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Use --overwrite to replace it."
        )

    loaded = load_from_disk(args.input_dir)
    if isinstance(loaded, DatasetDict):
        if "train" not in loaded:
            raise ValueError(f"Expected 'train' split in {args.input_dir}, got {list(loaded.keys())}")
        train_ds = loaded["train"]
    else:
        train_ds = loaded

    total_rows = len(train_ds)
    if total_rows < args.sample_size:
        raise ValueError(
            f"Requested sample_size={args.sample_size}, but dataset has only {total_rows} rows."
        )

    print(f"Sampling {args.sample_size} rows from {total_rows} total rows (seed={args.seed})...")
    sampled_train = train_ds.shuffle(seed=args.seed).select(range(args.sample_size))
    sampled = DatasetDict({"train": sampled_train})

    if os.path.exists(args.output_dir) and args.overwrite:
        print(f"Overwriting existing output: {args.output_dir}")

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    sampled.save_to_disk(args.output_dir)
    print(f"Saved sampled dataset to: {args.output_dir}")
    print(f"Sampled train rows: {len(sampled['train'])}")


if __name__ == "__main__":
    main()
