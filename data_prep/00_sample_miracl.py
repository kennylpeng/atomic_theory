from project_paths import resource_path as _paper_path, resource_location as _paper_location
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import argparse
import os
import random
from zlib import crc32

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm

DEFAULT_INPUT_DIR = str(_paper_path(get_path("datasets_dir")) / "miracl-corpus")
DEFAULT_OUTPUT_TITLE_DIR = str(_paper_path(get_path("datasets_dir")) / "miracl-corpus-title")
DEFAULT_OUTPUT_TEXT_DIR = str(_paper_path(get_path("datasets_dir")) / "miracl-corpus-text")
DEFAULT_SAMPLE_SIZE = 500_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MIRACL title/text datasets per language: deduplicated titles "
            "and a random text sample."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Root of miracl-corpus downloads (expects config=<lang> subdirs).",
    )
    parser.add_argument(
        "--output-title-dir",
        default=DEFAULT_OUTPUT_TITLE_DIR,
        help="Output root for deduplicated title datasets.",
    )
    parser.add_argument(
        "--output-text-dir",
        default=DEFAULT_OUTPUT_TEXT_DIR,
        help="Output root for sampled text datasets.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of text rows to sample per language.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for text sampling.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language codes to process (default: all config=* found).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-language outputs.",
    )
    return parser.parse_args()


def discover_languages(input_dir: str) -> list[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    languages = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("config="):
            languages.append(name.split("=", 1)[1])
    if not languages:
        raise FileNotFoundError(
            f"No config=<lang> subdirectories found under {input_dir}. "
            "Run 00_download_data.py for miracl/miracl-corpus first."
        )
    return languages


def load_train_split(config_dir: str) -> Dataset:
    loaded = load_from_disk(config_dir)
    if isinstance(loaded, DatasetDict):
        if "train" not in loaded:
            raise ValueError(f"Expected 'train' split in {config_dir}, got {list(loaded.keys())}")
        return loaded["train"]
    return loaded


def dedupe_titles(train_ds: Dataset) -> Dataset:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for row in tqdm(train_ds, desc="dedupe titles", leave=False):
        title = row["title"]
        if title in seen:
            continue
        seen.add(title)
        rows.append({"docid": row["docid"], "title": title})
    return Dataset.from_list(rows)


def sample_texts(train_ds: Dataset, sample_size: int, seed: int) -> Dataset:
    total_rows = len(train_ds)
    num_samples = min(sample_size, total_rows)
    if num_samples < sample_size:
        print(
            f"  Warning: requested {sample_size} texts but only {total_rows} available; "
            f"using all {num_samples}."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(total_rows), num_samples)
    sampled = train_ds.select(indices)
    keep_cols = [col for col in ("docid", "text") if col in sampled.column_names]
    return sampled.select_columns(keep_cols)


def save_split(output_root: str, language: str, dataset: Dataset, overwrite: bool) -> None:
    output_path = os.path.join(output_root, f"config={language}")
    if os.path.exists(output_path) and not overwrite:
        print(f"  Skipping save (exists): {output_path}")
        return

    if os.path.exists(output_path) and overwrite:
        print(f"  Overwriting: {output_path}")

    os.makedirs(output_root, exist_ok=True)
    DatasetDict({"train": dataset}).save_to_disk(output_path)
    print(f"  Saved {len(dataset)} rows to {output_path}")


def process_language(
    language: str,
    input_dir: str,
    output_title_dir: str,
    output_text_dir: str,
    sample_size: int,
    seed: int,
    overwrite: bool,
) -> None:
    config_dir = os.path.join(input_dir, f"config={language}")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Missing language config directory: {config_dir}")

    title_path = os.path.join(output_title_dir, f"config={language}")
    text_path = os.path.join(output_text_dir, f"config={language}")
    if (
        not overwrite
        and os.path.exists(title_path)
        and os.path.exists(text_path)
    ):
        print(f"Skipping {language}: title and text outputs already exist.")
        return

    print(f"Processing {language} from {config_dir}...")
    train_ds = load_train_split(config_dir)
    print(f"  Loaded {len(train_ds)} passages.")

    if overwrite or not os.path.exists(title_path):
        title_ds = dedupe_titles(train_ds)
        print(f"  Unique titles: {len(title_ds)}")
        save_split(output_title_dir, language, title_ds, overwrite=True)
    else:
        print(f"  Skipping title build (exists): {title_path}")

    if overwrite or not os.path.exists(text_path):
        text_ds = sample_texts(train_ds, sample_size=sample_size, seed=seed)
        save_split(output_text_dir, language, text_ds, overwrite=True)
    else:
        print(f"  Skipping text build (exists): {text_path}")


def main() -> None:
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be >= 1")

    languages = args.languages if args.languages else discover_languages(args.input_dir)
    print(f"Languages: {', '.join(languages)}")

    for language in languages:
        lang_seed = args.seed + crc32(language.encode())
        process_language(
            language=language,
            input_dir=args.input_dir,
            output_title_dir=args.output_title_dir,
            output_text_dir=args.output_text_dir,
            sample_size=args.sample_size,
            seed=lang_seed,
            overwrite=args.overwrite,
        )

    print("Done.")


if __name__ == "__main__":
    main()
