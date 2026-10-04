#!/usr/bin/env python3
"""Create hierarchy/wordfreq/rows.csv from the Python wordfreq package."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
from pathlib import Path

from wordfreq import top_n_list

from common import stable_id, stratified_half_split, write_meta, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=_paper_path("data/hierarchy"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--raw-n", type=int, default=500_000)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    words = []
    seen: set[str] = set()
    for token in top_n_list(args.language, args.raw_n):
        word = token.strip().lower()
        if word.isascii() and word.isalpha() and word not in seen:
            seen.add(word)
            words.append(word)
            if len(words) == args.rows:
                break
    if len(words) < args.rows:
        raise RuntimeError(f"Only found {len(words)} qualifying words")
    keys = [word[:2] if len(word) >= 2 else word for word in words]
    splits = stratified_half_split(keys, args.seed)
    rows = [
        {"row_idx": i, "id": f"wordfreq_{args.language}_ascii:{stable_id('wordfreq', word, 16)}",
         "text": word, "split": splits[i]}
        for i, word in enumerate(words)
    ]
    out = args.output_root / "wordfreq"
    write_rows(out / "rows.csv", rows, ["row_idx", "id", "text", "split"])
    write_meta(out / "source_meta.json", {
        "source_package": "wordfreq", "language": args.language,
        "raw_n_requested": args.raw_n, "rows_written": len(rows),
        "normalization": "strip().lower(); isascii() and isalpha()",
        "split": "stable two-letter-prefix-stratified half split", "split_seed": args.seed,
    })


if __name__ == "__main__":
    main()
