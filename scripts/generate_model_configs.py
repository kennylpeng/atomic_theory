#!/usr/bin/env python3
"""Generate the complete portable SAE experiment-config matrix."""

from __future__ import annotations
from project_paths import resource_path as _paper_path, resource_location as _paper_location

import argparse
from pathlib import Path

import yaml

FULL_DATASETS = [
    "emotion",
    "fever",
    "gooaq",
    "hotpotqa",
    "msmarco",
    "natural-questions",
    "nfcorpus",
    "paq_10m",
    "scifact",
    "squad",
    "trivia-qa",
    "WebInstructSub",
    "miracl-corpus-text",
    "miracl-corpus-title",
]

DATASET_VARIANTS = {
    "all": FULL_DATASETS,
    "wikipedia": [
        "fever",
        "hotpotqa",
        "natural-questions",
        "paq_10m",
        "squad",
        "trivia-qa",
    ],
    "no-wikipedia": [
        "emotion",
        "gooaq",
        "msmarco",
        "nfcorpus",
        "scifact",
        "WebInstructSub",
    ],
    "random-1": [
        "emotion",
        "fever",
        "gooaq",
        "natural-questions",
        "paq_10m",
        "scifact",
    ],
    "random-2": [
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "squad",
        "trivia-qa",
        "WebInstructSub",
    ],
}

WIDTHS = {
    512: 32,
    1024: 32,
    2048: 32,
    4096: 32,
    8192: 64,
    16384: 64,
    32768: 64,
    65536: 128,
    131072: 128,
}

FULL_SEEDS = {
    "gemini": {
        512: 1138983768,
        1024: 1643307215,
        2048: 1903409744,
        4096: 296799549,
        8192: 1244342741,
        16384: 707885418,
        32768: 1480725346,
        65536: 1317536270,
        131072: 172912,
    },
    "nemotron": {
        512: 1588734130,
        1024: 515450099,
        2048: 1277871182,
        4096: 578010941,
        8192: 1357330459,
        16384: 1610719394,
        32768: 1923885051,
        65536: 150416361,
        131072: 1728433352,
    },
}

SUBSET_SEEDS = {
    "gemini": {
        "wikipedia": 752675140,
        "no-wikipedia": 65683026,
        "random-1": 1028554591,
        "random-2": 669991270,
    },
    "nemotron": {
        "wikipedia": 1058356814,
        "no-wikipedia": 223132717,
        "random-1": 222067858,
        "random-2": 616594595,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all portable model experiment YAML files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_paper_path("generated_model_configs"),
        help="Destination directory (default: generated_model_configs).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace configs with matching names."
    )
    parser.add_argument(
        "--include-controls", action="store_true",
        help="Also generate the six fixed-k and three alternate-seed paper recipes.",
    )
    return parser.parse_args()


def compact_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def compact_steps(steps: int) -> str:
    return f"{steps // 1_000_000}m" if steps % 1_000_000 == 0 else str(steps)


def experiment_name(
    model: str,
    dataset_variant: str,
    width: int,
    top_k: int,
    aux_penalty: float,
    steps: int,
    seed: int,
) -> str:
    return (
        f"{model}_{dataset_variant}_d{width}_k{top_k}"
        f"_aux{compact_number(aux_penalty)}_steps{compact_steps(steps)}_seed{seed}"
    )


def build_config(
    *,
    name: str,
    datasets: list[str],
    width: int,
    top_k: int,
    aux_penalty: float,
    steps: int,
    seed: int,
) -> dict:
    return {
        "seed": seed,
        "device": "cuda",
        "data": {
            "datasets": datasets,
            "split": None,
            "text_col": None,
            "batches_per_shard": 4,
            "contiguous_sampling": True,
            "shuffle_within_batch": True,
            "staging_mode": "full",
            "min_ready_shards": 32,
            "staging_max_workers": 16,
            "staging_progress_interval": 25,
            "shard_refresh_interval_batches": 200,
        },
        "training": {
            "batch_size": 2048,
            "steps": steps,
            "num_tokens": None,
            "lr": 5e-4,
            "beta1": 0.9,
            "beta2": 0.99,
            "max_grad_norm": 1.0,
            "perf_log_freq": 1_000_000_000,
            "log_freq": 100,
            "checkpoint_freq": 100_000 if steps == 2_000_000 else 50_000,
            "resume_optimizer": True,
            "resume_rng_state": True,
            "save_training_state": True,
            "prefetch_batches": True,
            "prefetch_workers": 2,
            "prefetch_depth": 8,
        },
        "model": {
            "type": "topk",
            "dtype": "float32",
            "dict_size": width,
            "top_k": top_k,
            "l1_coeff": 0.0,
            "n_batches_to_dead": 512,
            "top_k_aux": 32,
            "aux_penalty": aux_penalty,
            "input_unit_norm": False,
            "bandwidth": 1.0,
        },
        "logging": {"name": name, "wandb_project": "saes"},
    }


def experiments():
    for model, seeds in FULL_SEEDS.items():
        for width, top_k in WIDTHS.items():
            is_gemini_largest = model == "gemini" and width == 131072
            steps = 2_000_000 if is_gemini_largest else 1_000_000
            aux_penalty = 1.0 if is_gemini_largest else 0.25
            seed = seeds[width]
            yield model, "all", width, top_k, aux_penalty, steps, seed

        for variant, seed in SUBSET_SEEDS[model].items():
            yield model, variant, 16384, 64, 0.25, 1_000_000, seed


def control_experiments():
    for model in ("gemini", "nemotron"):
        for width in (512, 4096, 32768):
            yield model, "all", width, 128, 0.25, 1_000_000, FULL_SEEDS[model][width]
    for width, seed in ((512, 1226913470), (4096, 1847620082), (32768, 885798999)):
        yield "gemini", "all", width, WIDTHS[width], 0.25, 1_000_000, seed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    recipes = list(experiments())
    if args.include_controls:
        recipes.extend(control_experiments())
    for model, variant, width, top_k, aux_penalty, steps, seed in recipes:
        name = experiment_name(
            model, variant, width, top_k, aux_penalty, steps, seed
        )
        output = args.output_dir / f"{name}.yaml"
        if output.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite {output}; pass --force to replace it.")
        config = build_config(
            name=name,
            datasets=DATASET_VARIANTS[variant],
            width=width,
            top_k=top_k,
            aux_penalty=aux_penalty,
            steps=steps,
            seed=seed,
        )
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        written.append(output)

    print(f"Wrote {len(written)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
