from project_paths import resource_path as _paper_path, resource_location as _paper_location
import sys
from pathlib import Path
sys.path.insert(0, str(_paper_path(__file__).resolve().parents[1]))
from project_paths import get_path

import json
import os
from datetime import datetime, timezone

from datasets import DatasetDict, get_dataset_config_names, load_dataset

DATASET_DIR = get_path("datasets_dir")
STATE_FILE = "dataset_download_state.json"

hf_datasets = [
    "mteb/emotion",
    "BeIR/fever",
    "sentence-transformers/gooaq",
    "mteb/hotpotqa",
    "TIGER-Lab/WebInstructSub",
    "mteb/msmarco",
    "mteb/nfcorpus",
    "sentence-transformers/natural-questions",
    "sentence-transformers/paq",
    "sentence-transformers/squad",
    "mteb/scifact",
    "sentence-transformers/trivia-qa",
    "miracl/miracl-corpus",
]

# miracl/miracl-corpus still ships a legacy loading script (unsupported in datasets>=4).
# Load the published jsonl.gz shards directly instead (see miracl-corpus.py on the Hub).
MIRACL_CORPUS_LANG_FILES = {
    "ar": 5,
    "bn": 1,
    "de": 32,
    "en": 66,
    "es": 21,
    "fa": 5,
    "fi": 4,
    "fr": 30,
    "hi": 2,
    "id": 3,
    "ja": 14,
    "ko": 3,
    "ru": 20,
    "sw": 1,
    "te": 2,
    "th": 2,
    "yo": 1,
    "zh": 10,
}


def _safe_name(value):
    return value.replace("/", "__")


def _dataset_root_path(hf_dataset, dataset_dir=DATASET_DIR):
    return os.path.join(dataset_dir, hf_dataset.split("/")[-1])


def _dataset_output_path(hf_dataset, config_name=None, use_config_subdir=False, dataset_dir=DATASET_DIR):
    root = _dataset_root_path(hf_dataset, dataset_dir)
    if use_config_subdir:
        if config_name is None:
            config_name = "default"
        return os.path.join(root, f"config={_safe_name(config_name)}")
    return root


def _state_key(hf_dataset, config_name=None):
    if config_name is None:
        return hf_dataset
    return f"{hf_dataset}::{config_name}"


def _state_path(dataset_dir=DATASET_DIR):
    return os.path.join(dataset_dir, STATE_FILE)


def load_state(dataset_dir=DATASET_DIR):
    path = _state_path(dataset_dir)
    if not os.path.exists(path):
        return {}
    with open(_paper_location(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state, dataset_dir=DATASET_DIR):
    os.makedirs(dataset_dir, exist_ok=True)
    path = _state_path(dataset_dir)
    with open(_paper_location(path), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def mark_status(
    state,
    hf_dataset,
    status,
    config_name=None,
    use_config_subdir=False,
    dataset_dir=DATASET_DIR,
    error=None,
):
    key = _state_key(hf_dataset, config_name)
    state[key] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "path": _dataset_output_path(hf_dataset, config_name, use_config_subdir, dataset_dir),
        "config_name": config_name,
    }
    if error:
        state[key]["error"] = error
    save_state(state, dataset_dir)


def is_downloaded(state, hf_dataset, config_name=None, use_config_subdir=False, dataset_dir=DATASET_DIR):
    key = _state_key(hf_dataset, config_name)
    output_path = _dataset_output_path(hf_dataset, config_name, use_config_subdir, dataset_dir)
    if state.get(key, {}).get("status") == "completed" and os.path.exists(output_path):
        return True
    # If state file is missing/stale but data exists, treat it as completed.
    return os.path.exists(output_path)


def load_miracl_corpus(config_name):
    if config_name not in MIRACL_CORPUS_LANG_FILES:
        raise ValueError(
            f"Unknown miracl/miracl-corpus language config {config_name!r}. "
            f"Expected one of: {sorted(MIRACL_CORPUS_LANG_FILES)}"
        )
    num_shards = MIRACL_CORPUS_LANG_FILES[config_name]
    data_files = [
        f"hf://datasets/miracl/miracl-corpus/miracl-corpus-v1.0-{config_name}/docs-{i}.jsonl.gz"
        for i in range(num_shards)
    ]
    train = load_dataset("json", data_files=data_files, split="train")
    return DatasetDict({"train": train})


def load_hf_dataset(hf_dataset, config_name=None):
    if hf_dataset == "miracl/miracl-corpus":
        if config_name is None:
            raise ValueError("miracl/miracl-corpus requires a language config (e.g. 'en').")
        return load_miracl_corpus(config_name)
    if config_name is None:
        return load_dataset(hf_dataset)
    return load_dataset(hf_dataset, name=config_name)


def get_configs(hf_dataset):
    if hf_dataset == "miracl/miracl-corpus":
        return list(MIRACL_CORPUS_LANG_FILES)
    try:
        config_names = get_dataset_config_names(hf_dataset)
    except Exception as exc:
        print(f"Could not resolve configs for {hf_dataset}: {exc}. Falling back to default config.")
        return [None]

    if not config_names:
        return [None]
    if len(config_names) == 1 and config_names[0] == "default":
        return [None]
    return config_names


def download_hf_dataset_config(
    hf_dataset,
    config_name,
    state,
    use_config_subdir=False,
    dataset_dir=DATASET_DIR,
):
    os.makedirs(dataset_dir, exist_ok=True)
    config_label = config_name if config_name is not None else "default"
    if is_downloaded(state, hf_dataset, config_name, use_config_subdir, dataset_dir):
        print(f"Skipping {hf_dataset} [{config_label}]: already downloaded.")
        mark_status(
            state,
            hf_dataset,
            "completed",
            config_name,
            use_config_subdir,
            dataset_dir,
        )
        return

    mark_status(state, hf_dataset, "in_progress", config_name, use_config_subdir, dataset_dir)
    print(f"Downloading {hf_dataset} [{config_label}]...")
    try:
        ds = load_hf_dataset(hf_dataset, config_name)
        save_path = _dataset_output_path(hf_dataset, config_name, use_config_subdir, dataset_dir)
        print(f"Saving {hf_dataset} [{config_label}] to {save_path}...")
        ds.save_to_disk(save_path)
        mark_status(state, hf_dataset, "completed", config_name, use_config_subdir, dataset_dir)
        print(f"Done downloading {hf_dataset} [{config_label}].")
    except Exception as exc:
        mark_status(state, hf_dataset, "failed", config_name, use_config_subdir, dataset_dir, error=str(exc))
        print(f"Failed downloading {hf_dataset} [{config_label}]: {exc}")
        raise


state = load_state(DATASET_DIR)
for hf_dataset in hf_datasets:
    configs = get_configs(hf_dataset)
    use_config_subdir = len(configs) > 1
    for config_name in configs:
        download_hf_dataset_config(hf_dataset, config_name, state, use_config_subdir, DATASET_DIR)