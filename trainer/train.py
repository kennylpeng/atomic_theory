from project_paths import resource_path as _paper_path, resource_location as _paper_location
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import tqdm
import random

try:
    import yaml
except ImportError as exc:
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None

try:
    from .dataset import build_activation_store
    from .sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
    from .logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
except ImportError:
    from dataset import build_activation_store
    from sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
    from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint

TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"


class _PrefetchingBatchSource:
    """
    Single-batch asynchronous prefetcher around activation_store.next_batch().
    """

    def __init__(self, activation_store, enabled=True, workers=1, depth=1):
        self.activation_store = activation_store
        self.enabled = bool(enabled)
        self._executor = None
        self._futures = deque()
        self._workers = max(1, int(workers))
        self._depth = max(1, int(depth))

        if self.enabled:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._workers)
            for _ in range(self._depth):
                self._futures.append(self._executor.submit(self.activation_store.next_batch))

    def next_batch(self):
        if not self.enabled:
            return self.activation_store.next_batch()

        future = self._futures.popleft()
        batch = future.result()
        self._futures.append(self._executor.submit(self.activation_store.next_batch))
        return batch

    def close(self):
        if self._executor is None:
            return
        while self._futures:
            self._futures.popleft().cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._futures.clear()
        self._executor = None


def _is_gcs_path(path):
    return str(path).startswith("gs://")


def _ensure_gsutil_available():
    if shutil.which("gsutil") is None:
        raise RuntimeError("GCS output requested but `gsutil` is not available in PATH.")


def _sync_dir_to_gcs(local_dir, gcs_dir):
    _ensure_gsutil_available()
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", local_dir, gcs_dir],
        check=True,
    )


def _prepare_output_dirs(output_dir_arg):
    if _is_gcs_path(output_dir_arg):
        local_output_dir = tempfile.mkdtemp(prefix="sae_outputs_")
        return local_output_dir, output_dir_arg.rstrip("/")
    return os.path.abspath(output_dir_arg), None


def _compact_metrics_str(sae, sae_output, cfg):
    parts = []
    if "num_dead_features" in sae_output and "dict_size" in cfg:
        dead_ratio = float(sae_output["num_dead_features"].item()) / float(cfg["dict_size"])
        parts.append(f"d={dead_ratio:.3f}")
    elif hasattr(sae, "num_batches_not_active"):
        threshold = cfg.get("n_batches_to_dead", getattr(sae, "cfg", {}).get("n_batches_to_dead", None))
        if threshold is not None:
            dead_mask = sae.num_batches_not_active > threshold
            dead_ratio = float(dead_mask.float().mean().item())
            parts.append(f"d={dead_ratio:.3f}")
    parts.extend(
        [
            f"lossx100={100.0 * sae_output['loss'].item():.4f}",
            f"l0={sae_output['l0_norm']:.1f}",
            f"l2={sae_output['l2_loss']:.4f}",
            f"l1={sae_output['l1_loss']:.4f}",
        ]
    )
    return " ".join(parts)


def _checkpoint_progress_str(step, num_batches, metrics_str):
    return f"step={step + 1}/{num_batches} {metrics_str}"


def _get_rng_state():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_all"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_rng_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.astype(np.uint8, copy=False)).contiguous()
    if isinstance(value, (bytes, bytearray)):
        return torch.tensor(list(value), dtype=torch.uint8)
    if isinstance(value, (list, tuple)):
        try:
            return torch.tensor(value, dtype=torch.uint8)
        except Exception:
            return None
    return None


def _set_rng_state(state):
    if not state:
        return
    if "python_random" in state:
        try:
            random.setstate(state["python_random"])
        except Exception as exc:
            print(f"[warn] Failed to restore python RNG state: {exc}")
    if "numpy_random" in state:
        try:
            np.random.set_state(state["numpy_random"])
        except Exception as exc:
            print(f"[warn] Failed to restore numpy RNG state: {exc}")
    if "torch_random" in state:
        torch_state = _coerce_rng_tensor(state["torch_random"])
        if torch_state is None:
            print("[warn] Skipping torch RNG restore: unsupported state format.")
        else:
            try:
                torch.set_rng_state(torch_state)
            except Exception as exc:
                print(f"[warn] Failed to restore torch RNG state: {exc}")
    if "torch_cuda_random_all" in state and torch.cuda.is_available():
        raw_cuda_states = state["torch_cuda_random_all"]
        if not isinstance(raw_cuda_states, (list, tuple)):
            raw_cuda_states = [raw_cuda_states]
        cuda_states = []
        for cuda_state in raw_cuda_states:
            coerced = _coerce_rng_tensor(cuda_state)
            if coerced is None:
                print("[warn] Skipping one CUDA RNG state: unsupported format.")
                continue
            cuda_states.append(coerced)
        if cuda_states:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception as exc:
                print(f"[warn] Failed to restore CUDA RNG state: {exc}")


def _resume_if_requested(sae, optimizer, cfg):
    resume_path = cfg.get("resume_checkpoint")
    if not resume_path:
        return 0

    # Resume files are trusted training artifacts and include Python/NumPy RNG state.
    payload = torch.load(_paper_location(resume_path), map_location=cfg.get("device", "cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {resume_path}")

    if "model_state_dict" in payload:
        sae.load_state_dict(payload["model_state_dict"])
    elif "state_dict" in payload:
        sae.load_state_dict(payload["state_dict"])
    else:
        # Raw state_dict checkpoint format
        sae.load_state_dict(payload)

    if cfg.get("resume_optimizer", True):
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    if cfg.get("resume_rng_state", True):
        _set_rng_state(payload.get("rng_state"))

    start_step = int(payload.get("global_step", -1)) + 1
    print(f"Resumed from {resume_path} at step {start_step}")
    return max(0, start_step)

def train_sae(sae, activation_store, model, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    start_step = _resume_if_requested(sae, optimizer, cfg)
    log_freq = int(cfg.get("log_freq", 100))

    wandb_run = init_wandb(cfg)
    
    can_log_perf = (
        model is not None
        and hasattr(activation_store, "get_batch_tokens")
        and hasattr(activation_store, "get_activations")
    )
    save_training_state = bool(cfg.get("save_training_state", True))
    batch_source = _PrefetchingBatchSource(
        activation_store,
        enabled=bool(cfg.get("prefetch_batches", True)),
        workers=int(cfg.get("prefetch_workers", 1)),
        depth=int(cfg.get("prefetch_depth", 1)),
    )

    try:
        for i in range(start_step, num_batches):
            batch = batch_source.next_batch()
            batch = batch.to(dtype=cfg["dtype"])
            sae_output = sae(batch)
            should_log = log_freq > 0 and i % log_freq == 0
            should_checkpoint = i % cfg["checkpoint_freq"] == 0
            if should_log:
                log_wandb(sae_output, i, wandb_run)
            if can_log_perf and i % cfg["perf_log_freq"] == 0:
                log_model_performance(wandb_run, i, model, activation_store, sae)

            loss = sae_output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

            if should_checkpoint:
                metrics_str = _compact_metrics_str(sae, sae_output, cfg)
                print(_checkpoint_progress_str(i, num_batches, metrics_str), flush=True)
                save_checkpoint(
                    wandb_run,
                    sae,
                    cfg,
                    i,
                    optimizer=optimizer,
                    save_training_state=save_training_state,
                    rng_state=_get_rng_state(),
                )
    finally:
        batch_source.close()

    if num_batches > 0:
        final_step = num_batches - 1
        save_checkpoint(
            wandb_run,
            sae,
            cfg,
            final_step,
            optimizer=optimizer,
            save_training_state=save_training_state,
            rng_state=_get_rng_state(),
        )
    

def train_sae_group(saes, activation_store, model, cfgs):
    num_batches = cfgs[0]["num_tokens"] // cfgs[0]["batch_size"]
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) for sae, cfg in zip(saes, cfgs)]
    log_freqs = [int(cfg.get("log_freq", 100)) for cfg in cfgs]

    wandb_run = init_wandb(cfgs[0])

    batch_tokens = activation_store.get_batch_tokens()
    batch_source = _PrefetchingBatchSource(
        activation_store,
        enabled=bool(cfgs[0].get("prefetch_batches", True)),
        workers=int(cfgs[0].get("prefetch_workers", 1)),
        depth=int(cfgs[0].get("prefetch_depth", 1)),
    )

    try:
        for i in range(num_batches):
            batch = batch_source.next_batch()
            batch = batch.to(dtype=cfgs[0]["dtype"])
            counter = 0
            for sae, cfg, optimizer, log_freq in zip(saes, cfgs, optimizers, log_freqs):
                sae_output = sae(batch)
                loss = sae_output["loss"]
                should_log = log_freq > 0 and i % log_freq == 0
                should_checkpoint = i % cfg["checkpoint_freq"] == 0
                if should_log:
                    log_wandb(sae_output, i, wandb_run, index=counter)
                can_log_perf = (
                    model is not None
                    and hasattr(activation_store, "get_batch_tokens")
                    and hasattr(activation_store, "get_activations")
                )
                if can_log_perf and i % cfg["perf_log_freq"] == 0:
                    log_model_performance(wandb_run, i, model, activation_store, sae, index=counter, batch_tokens=batch_tokens)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
                sae.make_decoder_weights_and_grad_unit_norm()
                optimizer.step()
                optimizer.zero_grad()
                if should_checkpoint:
                    metrics_str = _compact_metrics_str(sae, sae_output, cfg)
                    print(_checkpoint_progress_str(i, num_batches, metrics_str), flush=True)
                    save_checkpoint(
                        wandb_run,
                        sae,
                        cfg,
                        i,
                        optimizer=optimizer,
                        save_training_state=bool(cfg.get("save_training_state", True)),
                        rng_state=_get_rng_state(),
                    )
                counter += 1
    finally:
        batch_source.close()
   
    if num_batches > 0:
        final_step = num_batches - 1
        for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
            save_checkpoint(
                wandb_run,
                sae,
                cfg,
                final_step,
                optimizer=optimizer,
                save_training_state=bool(cfg.get("save_training_state", True)),
                rng_state=_get_rng_state(),
            )


def _parse_dtype(dtype_name):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Choose from {sorted(mapping)}")
    return mapping[dtype_name]


def _require_yaml():
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load config files. Install `pyyaml` and retry."
        ) from _YAML_IMPORT_ERROR


def _parse_args():
    parser = argparse.ArgumentParser(description="Train SAE from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--setup-config",
        type=str,
        required=True,
        help=(
            "Path to a machine/user-specific YAML config. Its values are used as "
            "defaults and the experiment config supplied with --config takes precedence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for checkpoints and final artifacts.",
    )
    return parser.parse_args()


def _load_yaml_config(path):
    _require_yaml()
    with open(_paper_location(path), "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config file must parse to a mapping/object.")
    return raw_cfg


def _merge_configs(defaults, overrides):
    """Recursively merge config mappings without mutating either input."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_seed_value(seed_value):
    if isinstance(seed_value, str):
        normalized = seed_value.strip().lower()
        if normalized in {"random", "auto"}:
            return random.SystemRandom().randrange(0, 2**31 - 1)
    return int(seed_value)


def _resolve_runtime_cfg(raw_cfg, output_dir):
    train_cfg = raw_cfg.get("training", {})
    model_cfg = raw_cfg.get("model", {})
    logging_cfg = raw_cfg.get("logging", {})
    data_cfg = raw_cfg.get("data", raw_cfg.get("dataset", {}))

    batch_size = int(train_cfg.get("batch_size", raw_cfg.get("batch_size", 1024)))
    steps = train_cfg.get("steps", raw_cfg.get("steps"))
    num_tokens = train_cfg.get("num_tokens", raw_cfg.get("num_tokens"))
    if num_tokens is None:
        if steps is None:
            raise ValueError("Config must include either `training.num_tokens` or `training.steps`.")
        num_tokens = int(steps) * batch_size

    run_name = logging_cfg.get("name", raw_cfg.get("name"))
    if not run_name:
        run_name = os.path.basename(os.path.abspath(output_dir))

    dtype_name = str(model_cfg.get("dtype", raw_cfg.get("dtype", "bfloat16")))

    seed_value = raw_cfg.get("seed", train_cfg.get("seed", 42))

    cfg = {
        "num_tokens": int(num_tokens),
        "batch_size": batch_size,
        "lr": float(train_cfg.get("lr", raw_cfg.get("lr", 5e-4))),
        "beta1": float(train_cfg.get("beta1", raw_cfg.get("beta1", 0.9))),
        "beta2": float(train_cfg.get("beta2", raw_cfg.get("beta2", 0.99))),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", raw_cfg.get("max_grad_norm", 1.0))),
        "log_freq": int(train_cfg.get("log_freq", raw_cfg.get("log_freq", 100))),
        "perf_log_freq": int(train_cfg.get("perf_log_freq", raw_cfg.get("perf_log_freq", 10**9))),
        "checkpoint_freq": int(train_cfg.get("checkpoint_freq", raw_cfg.get("checkpoint_freq", 10**9))),
        "seed": _resolve_seed_value(seed_value),
        "device": str(raw_cfg.get("device", train_cfg.get("device", "cuda"))),
        "dtype": _parse_dtype(dtype_name),
        "model_type": str(model_cfg.get("type", raw_cfg.get("model_type", "topk"))),
        "dict_size": int(model_cfg.get("dict_size", raw_cfg.get("dict_size", 1024))),
        "top_k": int(model_cfg.get("top_k", raw_cfg.get("top_k", 16))),
        "l1_coeff": float(model_cfg.get("l1_coeff", raw_cfg.get("l1_coeff", 0.0))),
        "n_batches_to_dead": int(
            model_cfg.get("n_batches_to_dead", raw_cfg.get("n_batches_to_dead", 256))
        ),
        "top_k_aux": int(model_cfg.get("top_k_aux", raw_cfg.get("top_k_aux", 32))),
        "aux_penalty": float(model_cfg.get("aux_penalty", raw_cfg.get("aux_penalty", 1 / 32))),
        "input_unit_norm": bool(model_cfg.get("input_unit_norm", raw_cfg.get("input_unit_norm", False))),
        "bandwidth": float(model_cfg.get("bandwidth", raw_cfg.get("bandwidth", 1.0))),
        "wandb_project": logging_cfg.get("wandb_project", raw_cfg.get("wandb_project", "")),
        "wandb_entity": logging_cfg.get("wandb_entity", raw_cfg.get("wandb_entity")),
        "name": run_name,
        "checkpoint_dir": os.path.join(output_dir, "checkpoints"),
        "resume_checkpoint": train_cfg.get("resume_checkpoint", raw_cfg.get("resume_checkpoint")),
        "resume_optimizer": bool(train_cfg.get("resume_optimizer", raw_cfg.get("resume_optimizer", True))),
        "resume_rng_state": bool(train_cfg.get("resume_rng_state", raw_cfg.get("resume_rng_state", True))),
        "save_training_state": bool(
            train_cfg.get("save_training_state", raw_cfg.get("save_training_state", True))
        ),
        "prefetch_batches": bool(
            train_cfg.get("prefetch_batches", raw_cfg.get("prefetch_batches", True))
        ),
        "prefetch_workers": int(
            train_cfg.get("prefetch_workers", raw_cfg.get("prefetch_workers", 1))
        ),
        "prefetch_depth": int(
            train_cfg.get("prefetch_depth", raw_cfg.get("prefetch_depth", 1))
        ),
        "data": data_cfg,
    }
    return cfg


def _build_sae_model(cfg):
    model_map = {
        "topk": TopKSAE,
        "batch_topk": BatchTopKSAE,
        "vanilla": VanillaSAE,
        "jumprelu": JumpReLUSAE,
    }
    model_type = cfg["model_type"].lower()
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {sorted(model_map)}")
    return model_map[model_type](cfg)


def _serialize_cfg_for_json(cfg):
    json_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_cfg[key] = value
        elif isinstance(value, torch.dtype):
            json_cfg[key] = str(value).replace("torch.", "")
        else:
            json_cfg[key] = str(value)
    return json_cfg


def _save_final_model(output_dir, model, cfg, raw_cfg):
    os.makedirs(output_dir, exist_ok=True)

    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, _paper_location(final_model_path))

    runtime_cfg_path = os.path.join(output_dir, "runtime_config.json")
    with open(_paper_location(runtime_cfg_path), "w", encoding="utf-8") as handle:
        json.dump(_serialize_cfg_for_json(cfg), handle, indent=2)

    _require_yaml()
    user_cfg_path = os.path.join(output_dir, "input_config.yaml")
    with open(_paper_location(user_cfg_path), "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_cfg, handle, sort_keys=False)


def main():
    args = _parse_args()
    setup_cfg = _load_yaml_config(args.setup_config)
    experiment_cfg = _load_yaml_config(args.config)
    raw_cfg = _merge_configs(setup_cfg, experiment_cfg)

    output_dir, gcs_output_dir = _prepare_output_dirs(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cfg = _resolve_runtime_cfg(raw_cfg, output_dir)
    if gcs_output_dir is not None:
        cfg["gcs_output_dir"] = gcs_output_dir

    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    print(f"Using seed: {cfg['seed']}")

    store = build_activation_store(cfg)
    cfg["act_size"] = int(store.d_in)
    sae_model = _build_sae_model(cfg)

    print(f"Found {len(store.shards)} shard(s)")
    print(f"Embedding dim: {store.d_in}")
    print(f"Training steps: {cfg['num_tokens'] // cfg['batch_size']}")
    run_error = None
    try:
        train_sae(sae_model, store, model=None, cfg=cfg)
        _save_final_model(output_dir, sae_model, cfg, deepcopy(raw_cfg))
        print(f"Training complete. Artifacts written to: {output_dir}")
    except Exception as exc:
        run_error = exc
        raise
    finally:
        if gcs_output_dir is not None:
            print(f"Syncing outputs to GCS: {gcs_output_dir}")
            try:
                _sync_dir_to_gcs(output_dir, gcs_output_dir)
                print(f"Synced outputs to GCS: {gcs_output_dir}")
            except Exception as sync_exc:
                print(f"[warn] Failed to sync outputs to GCS: {sync_exc}")
                if run_error is None:
                    raise


if __name__ == "__main__":
    main()