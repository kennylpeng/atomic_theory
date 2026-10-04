from project_paths import resource_path as _paper_path, resource_location as _paper_location
import wandb
import torch
import gc
from functools import partial
import os
import json
import shutil
import subprocess
from typing import Optional


_GPU_LOG_INTERVAL = int(os.environ.get("SAE_GPU_LOG_INTERVAL", "10"))
_GPU_LOG_WARNED = False

def _query_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return {}

    query_fields = [
        "index",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + chr(44).join(query_fields),
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return {}

    metrics = {}
    for row_num, raw_line in enumerate(result.stdout.splitlines()):
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) != len(query_fields):
            continue
        try:
            gpu_index = int(values[0])
            utilization = float(values[1])
            memory_used = float(values[2])
            memory_total = float(values[3])
            power_draw = float(values[4])
        except ValueError:
            continue

        prefix = f"gpu/{gpu_index}"
        metrics[f"{prefix}/utilization_percent"] = utilization
        metrics[f"{prefix}/memory_used_mb"] = memory_used
        metrics[f"{prefix}/memory_total_mb"] = memory_total
        metrics[f"{prefix}/memory_utilization_percent"] = 100.0 * memory_used / memory_total if memory_total else 0.0
        metrics[f"{prefix}/power_watts"] = power_draw
        if row_num == 0:
            metrics["gpu/utilization_percent"] = utilization
            metrics["gpu/memory_used_mb"] = memory_used
            metrics["gpu/memory_total_mb"] = memory_total
            metrics["gpu/memory_utilization_percent"] = metrics[f"{prefix}/memory_utilization_percent"]
            metrics["gpu/power_watts"] = power_draw
    return metrics

def get_gpu_stats_for_step(step):
    global _GPU_LOG_WARNED
    if _GPU_LOG_INTERVAL <= 0 or step % _GPU_LOG_INTERVAL != 0:
        return {}
    metrics = _query_nvidia_smi()
    if not metrics and not _GPU_LOG_WARNED:
        print("[warn] GPU stats unavailable from nvidia-smi; W&B will only have built-in system metrics.")
        _GPU_LOG_WARNED = True
    return metrics

def init_wandb(cfg):
    project = cfg.get("wandb_project")
    if not project:
        return None

    run_name = cfg.get("name", "sae-run")
    entity = cfg.get("wandb_entity")
    try:
        return wandb.init(project=project, entity=entity, name=run_name, config=cfg, reinit=True)
    except Exception as exc:
        print(f"[warn] W&B init failed, continuing without W&B: {exc}")
        return None

def log_wandb(output, step, wandb_run, index=None):
    if wandb_run is None:
        return

    metrics_to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    log_dict.update(get_gpu_stats_for_step(step))
    wandb_run.log(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if wandb_run is None:
        return

    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[:sae.cfg["batch_size"] // sae.cfg["seq_len"]]
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.cfg["act_size"])

    sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(
    wandb_run,
    sae,
    cfg,
    step,
    optimizer: Optional[torch.optim.Optimizer] = None,
    save_training_state: bool = False,
    rng_state: Optional[dict] = None,
):
    run_name = cfg.get("name", "sae-run")
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    save_dir = os.path.join(checkpoint_dir, f"{run_name}_{step}")
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), _paper_location(sae_path))

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(_paper_location(config_path), "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    training_state_path = None
    if save_training_state:
        optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        training_state = {
            "model_state_dict": sae.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "global_step": int(step),
            "config": cfg,
            "rng_state": rng_state,
        }
        training_state_path = os.path.join(save_dir, "training_state.pt")
        torch.save(training_state, _paper_location(training_state_path))
        del training_state, optimizer_state_dict

    gcs_output_dir = cfg.get("gcs_output_dir")
    log_checkpoint_artifacts = bool(
        cfg.get("log_checkpoint_artifacts_to_wandb", not bool(gcs_output_dir))
    )

    # Create and log artifact
    if wandb_run is not None and log_checkpoint_artifacts:
        artifact = wandb.Artifact(
            name=f"{run_name}_{step}",
            type="model",
            description=f"Model checkpoint at step {step}",
        )
        artifact.add_file(sae_path)
        artifact.add_file(config_path)
        if training_state_path is not None:
            artifact.add_file(training_state_path)
        wandb_run.log_artifact(artifact)

    uploaded_to_gcs = False
    if gcs_output_dir:
        gcs_checkpoint_dir = f"{str(gcs_output_dir).rstrip('/')}/checkpoints/{run_name}_{step}"
        if shutil.which("gsutil") is None:
            print("[warn] Skipping checkpoint GCS upload: `gsutil` not found in PATH.")
        else:
            try:
                subprocess.run(
                    ["gsutil", "-m", "rsync", "-r", save_dir, gcs_checkpoint_dir],
                    check=True,
                )
                uploaded_to_gcs = True
                print(f"Checkpoint uploaded to GCS: {gcs_checkpoint_dir}")
            except Exception as exc:
                print(f"[warn] Failed to upload checkpoint to GCS: {exc}")

    print(f"Checkpoint saved at step {step}: {save_dir}")
    if uploaded_to_gcs:
        try:
            shutil.rmtree(save_dir)
            print(f"Deleted local checkpoint after GCS upload: {save_dir}")
        except Exception as exc:
            print(f"[warn] Failed to delete local checkpoint after GCS upload: {exc}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
